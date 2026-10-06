"""Trial registry + selection-aware inference (Module 5's honesty machinery).

The centerpiece is the factor-zoo validation: the best of N pure-noise
strategies looks significant to the naive PSR and must be killed by the
deflated Sharpe — while a genuinely skilled strategy survives its own
selection multiplicity.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest
import torch

from nec_moe import (
    TrialRegistry,
    benjamini_hochberg,
    bonferroni,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    ic_pvalue,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from nec_moe.evaluation import ic_summary
from nec_moe.registry import PROVENANCE_KEYS

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: every registry row must carry its provenance (brief 06 A.6)
PROV = dict.fromkeys(PROVENANCE_KEYS) | {"data_source": "synthetic"}


def test_registry_roundtrip_and_persistence(tmp_path: Path):
    path = tmp_path / "trials.jsonl"
    reg = TrialRegistry(path)
    reg.log("sweep", {"ic": 0.02, "sharpe": 0.11}, config={"prior": "soft", **PROV}, seed=0)
    reg.log("sweep", {"ic": -0.01, "sharpe": -0.05}, config=PROV, seed=1)
    reg.log("other", {"ic": 0.5}, config=PROV)
    # reopen: the registry is the file, not the object
    reg2 = TrialRegistry(path)
    assert reg2.n_trials() == 3
    assert reg2.n_trials("sweep") == 2
    assert reg2.metric_values("sweep", "sharpe") == [0.11, -0.05]
    assert reg2.trials("sweep")[0].config == {"prior": "soft", **PROV}


def test_registry_best_records_selection_event(tmp_path: Path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    for i, s in enumerate([0.1, 0.3, 0.2]):
        reg.log("sweep", {"sharpe": s}, config=PROV, seed=i)
    pick = reg.best("sweep", "sharpe")
    assert pick.metrics["sharpe"] == pytest.approx(0.3)
    events = reg.selection_events("sweep")
    assert len(events) == 1
    assert events[0].config["n_candidates"] == 3
    assert events[0].config["selected_trial_id"] == pick.trial_id
    # the selection row is a registry entry too, and carries the provenance
    assert {k: events[0].config[k] for k in PROV} == PROV
    # selection events never contaminate the trial family itself
    assert reg.n_trials("sweep") == 3
    # and the reserved suffix cannot be logged directly
    with pytest.raises(ValueError, match="reserved"):
        reg.log("sweep#selection", {"x": 1.0}, config=PROV)


def test_registry_refuses_a_row_without_provenance(tmp_path: Path):
    """Brief 06 A.6: no result may be mistaken for another source's, so a row
    without ``data_source`` or ``post_delisting_return`` is refused."""
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="provenance"):
        reg.log("sweep", {"ic": 0.1})
    with pytest.raises(ValueError, match="post_delisting_return"):
        reg.log("sweep", {"ic": 0.1}, config={"data_source": "synthetic"})
    assert reg.n_trials() == 0


def test_registry_rejects_nonfinite_metrics(tmp_path: Path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="not finite"):
        reg.log("sweep", {"ic": float("nan")}, config=PROV)


# --------------------------------------------------------------------------- #
# PSR / expected-max / DSR
# --------------------------------------------------------------------------- #


def test_psr_at_own_sharpe_is_half():
    g = torch.Generator().manual_seed(0)
    r = 0.001 + 0.01 * torch.randn(500, generator=g)
    sr = sharpe_ratio(r)
    assert probabilistic_sharpe_ratio(r, sr_benchmark=sr) == pytest.approx(0.5)


def test_psr_monotonicity():
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(400, generator=g)
    weak = 0.0005 + 0.01 * noise
    strong = 0.003 + 0.01 * noise
    assert probabilistic_sharpe_ratio(strong) > probabilistic_sharpe_ratio(weak)
    # longer track record, same per-period SR -> more confidence
    long_r = torch.cat([strong, 0.003 + 0.01 * torch.randn(1600, generator=g)])
    assert probabilistic_sharpe_ratio(long_r) > probabilistic_sharpe_ratio(strong)


def test_expected_max_sharpe_properties():
    assert expected_max_sharpe(1, 0.01) == 0.0  # no selection, no hurdle
    assert expected_max_sharpe(10, 0.0) == 0.0  # identical trials, no hurdle
    e10 = expected_max_sharpe(10, 0.01)
    e100 = expected_max_sharpe(100, 0.01)
    assert 0 < e10 < e100  # hurdle grows with the number of trials
    assert expected_max_sharpe(10, 0.04) == pytest.approx(2 * e10)  # sqrt(V) scaling


def test_factor_zoo_deflation():
    """Best of 60 pure-noise strategies: naive PSR endorses it, DSR kills it —
    and a genuinely skilled strategy survives the same multiplicity."""
    g = torch.Generator().manual_seed(7)
    trials = [0.01 * torch.randn(750, generator=g) for _ in range(60)]
    srs = [sharpe_ratio(r) for r in trials]
    best = trials[max(range(60), key=lambda i: srs[i])]
    sr_var = statistics.pvariance(srs)

    naive = probabilistic_sharpe_ratio(best)  # selection ignored
    d = deflated_sharpe_ratio(best, n_trials=60, sr_variance=sr_var)
    assert naive > 0.95, naive  # the lucky winner looks 'significant'
    assert d.dsr < 0.75, d  # ...until its selection multiplicity is priced in
    assert d.expected_max_sr > 0.5 * d.sharpe  # hurdle comparable to the 'edge'

    # true skill of the same magnitude as the noise hurdle survives deflation
    skilled = 0.0012 + 0.01 * torch.randn(750, generator=g)
    d_skill = deflated_sharpe_ratio(skilled, n_trials=60, sr_variance=sr_var)
    assert d_skill.dsr > d.dsr + 0.15, (d_skill.dsr, d.dsr)


def test_moment_guards():
    with pytest.raises(ValueError, match=">= 4"):
        sharpe_ratio(torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError, match="zero variance"):
        sharpe_ratio(torch.ones(10))


# --------------------------------------------------------------------------- #
# p-value families
# --------------------------------------------------------------------------- #


def test_ic_pvalue():
    g = torch.Generator().manual_seed(2)
    ics = torch.full((100,), 0.05) + 0.02 * torch.randn(100, generator=g)
    s = ic_summary(ics)
    p = ic_pvalue(s)
    assert 0.0 <= p < 0.05  # strong, consistent IC -> tiny p
    null = ic_summary(0.02 * torch.randn(100, generator=torch.Generator().manual_seed(3)))
    assert ic_pvalue(null) > 0.05


def test_bonferroni_math():
    reject, adj = bonferroni([0.001, 0.02, 0.4], alpha=0.05)
    assert adj == pytest.approx([0.003, 0.06, 1.0])
    assert reject == [True, False, False]


def test_benjamini_hochberg_classic_example():
    # textbook BH at alpha=0.05: ordered p (.01,.02,.03,.04,.05) vs k/N*alpha
    pvals = [0.01, 0.04, 0.03, 0.005]
    reject, qvals = benjamini_hochberg(pvals, alpha=0.05)
    # ordered: .005 <= .0125, .01 <= .025, .03 <= .0375, .04 <= .05 -> all reject
    assert reject == [True, True, True, True]
    # monotone q-values, each >= its raw p
    assert all(q >= p for q, p in zip(qvals, pvals, strict=True))
    # a family with one clear discovery among junk
    reject2, q2 = benjamini_hochberg([0.001, 0.8, 0.9, 0.7, 0.95], alpha=0.10)
    assert reject2 == [True, False, False, False, False]
    assert q2[0] == pytest.approx(0.005)


def test_bh_no_stricter_than_bonferroni():
    g = torch.Generator().manual_seed(4)
    pvals = torch.rand(40, generator=g).tolist() + [1e-5, 3e-4]
    bon_rej, _ = bonferroni(pvals, alpha=0.05)
    bh_rej, _ = benjamini_hochberg(pvals, alpha=0.05)
    assert all(bh for bon, bh in zip(bon_rej, bh_rej, strict=True) if bon)  # BH ⊇ Bonferroni


def test_pvalue_guards():
    with pytest.raises(ValueError, match="empty"):
        bonferroni([], 0.05)
    with pytest.raises(ValueError, match="outside"):
        benjamini_hochberg([0.5, 1.2], 0.05)


# --------------------------------------------------------------------------- #
# Integration: the thesis-sweep workflow
# --------------------------------------------------------------------------- #


def test_registry_to_corrections_workflow(tmp_path: Path):
    """The intended pattern: log every config of a sweep, derive the p-value
    family from the registry, correct it, and let best() record the pick."""
    g = torch.Generator().manual_seed(9)
    reg = TrialRegistry(tmp_path / "sweep.jsonl")
    families = {}
    for i, true_ic in enumerate([0.06, 0.0, 0.0, 0.0]):  # one real signal, 3 nulls
        ics = true_ic + 0.05 * torch.randn(150, generator=g)
        s = ic_summary(ics)
        reg.log(
            "routing_sweep", {"mean_ic": s.mean_ic, "p": ic_pvalue(s)}, config=PROV, seed=i
        )
        families[i] = ic_pvalue(s)

    pvals = [r.metrics["p"] for r in reg.trials("routing_sweep")]
    assert len(pvals) == reg.n_trials("routing_sweep") == 4
    reject, _ = benjamini_hochberg(pvals, alpha=0.10)
    assert reject[0] and sum(reject) <= 2  # the planted signal survives; junk mostly dies

    pick = reg.best("routing_sweep", "mean_ic")
    assert pick.seed == 0
    assert reg.selection_events("routing_sweep")[0].config["n_candidates"] == 4
