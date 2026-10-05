"""The gate weight (brief 08 B, decision Q27) and the Q24 expert-weight diagnostic.

``gate_weight_probs`` is checked on pure numbers (no statsmodels fit) with the
brief's example matrix A = [[0.98, 0.02], [0.05, 0.95]]; the Hamilton gate's
plumbing (horizon from the panel, the same rule on training and test dates)
and the registry field on small fitted gates; the diagnostic on a planted
case and through the harness.

One number in the brief is not what its own formula gives: with certain calm
(xi = [1, 0]) and h = 5 the brief states a stress weight of 0.0547133, but
(1/5) sum_{j=1..5} [xi A^j]_stress = 0.0546859. A has eigenvalues 1 and
0.93 and stationary stress share 2/7, so from calm the j-step stress
probability is (2/7)(1 - 0.93^j) exactly; the test pins that closed form. The
brief's stress-from-stress value (0.8632852) agrees with the same closed form,
(2/7) + (5/7) 0.93^j averaged over j = 1..5.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import small_config

from nec_moe import (
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    nec_arm,
    run_sweep,
    trial_provenance,
    walk_forward_evaluate,
)
from nec_moe.diagnostics import expert_weight_diagnostics
from nec_moe.markov_gate import MarkovSwitchingRegimePrior, build_markov_gate, gate_weight_probs
from nec_moe.model import NECModel
from nec_moe.train import DeadParameterWarning

A = np.array([[0.98, 0.02], [0.05, 0.95]])
STRESS, CALM = np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])


def _window_closed_form(start_stress: bool, h: int) -> float:
    """Stress weight of the window average for A, from the spectral form."""
    pi_s, lam = 2.0 / 7.0, 0.93
    js = np.arange(1, h + 1)
    if start_stress:
        return float(np.mean(pi_s + (1.0 - pi_s) * lam**js))
    return float(np.mean(pi_s * (1.0 - lam**js)))


def test_predicted_from_certain_stress():
    assert gate_weight_probs(STRESS, A, "predicted", None)[0, 1] == pytest.approx(0.95, abs=1e-12)


def test_window_from_certain_stress_matches_the_brief():
    w = gate_weight_probs(STRESS, A, "window", 5)
    assert w[0, 1] == pytest.approx(0.8632852, abs=1e-6)
    assert w[0, 1] == pytest.approx(_window_closed_form(True, 5), abs=1e-12)


def test_window_from_certain_calm():
    w = gate_weight_probs(CALM, A, "window", 5)
    assert w[0, 1] == pytest.approx(_window_closed_form(False, 5), abs=1e-12)
    assert w[0, 1] == pytest.approx(0.0546859, abs=1e-6)
    # the brief's 0.0547133 is 2.7e-5 off its own formula (module docstring)
    assert abs(w[0, 1] - 0.0547133) > 1e-6


def test_window_with_one_step_is_predicted_and_with_identity_is_filtered():
    g = np.random.default_rng(0)
    xi = g.dirichlet([1.0, 1.0], size=50)
    assert np.allclose(
        gate_weight_probs(xi, A, "window", 1), gate_weight_probs(xi, A, "predicted", None)
    )
    assert np.allclose(gate_weight_probs(xi, np.eye(2), "window", 5), xi)
    assert np.allclose(gate_weight_probs(xi, A, "filtered", None), xi)


def test_rows_sum_to_one():
    g = np.random.default_rng(1)
    xi = g.dirichlet([1.0, 1.0, 1.0], size=40)
    a3 = g.dirichlet([5.0, 1.0, 1.0], size=3)
    for kind in ("filtered", "predicted", "window"):
        assert np.allclose(gate_weight_probs(xi, a3, kind, 5).sum(axis=1), 1.0)


def test_window_without_a_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        gate_weight_probs(STRESS, A, "window", None)
    with pytest.raises(ValueError, match="gate_weight"):
        gate_weight_probs(STRESS, A, "smoothed", 5)


# --------------------------------------------------------------------------- #
# The Hamilton gate's plumbing
# --------------------------------------------------------------------------- #


def _sticky_panel(n_dates: int = 220, n_entities: int = 6, seed: int = 3):
    return SyntheticRegimePanel(
        SyntheticSpec(
            regime_process="markov", transition_stay=0.96,
            vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=seed,
        )
    ).generate(n_dates, n_entities)


def _markov_cfg(**gate):
    cfg = small_config(prior_kind="markov", sigma_init=1.5, batch_size=256, freeze_gate=True)
    kw = dict(series="sequence_channel", series_channel=0, search_reps=3) | gate
    return dataclasses.replace(cfg, markov_gate=dataclasses.replace(cfg.markov_gate, **kw))


def test_default_is_window_and_config_is_validated():
    cfg = _markov_cfg()
    assert cfg.markov_gate.gate_weight == "window"
    with pytest.raises(ValueError, match="gate_weight"):
        _markov_cfg(gate_weight="smoothed").validate()


def test_the_gate_reads_its_horizon_from_the_panel_and_refuses_a_mismatch():
    panel = _sticky_panel()
    assert panel.horizon == 1  # the synthetic target declares no horizon
    cfg = _markov_cfg()
    gate = build_markov_gate(cfg)
    assert gate.horizon == cfg.data.horizon_periods == 1
    gate.fit(panel)
    assert gate.horizon == 1
    with pytest.raises(ValueError, match="horizon"):
        MarkovSwitchingRegimePrior(2, cfg.markov_gate, horizon=5).fit(panel)


@pytest.mark.parametrize("kind", ["filtered", "predicted", "window"])
def test_training_and_test_dates_get_the_same_kind_of_weight(kind: str):
    """The fitted rows and the causally applied rows are both the gate-weight
    transform of the filter, so serving the train block from a fit equals
    serving it from the extended filter (the filter at t sees through t)."""
    panel = _sticky_panel()
    dates = torch.unique(panel.date, sorted=True)
    train = panel.subset_dates(dates[:150])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate = MarkovSwitchingRegimePrior(2, _markov_cfg(gate_weight=kind).markov_gate)
        gate.fit(train)
        fitted = gate.gate_state()["log_table"].clone()
        gate.apply_causal(panel)
    table = gate.gate_state()["log_table"]
    assert torch.equal(table[: len(fitted)], fitted)
    assert table.shape[0] == len(dates)
    assert torch.allclose(table.exp().sum(dim=1), torch.ones(len(dates)), atol=1e-5)
    # the three kinds really differ on this panel
    if kind != "filtered":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = MarkovSwitchingRegimePrior(2, _markov_cfg(gate_weight="filtered").markov_gate)
            plain.fit(train)
        assert not torch.allclose(plain.gate_state()["log_table"].exp(), fitted.exp(), atol=1e-4)


def test_registry_records_gate_weight(tmp_path: Path):
    panel = _sticky_panel()
    cfg = _markov_cfg()
    assert trial_provenance(panel, cfg)["gate_weight"] == "window"
    assert trial_provenance(panel, small_config())["gate_weight"] is None  # no Hamilton gate
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_sweep(
            panel, [nec_arm("markov", cfg)], seeds=(0,), registry=reg, tag="gw", steps=3,
            n_folds=2, test_dates_per_fold=15, purge_dates=5, verbose=False,
        )
    rows = reg.trials("gw")
    assert rows and all(r.config["gate_weight"] == "window" for r in rows)
    starts = reg.trials(cfg.markov_gate.registry_tag)
    assert starts and all(r.config["gate_weight"] == "window" for r in starts)


# --------------------------------------------------------------------------- #
# The Q24 expert-weight diagnostic
# --------------------------------------------------------------------------- #


def test_diagnostic_is_one_when_responsibilities_equal_the_gate():
    g = torch.Generator().manual_seed(0)
    gate = torch.softmax(torch.randn(400, 2, generator=g), dim=1)
    res = torch.rand(400, generator=g)
    out = expert_weight_diagnostics(gate.clone(), gate, res)
    for k in range(2):
        assert out[f"ew_corr_resp_gate_{k}"] == pytest.approx(1.0, abs=1e-12)
        assert -1.0 <= out[f"ew_corr_resp_absres_{k}"] <= 1.0
        # bin means are increasing when the responsibility IS the gate
        means = [out[f"ew_resp_gate_bin{j}_{k}"] for j in range(5)]
        assert means == sorted(means)


def test_diagnostic_bins_follow_the_configured_edges_and_order():
    gate = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    resp = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.8, 0.2]])
    res = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = expert_weight_diagnostics(resp, gate, res, quantiles=(0.5,))
    # median edge of expert 0's gate prob is 0.25: rows 0, 1 below, 2, 3 above
    assert out["ew_resp_gate_bin0_0"] == pytest.approx(0.25)
    assert out["ew_resp_gate_bin1_0"] == pytest.approx(0.9)
    swapped = expert_weight_diagnostics(resp, gate, res, quantiles=(0.5,),
                                        order=torch.tensor([1, 0]))
    assert swapped["ew_resp_gate_bin0_1"] == out["ew_resp_gate_bin0_0"]


def test_diagnostic_omits_undefined_correlations():
    gate = torch.full((50, 2), 0.5)  # a uniform gate: correlation undefined
    resp = torch.softmax(torch.randn(50, 2), dim=1)
    out = expert_weight_diagnostics(resp, gate, torch.rand(50))
    assert "ew_corr_resp_gate_0" not in out and "ew_corr_resp_absres_0" in out
    assert all(math.isfinite(v) for v in out.values())
    with pytest.raises(ValueError, match="quantiles"):
        expert_weight_diagnostics(resp, gate, torch.rand(50), quantiles=(0.6, 0.4))


def test_diagnostic_reaches_every_fold_and_the_sweep_metrics(tmp_path: Path):
    panel = _sticky_panel()
    cfg = small_config(sigma_init=1.5)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = walk_forward_evaluate(
            panel, make_trainer, n_folds=2, test_dates_per_fold=15, purge_dates=5, steps=5,
        )
    for f in result.folds:
        ew = dict(f.expert_weights)
        assert {"ew_corr_resp_gate_0", "ew_corr_resp_gate_1"} <= set(ew)
        assert all(f"ew_resp_gate_bin{j}_0" in ew for j in range(5))

    reg = TrialRegistry(tmp_path / "t.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(
            panel, [nec_arm("soft", cfg)], seeds=(0,), registry=reg, tag="ew", steps=5,
            n_folds=2, test_dates_per_fold=15, purge_dates=5, verbose=False,
        )
    (row,) = reg.trials("ew")
    assert "ew_corr_resp_gate_0" in row.metrics and "ew_resp_gate_bin4_1" in row.metrics
