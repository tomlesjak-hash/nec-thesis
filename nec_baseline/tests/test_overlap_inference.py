"""Inference on overlapping forward returns (audit finding E-2): one test per claim.

A daily series built on ``h``-period forward returns overlaps itself by ``h-1``
periods, so neighbouring values share most of their return days and are not
independent. Every t-statistic, p-value and probabilistic Sharpe ratio built
on such a series must use a long-run (HAC) variance over h-1 lags, or it
overstates significance. The default kernel is Hansen-Hodrick, which the size
test below shows is close to nominal where Newey-West at the same lag is not.
On the real panel, 5-day targets gave a daily IC with lag-1 autocorrelation of
0.6 to 0.8 and an i.i.d. t-stat 1.6 to 1.8 times too large.
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
    NECModel,
    RidgeBaseline,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    deflated_sharpe_ratio,
    effective_sample_size,
    ic_summary,
    long_run_variance,
    nec_arm,
    probabilistic_sharpe_ratio,
    run_sweep,
    target_horizon,
    walk_forward_evaluate,
    walk_forward_evaluate_baseline,
)
from nec_moe.train import DeadParameterWarning


def _overlapping(n: int, h: int, mean: float, rng: np.random.Generator) -> torch.Tensor:
    """A daily series of overlapping h-period sums: an MA(h-1) process."""
    shocks = rng.standard_normal(n + h - 1)
    x = np.convolve(shocks, np.ones(h), mode="valid") / math.sqrt(h)
    return torch.from_numpy(x + mean)


@pytest.mark.parametrize("kernel", ["uniform", "bartlett"])
def test_long_run_variance_is_the_kernel_formula(kernel: str):
    x = torch.tensor([0.3, -0.1, 0.4, 0.2, -0.5, 0.1, 0.6, -0.2], dtype=torch.float64)
    xc = (x - x.mean()).numpy()
    t, lags = len(xc), 2
    by_hand = xc @ xc / (t - 1)
    for lag in range(1, lags + 1):
        w = 1.0 if kernel == "uniform" else 1 - lag / (lags + 1)
        by_hand += 2 * w * (xc[:-lag] @ xc[lag:]) / (t - 1)
    assert long_run_variance(x, lags, kernel) == pytest.approx(float(by_hand), abs=1e-12)
    # lags = 0 is exactly the unbiased sample variance, whatever the kernel
    assert long_run_variance(x, 0, kernel) == pytest.approx(float(x.var()), abs=1e-12)
    with pytest.raises(ValueError, match="lags"):
        long_run_variance(x, -1, kernel)
    with pytest.raises(ValueError, match="kernel"):
        long_run_variance(x, 1, "parzen")


def test_zero_lags_reproduce_the_iid_t_stat():
    ics = torch.tensor([0.10, 0.20, 0.00, 0.10, 0.05], dtype=torch.float64)
    s = ic_summary(ics)
    assert s.hac_lags == 0
    assert s.t_stat == pytest.approx(s.icir * math.sqrt(5), abs=1e-12)


def test_hac_t_stat_has_honest_size_on_overlapping_series():
    """The claim that matters: under the null (true mean 0), a 5% two-sided test
    on overlapping 5-period series rejects far too often with the i.i.d.
    t-stat; the default Hansen-Hodrick SE at lag h-1 is close to nominal, and
    Newey-West at the same lag sits in between (see HAC_KERNELS)."""
    rng = np.random.default_rng(0)
    h, n, sims = 5, 500, 400
    iid = hh = nw = 0
    for _ in range(sims):
        x = _overlapping(n, h, 0.0, rng)
        iid += abs(ic_summary(x, hac_lags=0).t_stat) > 1.96
        hh += abs(ic_summary(x, hac_lags=h - 1).t_stat) > 1.96
        nw += abs(ic_summary(x, hac_lags=h - 1, hac_kernel="bartlett").t_stat) > 1.96
    assert iid / sims > 0.25, iid / sims  # the i.i.d. test is badly oversized
    assert hh / sims < 0.08, hh / sims  # the default is close to nominal
    assert hh < nw < iid  # and Newey-West at lag h-1 still over-rejects


def test_a_non_positive_uniform_estimate_falls_back_to_bartlett_and_says_so():
    """Hansen-Hodrick is not guaranteed non-negative. On a strongly
    alternating series it goes negative; the summary must then use Newey-West
    and record that it did, rather than divide by a negative variance."""
    rng = np.random.default_rng(2)
    alt = np.where(np.arange(400) % 2 == 0, 1.0, -1.0)
    x = torch.from_numpy(alt + 0.5 * rng.standard_normal(400) + 0.05)
    assert long_run_variance(x, 3, "uniform") < 0
    s = ic_summary(x, hac_lags=3)
    assert s.hac_kernel == "bartlett"
    assert math.isfinite(s.t_stat)
    assert ic_summary(_overlapping(200, 5, 0.0, rng), hac_lags=4).hac_kernel == "uniform"


def test_target_horizon_is_read_from_the_target_name():
    assert target_horizon("fwd_ret_5d") == 5
    assert target_horizon("fwd_resid_ret_20d") == 20
    assert target_horizon("y_synth") is None
    assert target_horizon("fwd_return") is None


def _panel(target: str | None = None):
    panel = SyntheticRegimePanel(
        SyntheticSpec(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    ).generate(120, 6)
    if target is not None:
        panel = dataclasses.replace(
            panel, schema=dataclasses.replace(panel.schema, target=target)
        )
    return panel


@pytest.mark.parametrize(
    ("target", "override", "expected"),
    [("fwd_ret_5d", None, 4), (None, None, 0), ("fwd_ret_5d", 2, 2)],
)
def test_harness_uses_horizon_minus_one_lags_by_default(target, override, expected):
    """walk_forward_evaluate and its baseline twin read h from the panel's
    target: fwd_ret_5d gives 4 lags; a target that declares no horizon gives 0;
    an explicit hac_lags wins."""
    panel = _panel(target)
    common = dict(n_folds=2, test_dates_per_fold=15, purge_dates=5, hac_lags=override)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        nec = walk_forward_evaluate(
            panel, lambda: Trainer(NECModel(small_config(sigma_init=1.5))),
            steps=5, **common,
        )
    base = walk_forward_evaluate_baseline(panel, lambda: RidgeBaseline(), **common)
    for res in (nec, base):
        assert res.pooled_ic.hac_lags == expected
        assert all(f.ic.hac_lags == expected for f in res.folds)


def test_sweep_rows_record_the_lags_their_p_values_rest_on(tmp_path: Path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(
            _panel("fwd_ret_5d"), [nec_arm("soft", small_config(sigma_init=1.5))],
            seeds=(0,), registry=reg, tag="t", steps=5, n_folds=2,
            test_dates_per_fold=15, purge_dates=5, verbose=False,
        )
    assert reg.trials("t")[0].metrics["ic_hac_lags"] == 4.0


def test_psr_and_dsr_count_overlapping_periods_honestly():
    """A daily series of overlapping 5-period returns holds about one
    independent observation per 5 days; the PSR must be less confident about
    it than the i.i.d. formula, and lags = 0 must reproduce the i.i.d. PSR."""
    rng = np.random.default_rng(1)
    x = _overlapping(1000, 5, 0.05, rng)
    t_eff = effective_sample_size(x, 4)
    assert 0.12 * 1000 < t_eff < 0.35 * 1000
    assert effective_sample_size(x, 0) == 1000.0
    # less confident means closer to 0.5, whichever side the sample fell on
    psr_iid = probabilistic_sharpe_ratio(x)
    psr_hac = probabilistic_sharpe_ratio(x, hac_lags=4)
    assert abs(psr_hac - 0.5) < abs(psr_iid - 0.5)
    assert probabilistic_sharpe_ratio(x, hac_lags=0) == psr_iid
    d = deflated_sharpe_ratio(x, n_trials=10, sr_variance=0.01, hac_lags=4)
    d_iid = deflated_sharpe_ratio(x, n_trials=10, sr_variance=0.01)
    assert d.hac_lags == 4 and d.effective_periods == pytest.approx(t_eff)
    assert abs(d.dsr - 0.5) < abs(d_iid.dsr - 0.5)
