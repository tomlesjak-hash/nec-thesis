"""Reporting added for the first real-data run (brief 04 C.4): one test per claim.

- per-fold wall clock is recorded, split into gate / base / expert training,
  and excluded from equality (it describes the run, not the result);
- the correction's signed distribution and the out-of-sample distance between
  the fitted corrections are reported;
- the fitted gate's means and full transition matrix reach its metrics, in
  canonical order.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import pytest
from conftest import small_base_config, small_config

from nec_moe import (
    BaseCache,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    walk_forward_evaluate,
)
from nec_moe.train import DeadParameterWarning


def _panel():
    return SyntheticRegimePanel(
        SyntheticSpec(regime_process="markov", transition_stay=0.96,
                      vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    ).generate(160, 8)


def _run(cfg, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return walk_forward_evaluate(
            _panel(), lambda: Trainer(NECModel(cfg)), n_folds=2,
            test_dates_per_fold=12, purge_dates=3, steps=10,
            base_cache=BaseCache(), seed=0, **kw,
        )


def _residual_markov_cfg():
    cfg = small_config(
        prior_kind="markov", correction_mode=True, zero_init_head=False,
        base=small_base_config(), sigma_init=1.0, lr=3e-3, batch_size=256,
        freeze_gate=True,
    )
    return dataclasses.replace(
        cfg, markov_gate=dataclasses.replace(
            cfg.markov_gate, series="sequence_channel", series_channel=0,
            start_vol_quantiles=(0.6,), start_vol_windows=(10,),
            start_persistences=(0.95,), start_draws_per_centre=2,
            start_min_history=5, start_min_group_size=5,
        ),
    )


def test_fold_timing_is_recorded_and_split_three_ways():
    result = _run(_residual_markov_cfg())
    for fold in result.folds:
        t = dict(fold.timing)
        assert set(t) == {"gate_fit_s", "base_fit_s", "expert_train_s"}
        assert all(math.isfinite(v) and v >= 0 for v in t.values())
        assert t["gate_fit_s"] > 0 and t["expert_train_s"] > 0


def test_timing_does_not_enter_fold_equality():
    """Two identical runs differ in wall clock and must still compare equal —
    the property the bit-exact resume tests rely on."""
    fold = _run(small_config(sigma_init=1.5, lr=3e-3, batch_size=256)).folds[0]
    slower = dataclasses.replace(fold, timing=(("expert_train_s", 1e9),))
    assert slower == fold


def test_correction_distribution_and_pairwise_distance_are_reported():
    result = _run(_residual_markov_cfg())
    for fold in result.folds:
        c = dict(fold.correction)
        qs = [c[f"correction_q{q:02d}"] for q in (5, 25, 50, 75, 95)]
        assert qs == sorted(qs)  # a signed distribution, in order
        assert c["correction_min_pairwise_distance"] >= 0
        assert c["correction_mean_pairwise_distance"] >= c["correction_min_pairwise_distance"]


def test_gate_means_and_full_transition_reach_the_metrics():
    result = _run(_residual_markov_cfg())
    for fold in result.folds:
        g = dict(fold.gate_metrics)
        assert {"gate_mean_0", "gate_mean_1"} <= set(g)
        for i in range(2):
            row = [g[f"gate_transition_{i}_{j}"] for j in range(2)]
            assert sum(row) == pytest.approx(1.0, abs=1e-9)  # row-stochastic
            assert g[f"gate_stay_prob_{i}"] == g[f"gate_transition_{i}_{i}"]
        assert g["gate_variance_0"] <= g["gate_variance_1"]  # canonical order


def test_non_residual_folds_carry_no_correction_fields():
    """Reported where they mean something, absent (not zero) where they do not."""
    fold = _run(small_config(sigma_init=1.5, lr=3e-3, batch_size=256)).folds[0]
    assert fold.correction == () and fold.gate_metrics == ()


def test_pooled_base_ic_is_the_base_scored_over_the_same_dates():
    """Pooled over every test date of every fold by the same code as the
    mixture's pooled IC; for equal-length folds its mean is the fold mean."""
    result = _run(_residual_markov_cfg())
    pb = result.pooled_base_ic
    assert pb is not None
    assert pb.n_dates == result.pooled_ic.n_dates
    per_fold = [f.base_ic.mean_ic for f in result.folds]
    assert pb.mean_ic == pytest.approx(sum(per_fold) / len(per_fold), abs=1e-12)


def test_no_pooled_base_ic_without_a_base():
    result = _run(small_config(sigma_init=1.5, lr=3e-3, batch_size=256))
    assert result.pooled_base_ic is None
