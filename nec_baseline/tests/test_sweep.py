"""Multi-seed sweep runner: the protocol layer over the harness + registry.

The end-to-end test runs a miniature version of the real thesis workflow —
NEC arms and a baseline arm, crossed with seeds, on the planted sign-flip
panel — and checks that the report separates the arms the way the planted
truth demands, that every trial landed in the registry, and that deterministic
baselines show zero seed variance.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from conftest import small_config

from nec_moe import (
    RidgeBaseline,
    SweepArm,
    SyntheticRegimePanel,
    SyntheticSpec,
    TrialRegistry,
    baseline_arm,
    corrected_claims,
    nec_arm,
    run_sweep,
)


def _panel():
    spec = SyntheticSpec(
        vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3
    )
    return SyntheticRegimePanel(spec).generate(n_dates=160, n_entities=8)


def _vol_proxy(p):
    return p.x_seq[:, :, 0].std(dim=1)


def test_run_sweep_end_to_end(tmp_path: Path):
    panel = _panel()
    reg = TrialRegistry(tmp_path / "sweep.jsonl")
    cfg = small_config(sigma_init=1.5, lr=3e-3, batch_size=256)
    arms = [
        nec_arm("soft", cfg, warmstart_key=_vol_proxy),
        nec_arm("uniform", small_config(prior_kind="uniform", sigma_init=1.5,
                                        lr=3e-3, batch_size=256),
                warmstart_key=_vol_proxy),
        baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)),
    ]
    report = run_sweep(
        panel, arms, seeds=(0, 1), registry=reg, tag="mini_sweep",
        steps=200, n_folds=2, test_dates_per_fold=20, purge_dates=5,
        verbose=False,
    )

    # every (arm, seed) run became exactly one registry row with its identity
    assert reg.n_trials("mini_sweep") == 6
    by_arm = {}
    for r in reg.trials("mini_sweep"):
        by_arm.setdefault(r.config["arm"], []).append(r)
    assert {k: len(v) for k, v in by_arm.items()} == {
        "soft": 2, "uniform": 2, "ridge": 2,
    }
    assert "nec_config" in by_arm["soft"][0].config  # reconstructible
    assert {r.seed for r in by_arm["soft"]} == {0, 1}

    # the report table separates the arms as the planted truth demands
    frame = report.to_frame()
    assert list(frame.index) == ["soft", "uniform", "ridge"]
    assert frame.loc["soft", "mean_ic"] > 0.3
    assert abs(frame.loc["uniform", "mean_ic"]) < 0.15
    assert abs(frame.loc["ridge", "mean_ic"]) < 0.15
    assert frame.loc["soft", "nll"] < frame.loc["uniform", "nll"]  # routing helps
    # deterministic baseline: zero seed variance, and that is informative
    assert frame.loc["ridge", "mean_ic_std"] == pytest.approx(0.0, abs=1e-12)
    assert frame.loc["soft", "n_seeds"] == 2
    assert all(math.isfinite(v) for v in frame["median_p"])

    # corrected claim family across ARMS (seeds are replicates)
    claims = corrected_claims(report, alpha=0.10)
    assert set(claims) == {"soft", "uniform", "ridge"}
    assert claims["soft"][0] is True  # the planted signal survives correction
    assert all(0.0 <= q <= 1.0 for _, q in claims.values())


def test_nec_arm_seeding_is_real_and_deterministic():
    cfg = small_config()
    arm = nec_arm("soft", cfg)
    a0 = arm.build_trainer(0).model
    a0_again = arm.build_trainer(0).model
    a1 = arm.build_trainer(1).model
    w = lambda m: m.experts.experts[0].head.weight  # noqa: E731
    assert torch.equal(w(a0), w(a0_again))  # same seed -> identical init
    assert not torch.equal(w(a0), w(a1))  # different seed -> different init
    assert a0.cfg.train.seed == 0 and a1.cfg.train.seed == 1  # planted in config


def test_arm_and_sweep_validation(tmp_path: Path):
    cfg = small_config()
    with pytest.raises(ValueError, match="exactly one"):
        SweepArm(name="bad")
    with pytest.raises(ValueError, match="exactly one"):
        SweepArm(name="bad", build_trainer=lambda s: None,
                 build_baseline=lambda s: None)
    with pytest.raises(ValueError, match="NEC arms only"):
        SweepArm(name="bad", build_baseline=lambda s: RidgeBaseline(),
                 warmstart_key=_vol_proxy)

    panel = _panel()
    reg = TrialRegistry(tmp_path / "t.jsonl")
    dup = [nec_arm("x", cfg), nec_arm("x", cfg)]
    with pytest.raises(ValueError, match="duplicate arm names"):
        run_sweep(panel, dup, seeds=(0,), registry=reg, tag="t",
                  steps=1, n_folds=2, test_dates_per_fold=10, purge_dates=5)
    with pytest.raises(ValueError, match="at least one"):
        run_sweep(panel, [], seeds=(0,), registry=reg, tag="t",
                  steps=1, n_folds=2, test_dates_per_fold=10, purge_dates=5)
