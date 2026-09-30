"""Tests that pin bugs found by the code audit of 2026-09-25 (brief 05).

Every test here asserts the CORRECT behaviour. A test whose bug is still open
is marked ``xfail(strict=True)`` with the finding ID from
``Master Thesis/Code_Audit_2026-09-25.md``: the suite stays green while the
bug stays pinned, and when a fix lands the test starts passing, strict mode
turns that into a failure, and the marker has to be removed by hand, so a fix
can never go unnoticed.

Fixed (marker removed, the test now guards the fix): D-1, E-1, M-1.
Still open: B-1, G-2, G-3.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    BaseCache,
    Batch,
    NECModel,
    SpellUniverse,
    StageBSpec,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    assemble_panel,
    base_and_correction,
    filter_point_in_time,
    fit_base,
    market_frame,
    nec_arm,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe.config import DataConfig
from nec_moe.train import DeadParameterWarning


def _sticky_panel(n_dates: int = 160, n_entities: int = 8, seed: int = 3):
    return SyntheticRegimePanel(
        SyntheticSpec(regime_process="markov", transition_stay=0.96,
                      vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=seed)
    ).generate(n_dates, n_entities)


# --------------------------------------------------------------------------- #
# D-1: snapshot ranks are computed before the point-in-time filter
# --------------------------------------------------------------------------- #


def _daily(seed: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    ret = g.normal(0.0004, 0.02, len(dates))
    vol = 1e6 * np.exp(g.normal(0.0, 0.4, len(dates)))
    return pd.DataFrame(
        {"ret": ret, "volume": vol, "dollar_volume": 100.0 * np.exp(np.cumsum(ret)) * vol},
        index=dates,
    )


# D-1, fixed: filter_point_in_time re-ranks a rank-normalized panel among
# each date's members.
def test_D1_point_in_time_ranks_use_only_that_dates_members():
    dates = pd.bdate_range("2019-01-02", periods=330)
    permnos = [str(10001 + i) for i in range(7)]
    daily = {p: _daily(10 + i, dates) for i, p in enumerate(permnos)}
    spec = StageBSpec(seq_len=10, min_names_per_date=5)
    mkt = market_frame(pd.Series(np.random.default_rng(99).normal(0, 0.01, len(dates)),
                                 index=dates), spec)
    panel = assemble_panel(daily, mkt, spec)

    # the last PERMNO joins the index late: before 2020-01-02 it is not a member
    universe = SpellUniverse(pd.DataFrame({
        "entity": permnos,
        "start": pd.to_datetime(["2010-01-04"] * 6 + ["2020-01-02"]),
        "end": pd.to_datetime(["2030-12-31"] * 7),
    }))
    pit = filter_point_in_time(panel, universe)

    # on every date, each snapshot column must be the rank WITHIN that date's
    # members: n values spread exactly over [-0.5, 0.5]
    for d in torch.unique(pit.date):
        block = pit.x_snap[pit.date == d]
        n = block.shape[0]
        ideal = torch.arange(n, dtype=torch.float32) / (n - 1) - 0.5
        got = block.sort(dim=0).values
        assert torch.allclose(got, ideal.unsqueeze(1).expand_as(got), atol=1e-6), (
            pit.date_labels[int(d)]
        )


# --------------------------------------------------------------------------- #
# B-1: an arm's result depends on whether the base was already cached
# --------------------------------------------------------------------------- #


# B-1, fixed: fit_base leaves the global RNG untouched and the harness seeds
# the expert stage explicitly (evaluation.expert_stage_seed).
def test_B1_arm_result_does_not_depend_on_base_cache_state():
    panel = _sticky_panel()
    cfg = small_config(
        prior_kind="uniform", correction_mode=True, zero_init_head=True,
        base=small_base_config(), expert_dropout=0.05, sigma_init=1.0,
        lr=3e-3, batch_size=256, freeze_gate=True,
    )
    arm = nec_arm("uniform", cfg)

    def run(cache: BaseCache):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeadParameterWarning)
            return walk_forward_evaluate(
                panel, lambda: arm.build_trainer(0), n_folds=2,
                test_dates_per_fold=12, purge_dates=3, steps=40,
                base_cache=cache, seed=0,
            )

    cold = run(BaseCache())  # the base is fitted inside this run
    warm_cache = BaseCache()
    run(warm_cache)
    warm = run(warm_cache)  # identical arm, the base comes from the cache
    for a, b in zip(cold.folds, warm.folds, strict=True):
        assert a.base_ic == b.base_ic  # the same base in both runs ...
        assert a.nll == b.nll  # ... so the same result is required


# --------------------------------------------------------------------------- #
# E-1: the NLL "improvement over the base" is not about the correction
# --------------------------------------------------------------------------- #


# E-1, fixed: the base is scored as the same model with every correction
# forced to zero (NECModel.corrections_disabled).
def test_E1_nll_improvement_is_zero_when_every_correction_is_zero():
    panel = _sticky_panel(120, 8)
    cfg = small_config(
        prior_kind="uniform", correction_mode=True, zero_init_head=True,
        base=small_base_config(), freeze_gate=True,
    )
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    trainer.model.attach_base(fit_base(panel, cfg.base, seed=0).model)
    with torch.no_grad():  # experts disagree on noise, corrections stay zero
        trainer.model.experts.log_sigma.copy_(torch.tensor([0.6, 2.4]).log())

    mixture = trainer.evaluate(panel.full_batch())
    _, base_nll, corr = base_and_correction(trainer, panel)
    assert dict(corr)["correction_max_abs"] == 0.0
    assert base_nll is not None
    assert base_nll - mixture == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# M-1: the backprop-HMM prior conditions on targets not yet realised
# --------------------------------------------------------------------------- #


# M-1, fixed: with an h-period target the prior at t is the posterior from
# h dates back carried forward h steps (Trainer._lagged_context).
def test_M1_stateful_prior_ignores_targets_not_yet_realised():
    base = small_config(prior_kind="hmm")
    data = dataclasses.replace(base.data, target="fwd_ret_5d")  # horizon 5
    cfg = dataclasses.replace(base, data=data)
    assert isinstance(cfg.data, DataConfig)
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    panel = SyntheticRegimePanel(SyntheticSpec(regime_process="markov", seed=11)).generate(
        n_dates=10, n_entities=3
    )
    seq = panel.time_sequence()
    ref = trainer.evaluate_sequence(seq)

    t = 8  # the target dated t-1 = 7 covers (7, 12]: unknown at date 8
    moved = list(seq)
    b = seq[t - 1]
    moved[t - 1] = Batch(b.x_seq, b.x_snap, b.y + 5.0, b.regime, b.date)
    alt = trainer.evaluate_sequence(moved)
    assert torch.equal(ref.log_prior[t], alt.log_prior[t])


# --------------------------------------------------------------------------- #
# G-2: the autoregressive Hamilton gate (order > 0) cannot be fitted
# --------------------------------------------------------------------------- #


# G-2, fixed: the AR gate's filtered probabilities are aligned to dates[order:].
def test_G2_markov_autoregression_gate_fits():
    pytest.importorskip("statsmodels")
    from nec_moe import MarkovSwitchingRegimePrior

    cfg = small_config(prior_kind="markov")
    gate = dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", order=1,
        start_vol_quantiles=(0.6,), start_vol_windows=(10,), start_persistences=(0.95,),
        start_draws_per_centre=1, start_min_history=5, start_min_group_size=5,
    )
    prior = MarkovSwitchingRegimePrior(2, gate)
    prior.fit(_sticky_panel(300, 4))
    assert prior.fitted


# --------------------------------------------------------------------------- #
# G-3: a fold interrupted mid-fit cannot resume with a precomputed gate
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    strict=True,
    reason="G-3: the resume branch of walk_forward_evaluate loads the trainer "
    "checkpoint but never re-fits the gate, and PrecomputedRegimePrior's table "
    "is not in state_dict, so the resumed forward pass raises",
)
def test_G3_mid_fold_resume_with_a_precomputed_gate_matches_uninterrupted(tmp_path: Path):
    pytest.importorskip("statsmodels")
    from nec_moe.evaluation import _fit_gate

    panel = _sticky_panel(160, 6)
    cfg = small_config(
        prior_kind="markov", sigma_init=1.5, lr=3e-3, batch_size=256,
        freeze_gate=True, checkpoint_every=10,
    )
    cfg = dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", start_vol_quantiles=(0.6,),
        start_vol_windows=(10,), start_persistences=(0.95,), start_draws_per_centre=2,
        start_min_history=5, start_min_group_size=5,
    ))

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    kw = dict(n_folds=2, test_dates_per_fold=10, purge_dates=5, steps=40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        ref = walk_forward_evaluate(panel, make_trainer, **kw)

        # replay the harness up to a crash 20 steps into fold 0
        rdir = tmp_path / "wf"
        rdir.mkdir()
        folds = walk_forward_folds(panel.date, n_folds=2, test_dates_per_fold=10,
                                   purge_dates=5)
        train0 = panel.subset_dates(folds[0].train_dates)
        crashed = make_trainer()
        _fit_gate(crashed, panel, train0, folds[0])
        crashed.fit(train0, steps=20, checkpoint_path=rdir / "fold_0_trainer.pt")

        resumed = walk_forward_evaluate(panel, make_trainer, resume_dir=rdir, **kw)
    assert resumed == ref
