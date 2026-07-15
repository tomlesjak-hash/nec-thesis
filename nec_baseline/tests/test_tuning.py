"""Hyperparameter tuning: purged validation tail + logged selection.

The planted-truth test gives the tuner one candidate that can route (soft with
warm-start) and one that cannot (uniform) — it must pick soft, log every trial
plus the arm aggregates plus exactly one selection event with the candidate
count, and never touch dates outside the training window it was given.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import small_config

from nec_moe import (
    SyntheticRegimePanel,
    SyntheticSpec,
    TrialRegistry,
    nec_arm,
    tune,
    validation_tail,
    walk_forward_folds,
)


def _panel():
    spec = SyntheticSpec(
        vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3
    )
    return SyntheticRegimePanel(spec).generate(n_dates=160, n_entities=8)


def _vol_proxy(p):
    return p.x_seq[:, :, 0].std(dim=1)


def test_validation_tail_is_the_purged_last_block():
    date = torch.arange(100).repeat_interleave(4)
    tail = validation_tail(date, val_dates=15, purge_dates=5)
    assert tail.test_dates.tolist() == list(range(85, 100))  # the last 15 dates
    assert int(tail.train_dates.max()) == 79  # 5 purged dates in between
    assert len(tail.purged_dates) == 5
    # the same label-overlap guarantee as the outer split
    assert int(tail.train_dates.max()) + 5 < int(tail.test_dates.min())


def test_tune_selects_planted_winner_and_logs_everything(tmp_path: Path):
    panel = _panel()
    # tune INSIDE the first outer fold's training window, as the protocol says
    outer = walk_forward_folds(
        panel.date, n_folds=2, test_dates_per_fold=20, purge_dates=5
    )
    train_panel = panel.subset_dates(outer[0].train_dates)

    reg = TrialRegistry(tmp_path / "tune.jsonl")
    candidates = [
        nec_arm("soft", small_config(sigma_init=1.5, lr=3e-3, batch_size=256),
                warmstart_key=_vol_proxy),
        nec_arm("uniform",
                small_config(prior_kind="uniform", sigma_init=1.5, lr=3e-3,
                             batch_size=256),
                warmstart_key=_vol_proxy),
    ]
    result = tune(
        train_panel, candidates, seeds=(0, 1), registry=reg, tag="lr_tune",
        steps=200, val_dates=15, purge_dates=5, verbose=False,
    )

    # the candidate that can route wins on the tail
    assert result.winner == "soft"
    assert result.scores["soft"] > result.scores["uniform"] + 0.2
    assert result.n_candidates == 2

    # provenance: per-run trials, arm aggregates, exactly one selection event
    assert reg.n_trials("lr_tune") == 4  # 2 candidates x 2 seeds
    assert reg.n_trials("lr_tune.arms") == 2
    events = reg.selection_events("lr_tune.arms")
    assert len(events) == 1
    assert events[0].config["n_candidates"] == 2

    # no contamination: everything the tuner saw predates the outer test block
    outer_test_start = int(outer[0].test_dates.min())
    assert int(train_panel.date.max()) < outer_test_start


def test_tune_min_mode_on_nll(tmp_path: Path):
    panel = _panel()
    train_panel = panel.subset_dates(
        walk_forward_folds(panel.date, n_folds=1, test_dates_per_fold=20,
                           purge_dates=5)[0].train_dates
    )
    reg = TrialRegistry(tmp_path / "t.jsonl")
    candidates = [
        nec_arm("soft", small_config(sigma_init=1.5, lr=3e-3, batch_size=256),
                warmstart_key=_vol_proxy),
        nec_arm("uniform",
                small_config(prior_kind="uniform", sigma_init=1.5, lr=3e-3,
                             batch_size=256),
                warmstart_key=_vol_proxy),
    ]
    result = tune(
        train_panel, candidates, seeds=(0,), registry=reg, tag="nll_tune",
        steps=150, val_dates=12, purge_dates=5, metric="nll", mode="min",
        verbose=False,
    )
    assert result.winner == "soft"  # routing lowers held-out NLL too
    assert result.scores["soft"] < result.scores["uniform"]


def test_tune_guards(tmp_path: Path):
    panel = _panel()
    reg = TrialRegistry(tmp_path / "t.jsonl")
    one = [nec_arm("only", small_config())]
    with pytest.raises(ValueError, match=">= 2 candidates"):
        tune(panel, one, seeds=(0,), registry=reg, tag="t", steps=1,
             val_dates=10, purge_dates=5)
    two = one + [nec_arm("other", small_config())]
    with pytest.raises(ValueError, match="mode"):
        tune(panel, two, seeds=(0,), registry=reg, tag="t", steps=1,
             val_dates=10, purge_dates=5, mode="best")
