"""Brief 09 B: calendar-year folds, the pilot folds and their guard, and no
held-out tail in main mode (audit B-2). Synthetic panels on a synthetic
business-day calendar only.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import pandas as pd
import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    BaseCache,
    FoldConfig,
    NECModel,
    PreTestViolation,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    assert_pre_test_panel,
    calendar_year_folds,
    fit_base,
    pilot_folds,
    pilot_slice,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe.train import DeadParameterWarning

PURGE = 5
START, END = "2004-01-01", "2012-12-31"


def _calendar_panel(start: str = START, end: str = END, entities: int = 3, seed: int = 0):
    n = len(pd.bdate_range(start, end))
    spec = SyntheticSpec(calendar_start=start, seed=seed, regime_process="markov")
    return SyntheticRegimePanel(spec).generate(n, entities)


@pytest.fixture(scope="module")
def panel():
    return _calendar_panel()


def _years(panel, dates: torch.Tensor) -> list[int]:
    return [int(panel.date_labels[int(d)][:4]) for d in dates]


def test_fold_config_defaults_are_the_decision():
    fc = FoldConfig().validate()
    assert fc.fold_scheme == "calendar_year"
    assert (fc.first_test_year, fc.last_test_year) == (2010, 2024)
    assert fc.pilot_validation_years == (2007, 2008, 2009)
    assert fc.test_years == tuple(range(2010, 2025)) and len(fc.test_years) == 15
    for bad in (dict(fold_scheme="rolling"), dict(last_test_year=2009),
                dict(pilot_validation_years=(2008, 2007)),
                dict(pilot_validation_years=(2008, 2010))):
        with pytest.raises(ValueError):
            dataclasses.replace(fc, **bad).validate()


def test_main_folds_on_a_multi_year_calendar(panel):
    folds = calendar_year_folds(panel, first_test_year=2010, last_test_year=2012,
                                purge_dates=PURGE)
    assert [f.test_year for f in folds] == [2010, 2011, 2012]
    assert [f.fold for f in folds] == [0, 1, 2] and {f.kind for f in folds} == {"main"}
    dates = torch.unique(panel.date, sorted=True)
    for f in folds:
        y = f.test_year
        # test: every trading day of Y
        assert set(_years(panel, f.test_dates)) == {y}
        assert f.test_dates.numel() == len(pd.bdate_range(f"{y}-01-01", f"{y}-12-31"))
        # train: every date from the sample start to the end of Y-1 minus the purge
        before = dates[torch.tensor(_years(panel, dates)) < y]
        assert torch.equal(f.train_dates, before[:-PURGE])
        assert torch.equal(f.purged_dates, before[-PURGE:])
        assert int(f.train_dates[0]) == int(dates[0])  # expanding from the start
        assert max(_years(panel, f.purged_dates)) == y - 1


def test_no_test_date_appears_in_any_training_block(panel):
    folds = calendar_year_folds(panel, first_test_year=2006, last_test_year=2012,
                                purge_dates=PURGE)
    tests = torch.cat([f.test_dates for f in folds])
    for f in folds:
        assert not torch.isin(f.train_dates, f.test_dates).any()
        # nor any later test block, and no label window reaches the test year:
        # the last training date is at least PURGE dates before the test block
        later = torch.cat([g.test_dates for g in folds if g.test_year >= f.test_year])
        assert not torch.isin(f.train_dates, later).any()
        assert int(f.train_dates.max()) + PURGE < int(f.test_dates.min())
    assert torch.unique(tests).numel() == tests.numel()  # test years do not overlap


def test_count_scheme_is_the_old_split(panel):
    old = walk_forward_folds(panel.date, n_folds=2, test_dates_per_fold=30, purge_dates=PURGE)
    assert {f.kind for f in old} == {"count"} and all(f.test_year is None for f in old)


def test_pilot_folds_use_only_pre_test_data(panel):
    sliced = pilot_slice(panel, first_test_year=2010, purge_dates=PURGE)
    assert max(_years(sliced, torch.unique(sliced.date))) == 2009
    # the last PURGE dates of 2009 are gone: their labels are 2010 returns
    dates_2009 = [d for d in pd.bdate_range("2009-01-01", "2009-12-31")]
    kept_2009 = [lab for lab in (sliced.date_labels[int(d)] for d in torch.unique(sliced.date))
                 if lab.startswith("2009")]
    assert len(kept_2009) == len(dates_2009) - PURGE
    assert kept_2009[-1] == str(dates_2009[-PURGE - 1].date())
    folds = pilot_folds(sliced, validation_years=(2007, 2008, 2009), first_test_year=2010,
                        purge_dates=PURGE)
    assert [f.test_year for f in folds] == [2007, 2008, 2009]
    assert {f.kind for f in folds} == {"pilot"}
    for f in folds:
        assert set(_years(sliced, f.test_dates)) == {f.test_year}
        assert max(_years(sliced, f.train_dates)) == f.test_year - 1
        assert int(f.train_dates.max()) + PURGE < int(f.test_dates.min())
    assert folds[-1].test_dates.numel() == len(dates_2009) - PURGE


def test_the_pilot_raises_if_handed_a_test_period_date(panel):
    with pytest.raises(PreTestViolation, match="2010"):
        assert_pre_test_panel(panel, 2010)
    with pytest.raises(PreTestViolation, match="2010"):  # the whole panel, unsliced
        pilot_folds(panel, validation_years=(2007, 2008, 2009), first_test_year=2010,
                    purge_dates=PURGE)
    # one 2010 date smuggled into an otherwise clean slice is enough
    sliced = pilot_slice(panel, first_test_year=2010, purge_dates=PURGE)
    dates = torch.unique(panel.date, sorted=True)
    first_2010 = next(d for d in dates if panel.date_labels[int(d)].startswith("2010"))
    leaky = panel.subset_dates(torch.cat([torch.unique(sliced.date), first_2010.view(1)]))
    with pytest.raises(PreTestViolation):
        pilot_folds(leaky, validation_years=(2007, 2008, 2009), first_test_year=2010,
                    purge_dates=PURGE)
    with pytest.raises(PreTestViolation):  # a validation year at the boundary
        pilot_folds(sliced, validation_years=(2008, 2010), first_test_year=2010,
                    purge_dates=PURGE)


def test_calendar_folds_need_a_calendar():
    bare = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(50, 3)
    with pytest.raises(ValueError, match="date_labels"):
        calendar_year_folds(bare, first_test_year=2010, last_test_year=2010, purge_dates=1)


# --------------------------------------------------------------------------- #
# B.3: no held-out tail in main mode (audit B-2)
# --------------------------------------------------------------------------- #


def test_base_trains_on_the_full_block_without_a_tail(panel):
    block = panel.subset_dates(torch.unique(panel.date)[:300])
    cfg = small_base_config(steps=40)
    assert cfg.val_fraction == 0.0 and cfg.early_stopping_patience is None  # the default
    fit = fit_base(block, cfg, seed=0)
    assert math.isnan(fit.val_loss) and "base_val_mse" not in fit.metrics()
    assert fit.steps_run == cfg.steps and not fit.stopped_early
    # the last dates of the block are trained on: changing their targets
    # changes the fitted base (a held-out tail would leave it untouched)
    tail = torch.unique(block.date)[-60:]
    moved = dataclasses.replace(block, y=torch.where(torch.isin(block.date, tail),
                                                     block.y + 5.0, block.y))
    refit = fit_base(moved, cfg, seed=0)
    x = block.x_snap[:20]
    assert not torch.allclose(fit.model(x), refit.model(x))
    with pytest.raises(ValueError, match="validation tail"):
        fit_base(block, dataclasses.replace(cfg, early_stopping_patience=3), seed=0)
    with pytest.raises(ValueError, match="validation tail"):
        small_config(base=dataclasses.replace(cfg, early_stopping_patience=3)).validate()


def _residual_cfg(base):
    return small_config(prior_kind="soft", correction_mode=True, base=base, lr=3e-3,
                        batch_size=64, freeze_gate=False)


def test_main_folds_refuse_a_base_validation_tail():
    panel = _calendar_panel("2008-01-01", "2010-12-31", entities=4)
    folds = calendar_year_folds(panel, first_test_year=2010, last_test_year=2010,
                                purge_dates=PURGE)
    tailed = small_base_config(steps=20, val_fraction=0.2)
    with pytest.raises(ValueError, match="val_fraction=0"):
        walk_forward_evaluate(panel, lambda: Trainer(NECModel(_residual_cfg(tailed))),
                              folds=folds, purge_dates=PURGE, steps=2)
    # count folds keep the development option of a tail
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        walk_forward_evaluate(panel, lambda: Trainer(NECModel(_residual_cfg(tailed))),
                              n_folds=1, test_dates_per_fold=40, purge_dates=PURGE, steps=2)


def test_walk_forward_runs_on_calendar_folds_and_records_the_year():
    panel = _calendar_panel("2008-01-01", "2011-12-30", entities=4)
    folds = calendar_year_folds(panel, first_test_year=2010, last_test_year=2011,
                                purge_dates=PURGE)
    cfg = _residual_cfg(small_base_config(steps=20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        res = walk_forward_evaluate(panel, lambda: Trainer(NECModel(cfg)), folds=folds,
                                    purge_dates=PURGE, steps=3, base_cache=BaseCache())
    assert [f.test_year for f in res.folds] == [2010, 2011]
    assert [f.n_test for f in res.folds] == [len(panel.subset_dates(f.test_dates))
                                             for f in folds]


def test_run_experiment_evaluates_on_calendar_year_folds(tmp_path):
    """Evaluate mode's default fold scheme is the calendar-year one; on a
    synthetic calendar 2008-2011 with test years 2010-2011 it runs two annual
    folds, each recorded with its year in the run's aggregate metrics."""
    import run_experiment as rx

    exp = rx.Experiment(
        tag="cal", campaign="unit", mode="evaluate", out_dir=str(tmp_path / "results"),
        per_security_dir=str(tmp_path / "Data" / "derived" / "runs"), data="synthetic",
        synth_dates=len(pd.bdate_range("2008-01-01", "2011-12-30")), synth_entities=4,
        synth_calendar_start="2008-01-01", first_test_year=2010, last_test_year=2011,
        encoder_hidden=8, expert_hidden_dims=(8, 4), expert_dropout=0.0, steps=3,
        sigma_freeze_steps=0, seeds=(0,), include_ridge=True, include_mlp=False,
        backtest_quantiles=None, figures=False, calibration=False,
    )
    assert exp.fold_scheme == "calendar_year" and exp.base_val_fraction == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = rx.main(exp)
    frame = pd.read_csv(result["run_dir"] / "metrics" / f"{exp.prior}_seed0_folds.csv")
    assert list(frame["test_year"]) == [2010, 2011]
