"""Brief 10 A (Q22, decided 2026-10-08): the gate's filter runs through the
purge gap.

The gate is fitted on the training block only, as before. Its frozen filter
then runs over every date from the first training date to the last test date,
gap included, so one step of ``A`` is one trading day at the fold boundary as
everywhere else. The purge keeps removing the gap days' rows from the training
of the base and the experts, whose labels reach into the test block.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    MarkovSwitchingRegimePrior,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    evaluation,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe import base as base_module
from nec_moe import baum_welch as bw
from nec_moe import markov_gate as markov_gate_module
from nec_moe.evaluation import _fit_gate, gate_span_dates
from nec_moe.markov_gate import date_level_series
from nec_moe.priors import PriorContext
from nec_moe.train import DeadParameterWarning

pytest.importorskip("statsmodels")

PURGE = 5


def _panel(n_dates: int = 220, n_entities: int = 6, seed: int = 3):
    return SyntheticRegimePanel(
        SyntheticSpec(
            regime_process="markov", transition_stay=0.96,
            vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=seed,
        )
    ).generate(n_dates, n_entities)


def _cfg(backend: str = "statsmodels", **overrides):
    cfg = small_config(
        prior_kind="markov", sigma_init=1.5, lr=3e-3, batch_size=64,
        freeze_gate=True, **overrides,
    )
    return dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", series_channel=0,
        search_reps=2, fit_backend=backend,
    ))


def _residual_cfg():
    return _cfg(correction_mode=True, zero_init_head=True,
                base=small_base_config(steps=20))


def _fold(panel, n_folds: int = 1):
    return walk_forward_folds(
        panel.date, n_folds=n_folds, test_dates_per_fold=30, purge_dates=PURGE
    )


def _gap(fold) -> torch.Tensor:
    lo, hi = int(fold.train_dates.max()), int(fold.test_dates.min())
    return torch.arange(lo + 1, hi)


def _fitted_gate(panel, fold, cfg):
    trainer = Trainer(NECModel(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_gate(trainer, panel, panel.subset_dates(fold.train_dates), fold)
    return trainer.model.prior


def _served(prior, dates: torch.Tensor) -> torch.Tensor:
    """The gate weight served for each of ``dates`` (one row per date)."""
    return prior(torch.zeros(len(dates), prior.n_experts),
                 PriorContext(date=dates)).log_prior.exp()


def _with_channel(panel, date: int, value: float):
    """A copy of ``panel`` whose gate series (channel 0, last step) is
    ``value`` on ``date``; nothing else changes."""
    x_seq = panel.x_seq.clone()
    x_seq[panel.date == date, -1, 0] = value
    return dataclasses.replace(panel, x_seq=x_seq)


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #


def test_gate_span_is_every_date_from_first_train_to_last_test_gap_included():
    panel = _panel()
    fold = _fold(panel, n_folds=2)[0]  # not the last fold: dates follow it
    span = gate_span_dates(panel, fold)
    gap = _gap(fold)
    assert len(gap) == PURGE
    assert torch.equal(
        span, torch.arange(int(fold.train_dates.min()), int(fold.test_dates.max()) + 1)
    )
    assert torch.isin(gap, span).all()
    assert int(span.max()) < int(panel.date.max())  # nothing after the test block


# --------------------------------------------------------------------------- #
# test-date probabilities = one frozen filter pass over the contiguous series
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["statsmodels", "native"])
def test_test_dates_equal_one_frozen_pass_over_the_contiguous_series(backend):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    panel = _panel()
    fold = _fold(panel)[0]
    cfg = _cfg(backend)
    prior = _fitted_gate(panel, fold, cfg)
    mg, fit = cfg.markov_gate, prior.fit_result
    assert fit is not None and fit.backend == backend

    # one pass, by hand, frozen parameters, over every date from the first
    # training date to the last test date
    contiguous = panel.subset_dates(
        torch.arange(int(fold.train_dates.min()), int(fold.test_dates.max()) + 1)
    )
    dates, series = date_level_series(contiguous, mg)
    if backend == "native":
        by_hand, _ = bw.filter_series(series, prior.native_params())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = MarkovRegression(
                series, k_regimes=2, trend=mg.trend,
                switching_trend=mg.switching_trend,
                switching_variance=mg.switching_variance,
            ).filter(fit.params)
        by_hand = np.asarray(run.filtered_marginal_probabilities, dtype=float)
        by_hand = by_hand[:, np.asarray(fit.permutation, dtype=int)]
    test = torch.isin(dates, fold.test_dates).numpy()
    assert test.sum() == len(fold.test_dates)
    np.testing.assert_allclose(
        prior.filtered_probabilities(fold.test_dates), by_hand[test], atol=1e-10
    )
    # the served gate weight is the configured weight of that same pass
    weight = markov_gate_module.gate_weight_probs(
        by_hand, fit.transition, mg.gate_weight, panel.horizon
    )
    assert torch.allclose(
        _served(prior, fold.test_dates),
        torch.from_numpy(weight[test]).to(torch.float32), atol=1e-6,
    )
    # the gap dates get filtered probabilities from the same causal pass
    gap = _gap(fold)
    np.testing.assert_allclose(
        prior.filtered_probabilities(gap),
        by_hand[torch.isin(dates, gap).numpy()], atol=1e-10,
    )

    # and the gap matters: a pass that skips it disagrees on the test block
    skipped = panel.subset_dates(torch.cat([fold.train_dates, fold.test_dates]))
    s_dates, s_series = date_level_series(skipped, mg)
    if backend == "native":
        skip, _ = bw.filter_series(s_series, prior.native_params())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skip = np.asarray(MarkovRegression(
                s_series, k_regimes=2, trend=mg.trend,
                switching_trend=mg.switching_trend,
                switching_variance=mg.switching_variance,
            ).filter(fit.params).filtered_marginal_probabilities, dtype=float)
        skip = skip[:, np.asarray(fit.permutation, dtype=int)]
    assert not np.allclose(skip[torch.isin(s_dates, fold.test_dates).numpy()],
                           by_hand[test], atol=1e-10)


@pytest.mark.parametrize("backend", ["statsmodels", "native"])
def test_fitted_training_rows_are_not_overwritten_by_the_span(backend):
    panel = _panel()
    fold = _fold(panel)[0]
    cfg = _cfg(backend)
    trainer = Trainer(NECModel(cfg))
    prior = trainer.model.prior
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior.fit(panel.subset_dates(fold.train_dates))
    before = _served(prior, fold.train_dates).clone()
    prior.apply_causal(panel.subset_dates(gate_span_dates(panel, fold)))
    assert torch.equal(_served(prior, fold.train_dates), before)


# --------------------------------------------------------------------------- #
# the gap moves the gate, never training
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["statsmodels", "native"])
def test_a_gap_day_market_return_moves_the_first_test_date_gate_weight(backend):
    panel = _panel()
    fold = _fold(panel)[0]
    cfg = _cfg(backend)
    last_gap = int(_gap(fold).max())
    first_test = fold.test_dates[:1]
    # the first test date's own observation set to the calm mean, in both
    # panels: in this draw it is an extreme value that saturates the filter
    # there whatever came before, which would hide the gap's effect
    panel = _with_channel(panel, int(first_test), 0.0)

    plain = _fitted_gate(panel, fold, cfg)
    shocked = _fitted_gate(_with_channel(panel, last_gap, 25.0), fold, cfg)
    # same training block, so the same fit and the same training rows
    assert np.array_equal(plain.fit_result.params, shocked.fit_result.params)
    assert torch.equal(_served(plain, fold.train_dates), _served(shocked, fold.train_dates))
    # a huge gap-day move is a stress observation the first test date now sees
    moved = _served(shocked, first_test) - _served(plain, first_test)
    assert float(moved.abs().max()) > 1e-3


def _training_spy(monkeypatch) -> dict[str, list[torch.Tensor]]:
    """Record the dates of every panel the base and the experts train on."""
    seen: dict[str, list[torch.Tensor]] = {"base": [], "experts": []}
    real_fit_base, real_fit = base_module.fit_base, Trainer.fit

    def fit_base(panel, cfg, *, seed):
        seen["base"].append(torch.unique(panel.date))
        return real_fit_base(panel, cfg, seed=seed)

    def fit(self, panel, *args, **kwargs):
        seen["experts"].append(torch.unique(panel.date))
        return real_fit(self, panel, *args, **kwargs)

    monkeypatch.setattr(base_module, "fit_base", fit_base)
    monkeypatch.setattr(Trainer, "fit", fit)
    return seen


def _run(panel, cfg):
    def make_trainer() -> Trainer:
        return Trainer(NECModel(cfg))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        warnings.simplefilter("ignore", UserWarning)
        return walk_forward_evaluate(
            panel, make_trainer, n_folds=1, test_dates_per_fold=30,
            purge_dates=PURGE, steps=15, seed=0,
        )


def test_gap_dates_are_not_in_the_training_rows_of_the_base_or_the_experts(monkeypatch):
    panel = _panel()
    fold = _fold(panel)[0]
    gap = _gap(fold)
    seen = _training_spy(monkeypatch)
    covered: list[torch.Tensor] = []
    real_fit_gate = evaluation._fit_gate

    def fit_gate(trainer, panel, train, fold):
        real_fit_gate(trainer, panel, train, fold)
        covered.append(trainer.model.prior.covers(gap))

    monkeypatch.setattr(evaluation, "_fit_gate", fit_gate)
    _run(panel, _residual_cfg())
    assert seen["base"] and seen["experts"]
    for dates in seen["base"] + seen["experts"]:
        assert not torch.isin(gap, dates).any()
        assert int(dates.max()) == int(fold.train_dates.max())
    # while the gate does cover them
    assert covered and bool(covered[0].all())


def test_a_gap_day_target_changes_nothing(monkeypatch):
    """The gap days' labels reach into the test block; the purge keeps them
    out of training, so changing them changes no trained number."""
    panel = _panel()
    fold = _fold(panel)[0]
    cfg = _residual_cfg()

    def with_targets(dates: torch.Tensor, value: float):
        y = panel.y.clone()
        y[torch.isin(panel.date, dates)] = value
        y_daily = None if panel.y_daily is None else panel.y_daily.clone()
        if y_daily is not None:
            y_daily[torch.isin(panel.date, dates)] = value
        return dataclasses.replace(panel, y=y, y_daily=y_daily)

    def summary(result):
        f = result.folds[0]
        return (f.nll, f.base_nll, f.ic.mean_ic, f.base_ic.mean_ic, f.n_train)

    plain = summary(_run(panel, cfg))
    assert summary(_run(with_targets(_gap(fold), 9.0), cfg)) == plain
    # the comparison has power: the same change on training dates does move it
    assert summary(_run(with_targets(fold.train_dates[-PURGE:], 9.0), cfg)) != plain


# --------------------------------------------------------------------------- #
# the fit still sees the training block only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("where", ["first_test", "after_test"])
def test_the_gate_fit_still_refuses_a_date_in_or_after_the_test_block(where):
    panel = _panel()
    fold = _fold(panel, n_folds=2)[0]
    extra = (fold.test_dates[:1] if where == "first_test"
             else torch.tensor([int(fold.test_dates.max()) + 1]))
    leaky = panel.subset_dates(torch.cat([fold.train_dates, extra]))
    trainer = Trainer(NECModel(_cfg()))
    with pytest.raises(AssertionError):
        _fit_gate(trainer, panel, leaky, fold)
    assert not trainer.model.prior.fitted


def test_the_pilot_scores_only_the_validation_dates_over_the_span(monkeypatch):
    """F.2 with Q22: the predictive density is evaluated over the contiguous
    span, gap included, and only the validation dates are summed."""
    from nec_moe import pilot as pl
    from nec_moe.pilot import PilotGrid

    panel = _panel(n_dates=260)
    folds = _fold(panel, n_folds=2)
    cfg = _cfg("native")
    calls: list[torch.Tensor] = []
    real = MarkovSwitchingRegimePrior.predictive_log_density

    def spy(self, p):
        calls.append(torch.unique(p.date))
        return real(self, p)

    monkeypatch.setattr(MarkovSwitchingRegimePrior, "predictive_log_density", spy)
    grid = dataclasses.replace(PilotGrid(), gate_half_life_years=(None,))
    assert grid.gate_half_lives == (float("inf"),)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report, _ = pl.gate_memory_selection(panel, folds, cfg, grid)
    assert len(calls) == len(folds) == 2
    for c, f in zip(calls, folds, strict=True):
        assert torch.equal(c, gate_span_dates(panel, f))
    # the score is the sum over the validation dates of the same pass
    total = 0.0
    for f in folds:
        gate = MarkovSwitchingRegimePrior(2, cfg.markov_gate,
                                          horizon=cfg.data.horizon_periods)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gate.fit(panel.subset_dates(f.train_dates))
        span = panel.subset_dates(gate_span_dates(panel, f))
        dates, log_c = real(gate, span)
        total += float(log_c[torch.isin(dates, f.test_dates).numpy()].sum())
    assert report["results"][0]["predictive_loglik"] == pytest.approx(total, rel=1e-12)
