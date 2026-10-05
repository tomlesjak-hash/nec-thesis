"""Brief 09 C: calendar decay for the base, regime-clock decay for the
experts, the weighted losses and their summaries. Synthetic data only.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    BaseCache,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    calendar_decay_weights,
    fit_base,
    fold_decay_weights,
    fold_metrics_frame,
    kish_ess,
    regime_clock_components,
    regime_clock_exponents,
    regime_clock_weights,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe.decay import weight_summary, weighted_mean
from nec_moe.evaluation import _fit_gate, _gate_training_block
from nec_moe.train import DeadParameterWarning


def test_calendar_weights_halve_every_half_life():
    codes = torch.tensor([0, 0, 3, 5, 7, 7])
    w = calendar_decay_weights(codes, 2.0)
    assert w is not None and w.dtype == torch.float64
    # age = T - s in trading days, T = 7: rows on T weigh 1, two days back 1/2
    expected = torch.tensor([2 ** -3.5, 2 ** -3.5, 2 ** -2.0, 2 ** -1.0, 1.0, 1.0],
                            dtype=torch.float64)
    assert torch.allclose(w, expected, rtol=0, atol=1e-15)
    assert calendar_decay_weights(codes, None) is None  # equal weights: no weighting
    assert calendar_decay_weights(codes, math.inf) is None


def test_regime_clock_counts_only_later_same_regime_experience():
    xi = np.array([[0.2, 0.8], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    n = regime_clock_exponents(xi)
    # n_k(s, T) = sum over u = s+1 .. T of xi_u(k), the last row has none
    assert np.allclose(n, [[1.5, 1.5], [0.5, 1.5], [0.0, 1.0], [0.0, 0.0]])
    h = 3.0
    rho = 2 ** (-1 / h)
    w = regime_clock_weights(xi, h)
    assert w[0] == pytest.approx(0.2 * rho ** 1.5 + 0.8 * rho ** 1.5, rel=1e-14)
    assert w[1] == pytest.approx(rho ** 0.5, rel=1e-14)
    assert w[-1] == pytest.approx(1.0)
    assert np.allclose(regime_clock_components(xi, h).sum(axis=1), w)
    assert np.allclose(regime_clock_weights(xi, math.inf), 1.0)


def test_a_certain_regime_reduces_to_calendar_decay():
    """C.4: with a gate certainly in regime k throughout, the regime clock is
    the calendar: every later day is a day of k."""
    t_len, h = 400, 37.0
    for k in (0, 1):
        xi = np.zeros((t_len, 2))
        xi[:, k] = 1.0
        clock = regime_clock_weights(xi, h)
        calendar = calendar_decay_weights(torch.arange(t_len), h)
        assert calendar is not None
        assert np.allclose(clock, calendar.numpy(), rtol=1e-12, atol=0)


def test_a_rare_regime_row_decays_slower_than_a_common_one_of_the_same_age():
    """C.4: 1,000 later days of which 300 are stress (3 in every 10) and
    H = 500 regime-days. A stress row has aged 300 regime-days, weight
    2^(-300/500) = 0.6597540; a calm row of the same calendar age has aged
    700, weight 2^(-700/500) = 0.3789291."""
    later = np.zeros((1000, 2))
    stress = (np.arange(1000) % 10) < 3
    later[stress, 1] = 1.0
    later[~stress, 0] = 1.0
    stress_row = np.vstack([[0.0, 1.0], later])
    calm_row = np.vstack([[1.0, 0.0], later])
    w_stress = regime_clock_weights(stress_row, 500.0)[0]
    w_calm = regime_clock_weights(calm_row, 500.0)[0]
    assert w_stress == pytest.approx(0.6597539553864471, rel=1e-12)
    assert w_calm == pytest.approx(0.3789291416275995, rel=1e-12)
    assert w_stress > w_calm
    # calendar decay at the same half-life would give both 2^(-1000/500) = 0.25
    assert calendar_decay_weights(torch.arange(1001), 500.0)[0].item() == pytest.approx(0.25)


def test_weighted_mean_is_normalised_by_the_sum():
    loss = torch.tensor([1.0, 2.0, 4.0])
    w = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    assert weighted_mean(loss, w).item() == pytest.approx(2.0)
    # scaling every weight leaves the loss (and its gradient) unchanged, so the
    # step size does not depend on the half-life's overall level
    assert weighted_mean(loss, 7.0 * w).item() == pytest.approx(weighted_mean(loss, w).item())
    assert torch.equal(weighted_mean(loss, None), loss.mean())


def test_summaries_are_aggregates():
    dates = torch.arange(6)
    labels = ("2007-12-28", "2007-12-31", "2008-01-02", "2008-01-03", "2008-01-04",
              "2008-01-07")
    w = np.array([0.25, 0.5, 0.5, 1.0, 1.0, 1.0])
    rows = np.array([2, 2, 2, 2, 2, 2])
    comps = np.stack([w * 0.5, w * 0.5], axis=1)
    out = weight_summary("expert_weight", w, dates, rows, labels, comps)
    assert out["expert_weight_ess_dates"] == pytest.approx(kish_ess(w))
    assert out["expert_weight_ess_rows"] == pytest.approx(2 * kish_ess(w))
    assert kish_ess(np.ones(10)) == pytest.approx(10.0)
    assert out["expert_weight_mass_2007"] + out["expert_weight_mass_2008"] == pytest.approx(1.0)
    assert out["expert_weight_mass_2007"] == pytest.approx(0.75 / 4.25)
    assert {"expert_weight_ess_regime_0", "expert_weight_ess_regime_1"} <= set(out)
    assert all(isinstance(v, float) for v in out.values())


# --------------------------------------------------------------------------- #
# The weights in the training loops
# --------------------------------------------------------------------------- #


def _panel(n_dates: int = 160, seed: int = 3):
    return SyntheticRegimePanel(
        SyntheticSpec(regime_process="markov", transition_stay=0.96, vol_levels=(0.5, 2.5),
                      beta_scale=2.0, noise_std=0.4, seed=seed,
                      calendar_start="2006-06-01")
    ).generate(n_dates, 8)


def _markov_cfg(**train):
    cfg = small_config(prior_kind="markov", correction_mode=True, zero_init_head=False,
                       base=small_base_config(), sigma_init=1.0, lr=3e-3, batch_size=64,
                       freeze_gate=True, expert_dropout=0.1, **train)
    return dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", series_channel=0,
        start_vol_quantiles=(0.6,), start_vol_windows=(10,), start_persistences=(0.95,),
        start_draws_per_centre=1, start_min_history=5, start_min_group_size=5,
    ))


def test_equal_weights_train_the_base_bit_identically():
    panel = _panel()
    cfg = small_base_config(steps=50, dropout=0.1)
    plain = fit_base(panel, cfg, seed=0)
    for h in (None, math.inf):
        again = fit_base(panel, dataclasses.replace(cfg, decay_half_life_days=h), seed=0)
        for a, b in zip(plain.model.state_dict().values(), again.model.state_dict().values(),
                        strict=True):
            assert torch.equal(a, b)
    decayed = fit_base(panel, dataclasses.replace(cfg, decay_half_life_days=20.0), seed=0)
    assert not torch.equal(plain.model(panel.x_snap[:5]), decayed.model(panel.x_snap[:5]))


def _evaluate(cfg, panel):
    trainers: list[Trainer] = []

    def make():
        torch.manual_seed(0)
        trainers.append(Trainer(NECModel(cfg)))
        return trainers[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        res = walk_forward_evaluate(panel, make, n_folds=2, test_dates_per_fold=12,
                                    purge_dates=3, steps=15, base_cache=BaseCache(), seed=0)
    return res, trainers


def test_an_infinite_regime_clock_trains_the_experts_bit_identically():
    """C.4: H = infinity is equal weights, and training is bit-identical to
    the unweighted run (dropout on, so the RNG stream must match too)."""
    pytest.importorskip("statsmodels")
    panel = _panel()
    plain, t_plain = _evaluate(_markov_cfg(), panel)
    clock, t_clock = _evaluate(_markov_cfg(expert_decay="regime_clock",
                                           expert_decay_half_life=math.inf), panel)
    for a, b in zip(t_plain, t_clock, strict=True):
        for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values(),
                        strict=True):
            assert torch.equal(x, y)
    assert [f.nll for f in plain.folds] == [f.nll for f in clock.folds]
    # the infinite clock still reports its (equal-weight) diagnostics
    assert dict(clock.folds[0].weights)["expert_weight_ess_dates"] > 0
    assert plain.folds[0].weights == ()
    # a finite half-life changes the training and is reported per fold
    finite, _ = _evaluate(_markov_cfg(expert_decay="regime_clock",
                                      expert_decay_half_life=20.0), panel)
    assert [f.nll for f in finite.folds] != [f.nll for f in plain.folds]
    frame = fold_metrics_frame(finite)
    assert {"expert_weight_ess_dates", "expert_weight_ess_regime_0",
            "expert_weight_ess_regime_1"} <= set(frame.columns)
    assert any(c.startswith("expert_weight_mass_") for c in frame.columns)


def test_the_weights_depend_only_on_data_up_to_the_last_training_date():
    """C.4: changing every date after T (the purge gap and the test block)
    leaves the fold's weights unchanged: the gate is fitted on the training
    block and only its filtered probabilities up to T enter."""
    pytest.importorskip("statsmodels")
    panel = _panel()
    fold = walk_forward_folds(panel.date, n_folds=1, test_dates_per_fold=20,
                              purge_dates=3)[0]
    later = ~torch.isin(panel.date, fold.train_dates)
    shocked = dataclasses.replace(panel, x_seq=torch.where(later.view(-1, 1, 1),
                                                           panel.x_seq * 5.0 + 1.0, panel.x_seq),
                                  y=torch.where(later, panel.y * -3.0, panel.y))
    cfg = _markov_cfg(expert_decay="regime_clock", expert_decay_half_life=15.0)
    cfg = dataclasses.replace(cfg, base=dataclasses.replace(cfg.base, decay_half_life_days=30.0))
    weights = []
    for p in (panel, shocked):
        trainer = Trainer(NECModel(cfg))
        train = p.subset_dates(fold.train_dates)
        _fit_gate(trainer, p, train, fold)
        expert_train, _ = _gate_training_block(trainer, train)
        weights.append(fold_decay_weights(trainer, train, expert_train))
    (w_a, s_a), (w_b, s_b) = weights
    assert w_a is not None and torch.equal(w_a, w_b)
    assert s_a == s_b
    # every stock of a date shares the date's weight; the last date weighs most
    t_last = int(fold.train_dates.max())
    rows_last = panel.subset_dates(fold.train_dates).date == t_last
    assert torch.all(w_a[rows_last] == w_a[rows_last][0])


def test_regime_clock_needs_the_fitted_gate():
    with pytest.raises(ValueError, match="filtered regime probabilities"):
        small_config(prior_kind="soft", expert_decay="regime_clock",
                     expert_decay_half_life=10.0).validate()
    with pytest.raises(ValueError, match="expert_decay"):
        small_config(expert_decay="per_expert").validate()
    with pytest.raises(ValueError, match="decay_half_life_days"):
        small_config(base=small_base_config(decay_half_life_days=0.0)).validate()


def test_mass_by_year_uses_the_calendar():
    labels = tuple(str(d.date()) for d in pd.bdate_range("2006-12-25", periods=10))
    out = weight_summary("base_weight", np.ones(10), torch.arange(10), np.ones(10), labels)
    assert out["base_weight_mass_2006"] == pytest.approx(0.5)
    assert out["base_weight_mass_2007"] == pytest.approx(0.5)
