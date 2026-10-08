"""The gate interface and the Hamilton gate (brief 03): one test per claim.

The properties that make a separately-fitted, frozen gate trustworthy, each of
which fails *silently* if broken:

1. ``fit`` runs once per fold, on a panel that stops strictly before the test
   block — so the gate inherits the harness's purge discipline;
2. a precomputed prior refuses to serve anything before it is fitted;
3. the parameters applied to the test block are the training fit's, unchanged;
4. the gate path never touches ``smoothed_marginal_probabilities``;
5. on data simulated from a known two-state process the fit recovers the
   transition matrix and expected durations after canonical reordering;
6. that canonical ordering is stable across starting values;
7. the frozen-apply path agrees with running the filter by hand.

Kim & Nelson (1999, ch. 4) is the reference for why (3), (4) and (7) are
modelling requirements rather than implementation details: the filtered
probability at *t* conditions on the series through *t*, the smoothed one
conditions on the whole sample, and only the first may reach a gate.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import small_config

from nec_moe import (
    SERIES_REGISTRY,
    MarkovSwitchingRegimePrior,
    NECModel,
    PrecomputedRegimePrior,
    PriorContext,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    date_level_series,
    gate_span_dates,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe import markov_gate as markov_gate_module
from nec_moe.diagnostics import expected_durations
from nec_moe.train import DeadParameterWarning

pytest.importorskip("statsmodels")


def _sticky_panel(n_dates: int = 300, n_entities: int = 8, stay: float = 0.96, seed: int = 3):
    """A panel whose regimes actually persist, so a Markov fit has a target."""
    return SyntheticRegimePanel(
        SyntheticSpec(
            regime_process="markov", transition_stay=stay,
            vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=seed,
        )
    ).generate(n_dates, n_entities)


def _markov_cfg(**gate_overrides):
    cfg = small_config(
        prior_kind="markov", sigma_init=1.5, lr=3e-3, batch_size=256,
        freeze_gate=True,
    )
    gate = dict(series="sequence_channel", series_channel=0, search_reps=3)
    gate.update(gate_overrides)
    return dataclasses.replace(
        cfg, markov_gate=dataclasses.replace(cfg.markov_gate, **gate)
    )


def _simulate_markov(
    n: int, stay: tuple[float, float], sigma: tuple[float, float], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth two-state Markov switching series (mean zero, switching vol)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.uniform() < stay[s[t - 1]] else 1 - s[t - 1]
    return rng.normal(0.0, np.asarray(sigma)[s]), s


# --------------------------------------------------------------------------- #
# 1. fit is called once per fold, on the training block only
# --------------------------------------------------------------------------- #


def test_fit_called_once_per_fold_on_training_block_only():
    panel = _sticky_panel(n_dates=200, n_entities=6)
    seen: list[tuple[int, int]] = []

    class SpyPrior(PrecomputedRegimePrior):
        def fit(self, train_panel):
            seen.append(
                (int(train_panel.date.min()), int(train_panel.date.max()))
            )
            dates = torch.unique(train_panel.date, sorted=True)
            table = torch.log(
                torch.full((len(dates), self.n_experts), 1.0 / self.n_experts)
            )
            self.set_fitted_table(dates, table)

        def apply_causal(self, p):
            dates = torch.unique(p.date, sorted=True)
            self.extend_causal_table(
                dates,
                torch.log(torch.full((len(dates), self.n_experts), 0.5)),
            )

    cfg = small_config(sigma_init=1.5, lr=3e-3, batch_size=256, freeze_gate=True)
    folds = walk_forward_folds(
        panel.date, n_folds=3, test_dates_per_fold=10, purge_dates=5
    )

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        model = NECModel(cfg)
        model.prior = SpyPrior(cfg.experts.n_experts)
        return Trainer(model, cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        walk_forward_evaluate(
            panel, make_trainer, n_folds=3, test_dates_per_fold=10,
            purge_dates=5, steps=3,
        )

    assert len(seen) == 3, seen  # exactly once per fold
    for (lo, hi), fold in zip(seen, folds, strict=True):
        assert hi < int(fold.test_dates.min()), (hi, int(fold.test_dates.min()))
        assert lo == int(fold.train_dates.min())
        # the purge gap is inherited, not re-implemented
        assert int(fold.test_dates.min()) - hi > 1


def test_gate_fit_panel_reaching_the_test_block_is_refused():
    """The guard behind test 1: if a caller ever hands fit() a panel that
    overlaps the test block, that is an assertion failure, not a warning."""
    from nec_moe.evaluation import _fit_gate

    panel = _sticky_panel(n_dates=80, n_entities=4)
    folds = walk_forward_folds(
        panel.date, n_folds=2, test_dates_per_fold=10, purge_dates=5
    )
    cfg = small_config(sigma_init=1.5)
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    leaky = panel  # the WHOLE panel: reaches into the test block
    with pytest.raises(AssertionError, match="must not see"):
        _fit_gate(trainer, panel, leaky, folds[0])


# --------------------------------------------------------------------------- #
# 2. forward before fit raises
# --------------------------------------------------------------------------- #


def test_precomputed_prior_forward_before_fit_raises():
    prior = PrecomputedRegimePrior(2)
    with pytest.raises(ValueError, match="before fit"):
        prior(torch.zeros(3, 2), PriorContext(date=torch.tensor([0, 1, 2])))


def test_precomputed_prior_needs_dates_and_refuses_unseen_ones():
    prior = PrecomputedRegimePrior(2)
    table = torch.log(torch.tensor([[0.8, 0.2], [0.3, 0.7]]))
    prior.set_fitted_table(torch.tensor([0, 1]), table)

    with pytest.raises(ValueError, match="looked up by date"):
        prior(torch.zeros(2, 2))  # no PriorContext at all
    with pytest.raises(ValueError, match="outside the fitted information set"):
        prior(torch.zeros(1, 2), PriorContext(date=torch.tensor([99])))

    out = prior(torch.zeros(3, 2), PriorContext(date=torch.tensor([1, 0, 1])))
    assert torch.allclose(out.log_prior.exp()[0], torch.tensor([0.3, 0.7]), atol=1e-6)
    assert not out.log_prior.requires_grad  # contributes no gradient


def test_causal_table_never_overwrites_a_fitted_row():
    """The two date populations cannot be confused: apply may add later dates,
    never rewrite one the fit already saw."""
    prior = PrecomputedRegimePrior(2)
    fitted = torch.log(torch.tensor([[0.9, 0.1], [0.8, 0.2]]))
    prior.set_fitted_table(torch.tensor([0, 1]), fitted)
    prior.extend_causal_table(
        torch.tensor([0, 1, 2, 3]),
        torch.log(torch.tensor([[0.1, 0.9]] * 4)),
    )
    # dates 0 and 1 keep their FITTED values; 2 and 3 are added
    got = prior(
        torch.zeros(4, 2), PriorContext(date=torch.tensor([0, 1, 2, 3]))
    ).log_prior.exp()
    assert torch.allclose(got[0], torch.tensor([0.9, 0.1]), atol=1e-6)
    assert torch.allclose(got[2], torch.tensor([0.1, 0.9]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. + 7. Frozen parameters, and the applied filter matches a manual one
# --------------------------------------------------------------------------- #


def test_applied_parameters_are_identical_to_the_training_fit():
    panel = _sticky_panel(n_dates=220, n_entities=6)
    folds = walk_forward_folds(
        panel.date, n_folds=1, test_dates_per_fold=30, purge_dates=5
    )
    train = panel.subset_dates(folds[0].train_dates)
    cfg = _markov_cfg()
    prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
    prior.fit(train)
    fitted_params = prior.fit_result.params.copy()

    prior.apply_causal(panel)
    # element-wise identity, not "close": a re-estimate would differ in the
    # last digits and pass an allclose check
    assert np.array_equal(prior.fit_result.params, fitted_params)


def test_frozen_apply_matches_a_manual_filter_run():
    """Test 7: the probabilities the gate serves for a test date are exactly
    those of a filter run by hand with the same frozen parameters, moved to
    the configured gate weight (Q27; the default window average)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    panel = _sticky_panel(n_dates=220, n_entities=6)
    folds = walk_forward_folds(
        panel.date, n_folds=1, test_dates_per_fold=30, purge_dates=5
    )
    train = panel.subset_dates(folds[0].train_dates)
    cfg = _markov_cfg()
    prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
    prior.fit(train)
    full = panel.subset_dates(gate_span_dates(panel, folds[0]))  # Q22: gap included
    prior.apply_causal(full)

    # by hand: same series, same model spec, same frozen params, filter only
    dates, series = date_level_series(full, cfg.markov_gate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manual = MarkovRegression(
            series, k_regimes=2, trend=cfg.markov_gate.trend,
            switching_trend=cfg.markov_gate.switching_trend,
            switching_variance=cfg.markov_gate.switching_variance,
        ).filter(prior.fit_result.params)
    probs = np.asarray(manual.filtered_marginal_probabilities, dtype=float)
    probs = probs[:, np.asarray(prior.fit_result.permutation, dtype=int)]
    assert cfg.markov_gate.gate_weight == "window"
    probs = markov_gate_module.gate_weight_probs(
        probs, prior.fit_result.transition, "window", panel.horizon
    )

    test = panel.subset_dates(folds[0].test_dates)
    served = prior(
        torch.zeros(len(test), 2), PriorContext(date=test.date)
    ).log_prior.exp()
    first_test = int(torch.searchsorted(dates, test.date.min()))
    expected = torch.from_numpy(probs[first_test:]).to(torch.float32)
    # one row per test date, served once per row of the cross-section
    by_date = served[
        torch.tensor([int((test.date == d).nonzero()[0]) for d in torch.unique(test.date)])
    ]
    assert torch.allclose(by_date, expected, atol=1e-5)


def test_filtering_over_the_extended_series_leaves_training_dates_unchanged():
    """Why extending the series is legal at all: the filter at t conditions
    only on the series through t, so adding later dates cannot move an earlier
    filtered probability. If this failed, the whole apply path would be
    look-ahead."""
    panel = _sticky_panel(n_dates=200, n_entities=6)
    folds = walk_forward_folds(
        panel.date, n_folds=1, test_dates_per_fold=30, purge_dates=5
    )
    train = panel.subset_dates(folds[0].train_dates)
    cfg = _markov_cfg()
    prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
    prior.fit(train)
    train_only = prior(
        torch.zeros(len(train), 2), PriorContext(date=train.date)
    ).log_prior.clone()

    prior.apply_causal(panel)
    after_extension = prior(
        torch.zeros(len(train), 2), PriorContext(date=train.date)
    ).log_prior
    assert torch.equal(train_only, after_extension)


# --------------------------------------------------------------------------- #
# 4. The gate path never touches the smoother
# --------------------------------------------------------------------------- #


def test_gate_path_never_references_smoothed_probabilities():
    """A string-level assertion, which is the right strength here: the failure
    mode is a careless edit reaching for the more natural-looking attribute,
    not a subtle one."""
    source = inspect.getsource(markov_gate_module)
    code = "\n".join(
        line for line in source.splitlines()
        if not line.strip().startswith("#")
    )
    # the docstring names it in order to forbid it; strip docstrings, then look
    body = code.split('"""')
    executable = "".join(body[i] for i in range(0, len(body), 2))
    assert "smoothed_marginal_probabilities" not in executable
    assert "filtered_marginal_probabilities" in executable
    assert ".smooth(" not in executable


# --------------------------------------------------------------------------- #
# 5. Recovery of a known process, after canonical reordering
# --------------------------------------------------------------------------- #


def test_recovers_known_transition_matrix_and_durations():
    """Ground truth: stay probabilities (0.98, 0.90) -> expected durations
    (50, 10), with the calm regime first after the canonical sort."""
    stay, sigma = (0.98, 0.90), (0.5, 2.5)
    series, _ = _simulate_markov(2500, stay, sigma, seed=7)
    panel = _panel_from_series(series)
    cfg = _markov_cfg(search_reps=6)
    prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
    prior.fit(panel)
    fit = prior.fit_result

    # canonical order: ascending variance, so regime 0 is the calm one
    assert fit.variances[0] < fit.variances[1]
    assert fit.variances[0] == pytest.approx(sigma[0] ** 2, rel=0.35)
    assert fit.variances[1] == pytest.approx(sigma[1] ** 2, rel=0.35)
    # transition matrix is row-stochastic in OUR convention
    assert np.allclose(fit.transition.sum(axis=1), 1.0, atol=1e-6)
    assert fit.transition[0, 0] == pytest.approx(stay[0], abs=0.05)
    assert fit.transition[1, 1] == pytest.approx(stay[1], abs=0.07)
    # durations, and agreement with the package's own diagnostic
    true_durations = [1 / (1 - stay[0]), 1 / (1 - stay[1])]
    assert fit.expected_durations[0] == pytest.approx(true_durations[0], rel=0.35)
    assert fit.expected_durations[1] == pytest.approx(true_durations[1], rel=0.35)
    ours = expected_durations(torch.from_numpy(fit.transition))
    assert np.allclose(ours.numpy(), fit.expected_durations, rtol=1e-4)


def test_transition_matrix_orientation_is_row_stochastic():
    """statsmodels returns a COLUMN-stochastic matrix; ours is row-stochastic.
    An asymmetric chain is the only way to see the difference, and getting it
    wrong silently transposes every regime's persistence."""
    stay, sigma = (0.99, 0.80), (0.4, 2.0)
    series, _ = _simulate_markov(3000, stay, sigma, seed=11)
    panel = _panel_from_series(series)
    prior = MarkovSwitchingRegimePrior(2, _markov_cfg(search_reps=4).markov_gate)
    prior.fit(panel)
    a = prior.fit_result.transition
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-6)  # rows, not columns
    # the very sticky state is the calm one here, and canonical order puts it
    # first; a transposed matrix would report its persistence as the other's
    assert a[0, 0] > a[1, 1]
    assert a[0, 0] == pytest.approx(stay[0], abs=0.04)


# --------------------------------------------------------------------------- #
# 6. Canonical ordering is stable across starting values
# --------------------------------------------------------------------------- #


def test_canonical_ordering_is_stable_across_seeds():
    """Two fits of the same data from different random starts must produce the
    same regime labelling after the sort — otherwise cross-fold per-regime
    statistics are averaging different things."""
    series, _ = _simulate_markov(2000, (0.97, 0.93), (0.5, 2.5), seed=5)
    panel = _panel_from_series(series)
    fits = []
    for start_seed in (0, 12345):
        cfg = _markov_cfg(search_reps=4, start_seed=start_seed)
        prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
        prior.fit(panel)
        fits.append(prior.fit_result)

    a, b = fits
    # same canonical content, whatever raw labels the optimizer produced
    assert np.allclose(a.variances, b.variances, rtol=0.05)
    assert np.allclose(a.transition, b.transition, atol=0.05)
    assert a.variances[0] < a.variances[1] and b.variances[0] < b.variances[1]


def _turbulent_first_start(model, series, cfg):
    """One informed centre with the two regimes' moments swapped, so the
    optimizer starts, and stays, with the HIGH-variance regime labelled 0.

    The informed schemes always label the calmest group 0, which makes the
    canonical sort a no-op in every other test here (audit finding G-6, where
    removing the sort passed the whole suite). This start is what gives the
    sort real work to do.
    """
    centre = markov_gate_module.informed_centre(model, series, 0.5, 20, 0.95, cfg)
    names = list(model.param_names)
    swapped = centre.copy()
    for a, b in (("const[0]", "const[1]"), ("sigma2[0]", "sigma2[1]")):
        i, j = names.index(a), names.index(b)
        swapped[i], swapped[j] = centre[j], centre[i]
    markov_gate_module.validate_start(names, swapped, cfg.k_regimes)
    return [swapped]


def test_canonical_reordering_relabels_turbulent_first_raw_output(monkeypatch):
    """Audit G-6 / critical property P13. When statsmodels hands back the
    turbulent regime as raw label 0, the gate must still store the calm regime
    first, and must permute EVERYTHING it stores the same way: the moments,
    the transition matrix, the durations and the filtered-probability table.

    Asymmetric truth (a very sticky calm regime, a short turbulent one) is what
    makes a missed or partial permutation visible.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    stay, sigma = (0.99, 0.80), (0.4, 2.0)
    series, truth = _simulate_markov(3000, stay, sigma, seed=11)
    panel = _panel_from_series(series)
    monkeypatch.setitem(
        markov_gate_module.START_SCHEME_REGISTRY, "turbulent_first", _turbulent_first_start
    )
    cfg = _markov_cfg(start_scheme="turbulent_first")
    prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
    prior.fit(panel)
    fit = prior.fit_result

    # the optimizer kept the swapped labels: without this the test proves nothing
    assert fit.permutation == (1, 0), fit.permutation
    # moments, transition and durations all come out calm-first
    assert fit.variances[0] < fit.variances[1]
    assert fit.variances[0] == pytest.approx(sigma[0] ** 2, rel=0.35)
    assert fit.transition[0, 0] == pytest.approx(stay[0], abs=0.04)
    assert fit.transition[1, 1] == pytest.approx(stay[1], abs=0.08)
    assert np.allclose(fit.transition.sum(axis=1), 1.0, atol=1e-6)
    assert fit.expected_durations[0] > fit.expected_durations[1]
    g = fit.metrics()
    assert g["gate_variance_0"] < g["gate_variance_1"]
    assert g["gate_transition_0_0"] == fit.transition[0, 0]

    # the stored table is the raw filter with its columns permuted the same way
    dates = torch.unique(panel.date, sorted=True)
    served = prior(torch.zeros(len(dates), 2), PriorContext(date=dates)).log_prior.exp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = MarkovRegression(
            date_level_series(panel, cfg.markov_gate)[1], k_regimes=2,
            trend=cfg.markov_gate.trend,
            switching_trend=cfg.markov_gate.switching_trend,
            switching_variance=cfg.markov_gate.switching_variance,
        ).filter(fit.params)
    raw_probs = np.asarray(raw.filtered_marginal_probabilities, dtype=float)
    # the served rows are the permuted filter moved to the gate weight with
    # the permuted transition matrix (Q27): both must use the same order
    expected = torch.from_numpy(
        markov_gate_module.gate_weight_probs(
            raw_probs[:, list(fit.permutation)], fit.transition,
            cfg.markov_gate.gate_weight, panel.horizon,
        )
    ).to(torch.float32)
    assert torch.allclose(served, expected, atol=1e-5)
    # ... and column 0 really is the calm regime on the true path
    calm = torch.from_numpy(truth == 0)
    assert float(served[calm, 0].mean()) > 0.8
    assert float(served[~calm, 0].mean()) < 0.5


def test_ordering_rule_is_config_and_degenerate_sorts_are_refused():
    """The ordering is a declared rule, not an assumption — and a sort key
    that cannot vary across regimes is refused rather than producing an
    arbitrary order wearing a canonical name (Stephens 2000: identifiability
    constraints are a convention, not a solution)."""
    cfg = _markov_cfg(order_by="mean")
    cfg.validate()  # mean + switching_trend=True is fine
    with pytest.raises(ValueError, match="order_by='variance' requires"):
        _markov_cfg(switching_variance=False).validate()
    with pytest.raises(ValueError, match="order_by='mean' requires"):
        _markov_cfg(order_by="mean", switching_trend=False).validate()
    with pytest.raises(ValueError, match="unknown markov_gate.order_by"):
        _markov_cfg(order_by="banana").validate()


# --------------------------------------------------------------------------- #
# Section 2 (what it is fitted on) and section 5 (multi-start as selection)
# --------------------------------------------------------------------------- #


def test_series_is_a_registry_choice_and_must_be_causal():
    assert set(SERIES_REGISTRY) == {
        "crsp_market_log_return", "market_excess_return", "sequence_feature",
        "sequence_channel",
    }
    panel = _sticky_panel(n_dates=60, n_entities=4)
    cfg = _markov_cfg()
    dates, series = date_level_series(panel, cfg.markov_gate)
    assert len(series) == len(torch.unique(panel.date))
    assert np.isfinite(series).all()
    with pytest.raises(ValueError, match="unknown markov_gate.series"):
        date_level_series(
            panel, dataclasses.replace(cfg.markov_gate, series="nope")
        )
    # the named-feature path needs named features, and says so
    with pytest.raises(ValueError, match="not a sequence feature"):
        date_level_series(
            panel,
            dataclasses.replace(cfg.markov_gate, series="sequence_feature"),
        )
    # the French and CRSP paths need calendar dates to align on, and say so
    for key in ("market_excess_return", "crsp_market_log_return"):
        with pytest.raises(ValueError, match="no date_labels"):
            date_level_series(panel, dataclasses.replace(cfg.markov_gate, series=key))


def _french_fixture_dir(tmp_path: Path, rows: list[tuple[str, float]]) -> Path:
    """A cache dir holding a minimal French daily-factors zip (offline)."""
    import zipfile

    body = "".join(f"{d.replace('-', '')},{r * 100:.4f},0.00,0.00,0.00\n" for d, r in rows)
    csv = (
        "This file was created by CMPT_ME_BEME_RETS_DAILY\n\n"
        ",Mkt-RF,SMB,HML,RF\n" + body + "\nCopyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    with zipfile.ZipFile(tmp_path / "ff_factors_daily.zip", "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.CSV", csv)
    return tmp_path


def test_market_excess_return_is_french_mkt_rf_aligned_by_date(tmp_path: Path):
    """The key brief 03 §2 specified: French Mkt-RF, aligned by calendar date
    through the panel's date labels, with a missing date refused rather than
    filled."""
    panel = _sticky_panel(n_dates=6, n_entities=3)
    labels = ["2020-03-02", "2020-03-03", "2020-03-04",
              "2020-03-05", "2020-03-06", "2020-03-09"]
    panel = dataclasses.replace(panel, date_labels=labels)
    rets = [0.0461, -0.0281, 0.0422, -0.0339, -0.0171, -0.0760]
    ctx = _french_fixture_dir(tmp_path, list(zip(labels, rets, strict=True)))
    cfg = dataclasses.replace(
        _markov_cfg().markov_gate, series="market_excess_return", context_dir=str(ctx)
    )
    dates, series = date_level_series(panel, cfg)
    assert np.allclose(series, rets, atol=1e-9)  # decimal, in date order

    # one date missing from the factor file -> refused, not forward-filled
    short = tmp_path / "short"
    short.mkdir()
    _french_fixture_dir(short, list(zip(labels[:-1], rets[:-1], strict=True)))
    with pytest.raises(ValueError, match="no French factor row"):
        date_level_series(panel, dataclasses.replace(cfg, context_dir=str(short)))


def test_every_start_is_logged_as_a_trial(tmp_path: Path):
    """Brief 03 §5: best-of-N is a selection event, so the multiplicity has to
    reach the registry rather than living inside the fit."""
    panel = _sticky_panel(n_dates=180, n_entities=6)
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    cfg = _markov_cfg(registry_tag="gate_starts")
    mg = cfg.markov_gate
    # the scheme decides the count: centres x draws for informed_jitter
    # (search_reps only governs default_jitter)
    n_expected = (
        len(mg.start_vol_quantiles) * len(mg.start_vol_windows)
        * len(mg.start_persistences) * mg.start_draws_per_centre
    )

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = walk_forward_evaluate(
            panel, make_trainer, n_folds=2, test_dates_per_fold=15,
            purge_dates=5, steps=5, registry=reg,
        )
    starts = reg.trials("gate_starts")
    assert len(starts) >= 2  # at least the converged starts of both folds
    assert {r.config["arm"] for r in starts} == {"gate_starts"}
    assert any(r.config["chosen"] for r in starts)
    assert all(math.isfinite(r.metrics["llf"]) for r in starts)

    # and the fold carries the permutation + the fitted gate's own summary
    for fold in result.folds:
        assert sorted(fold.gate_permutation) == [0, 1]
        g = dict(fold.gate_metrics)
        assert g["gate_n_starts"] == n_expected
        assert g["gate_n_converged"] >= 1
        assert g["gate_expected_duration_0"] > 1.0
        assert 0.0 < g["gate_stay_prob_0"] < 1.0


def test_non_convergence_is_reported_not_silently_absorbed():
    """A fit where nothing converges must raise, not return a bad model."""
    cfg = _markov_cfg(search_reps=2, maxiter=1)
    prior = MarkovSwitchingRegimePrior(2, cfg.markov_gate)
    panel = _panel_from_series(np.zeros(200))  # degenerate: no regimes at all
    with pytest.raises((RuntimeError, ValueError)):
        prior.fit(panel)


# --------------------------------------------------------------------------- #
# Integration: the Hamilton gate drives a real fold
# --------------------------------------------------------------------------- #


def test_markov_gate_runs_through_the_harness_and_is_date_level():
    panel = _sticky_panel(n_dates=220, n_entities=8)
    cfg = _markov_cfg(search_reps=3)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = walk_forward_evaluate(
            panel, make_trainer, n_folds=2, test_dates_per_fold=15,
            purge_dates=5, steps=25,
        )
    assert len(result.folds) == 2
    for f in result.folds:
        assert math.isfinite(f.nll)

    # the gate is a DATE-level variable: identical for every name on a date
    folds = walk_forward_folds(
        panel.date, n_folds=2, test_dates_per_fold=15, purge_dates=5
    )
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    train = panel.subset_dates(folds[0].train_dates)
    trainer.model.prior.fit(train)
    trainer.model.prior.apply_causal(panel.subset_dates(gate_span_dates(panel, folds[0])))
    test = panel.subset_dates(folds[0].test_dates)
    pi = trainer.model(
        test.x_seq, test.x_snap, PriorContext(date=test.date)
    ).prior.log_prior.exp()
    for d in torch.unique(test.date):
        rows = pi[test.date == d]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-7)


def _panel_from_series(series: np.ndarray):
    """Wrap a univariate series as a panel whose channel 0 carries it.

    The gate reads a date-level series off the panel, so a recovery test needs
    a panel whose series *is* the simulated process. Entities are duplicates:
    the gate is date-level, so the cross-section is irrelevant to it.
    """
    n_dates, n_entities, seq_len = len(series), 3, 8
    x_seq = torch.zeros(n_dates * n_entities, seq_len, 2)
    vals = torch.from_numpy(series).to(torch.float32)
    date = torch.repeat_interleave(torch.arange(n_dates), n_entities)
    x_seq[:, -1, 0] = vals.repeat_interleave(n_entities)
    from nec_moe import FeatureSchema, SyntheticPanel

    g = torch.Generator().manual_seed(0)
    n = n_dates * n_entities
    return SyntheticPanel(
        x_seq=x_seq,
        x_snap=torch.randn(n, 3, generator=g),
        y=torch.randn(n, generator=g),
        date=date,
        entity=torch.arange(n_entities).repeat(n_dates),
        schema=FeatureSchema(("seq_0", "seq_1"), ("snap_0", "snap_1", "snap_2"), "y"),
        regime=torch.zeros(n, dtype=torch.long),
        y_clean=torch.zeros(n),
        betas=torch.zeros(2, 3),
        transition=None,
        spec=SyntheticSpec(seed=0),
    )
