"""Starting values for the Hamilton fit (brief 04 section B): one test per claim.

The diagnosis being fixed: the old scheme added absolute noise to statsmodels'
*constrained* start vector, where transition probabilities, daily means and
daily variances live on scales orders of magnitude apart — so on realistic
returns most starts were invalid and raised, and every start was centred on
an uninformed point with no persistence. The claims pinned here:

1. informed starts are valid before any fitting happens;
2. on a daily-scale series, informed_jitter converges >= 95% of its starts;
3. its best log-likelihood is at least the old scheme's;
4. the informed centre uses no data after each date;
5. seeded runs are reproducible;
6. default_jitter reproduces the pre-change behaviour exactly — checked
   against a verbatim copy of the old loop, not a paraphrase of it.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest
from conftest import small_config

from nec_moe import MarkovSwitchingRegimePrior
from nec_moe.markov_gate import (
    START_SCHEME_REGISTRY,
    causal_vol_assignment,
    date_level_series,
    distinct_optima,
    validate_start,
)

pytest.importorskip("statsmodels")

from statsmodels.tsa.regime_switching.markov_regression import (  # noqa: E402
    MarkovRegression,
)
from test_markov_gate import _panel_from_series  # noqa: E402


def _daily_series(kind: str = "well_separated", n: int = 2000, seed: int = 0) -> np.ndarray:
    """A two-state series at DAILY return scale — the scale the bug lives at."""
    spec = {
        "well_separated": ((0.98, 0.95), (4e-4, -8e-4), (0.006, 0.020)),
        "mean_switch": ((0.97, 0.97), (1.2e-3, -1.2e-3), (0.010, 0.010)),
    }[kind]
    stay, mu, sd = spec
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.uniform() < stay[s[t - 1]] else 1 - s[t - 1]
    return rng.normal(np.asarray(mu)[s], np.asarray(sd)[s])


def _gate_cfg(**overrides):
    cfg = small_config(prior_kind="markov", freeze_gate=True)
    gate = dict(series="sequence_channel", series_channel=0)
    gate.update(overrides)
    return dataclasses.replace(cfg.markov_gate, **gate)


def _model(series: np.ndarray, cfg) -> MarkovRegression:
    return MarkovRegression(
        series, k_regimes=cfg.k_regimes, trend=cfg.trend,
        switching_variance=cfg.switching_variance,
        switching_trend=cfg.switching_trend,
    )


def _fit(series: np.ndarray, **overrides):
    cfg = _gate_cfg(**overrides)
    prior = MarkovSwitchingRegimePrior(cfg.k_regimes, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior.fit(_panel_from_series(series))
    return prior.fit_result


# --------------------------------------------------------------------------- #
# 1. Valid before fitting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["informed_grid", "informed_jitter"])
@pytest.mark.parametrize("k", [2, 3])
def test_informed_starts_are_valid_before_fitting(scheme: str, k: int):
    series = _daily_series()
    cfg = _gate_cfg(start_scheme=scheme, k_regimes=k)
    model = _model(series, cfg)
    names = list(model.param_names)
    starts = START_SCHEME_REGISTRY[scheme](model, series, cfg)
    assert starts and all(s is not None for s in starts)
    for vec in starts:
        validate_start(names, vec, k)  # raises on any violation
        sig = [vec[i] for i, n in enumerate(names) if n.startswith("sigma2")]
        prob = [vec[i] for i, n in enumerate(names) if n.startswith("p[")]
        assert all(v > 0 for v in sig)
        assert all(0.0 <= v <= 1.0 for v in prob)
        # each origin's full row, with the implicit last destination, sums to 1
        for src in range(k):
            free = sum(
                vec[i] for i, n in enumerate(names)
                if n.startswith(f"p[{src}->")
            )
            assert 0.0 <= free <= 1.0 + 1e-12
            assert free + (1.0 - free) == pytest.approx(1.0)


def test_the_old_scheme_is_what_produces_invalid_starts():
    """The contrast that makes test 1 mean something: absolute noise in the
    constrained space does produce invalid starts at daily scale."""
    series = _daily_series()
    cfg = _gate_cfg(start_scheme="default_jitter", start_jitter=0.05, search_reps=40)
    model = _model(series, cfg)
    names = list(model.param_names)
    invalid = 0
    for vec in START_SCHEME_REGISTRY["default_jitter"](model, series, cfg)[1:]:
        try:
            validate_start(names, vec, 2)
        except ValueError:
            invalid += 1
    assert invalid > 0


def test_validate_start_rejects_each_kind_of_violation():
    names = ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2[0]", "sigma2[1]"]
    good = np.array([0.95, 0.05, 0.0, 0.0, 1e-4, 4e-4])
    validate_start(names, good, 2)
    for i, value, match in [(4, -1e-5, "not > 0"), (0, 1.2, r"not in \[0, 1\]")]:
        bad = good.copy()
        bad[i] = value
        with pytest.raises(ValueError, match=match):
            validate_start(names, bad, 2)


# --------------------------------------------------------------------------- #
# 2. + 3. Convergence rate and best log-likelihood
# --------------------------------------------------------------------------- #


def test_informed_jitter_converges_at_least_95_percent():
    fit = _fit(_daily_series(), start_scheme="informed_jitter")
    assert fit.n_converged / fit.n_starts >= 0.95, (fit.n_converged, fit.n_starts)
    assert fit.start_status.count("failed") == 0


def test_informed_jitter_best_is_at_least_default_jitters():
    series = _daily_series("mean_switch", seed=1)
    informed = _fit(series, start_scheme="informed_jitter")
    old = _fit(series, start_scheme="default_jitter", start_jitter=0.05)
    assert informed.llf >= old.llf - 1e-6, (informed.llf, old.llf)


# --------------------------------------------------------------------------- #
# 4. No future data at initialisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("window", [20, 60])
def test_informed_split_uses_no_data_after_each_date(window: int):
    series = _daily_series(n=800)
    t = 500
    before = causal_vol_assignment(series, window, 0.75, 2, min_history=20)
    perturbed = series.copy()
    perturbed[t + 1:] += np.random.default_rng(99).normal(0, 0.05, len(series) - t - 1)
    after = causal_vol_assignment(perturbed, window, 0.75, 2, min_history=20)
    assert np.array_equal(before[: t + 1], after[: t + 1])
    # and the perturbation did matter later, or the test would prove nothing
    assert not np.array_equal(before[t + 1:], after[t + 1:])


def test_a_centred_window_would_fail_the_same_check():
    """Guard the guard: the property above is not automatic — a centred
    rolling window, the mistake the brief names, violates it."""
    import pandas as pd

    series = _daily_series(n=800)
    t, w = 500, 20

    def centred(y):
        return pd.Series(y).rolling(w, center=True, min_periods=w).std().to_numpy()

    perturbed = series.copy()
    perturbed[t + 1:] += 0.05
    assert not np.allclose(centred(series)[: t + 1], centred(perturbed)[: t + 1],
                           equal_nan=True)


# --------------------------------------------------------------------------- #
# 5. Reproducibility
# --------------------------------------------------------------------------- #


def test_seeded_runs_are_reproducible():
    series = _daily_series(n=1200)
    small = dict(start_vol_quantiles=(0.75,), start_vol_windows=(20,),
                 start_persistences=(0.97,), start_draws_per_centre=4)
    a = _fit(series, start_scheme="informed_jitter", start_seed=7, **small)
    b = _fit(series, start_scheme="informed_jitter", start_seed=7, **small)
    assert np.array_equal(np.asarray(a.start_llfs), np.asarray(b.start_llfs), equal_nan=True)
    assert np.array_equal(a.params, b.params)
    assert a.start_status == b.start_status

    cfg = _gate_cfg(start_scheme="informed_jitter", start_seed=7, **small)
    model = _model(series, cfg)
    s1 = START_SCHEME_REGISTRY["informed_jitter"](model, series, cfg)
    s2 = START_SCHEME_REGISTRY["informed_jitter"](
        model, series, dataclasses.replace(cfg, start_seed=8)
    )
    assert not all(np.array_equal(x, y) for x, y in zip(s1, s2, strict=True))


# --------------------------------------------------------------------------- #
# 6. default_jitter is the pre-change behaviour, exactly
# --------------------------------------------------------------------------- #


def _legacy_multi_start(model, c):
    """VERBATIM the pre-brief-04 loop (MarkovSwitchingRegimePrior at 1bfe7ca),
    with only its return value reshaped for comparison."""
    rng = np.random.default_rng(c.start_seed)
    best, best_llf, best_i = None, -np.inf, -1
    llfs: list[float] = []
    starts: list = []
    for i in range(max(c.search_reps, 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if i == 0:
                    start = None
                else:  # the old _jitter(model, rng, c.start_jitter)
                    s0 = np.asarray(model.start_params, dtype=float)
                    start = s0 + rng.normal(0.0, c.start_jitter, size=s0.shape)
                starts.append(start)
                res = model.fit(start_params=start, maxiter=c.maxiter, search_reps=0, disp=0)
            except Exception:
                llfs.append(float("nan"))
                continue
        converged = bool(res.mle_retvals.get("converged", False))
        llf = float(res.llf)
        llfs.append(llf)
        if converged and llf > best_llf:
            best, best_llf, best_i = res, llf, i  # noqa: F841 - kept verbatim
    return starts, llfs, best_llf, best_i


def test_default_jitter_reproduces_the_pre_change_behaviour_exactly():
    raw = _daily_series(n=1500)
    cfg = _gate_cfg(start_scheme="default_jitter", start_jitter=0.05, search_reps=12,
                    start_seed=3)
    # the legacy loop must see EXACTLY the series the prior fits on — which
    # has been through the panel (float32) — or the comparison is between two
    # datasets and differs in the ninth significant digit for that reason alone
    _, series = date_level_series(_panel_from_series(raw), cfg)
    model = _model(series, cfg)
    legacy_starts, legacy_llfs, legacy_best, legacy_i = _legacy_multi_start(model, cfg)

    new_starts = START_SCHEME_REGISTRY["default_jitter"](model, series, cfg)
    assert len(new_starts) == len(legacy_starts)
    for new, old in zip(new_starts, legacy_starts, strict=True):
        assert (new is None and old is None) or np.array_equal(new, old)

    fit = _fit(raw, start_scheme="default_jitter", start_jitter=0.05,
               search_reps=12, start_seed=3)
    assert np.array_equal(np.asarray(fit.start_llfs), np.asarray(legacy_llfs), equal_nan=True)
    assert fit.llf == legacy_best
    assert fit.chosen_start == legacy_i


def test_default_scheme_is_informed_jitter():
    assert _gate_cfg().start_scheme == "informed_jitter"
    assert set(START_SCHEME_REGISTRY) == {"default_jitter", "informed_grid", "informed_jitter"}


# --------------------------------------------------------------------------- #
# B.3: what is recorded per fit
# --------------------------------------------------------------------------- #


def test_failed_and_nonconverged_are_counted_separately():
    series = _daily_series(n=1500)
    fit = _fit(series, start_scheme="default_jitter", start_jitter=0.05, search_reps=20)
    m = fit.metrics()
    outcomes = m["gate_n_failed"] + m["gate_n_nonconverged"] + m["gate_n_converged"]
    assert outcomes == m["gate_n_starts"]
    assert m["gate_n_failed"] == fit.start_status.count("failed")
    assert m["gate_n_failed"] > 0  # the old scheme's invalid starts raise

    # A starved optimizer produces the OTHER kind of failure: valid starts that
    # ran and did not converge. Counted as such, not lumped with the raised
    # ones, and the fit refuses to fall back to a non-converged result.
    with pytest.raises(RuntimeError, match=r"\(0 raised, 8 did not converge\)"):
        _fit(series, start_scheme="informed_grid", maxiter=1)


def test_distinct_optima_cluster_by_gap_not_by_rounding():
    assert distinct_optima([100.0, 99.6, 99.2], tol=1.0) == [100.0]
    assert distinct_optima([100.0, 98.5, 98.4, 90.0], tol=1.0) == [100.0, 98.5, 90.0]
    # 99.49 and 99.51 straddle a rounding boundary but are one optimum
    assert len(distinct_optima([99.49, 99.51], tol=1.0)) == 1


def test_best_minus_second_is_omitted_with_a_single_optimum():
    """Undefined with one optimum; reporting 0 would read as a tie."""
    series = _daily_series(n=1500)
    fit = _fit(series, start_scheme="informed_jitter")
    m = fit.metrics()
    if m["gate_n_distinct_optima"] == 1:
        assert "gate_best_minus_second" not in m
    else:
        assert m["gate_best_minus_second"] > 0
    assert all(math.isfinite(v) for v in m.values())  # registry-safe


def test_date_level_series_feeds_the_starts():
    """The starts are built from the same series the fit sees."""
    series = _daily_series(n=300)
    dates, got = date_level_series(_panel_from_series(series), _gate_cfg())
    assert np.allclose(got, series.astype(np.float32), atol=1e-7)
