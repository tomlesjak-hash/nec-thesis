"""Brief 09 D: the native Baum-Welch and the regime-clock weighted gate fit.

D.1 validates the native estimator against statsmodels before any weighting:
on synthetic series (always) and on the real market series 2000-2008 (skips
without the data): log-likelihood, parameters and filtered probabilities. The
real-data checks are gate-fit diagnostics only: no prediction is scored.
D.5 pins the weighted fit's behaviour on synthetic series.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nec_moe import (
    FeatureSchema,
    MarkovGateConfig,
    Panel,
    SyntheticRegimePanel,
    SyntheticSpec,
)
from nec_moe import baum_welch as bw
from nec_moe.decay import decay_factor
from nec_moe.markov_gate import (
    GateNotSettledWarning,
    MarkovSwitchingRegimePrior,
    _row_stochastic_transition,
    gate_weight_probs,
    regime_clock_gate_weights,
)

statsmodels = pytest.importorskip("statsmodels")
from statsmodels.tsa.regime_switching.markov_regression import (  # noqa: E402
    MarkovRegression,
)


def _chain(t_len: int, stay: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    z = np.zeros(t_len, dtype=int)
    for t in range(1, t_len):
        z[t] = z[t - 1] if rng.random() < stay[z[t - 1]] else 1 - z[t - 1]
    return z


def _daily_series(t_len: int = 2500, seed: int = 1) -> np.ndarray:
    """Two-state daily-scale returns: calm 0.7% sd, stress 2% sd."""
    rng = np.random.default_rng(seed)
    z = _chain(t_len, (0.98, 0.95), rng)
    return rng.normal(np.where(z == 0, 0.0005, -0.001), np.where(z == 0, 0.007, 0.02))


def _sm_fit(x: np.ndarray):
    mod = MarkovRegression(x, k_regimes=2, trend="c", switching_variance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(search_reps=20, maxiter=500, disp=0)
    assert res.mle_retvals.get("converged")
    return mod, res


def _sm_filtered(res) -> np.ndarray:
    f = np.asarray(res.filtered_marginal_probabilities)
    return f.T if f.shape[0] < f.shape[1] else f


# --------------------------------------------------------------------------- #
# D.1 the unweighted native Baum-Welch against statsmodels
# --------------------------------------------------------------------------- #


def test_the_native_filter_is_statsmodels_filter():
    """At statsmodels' own fitted parameters, the native scaled forward pass
    gives the same log-likelihood and filtered probabilities: the E-step and
    the steady-state initial distribution are the same computation."""
    x = _daily_series()
    mod, res = _sm_fit(x)
    params = bw.params_from_statsmodels(list(mod.param_names), res.params, 2)
    assert np.allclose(params.transition, _row_stochastic_transition(res), atol=1e-15)
    filtered, log_c = bw.filter_series(x, params)
    assert log_c.sum() == pytest.approx(res.llf, abs=1e-8)
    assert np.abs(filtered - _sm_filtered(res)).max() < 1e-12
    vec = bw.params_to_statsmodels(params, list(mod.param_names))
    assert np.allclose(vec, res.params, rtol=0, atol=1e-15)


def test_native_baum_welch_reaches_the_statsmodels_optimum():
    x = _daily_series()
    mod, res = _sm_fit(x)
    start = bw.params_from_statsmodels(list(mod.param_names), mod.start_params, 2)
    fit = bw.baum_welch(x, start, variance_floor=1e-6 * x.var(), tol=1e-9, maxiter=5000)
    assert fit.converged
    # the native optimum is at least as high, and the same to optimiser tolerance
    assert fit.loglik >= res.llf - 1e-6 and fit.loglik - res.llf < 1e-4
    sm = bw.params_from_statsmodels(list(mod.param_names), res.params, 2)
    order_n, order_s = np.argsort(fit.params.variances), np.argsort(sm.variances)
    assert np.allclose(fit.params.variances[order_n], sm.variances[order_s], rtol=1e-4)
    assert np.allclose(fit.params.means[order_n], sm.means[order_s], rtol=1e-3, atol=1e-7)
    a_n = fit.params.transition[np.ix_(order_n, order_n)]
    a_s = sm.transition[np.ix_(order_s, order_s)]
    assert np.allclose(a_n, a_s, atol=1e-5)
    assert np.abs(fit.filtered[:, order_n] - _sm_filtered(res)[:, order_s]).max() < 1e-4
    # EM never lowers the likelihood (Baum's result, generalised to the
    # stationary rule's exact numerical A-step)
    lls = [t[0] for t in fit.trace]
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:], strict=False))


def test_the_textbook_free_initial_distribution_fits_at_least_as_well():
    x = _daily_series()
    mod, _ = _sm_fit(x)
    start = bw.params_from_statsmodels(list(mod.param_names), mod.start_params, 2)
    st = bw.baum_welch(x, start, initial="stationary", tol=1e-9)
    est = bw.baum_welch(x, start, initial="estimated", tol=1e-9)
    assert est.converged and est.loglik >= st.loglik - 1e-6  # one more free parameter
    assert est.params.initial.sum() == pytest.approx(1.0)


def _gate(**kw) -> MarkovSwitchingRegimePrior:
    cfg = MarkovGateConfig(series="sequence_channel", series_channel=0, **kw)
    return MarkovSwitchingRegimePrior(2, cfg, horizon=1)


def _synthetic_panel(n_dates: int = 1500, seed: int = 3):
    return SyntheticRegimePanel(SyntheticSpec(
        regime_process="markov", transition_stay=0.97, vol_levels=(0.5, 2.5), seed=seed,
    )).generate(n_dates, 4)


def test_the_native_gate_matches_the_statsmodels_gate():
    """Through the prior (multi-start from the same starting values): the
    log-likelihood, the canonical parameters, and the filtered probabilities
    on training and later dates agree; the frozen native filter reproduces
    the fit's own probabilities on the training dates exactly."""
    panel = _synthetic_panel()
    dates = torch.unique(panel.date)
    train = panel.subset_dates(dates[:1200])
    gates = {}
    for backend in ("statsmodels", "native"):
        g = _gate(fit_backend=backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g.fit(train)
        g.apply_causal(panel)
        gates[backend] = g
    s, n = gates["statsmodels"].fit_result, gates["native"].fit_result
    assert s is not None and n is not None and n.backend == "native"
    assert n.llf == pytest.approx(s.llf, abs=1e-4) and n.llf >= s.llf - 1e-6
    assert n.n_converged == n.n_starts == s.n_starts
    assert np.allclose(n.variances, s.variances, rtol=1e-4)
    assert np.allclose(n.means, s.means, rtol=1e-3, atol=1e-6)
    assert np.allclose(n.transition, s.transition, atol=1e-5)
    fs = gates["statsmodels"].filtered_probabilities(dates)
    fn = gates["native"].filtered_probabilities(dates)
    assert np.abs(fs - fn).max() < 1e-4
    # downstream unchanged: the served rows are the Q27 rule applied to them
    table = gates["native"](torch.zeros(len(dates), 2),
                            ctx=_ctx(dates)).log_prior.exp().numpy()
    expect = gate_weight_probs(fn, n.transition, "window", 1)
    assert np.allclose(table, expect / expect.sum(axis=1, keepdims=True), atol=1e-6)


def _ctx(dates):
    from nec_moe import PriorContext

    return PriorContext(date=dates)


# --------------------------------------------------------------------------- #
# D.2-D.5 the regime-clock weighted fit
# --------------------------------------------------------------------------- #


def test_an_infinite_gate_half_life_reproduces_the_unweighted_fit_exactly():
    panel = _synthetic_panel(900)
    train = panel.subset_dates(torch.unique(panel.date)[:800])
    full, clock = _gate(fit_backend="native"), _gate(
        fit_backend="native", memory="regime_clock", gate_half_life=math.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full.fit(train)
        clock.fit(train)
    a, b = full.fit_result, clock.fit_result
    assert a is not None and b is not None
    assert np.array_equal(a.params, b.params) and a.llf == b.llf
    assert np.array_equal(a.transition, b.transition) and np.array_equal(a.variances, b.variances)
    assert torch.equal(full._log_table, clock._log_table)
    assert b.weighted_settled is None  # no second pass was needed


def test_the_regime_clock_tracks_drifting_calm_and_keeps_stress():
    """D.5: the calm regime's variance drifts down over 5,000 dates while the
    stress regime keeps its parameters. The weighted fit tracks the recent
    calm variance and keeps the stress variance; full memory averages the
    drift away."""
    rng = np.random.default_rng(7)
    t_len = 5000
    z = _chain(t_len, (0.99, 0.97), rng)
    sd_calm = np.linspace(1.0, 0.5, t_len)
    x = rng.normal(0.0, np.where(z == 0, sd_calm, 3.0))
    v = x.var()
    start = bw.HMMParams(np.zeros(2), np.array([0.5 * v, 3 * v]),
                         np.array([[0.98, 0.02], [0.05, 0.95]]), np.full(2, 0.5))
    full = bw.baum_welch(x, start, variance_floor=1e-6 * v, tol=1e-8, maxiter=20000)
    omega = regime_clock_gate_weights(full.filtered, 250.0)
    clock = bw.baum_welch(x, full.params, weights=omega, variance_floor=1e-6 * v,
                          tol=1e-8, maxiter=20000)
    assert full.converged and clock.converged
    recent = sd_calm[-300:].mean() ** 2  # about 0.27
    calm_full, stress_full = np.sort(full.params.variances)
    calm_clock, stress_clock = np.sort(clock.params.variances)
    assert abs(calm_clock - recent) < 0.3 * abs(calm_full - recent)
    assert abs(calm_clock - recent) / recent < 0.25
    assert abs(stress_clock - 9.0) / 9.0 < 0.15  # stress kept (true variance 9)
    # the weighted objective is logged every iteration
    assert len(clock.trace) == clock.n_iter and all(math.isfinite(t[1]) for t in clock.trace)


def test_a_long_calm_end_does_not_remove_the_stress_state():
    """D.5 and theory notes section 7.3: stress episodes, then 4,000 calm
    dates. Calendar forgetting at H = 250 days forgets the episodes and the
    two states both become calm (the failure mode); the regime clock at
    H = 250 regime-days keeps a stress state near the true variance 9."""
    rng = np.random.default_rng(11)
    z1 = _chain(2500, (0.985, 0.95), rng)
    x = np.concatenate([rng.normal(0.0, np.where(z1 == 0, 1.0, 3.0)), rng.normal(0.0, 1.0, 4000)])
    v = x.var()
    start = bw.HMMParams(np.zeros(2), np.array([0.5 * v, 3 * v]),
                         np.array([[0.98, 0.02], [0.05, 0.95]]), np.full(2, 0.5))
    first = bw.baum_welch(x, start, variance_floor=1e-6 * v, tol=1e-8, maxiter=20000)
    h = 250.0
    clock = bw.baum_welch(x, first.params, weights=regime_clock_gate_weights(first.filtered, h),
                          variance_floor=1e-6 * v, tol=1e-8, maxiter=20000)
    age = len(x) - 1 - np.arange(len(x))
    calendar_w = np.repeat((decay_factor(h) ** age)[:, None], 2, axis=1)
    calendar = bw.baum_welch(x, first.params, weights=calendar_w, variance_floor=1e-6 * v,
                             tol=1e-8, maxiter=20000)
    assert max(calendar.params.variances) < 2.0  # no stress state left
    assert max(clock.params.variances) > 6.0  # the stress state survives
    assert min(clock.params.variances) == pytest.approx(1.0, rel=0.1)


def test_a_weighted_pass_that_does_not_settle_is_reported():
    panel = _synthetic_panel(900)
    train = panel.subset_dates(torch.unique(panel.date)[:800])
    g = _gate(fit_backend="native", memory="regime_clock", gate_half_life=50.0,
              weighted_maxiter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.fit(train)
    assert any(issubclass(w.category, GateNotSettledWarning) for w in caught)
    f = g.fit_result
    assert f is not None and f.weighted_settled is False and f.weighted_iterations == 1
    m = f.metrics()
    assert m["gate_weighted_settled"] == 0.0 and "gate_weighted_objective_final" in m
    settled = _gate(fit_backend="native", memory="regime_clock", gate_half_life=50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settled.fit(train)
    assert settled.fit_result is not None and settled.fit_result.weighted_settled is True
    assert settled.fit_result.metrics()["gate_weighted_settled"] == 1.0
    # canonical order kept after pass 2: regime 0 is the calmer one
    assert settled.fit_result.variances[0] < settled.fit_result.variances[1]


def test_the_memory_rules():
    with pytest.raises(NotImplementedError, match="order-0"):
        _gate(memory="regime_clock", fit_backend="native", order=1)
    with pytest.raises(ValueError, match="native"):
        _gate(memory="regime_clock", fit_backend="statsmodels")
    with pytest.raises(NotImplementedError):
        _gate(fit_backend="native", switching_trend=False, order_by="variance")
    for bad in (dict(gate_half_life=0.0), dict(initial_distribution="free")):
        with pytest.raises(ValueError):
            _gate(fit_backend="native", **bad)


# --------------------------------------------------------------------------- #
# D.1 on the real market series (gate-fit diagnostics only; skips without data)
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[1]
WINDOW = ("2000-01-01", "2008-12-31")  # the 2009 pilot fold's training years


def _date_panel(dates: pd.DatetimeIndex, metadata: dict | None = None) -> Panel:
    """A minimal one-row-per-date panel: the gate reads only the dates (and,
    for the CRSP series, the market series in ``metadata``)."""
    n = len(dates)
    return Panel(
        x_seq=torch.zeros(n, 1, 1), x_snap=torch.zeros(n, 1), y=torch.zeros(n),
        date=torch.arange(n), entity=torch.zeros(n, dtype=torch.long),
        schema=FeatureSchema(sequence_features=("s",), snapshot_features=("x",),
                             target="fwd_ret_1d"),
        date_labels=tuple(str(d.date()) for d in dates),
        metadata=dict(metadata or {}),
    )


def _compare_backends_on(series_cfg: dict, dates: pd.DatetimeIndex,
                         metadata: dict | None = None) -> None:
    panel = _date_panel(dates, metadata)
    fits = {}
    for backend in ("statsmodels", "native"):
        cfg = MarkovGateConfig(fit_backend=backend, **series_cfg)
        g = MarkovSwitchingRegimePrior(2, cfg, horizon=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g.fit(panel)
        fits[backend] = g
    s, n = fits["statsmodels"].fit_result, fits["native"].fit_result
    assert s is not None and n is not None
    print(f"\n[gate D.1] {series_cfg.get('series')}: {len(dates)} dates, llf statsmodels "
          f"{s.llf:.3f}, native {n.llf:.3f}, converged {s.n_converged}/{s.n_starts} and "
          f"{n.n_converged}/{n.n_starts}")
    assert n.llf == pytest.approx(s.llf, abs=1e-3) and n.llf >= s.llf - 1e-4
    assert np.allclose(n.variances, s.variances, rtol=1e-3)
    assert np.allclose(n.transition, s.transition, atol=1e-4)
    all_dates = torch.unique(panel.date)
    diff = np.abs(fits["native"].filtered_probabilities(all_dates)
                  - fits["statsmodels"].filtered_probabilities(all_dates)).max()
    assert diff < 1e-3


def _french_available() -> bool:
    return (REPO / "data_cache" / "ff_factors_daily.zip").exists()


@pytest.mark.skipif(not _french_available(), reason="French daily factors not cached")
def test_native_matches_statsmodels_on_the_market_excess_return_2000_2008():
    """The gate's current input series (French Mkt-RF; the choice is Q23,
    open), 2000-2008: both backends agree."""
    from nec_moe.context_data import load_french_factors

    f = load_french_factors(str(REPO / "data_cache"), momentum=False)
    dates = f.index[(f.index >= WINDOW[0]) & (f.index <= WINDOW[1])]
    _compare_backends_on({"series": "market_excess_return",
                          "context_dir": str(REPO / "data_cache")}, pd.DatetimeIndex(dates))


def _crsp_available() -> bool:
    from nec_moe.crsp import CRSPSpec, crsp_file_source

    try:
        crsp_file_source(CRSPSpec(), "IndDlySeriesData")
    except FileNotFoundError:
        return False
    return True


@pytest.mark.crsp_data
@pytest.mark.skipif(not _crsp_available(), reason="CRSP files not under Data/")
def test_native_matches_statsmodels_on_the_crsp_sp500_index_2000_2008():
    """The CRSP S&P 500 index return (INDNO 1000500; Q23, decided 2026-10-08:
    the gate's series), 2000-2008, through the registered key: both backends
    agree. A gate-fit diagnostic: nothing is scored."""
    from nec_moe.crsp import (
        MARKET_LOG_RETURN_KEY,
        CRSPSpec,
        market_log_return_by_date,
        read_index_returns,
    )

    ret = read_index_returns(CRSPSpec(), 1000500)
    ret = ret[(ret.index >= WINDOW[0]) & (ret.index <= WINDOW[1])]
    dates = pd.DatetimeIndex(ret.index)
    labels = [str(d.date()) for d in dates]
    stored = market_log_return_by_date(np.log1p(ret.astype(float)), labels)
    _compare_backends_on({"series": "crsp_market_log_return"}, dates,
                         metadata={MARKET_LOG_RETURN_KEY: stored})


def test_hmm_params_permute_consistently():
    p = bw.HMMParams(np.array([1.0, 2.0]), np.array([0.5, 4.0]),
                     np.array([[0.9, 0.1], [0.3, 0.7]]), np.array([0.75, 0.25]))
    q = p.permuted([1, 0])
    assert np.array_equal(q.transition, np.array([[0.7, 0.3], [0.1, 0.9]]))
    x = _daily_series(400)
    f_p, l_p = bw.filter_series(x, p)
    f_q, l_q = bw.filter_series(x, q)
    assert np.allclose(f_p[:, ::-1], f_q) and np.allclose(l_p, l_q)
    assert dataclasses.is_dataclass(p)


def test_the_decided_training_scheme_runs_end_to_end(tmp_path):
    """The control panel wires every brief 09 C/D setting through: a quick
    run with the native regime-clock gate, regime-clock expert decay and
    calendar base decay, on a synthetic panel."""
    import json

    import run_experiment as rx
    from nec_moe.train import DeadParameterWarning

    exp = rx.Experiment(
        tag="scheme", campaign="unit", mode="quick", out_dir=str(tmp_path / "results"),
        per_security_dir=str(tmp_path / "Data" / "derived" / "runs"), data="synthetic",
        synth_dates=400, synth_entities=6, synth_regime_process="markov", prior="markov",
        base_enabled=True, correction_mode=True, freeze_gate=True,
        gate_series="sequence_channel", gate_series_channel=0, gate_fit_backend="native",
        gate_memory="regime_clock", gate_half_life=100.0, gate_start_vol_quantiles=(0.6,),
        gate_start_vol_windows=(10,), gate_start_persistences=(0.95,),
        gate_start_draws_per_centre=1, gate_start_min_history=5, gate_start_min_group_size=5,
        expert_decay="regime_clock", expert_decay_half_life=60.0,
        base_decay_half_life_days=120.0, base_steps=30, base_batch_size=64,
        expert_hidden_dims=(8, 4), encoder_hidden=8, steps=10, figures=False,
        calibration=False,
    )
    cfg = rx._nec_config(exp, rx._build_panel(exp)[0], sigma_init=1.0)
    assert (cfg.markov_gate.fit_backend, cfg.markov_gate.memory) == ("native", "regime_clock")
    assert cfg.markov_gate.gate_half_life == 100.0 and cfg.base.decay_half_life_days == 120.0
    assert (cfg.train.expert_decay, cfg.train.expert_decay_half_life) == ("regime_clock", 60.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = rx.main(exp)
    metrics = json.loads((result["run_dir"] / "metrics" / "quick.json").read_text())
    assert metrics["expert_weight_ess_dates"] > 0 and metrics["base_weight_ess_dates"] > 0
