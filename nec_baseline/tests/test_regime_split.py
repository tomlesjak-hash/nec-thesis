"""Brief 11: the regime-split diagnostics, on synthetic data only.

Nothing here reads the real panel or the CRSP files: the scripts are run on
real data by Tom.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import sys
import tokenize
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nec_moe import baum_welch as bw
from nec_moe import regime_split as rs
from nec_moe.crsp import CRSPSpec

pytest.importorskip("statsmodels")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses need it
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def irs():
    mod = _load("ic_regime_split")
    yield mod
    sys.modules.pop("ic_regime_split_script", None)


def _two_regime(t_len: int = 1500, seed: int = 0, sd=(0.006, 0.02), stay=(0.98, 0.95),
                start: str = "2000-01-03"):
    """A planted two-state Markov switching series (zero means)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(t_len, dtype=int)
    for t in range(1, t_len):
        s[t] = s[t - 1] if rng.uniform() < stay[s[t - 1]] else 1 - s[t - 1]
    x = rng.normal(0.0, np.asarray(sd)[s])
    days = pd.bdate_range(start, periods=t_len)
    return pd.Series(x, index=days), s


def _fit(irs, log_ret, k=2, backend="statsmodels"):
    s = irs.Settings(k_list=(2,) if k == 2 else (2, k), fit_backend=backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return irs.fit_market(log_ret, s)[k]


# --------------------------------------------------------------------------- #
# Part A: the regime probabilities
# --------------------------------------------------------------------------- #


def test_a_planted_two_regime_series_is_recovered_in_variance_order(irs):
    log_ret, states = _two_regime()
    fit = _fit(irs, log_ret)
    assert fit.k == 2 and fit.variances[0] < fit.variances[1]  # calm first
    np.testing.assert_allclose(np.sqrt(fit.variances), [0.006, 0.02], rtol=0.15)
    np.testing.assert_allclose(np.diag(fit.transition), [0.98, 0.95], atol=0.03)
    # the filtered stress probability finds the planted stress days
    assert ((fit.stress > 0.5) == (states == 1)).mean() > 0.9
    assert fit.n_converged >= 1 and len(fit.start_llfs) == fit.n_starts
    assert fit.labels[0] == "2000-01-03" and len(fit.labels) == len(log_ret)


@pytest.mark.parametrize("backend", ["statsmodels", "native"])
def test_the_probabilities_are_filtered_never_smoothed(irs, backend):
    """By construction: equal to a forward filter with the fitted parameters,
    and not equal to the smoother's output."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    from nec_moe import MarkovSwitchingRegimePrior

    log_ret, _ = _two_regime(t_len=800, seed=1)
    fit = _fit(irs, log_ret, backend=backend)
    x = log_ret.to_numpy()
    if backend == "native":
        a = fit.transition
        evals, evecs = np.linalg.eig(a.T)
        pi0 = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
        pi0 = pi0 / pi0.sum()
        forward, _ = bw.filter_series(x, bw.HMMParams(fit.means, fit.variances, a, pi0))
        np.testing.assert_allclose(fit.filtered, forward, atol=1e-6)
    else:
        # the same fit, to read the frozen parameters and the canonical order
        s = irs.Settings(k_list=(2,))
        gate = MarkovSwitchingRegimePrior(2, irs.gate_config(s), horizon=s.gate_horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gate.fit(irs.market_panel(log_ret, s))
            model = MarkovRegression(x, k_regimes=2, trend="c", switching_variance=True)
            perm = np.asarray(gate.fit_result.permutation, dtype=int)
            filt = np.asarray(model.filter(gate.fit_result.params)
                              .filtered_marginal_probabilities)[:, perm]
            smooth = np.asarray(model.smooth(gate.fit_result.params)
                                .smoothed_marginal_probabilities)[:, perm]
        np.testing.assert_allclose(fit.filtered, filt, atol=1e-10)
        assert np.abs(fit.filtered - smooth).max() > 0.05
    # and nothing in the diagnostics' code asks for a smoother
    for src in (inspect.getsource(rs), (SCRIPTS / "ic_regime_split.py").read_text()):
        assert "smooth" not in _code_only(src).lower()


def _code_only(src: str) -> str:
    """The source without strings and comments (docstrings say "never
    smoothed"; only code could ask for a smoother)."""
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    return " ".join(t.string for t in toks
                    if t.type not in (tokenize.STRING, tokenize.COMMENT))


def test_the_window_weight_is_the_filtered_weight_moved_by_a_and_averaged(irs):
    log_ret, _ = _two_regime(t_len=600, seed=2)
    fit = _fit(irs, log_ret)
    for h in (1, 2, 5, 10):
        got = rs.regime_weights(fit, "window", h).to_numpy()
        steps = [fit.filtered @ np.linalg.matrix_power(fit.transition, j)
                 for j in range(1, h + 1)]
        np.testing.assert_allclose(got, np.mean(steps, axis=0), atol=1e-12)
        np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(rs.regime_weights(fit, "window", 1).to_numpy(),
                               fit.filtered @ fit.transition, atol=1e-12)
    filt = rs.regime_weights(fit, "filtered", 5)
    np.testing.assert_array_equal(filt.to_numpy(), fit.filtered)
    assert list(filt.columns) == ["calm", "stress"]
    with pytest.raises(ValueError, match="weight kind"):
        rs.regime_weights(fit, "smoothed", 5)
    with pytest.raises(ValueError, match="weight kind"):
        rs.regime_weights(fit, "predicted", 5)


def test_the_trailing_vol_state_uses_data_up_to_t_and_the_median():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 300), index=pd.bdate_range("2001-01-02", periods=300))
    state, median = rs.trailing_vol_state(r, 20)
    vol = pd.Series([r.iloc[t - 19:t + 1].std() for t in range(19, 300)], index=r.index[19:])
    assert state.iloc[:19].isna().all() and state.iloc[19:].notna().all()
    assert median == pytest.approx(float(vol.median()), rel=1e-9)
    # the state is 1 exactly where the trailing sd is above the median (the
    # date whose sd *is* the median, up to rounding, is left out)
    clear = (vol - median).abs() > 1e-12
    np.testing.assert_array_equal(state.iloc[19:][clear].to_numpy(),
                                  (vol[clear] > median).astype(float).to_numpy())
    # trailing: a later return moves no earlier sd
    r2 = r.copy()
    r2.iloc[200] = 0.5
    v1 = r.rolling(20, min_periods=20).std()
    v2 = r2.rolling(20, min_periods=20).std()
    pd.testing.assert_series_equal(v1.iloc[:200], v2.iloc[:200])


@pytest.mark.parametrize("end", ["2007-01-01", "2007-06-29", "2010-12-31"])
def test_the_guard_refuses_2007_with_no_override(irs, end):
    with pytest.raises(ValueError, match="no override"):
        irs.Settings(end=end).validate()
    fields = {f.lower() for f in irs.Settings.__dataclass_fields__}
    assert not any(w in f for f in fields for w in ("override", "allow", "force"))


def test_the_market_series_is_cut_at_end_before_the_log(irs, tmp_path):
    spec = CRSPSpec(crsp_dir=str(tmp_path))
    spec.extract_dir.mkdir(parents=True)
    days = pd.bdate_range("2006-12-20", "2007-01-10")
    pd.DataFrame({"DlyTotRet": np.linspace(-0.01, 0.01, len(days))}, index=days).to_parquet(
        spec.extract_dir / "market.parquet")
    out = irs.load_market(irs.Settings(start="2006-12-01"), spec)
    assert out.index.max() <= pd.Timestamp("2006-12-31")
    assert len(out) == (days <= pd.Timestamp("2006-12-31")).sum()
    np.testing.assert_allclose(out.to_numpy(), np.log1p(np.linspace(-0.01, 0.01,
                                                                    len(days))[: len(out)]))


def test_part_a_writes_aggregate_outputs_with_the_caveat(irs, tmp_path):
    log_ret, _ = _two_regime(t_len=700, seed=4, start="2003-01-02")
    # dates after 2006 in the input are dropped by the guard, never fitted
    late = pd.Series(0.0, index=pd.bdate_range("2007-01-02", periods=5))
    s = irs.Settings(k_list=(2, 3), fit_backend="native")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = irs.run_regimes(s, root=tmp_path, log_ret=pd.concat([log_ret, late]))
    out = tmp_path / "results" / "diagnostics"
    saved = json.loads((out / "market_regimes_2000_2006.json").read_text())
    assert saved == json.loads(json.dumps(report))
    assert saved["caveat"] == rs.IN_SAMPLE_CAVEAT
    assert saved["window"]["last"] <= "2006-12-31" and saved["window"]["dates"] == 700
    for k in ("2", "3"):
        fit = saved["fits"][k]
        assert len(fit["variances"]) == int(k) and fit["variances"] == sorted(fit["variances"])
        assert len(fit["multi_start"]["start_llfs"]) == fit["multi_start"]["n_starts"]
        assert sum(fit["share_of_days_most_likely"]) == pytest.approx(1.0)
    # aggregate only: no list as long as the series
    def lengths(x):
        if isinstance(x, dict):
            return [n for v in x.values() for n in lengths(v)]
        if isinstance(x, list):
            return [len(x), *(n for v in x for n in lengths(v))]
        return []
    assert max(lengths(saved)) < 100
    assert (out / "market_regimes_2000_2006.png").stat().st_size > 0


# --------------------------------------------------------------------------- #
# Part B: the IC split by regime
# --------------------------------------------------------------------------- #


def _calibrated_weights(t_len: int, seed: int, stay=(0.98, 0.98), sd=(0.006, 0.02)):
    """True states, and filtered stress probabilities under the true
    parameters (calibrated: E[state | weight] = weight)."""
    log_ret, states = _two_regime(t_len=t_len, seed=seed, sd=sd, stay=stay)
    a = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    pi0 = np.array([0.5, 0.5])
    filtered, _ = bw.filter_series(log_ret.to_numpy(),
                                   bw.HMMParams(np.zeros(2), np.asarray(sd) ** 2, a, pi0))
    return states, filtered


def test_a_planted_regime_flip_is_recovered_while_the_pooled_ic_is_near_zero():
    states, w = _calibrated_weights(4000, seed=5)
    rng = np.random.default_rng(5)
    a = 0.05
    ic = a * (1 - 2 * states) + rng.normal(0, 0.1, len(states))
    assert abs(states.mean() - 0.5) < 0.1
    res = rs.regime_regression(ic, w, 0, "uniform")
    np.testing.assert_allclose(res.coef, [a, -a], atol=0.01)
    assert res.contrast == pytest.approx(-2 * a, abs=0.015) and res.contrast_t < -10
    assert abs(ic.mean()) < 0.01  # pooled: near zero
    wmean, eff = rs.weighted_regime_ic(ic, w)
    assert wmean[0] > 0.02 and wmean[1] < -0.02  # same signs, shrunk by the soft weights
    assert (eff > 0).all() and (eff <= len(ic) + 1e-9).all()  # each at most T


def test_regime_independent_ic_holds_its_size():
    out = rs.contrast_size_simulation(n_sims=400, seed=11)
    print(f"\n[size] rejection at 5%: {out}")
    for choice in rs.HAC_CHOICES:
        assert 0.02 <= out[choice] <= 0.10, (choice, out[choice])
    assert abs(out["mean_contrast"]) < 0.25 * out["sd_contrast"]


def test_a_slow_ic_drift_oversizes_both_hac_choices():
    """A caveat for reading the t-statistics: with a persistent regime weight,
    a slow IC drift (independent of the regime, autocorrelation 0.98) makes
    x_t e_t autocorrelated far beyond h - 1 lags. Newey-West's 20 Bartlett
    lags cover only part of it: both choices over-reject, Newey-West less."""
    out = rs.contrast_size_simulation(n_sims=300, drift_sd=0.1, seed=12)
    print(f"\n[size, drifting IC] rejection at 5%: {out}")
    assert out["hansen_hodrick"] > 0.15 and out["newey_west"] > 0.12
    assert out["newey_west"] < out["hansen_hodrick"]


def test_the_weighted_mean_is_the_plain_mean_under_equal_weights():
    rng = np.random.default_rng(6)
    ic = rng.normal(0.01, 0.1, 300)
    w = np.full((300, 2), 0.5)
    wmean, eff = rs.weighted_regime_ic(ic, w)
    np.testing.assert_allclose(wmean, [ic.mean(), ic.mean()], atol=1e-15)
    np.testing.assert_allclose(eff, [300.0, 300.0])


def test_the_decomposition_is_exact_on_noise_free_data():
    _, w = _calibrated_weights(500, seed=7)
    w3 = np.random.default_rng(7).dirichlet([1.0, 1.0, 1.0], size=500)  # K = 3 rows
    for x, c in ((w, np.array([0.03, -0.02])), (w3, np.array([0.02, 0.0, -0.04]))):
        res = rs.regime_regression(x @ c, x, 4, "uniform")
        np.testing.assert_allclose(res.coef, c, atol=1e-12)
        np.testing.assert_allclose(res.se, 0.0, atol=1e-12)
        assert res.contrast == pytest.approx(c[-1] - c[0], abs=1e-12)


@pytest.mark.parametrize("lags,kernel", [(0, "uniform"), (4, "uniform"), (20, "bartlett")])
def test_the_sandwich_through_hac_variance_equals_the_matrix_sandwich(lags, kernel):
    """Oracle: (X'X)^-1 Omega (X'X)^-1 with Omega the kernel-weighted sum of
    the autocovariances of x_t e_t; ours carries hac_variance's T/(T-1)."""
    _, w = _calibrated_weights(900, seed=8)
    rng = np.random.default_rng(8)
    ic = 0.01 + np.convolve(rng.normal(0, 0.1, 904), np.ones(5) / 5, mode="valid")
    res = rs.regime_regression(ic, w, lags, kernel)
    xtx_inv = np.linalg.inv(w.T @ w)
    e = ic - w @ res.coef
    g = w * e[:, None]
    omega = g.T @ g
    for lag in range(1, lags + 1):
        wl = 1.0 if kernel == "uniform" else 1.0 - lag / (lags + 1.0)
        cross = g[lag:].T @ g[:-lag]
        omega += wl * (cross + cross.T)
    v = xtx_inv @ omega @ xtx_inv * len(ic) / (len(ic) - 1)
    np.testing.assert_allclose(res.se, np.sqrt(np.diag(v)), rtol=1e-9)
    a = np.array([-1.0, 1.0])
    assert res.contrast_se == pytest.approx(float(np.sqrt(a @ v @ a)), rel=1e-9)
    assert set(res.kernels) == {kernel}


def test_the_hac_choices_and_their_lags():
    assert rs.hac_choice("hansen_hodrick", 5, 20) == (4, "uniform")
    assert rs.hac_choice("hansen_hodrick", 1, 20) == (0, "uniform")
    assert rs.hac_choice("newey_west", 5, 20) == (20, "bartlett")
    assert rs.hac_choice("newey_west", 30, 20) == (29, "bartlett")
    with pytest.raises(ValueError):
        rs.hac_choice("white", 5, 20)


def test_holm_and_bh_match_a_hand_worked_example():
    from nec_moe import benjamini_hochberg, holm

    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    # sorted 0.005, 0.01, 0.03, 0.04, 0.2 times 5, 4, 3, 2, 1 -> 0.025, 0.04,
    # 0.09, 0.08 (running max 0.09), 0.2
    reject, adjusted = holm(p, 0.05)
    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.025, 0.2])
    assert reject == [True, False, False, True, False]
    # BH: p * 5 / rank -> 0.025, 0.025, 0.05, 0.05, 0.2 (already monotone)
    _, q = benjamini_hochberg(p, 0.10)
    np.testing.assert_allclose(q, [0.025, 0.05, 0.05, 0.025, 0.2])


def test_p_values_and_the_hard_split():
    assert rs.p_value(-1.6448536, "less") == pytest.approx(0.05, abs=1e-6)
    assert rs.p_value(1.959964, "two-sided") == pytest.approx(0.05, abs=1e-6)
    assert rs.p_value(-1.959964, "two-sided") == pytest.approx(0.05, abs=1e-6)
    assert np.isnan(rs.p_value(float("nan"), "less"))
    hard = rs.hard_split(np.array([0.2, 0.7, np.nan, 0.5]), 0.5)
    np.testing.assert_array_equal(hard[[0, 1, 3]], [[1, 0], [0, 1], [1, 0]])
    assert np.isnan(hard[2]).all()
    # no stress date at all: the regression is undefined, not a number
    res = rs.regime_regression(np.ones(50), rs.hard_split(np.zeros(50), 0.5), 0, "uniform")
    assert np.isnan(res.contrast) and np.isnan(res.coef).all()


def _synthetic_inputs(irs, t_len=600, n=50, seed=9, b=0.3, extra_2007=5):
    """A market whose regimes flip ret_5d's IC (+ in calm, - in stress); the
    other 17 signals are noise. Dates end 2006-12-29, plus some 2007 rows the
    guard must drop."""
    log_ret, states = _two_regime(t_len=t_len, seed=seed, stay=(0.98, 0.98),
                                  start="2004-09-13")
    assert log_ret.index[-1] == pd.Timestamp("2006-12-29")
    days = log_ret.index.append(pd.bdate_range("2007-01-01", periods=extra_2007))
    rng = np.random.default_rng(seed)
    names = [str(10000 + i) for i in range(n)]
    sig = {s: pd.DataFrame(rng.normal(size=(len(days), n)), index=days, columns=names)
           for s in irs.SIGNALS}
    beta = np.where(np.r_[states, np.zeros(extra_2007, dtype=int)] == 0, b, -b)
    r = np.full((len(days), n), np.nan)
    r[1:] = beta[:-1, None] * sig["ret_5d"].to_numpy()[:-1] + rng.normal(size=(len(days) - 1, n))
    daily = pd.DataFrame(r * 0.01, index=days, columns=names)
    universe = pd.DataFrame(True, index=days, columns=names)
    late = pd.Series(0.0, index=days[t_len:])
    return (sig, daily, universe, {"synthetic": True}), pd.concat([log_ret, late])


def test_part_b_end_to_end_on_synthetic_data(irs, tmp_path):
    inputs, log_ret = _synthetic_inputs(irs)
    s = irs.Settings(k_list=(2,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = irs.run_ic(s, root=tmp_path, inputs=inputs, log_ret=log_ret)
    out = tmp_path / "results" / "diagnostics"
    for name in ("ic_regime_2000_2006.csv", "ic_regime_2000_2006.json",
                 "ic_regime_2000_2006.png", "ic_regime_contrast_2000_2006.png"):
        assert (out / name).stat().st_size > 0
    # strict JSON (no NaN tokens), with the caveat and the declared families
    report = json.loads((out / "ic_regime_2000_2006.json").read_text(),
                        parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert report["caveat"] == rs.IN_SAMPLE_CAVEAT
    fam = report["families"]
    assert fam["primary"]["tests"] == 5
    assert fam["exploratory"]["tests"] == len(irs.SIGNALS) * len(s.horizons) * 1 - 5
    assert len(report["primary_results"]) == 5
    # the five primary tests: K = 2, h = 5, one-sided (less), Holm
    prim = table[table["family"] == "primary"]
    assert set(prim["signal"]) == {"ret_5d", "ret_20d", "ret_20d_ind_rel", "mom_12_1",
                                   "ind_mom_12_1"}
    assert (prim["K"] == 2).all() and (prim["h"] == 5).all()
    assert (prim["alternative"] == "less").all() and (prim["adjustment"] == "holm").all()
    np.testing.assert_allclose(prim["p"], [rs.p_value(x, "less") for x in prim["t_hh"]])
    expl = table[table["family"] == "exploratory"]
    assert (expl["alternative"] == "two-sided").all()
    assert (expl["adjustment"] == "benjamini_hochberg").all()
    # the planted flip: ret_5d's IC is positive in calm, negative in stress
    flip = table[(table["signal"] == "ret_5d") & (table["h"] == 1) & (table["K"] == 2)
                 & (table["split"] == "soft") & (table["weight_kind"] == "window")]
    c = flip.set_index("term")["estimate"]
    assert c["calm"] > 0.05 and c["stress"] < -0.05
    assert flip.set_index("term").loc["contrast", "t_hh"] < -5
    # the guard: 2007 rows dropped; each h loses its last h dates
    soft = table[(table["split"] == "soft") & (table["term"] == "contrast")]
    for h in s.horizons:
        assert (soft[soft["h"] == h]["n_dates"] == 600 - h).all()
    # every split is there: both weight kinds, the hard split, stress_vol
    assert set(table["split"]) == {"soft", "hard", "stress_vol"}
    assert set(table[table["split"] == "soft"]["weight_kind"]) == {"window", "filtered"}
    assert len({st for st in irs.PRIMARY_STYLES}) == 5  # distinct line styles


def test_the_primary_hypotheses_are_fixed(irs):
    assert [h["id"] for h in rs.PRIMARY_HYPOTHESES] == ["H1", "H2"]
    assert rs.PRIMARY_HYPOTHESES[0]["signals"] == ("ret_5d", "ret_20d", "ret_20d_ind_rel")
    assert rs.PRIMARY_HYPOTHESES[1]["signals"] == ("mom_12_1", "ind_mom_12_1")
    assert all(h["alternative"] == "contrast < 0" for h in rs.PRIMARY_HYPOTHESES)
    assert (rs.PRIMARY_SPEC["k"], rs.PRIMARY_SPEC["h"], rs.PRIMARY_SPEC["hac"]) == (
        2, 5, "hansen_hodrick")
    with pytest.raises(ValueError, match="primary hypotheses are fixed"):
        irs.Settings(signals=tuple(x for x in irs.SIGNALS if x != "mom_12_1")).validate()
    with pytest.raises(ValueError, match="primary hypotheses are fixed"):
        irs.Settings(horizons=(1, 2, 3)).validate()
    assert len(irs.SIGNALS) == 18 and irs.SIGNALS[-2:] == ("ret_20d_ind_rel", "ind_mom_12_1")


# --------------------------------------------------------------------------- #
# Part C: sector regimes
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sec():
    mod = _load("sector_regimes")
    yield mod
    sys.modules.pop("sector_regimes_script", None)


def _sector_world(t_len=700, seed=13, episode=(300, 450), sectors=(10, 20, 45), per=8,
                  thin_sector=None, start="2003-06-02"):
    """Calm market, quiet sectors, and one sector-only stress episode in the
    first sector; long rows as sector_rows makes them."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=t_len)
    m = rng.normal(0.0, 0.008, t_len)
    rows = []
    for j, code in enumerate(sectors):
        f_sd = np.full(t_len, 0.002)
        if j == 0:
            f_sd[episode[0]:episode[1]] = 0.02
        f = rng.normal(0.0, f_sd)
        names = per if code != thin_sector else 4
        for i in range(names):
            r = m + f + rng.normal(0.0, 0.004, t_len)
            for t in range(1, t_len):
                rows.append((days[t], f"{code}{i:02d}", code, r[t], 1.0 + i))
    frame = pd.DataFrame(rows, columns=["date", "permno", "sector", "ret", "me_prev"])
    return frame, pd.Series(np.log1p(m), index=days)


def test_a_sector_only_stress_episode_is_found_in_the_relative_series_not_the_market(irs, sec):
    rows, log_ret = _sector_world()
    s = sec.Settings()
    ret, _ = rs.sector_returns(rows, "value", s.min_sector_names)
    dates = log_ret.index[1:]
    relative = ret.reindex(dates).sub(log_ret.reindex(dates), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = sec.fit_sectors(relative, s)
        market = irs.fit_market(log_ret, s.regime_settings())[2]
    in_ep = np.zeros(len(dates), dtype=bool)
    in_ep[299:449] = True  # days 300..449 of the calendar, one date shifted
    p10 = fits[10].stress
    assert p10[in_ep].mean() > 0.8 and p10[~in_ep].mean() < 0.1
    pm = pd.Series(market.stress, index=list(market.labels)).loc[list(fits[10].labels)]
    assert abs(pm[in_ep].mean() - pm[~in_ep].mean()) < 0.15  # the market did not see it
    m = rs.sector_measures(fits[10], market, 0.5, 5)
    assert m["stress_episodes"] >= 1
    assert m["share_stressed_while_market_calm"] == pytest.approx(
        ((p10 > 0.5) & (pm.to_numpy() <= 0.5)).mean())
    assert m["corr_with_market_stress"] < 0.3


def test_value_weights_use_the_previous_days_market_equity():
    days = pd.bdate_range("2004-03-01", periods=4)
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=days)
    cap = pd.DataFrame({"A": [1.0, 9.0, 9.0, 9.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=days)
    sector = pd.DataFrame(10.0, index=days, columns=["A", "B"])
    universe = pd.DataFrame(True, index=days, columns=["A", "B"])
    rows = rs.sector_rows(ret, cap, sector, universe)
    assert rows["date"].min() == days[1]  # no previous day for the first date
    out, counts = rs.sector_returns(rows, "value", 2)
    # day 1: A's cap jumped to 9 that day, but the weights are day 0's (1:1)
    assert out.loc[days[1], 10] == pytest.approx(np.log1p(0.05))
    # day 2: day 1's caps, 9:1
    assert out.loc[days[2], 10] == pytest.approx(np.log1p(0.09))
    eq, _ = rs.sector_returns(rows, "equal", 2)
    assert eq.loc[days[2], 10] == pytest.approx(np.log1p(0.05))
    assert (counts[10] == 2).all()
    # a non-member row is not in the sector
    universe.loc[days[2], "B"] = False
    solo, n = rs.sector_returns(rs.sector_rows(ret, cap, sector, universe), "value", 2)
    assert np.isnan(solo.loc[days[2], 10]) and n.loc[days[2], 10] == 1


def test_thin_sectors_are_skipped_and_reported(sec):
    rows, log_ret = _sector_world(t_len=200, sectors=(10, 20), thin_sector=20)
    ret, counts = rs.sector_returns(rows, "value", 5)
    assert ret[20].isna().all() and ret[10].notna().all()
    thin = rs.thin_sectors(counts, log_ret.index[1:], 5)
    assert thin[10]["fitted"] and thin[10]["thin_dates"] == 0
    assert not thin[20]["fitted"] and thin[20]["thin_dates"] == 199
    assert thin[20]["median_names"] == 4.0
    # sectors without a single name (Real Estate before 2016) are reported too
    assert not thin[60]["fitted"] and thin[60]["median_names"] == 0.0
    assert set(thin) == set(rs.SECTOR_NAMES)
    with pytest.raises(ValueError, match="gap"):
        sec.fit_sectors(ret.reindex(log_ret.index[1:]), sec.Settings())


def test_stress_episodes_and_the_co_stress_matrix():
    p = np.array([0.6, 0.7, 0.2, 0.8, 0.9, 0.8, 0.9, 0.8, 0.1, 0.6])
    assert rs.stress_episodes(p, 0.5, 2) == 2
    assert rs.stress_episodes(p, 0.5, 3) == 1
    assert rs.stress_episodes(p, 0.5, 1) == 3  # the trailing one-day run counts
    labels = tuple(str(d.date()) for d in pd.bdate_range("2005-01-03", periods=10))

    def fit(stress):
        f = np.column_stack([1 - stress, stress])
        return rs.RegimeFit(2, labels, f, np.eye(2), np.zeros(2), np.ones(2), np.ones(2),
                            0.0, "x", 1, 1, 0, (0.0,), ("converged",), 1)

    co = rs.co_stress({10: fit(p), 20: fit(1 - p)}, 0.5)
    assert co.loc[10, 10] == pytest.approx((p > 0.5).mean())
    assert co.loc[10, 20] == co.loc[20, 10] == pytest.approx(((p > 0.5) & (1 - p > 0.5)).mean())


def test_part_c_guard_and_outputs(sec, tmp_path):
    with pytest.raises(ValueError, match="no override"):
        sec.Settings(end="2007-01-01").validate()
    rows, log_ret = _sector_world(t_len=500, episode=(200, 320), start="2004-11-01")
    assert log_ret.index[-1] <= pd.Timestamp("2006-12-31")
    late_rows = rows.assign(date=rows["date"] + pd.Timedelta(days=800))
    late_rows = late_rows[late_rows["date"] > pd.Timestamp("2006-12-31")]
    s = sec.Settings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sec.run_sectors(s, root=tmp_path, rows=pd.concat([rows, late_rows]),
                                 log_ret=log_ret)
    out = tmp_path / "results" / "diagnostics"
    for name in ("sector_regimes_2000_2006.csv", "sector_regimes_2000_2006.json",
                 "sector_regimes_2000_2006.png"):
        assert (out / name).stat().st_size > 0
    saved = json.loads((out / "sector_regimes_2000_2006.json").read_text(),
                       parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert saved["caveat"] == rs.IN_SAMPLE_CAVEAT
    assert saved["sector_dates"]["last"] <= "2006-12-31"
    assert {k for k, t in saved["thin_sectors"].items() if t["fitted"]} == {"10", "20", "45"}
    m = pd.DataFrame(saved["measures"])
    assert len(m) == 3 * 2 * 2  # sectors x {relative, raw} x {value, equal}
    assert set(m["series"]) == {"relative", "raw"} and set(m["weighting"]) == {"value", "equal"}
    assert "permno" not in m.columns  # sector level only
    co = saved["co_stress"]["value"]["relative"]
    assert co["sectors"] == [10, 20, 45] and len(co["share_both_stressed"]) == 3
    assert report["market_gate"]["k"] == 2
