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
