"""Brief 10 C: ``scripts/ic_decay.py``, on synthetic data only.

The script is never run on the real panel in development (Tom runs it). These
tests plant known signals in invented returns.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nec_moe.characteristics import MARKET_FEATURES, Q26_FEATURES
from nec_moe.crsp import CRSPExtract
from nec_moe.evaluation import ic_summary

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ic_decay.py"


@pytest.fixture(scope="module")
def ic():
    spec = importlib.util.spec_from_file_location("ic_decay_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass needs it
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _planted(t_len: int = 700, n: int = 150, beta: float = 0.3, seed: int = 0):
    """Signals and returns where the signal at t predicts only day t + 1.

    ``r[t+1, i] = beta * s[t, i] + market[t+1] + noise``; the signal is i.i.d.
    over time, so the days after t + 1 carry nothing about ``s[t]``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2001-01-02", periods=t_len)
    names = [str(10000 + i) for i in range(n)]
    s = rng.normal(size=(t_len, n))
    r = np.full((t_len, n), np.nan)
    market = rng.normal(size=(t_len - 1, 1))  # one common move per day
    r[1:] = beta * s[:-1] + market + rng.normal(size=(t_len - 1, n))
    universe = pd.DataFrame(rng.uniform(size=(t_len, n)) < 0.9, index=dates, columns=names)
    sig = pd.DataFrame(s, index=dates, columns=names)
    daily = pd.DataFrame(r, index=dates, columns=names)
    return sig, daily, universe


# --------------------------------------------------------------------------- #
# the signals and the guard
# --------------------------------------------------------------------------- #


def test_the_default_signals_are_the_two_str_two_momentum_and_the_market_block(ic):
    assert ic.DEFAULT_SIGNALS[:4] == ("ret_5d", "ret_20d", "mom_12_1", "prc_highprc_252d")
    assert ic.DEFAULT_SIGNALS[4:] == MARKET_FEATURES and len(MARKET_FEATURES) == 12
    assert set(ic.DEFAULT_SIGNALS) <= set(Q26_FEATURES)
    s = ic.Settings()
    assert (s.start, s.end, s.horizons) == ("2000-01-01", "2006-12-31", (1, 2, 3, 5, 10))
    assert s.stem == "ic_decay_2000_2006"


@pytest.mark.parametrize("end", ["2007-01-01", "2007-01-02", "2009-12-31", "2024-12-31"])
def test_the_guard_refuses_2007_and_later_without_override(ic, end, monkeypatch):
    with pytest.raises(ValueError, match="no override"):
        ic.Settings(end=end).validate()
    with pytest.raises(ValueError, match="no override"):
        ic.check_end(end)

    def never(*a, **k):
        raise AssertionError("data was loaded before the guard")

    monkeypatch.setattr(ic, "load_inputs", never)
    with pytest.raises(ValueError, match="no override"):
        ic.main(ic.Settings(end=end))
    # there is no field that could switch it off
    fields = {f.lower() for f in ic.Settings.__dataclass_fields__}
    assert not any(w in f for f in fields for w in ("override", "allow", "force", "unsafe"))


def test_the_last_allowed_end_passes_and_nothing_later_survives(ic):
    assert ic.Settings(end="2006-12-31").validate().end == "2006-12-31"
    days = pd.bdate_range("2006-12-20", "2007-01-10")
    frame = pd.DataFrame({"a": np.arange(len(days), dtype=float)}, index=days)
    kept = ic.restrict(frame, ic.Settings())
    assert kept.index.max() <= pd.Timestamp("2006-12-31")
    assert len(kept) == (days <= pd.Timestamp("2006-12-31")).sum()


def test_the_extract_is_cut_at_end_before_any_return_is_built(ic):
    days = pd.bdate_range("2006-12-26", "2007-01-05")
    stock = pd.DataFrame({"PERMNO": 1, "DlyCalDt": days, "DlyRet": 0.01})
    extract = CRSPExtract(
        info={}, stock=stock, adj=pd.DataFrame(), market=pd.Series(0.001, index=days),
        spells=pd.DataFrame(), delists=pd.DataFrame(), security_info=pd.DataFrame(),
    )
    cut = ic.cut_at(extract, "2006-12-31")
    assert cut.stock["DlyCalDt"].max() <= pd.Timestamp("2006-12-31")
    assert cut.market.index.max() <= pd.Timestamp("2006-12-31")
    with pytest.raises(ValueError, match="no override"):
        ic.cut_at(extract, "2007-01-02")


# --------------------------------------------------------------------------- #
# the target
# --------------------------------------------------------------------------- #


def test_forward_returns_sum_the_next_h_days_and_stop_at_the_frame_end(ic):
    days = pd.bdate_range("2003-01-01", periods=8)
    r = pd.DataFrame({"a": np.arange(1.0, 9.0), "b": [1.0, np.nan, 1, 1, 1, 1, 1, 1]},
                     index=days)
    f3 = ic.forward_log_returns(r, 3)
    assert f3["a"].iloc[0] == 2 + 3 + 4  # (t, t+3]: excludes day t
    assert np.isnan(f3["b"].iloc[0])  # a missing day inside the window
    assert f3["b"].iloc[1] == 3.0
    # the last h dates have no complete window inside the frame: missing,
    # never filled from beyond it
    assert f3.iloc[-3:].isna().all().all() and f3.iloc[:-3]["a"].notna().all()


def test_the_market_neutral_demeaning_holds(ic):
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2004-01-01", periods=40)
    fwd = pd.DataFrame(rng.normal(size=(40, 30)), index=days)
    fwd.iloc[3, 5] = np.nan
    universe = pd.DataFrame(rng.uniform(size=(40, 30)) < 0.8, index=days)
    y = ic.market_neutral(fwd, universe)
    # zero equal-weighted mean per date over the universe's valid names
    np.testing.assert_allclose(y.mean(axis=1).to_numpy(), 0.0, atol=1e-12)
    assert y.where(~universe).isna().all().all()  # nothing outside the universe
    assert np.isnan(y.iloc[3, 5])
    # the mean is over valid universe names only: by hand on one date
    d = 3
    keep = universe.iloc[d] & fwd.iloc[d].notna()
    np.testing.assert_allclose(y.iloc[d][keep], fwd.iloc[d][keep] - fwd.iloc[d][keep].mean())
    # a common move on a date changes nothing
    shocked = fwd.add(pd.Series(rng.normal(size=40) * 10, index=days), axis=0)
    pd.testing.assert_frame_equal(ic.market_neutral(shocked, universe), y, atol=1e-10)


# --------------------------------------------------------------------------- #
# the IC
# --------------------------------------------------------------------------- #


def test_spearman_uses_average_ranks_for_ties(ic):
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2005-01-03", periods=25)
    binary = pd.DataFrame((rng.uniform(size=(25, 40)) < 0.2).astype(float), index=days)
    target = pd.DataFrame(rng.normal(size=(25, 40)), index=days)
    target.iloc[0, :5] = np.nan
    got = ic.spearman_ic_by_date(binary, target)
    want = pd.Series({d: binary.loc[d].corr(target.loc[d], method="spearman") for d in days})
    pd.testing.assert_series_equal(got, want.loc[got.index], check_names=False,
                                   check_freq=False, atol=1e-12)
    # skipped: too few names, or a constant signal
    sparse = target.copy()
    sparse.iloc[1, 2:] = np.nan
    flat = binary.copy()
    flat.iloc[2] = 1.0
    out = ic.spearman_ic_by_date(flat, sparse, min_names=3)
    assert days[1] not in out.index and days[2] not in out.index


def test_a_next_day_signal_decays_with_h_as_planted(ic):
    sig, daily, universe = _planted()
    table = ic.ic_decay({"planted": sig}, daily, universe, (1, 2, 3, 5, 10))
    by_h = table.set_index("h")["mean_ic"]
    # strictly falling, and like 1/sqrt(h): the forward sum adds h - 1 days
    # that carry nothing about the signal (each adds one day's variance)
    assert by_h.is_monotonic_decreasing and by_h[1] > 0.2
    for h in (2, 3, 5, 10):
        assert by_h[h] == pytest.approx(by_h[1] / np.sqrt(h), abs=0.03)
    # each horizon loses its last h dates (no window past the frame's end)
    n = table.set_index("h")["n_dates"]
    assert all(n[h] == len(sig) - h for h in (1, 2, 3, 5, 10))


def test_a_pure_noise_signal_has_no_ic(ic):
    sig, daily, universe = _planted(beta=0.0, seed=3)
    table = ic.ic_decay({"noise": sig}, daily, universe, (1, 5))
    assert (table["mean_ic"].abs() < 0.01).all()


def test_the_t_statistic_is_hansen_hodrick_with_h_minus_1_lags(ic, monkeypatch):
    sig, daily, universe = _planted(t_len=300, n=60)
    seen: list[tuple[int, str]] = []
    real = ic.ic_summary

    def spy(ics, hac_lags=0, hac_kernel="uniform"):
        seen.append((hac_lags, hac_kernel))
        return real(ics, hac_lags=hac_lags, hac_kernel=hac_kernel)

    monkeypatch.setattr(ic, "ic_summary", spy)
    table = ic.ic_decay({"s": sig}, daily, universe, (1, 2, 5, 10))
    assert seen == [(0, "uniform"), (1, "uniform"), (4, "uniform"), (9, "uniform")]
    assert (table["hac_lags"] == table["h"] - 1).all()
    # the same number as the harness's estimator on the same daily ICs
    for h in (1, 5):
        target = ic.market_neutral(ic.forward_log_returns(daily, h), universe)
        ics = ic.spearman_ic_by_date(sig.where(universe), target)
        want = ic_summary(torch.tensor(ics.to_numpy()), hac_lags=h - 1, hac_kernel="uniform")
        got = table[table["h"] == h].iloc[0]
        assert got["t_stat"] == pytest.approx(want.t_stat, rel=1e-12)
        assert got["mean_ic"] == pytest.approx(want.mean_ic, rel=1e-12)


# --------------------------------------------------------------------------- #
# the outputs
# --------------------------------------------------------------------------- #


def test_the_outputs_are_aggregate_only(ic, tmp_path):
    sig, daily, universe = _planted(t_len=120, n=40)
    table = ic.ic_decay({"ret_5d": sig, "vol_shock": -sig}, daily, universe, (1, 2))
    paths = ic.write_outputs(table, ic.Settings(), {"dates": 120}, root=tmp_path)
    assert {p.name for p in paths.values()} == {
        "ic_decay_2000_2006.csv", "ic_decay_2000_2006.json", "ic_decay_2000_2006.png"}
    assert all(p.parent == tmp_path / "results" / "diagnostics" for p in paths.values())
    csv = pd.read_csv(paths["csv"])
    assert list(csv.columns) == ["signal", "h", "mean_ic", "t_stat", "n_dates", "hac_lags",
                                 "hac_kernel", "first_date", "last_date"]
    assert len(csv) == 4  # one row per (signal, h); no name, no per-security value
    report = json.loads(paths["json"].read_text())
    assert len(report["rows"]) == 4 and report["settings"]["end"] == "2006-12-31"
    assert paths["png"].stat().st_size > 0
