"""Brief 08 D: the Q26 feature set (57 inputs), on invented data.

Every characteristic is checked by hand on one invented stock; the per-date
ranks, the within-sector fallback, the 0-fill and the flags on a small
invented cross-section; the earnings timing on invented report dates; no
look-ahead for every characteristic (and for the fundamentals, with the
Compustat availability dates truncated too); and the full build on the CRSP
and Compustat fixtures together.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path

import cfz_fixture as cfz
import crsp_fixture as fx
import numpy as np
import pandas as pd
import pytest
import torch
from conftest import small_config

from nec_moe import (
    NECConfig,
    StageBSpec,
    TrialRegistry,
    assemble_panel,
    data_config_from_panel,
    market_frame,
    nec_arm,
    run_sweep,
    stock_features,
)
from nec_moe import compustat as cs
from nec_moe.characteristics import (
    CRSP_FLAGGED,
    FUND_FLAGGED,
    JKP_FEATURES,
    MARKET_FEATURES,
    Q26_CHARACTERISTICS,
    Q26_FEATURES,
    SECTOR_FEATURES,
    UNRANKED,
    cross_section,
    rank_to_unit,
    stock_characteristics,
)
from nec_moe.compustat import CompustatSpec
from nec_moe.config import ExpertConfig
from nec_moe.crsp import CRSPSpec, build_crsp_panel, extract_crsp, load_extract
from nec_moe.features import FeatureSpec, check_panel_input_mode
from nec_moe.train import DeadParameterWarning

FS = FeatureSpec()
N = 1700
IDX = pd.bdate_range("2014-01-01", periods=N)
T = N - 40  # the hand-checked date: every default window is full by then
RDQS = pd.to_datetime(["2019-02-14", "2019-05-02", "2019-08-01", "2019-10-31",
                       "2020-02-13", "2020-04-30"])


def _stock(seed: int = 0, *, gaps: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """One invented stock with every daily column, and the market."""
    g = np.random.default_rng(seed)
    m = pd.Series(g.normal(0.0003, 0.01, N), index=IDX)
    r = 0.8 * m.to_numpy() + g.normal(0.0002, 0.015, N)
    div = np.where(g.uniform(size=N) < 0.02, 0.002, 0.0)
    big_r = np.expm1(r)
    retx = big_r - div
    vol = np.round(1e6 * np.exp(g.normal(0, 0.4, N)))
    vol[g.uniform(size=N) < 0.01] = 0.0  # zero-volume days
    factor = np.where(np.arange(N) < 900, 2.0, 1.0)  # a split
    price = 40 * np.exp(np.cumsum(np.log1p(retx)))
    intraday = g.normal(0, 0.008, N)
    close = price
    open_ = close * np.exp(-intraday)
    half = 0.0005 + 0.002 * g.uniform(size=N)
    bid, ask = close * (1 - half), close * (1 + half)
    bid[g.uniform(size=N) < 0.02] = 0.0  # bad quotes: excluded
    bid[T - 5] = 0.0
    ask[T - 7] = bid[T - 7] * 0.99  # a crossed quote (bid > ask): excluded
    daily = pd.DataFrame({
        "ret": r, "retx": retx, "volume": vol, "dollar_volume": close * vol,
        "share_factor": factor, "open": open_, "close": close, "high": close * 1.01,
        "low": close * 0.99, "bid": bid, "ask": ask,
        "me": np.abs(close) * 5e3 / 1e6 * 1e3, "sector": 45.0,
        "be": 900.0, "ni_ttm": 80.0, "netdebt": 300.0, "niq_be": 0.02, "ocf_at": 0.07,
        "gp_at": 0.3, "ni_inc8q": 3.0, "at_gr1": 0.05, "sale_gr1": 0.08,
        "oaccruals_at": -0.01, "taccruals_at": 0.02, "debt_gr3": 0.1, "noa_at": 0.6,
        "cash_at": 0.1, "niq_su": 1.2, "saleq_su": -0.4,
    }, index=IDX)
    if gaps:
        for i in (T - 3, T - 30, T - 200, T - 900):
            daily.iloc[i, daily.columns.get_loc("ret")] = np.nan
            daily.iloc[i, daily.columns.get_loc("retx")] = np.nan
    rdq = pd.Series(pd.NaT, index=IDX, dtype="datetime64[ns]")
    for d in RDQS:
        rdq.iloc[IDX.searchsorted(d)] = d
    daily["rdq_date"] = rdq
    return daily, m


@pytest.fixture(scope="module")
def one():
    daily, m = _stock()
    return daily, m, stock_characteristics(daily, m, FS)


def _w(x: pd.Series, n: int, end: int = T) -> np.ndarray:
    return x.iloc[end - n + 1 : end + 1].to_numpy(dtype=float)


def _price_index(retx: pd.Series) -> pd.Series:
    p = np.exp(np.log1p(retx).fillna(0.0).cumsum())
    return p.where(retx.notna())


# --------------------------------------------------------------------------- #
# The 57 inputs
# --------------------------------------------------------------------------- #


def test_57_inputs_and_the_model_reads_them_from_the_schema():
    assert len(JKP_FEATURES) == 26 and len(MARKET_FEATURES) == 12
    assert len(Q26_CHARACTERISTICS) == 40 and len(SECTOR_FEATURES) == 11
    assert len(Q26_FEATURES) == 26 + 12 + 2 + 4 + 11 + 2 == 57
    assert len(set(Q26_FEATURES)) == 57
    assert StageBSpec().feature_set == "q26" and StageBSpec().snapshot_features == Q26_FEATURES
    daily = {str(10001 + i): _stock(i)[0] for i in range(6)}
    m = _stock(0)[1]
    spec = StageBSpec(seq_len=10)
    panel = assemble_panel(daily, market_frame(m, spec), spec, window=("2020-01-01", "2020-06-30"))
    assert panel.x_snap.shape[1] == 57 and panel.schema.snapshot_features == Q26_FEATURES
    data = data_config_from_panel(panel)
    assert data.d_snap == 57  # from the panel's schema, not a constant
    cfg = dataclasses.replace(small_config(), data=data)
    assert NECConfig.validate(cfg).expert_input_dim == 57
    assert panel.metadata["feature_set"] == "q26"


# --------------------------------------------------------------------------- #
# Every characteristic, by hand
# --------------------------------------------------------------------------- #


def test_return_characteristics_by_hand(one):
    daily, m, ch = one
    r = daily["ret"]
    d = IDX[T]
    assert ch.loc[d, "ret_5d"] == pytest.approx(np.nansum(_w(r, 5)))  # 4 of 5 present
    assert np.isnan(_w(r, 5)).sum() == 1
    assert ch.loc[d, "ret_20d"] == pytest.approx(np.nansum(_w(r, 20)))
    mom = r.iloc[T - 251 : T - 21 + 1]
    assert len(mom) == 231
    assert ch.loc[d, "mom_12_1"] == pytest.approx(np.nansum(mom))
    assert ch.loc[d, "rvol_21d"] == pytest.approx(np.nanstd(_w(r, 21), ddof=1))
    big = np.expm1(r)
    assert ch.loc[d, "rmax1_21d"] == pytest.approx(np.nanmax(_w(big, 21)))
    assert ch.loc[d, "vol_shock"] == pytest.approx(
        np.nanstd(_w(r, 5), ddof=1) / np.nanstd(_w(r, 60), ddof=1)
    )
    x = _w(r, 21)
    x = x[~np.isnan(x)]
    n, s = len(x), x.std(ddof=1)
    skew = n / ((n - 1) * (n - 2)) * np.sum(((x - x.mean()) / s) ** 3)
    assert ch.loc[d, "rskew_21d"] == pytest.approx(skew)


def test_price_index_characteristics_by_hand(one):
    daily, _, ch = one
    p = _price_index(daily["retx"])
    d = IDX[T]
    assert ch.loc[d, "prc_highprc_252d"] == pytest.approx(p.iloc[T] / np.nanmax(_w(p, 252)))
    assert ch.loc[d, "ma50_gap"] == pytest.approx(p.iloc[T] / np.nanmean(_w(p, 50)) - 1)
    # a split changes no ratio of P: DlyRetx is split-adjusted by construction
    assert ch["prc_highprc_252d"].dropna().between(0, 1 + 1e-12).all()


def test_beta_coskew_ivol_by_hand(one):
    daily, m, ch = one
    r, d = daily["ret"], IDX[T]
    r3 = r.rolling(3, min_periods=3).sum()
    m3 = m.rolling(3, min_periods=3).sum()
    a, b = _w(r3, 1260), _w(m3, 1260)
    ok = ~np.isnan(a) & ~np.isnan(b)
    corr = np.corrcoef(a[ok], b[ok])[0, 1]
    beta = corr * np.nanstd(_w(r, 252), ddof=1) / np.nanstd(_w(m, 252), ddof=1)
    assert ch.loc[d, "beta_bab"] == pytest.approx(beta, rel=1e-9)

    x, y = _w(r, 21), _w(m, 21)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    cosk = np.mean(x * y * y) / (np.sqrt(np.mean(x * x)) * np.mean(y * y))
    assert ch.loc[d, "coskew_21d"] == pytest.approx(cosk, rel=1e-8)

    x, y = _w(r, 21), _w(m, 21)
    ok = ~np.isnan(x) & ~np.isnan(y)
    slope, icept = np.polyfit(y[ok], x[ok], 1)
    resid = x[ok] - (slope * y[ok] + icept)
    assert ch.loc[d, "ivol_capm_21d"] == pytest.approx(resid.std(ddof=1), rel=1e-8)


def test_size_liquidity_and_quotes_by_hand(one):
    daily, _, ch = one
    d = IDX[T]
    assert ch.loc[d, "log_me"] == pytest.approx(math.log(daily.loc[d, "me"]))
    big = np.expm1(daily["ret"])
    dv = daily["dollar_volume"] / 1e6
    illiq = (big.abs() / dv).where(daily["volume"] > 0)
    assert ch.loc[d, "ami_126d"] == pytest.approx(np.nanmean(_w(illiq, 126)))
    bid, ask = _w(daily["bid"], 21), _w(daily["ask"], 21)
    ok = (bid > 0) & (bid <= ask)
    assert not ok.all()  # a bad quote sits in the window and is excluded
    assert ch.loc[d, "qspread_21d"] == pytest.approx(
        np.mean(((ask - bid) / ((ask + bid) / 2))[ok])
    )
    over = daily["ret"] - np.log(daily["close"] / daily["open"])
    assert ch.loc[d, "overnight_20d"] == pytest.approx(np.nansum(_w(over, 20)))


def test_volume_characteristics_by_hand(one):
    daily, _, ch = one
    d = IDX[T]
    v, f = _w(daily["volume"], 20), _w(daily["share_factor"], 20)
    adj = v * f / f[-1]
    assert ch.loc[d, "volume_z"] == pytest.approx((v[-1] - adj.mean()) / adj.std(ddof=1))
    # across the split: put every volume on the latest basis by hand
    s = 905  # 900 is the first post-split row
    v, f = _w(daily["volume"], 20, s), _w(daily["share_factor"], 20, s)
    adj = v * f / f[-1]
    assert ch.iloc[s]["volume_z"] == pytest.approx((v[-1] - adj.mean()) / adj.std(ddof=1))
    lv = np.log((daily["volume"] * daily["share_factor"]).where(daily["volume"] > 0))
    a, b = _w(daily["ret"], 60), _w(lv, 60)
    ok = ~np.isnan(a) & ~np.isnan(b)
    assert ch.loc[d, "ret_vol_corr_60d"] == pytest.approx(np.corrcoef(a[ok], b[ok])[0, 1])
    r5 = daily["ret"].rolling(5, min_periods=5).sum()
    vr = np.nanvar(_w(r5, 60), ddof=1) / (5 * np.nanvar(_w(daily["ret"], 60), ddof=1))
    assert ch.loc[d, "var_ratio_60d"] == pytest.approx(vr)


def test_seasonality_by_hand(one):
    daily, _, ch = one
    d = IDX[T]
    nxt = d + pd.offsets.BDay(1)
    big = np.expm1(daily["ret"])
    vals = []
    for k in range(2, 6):
        days = IDX[(IDX.year == nxt.year - k) & (IDX.month == nxt.month)]
        month = big.reindex(days)
        assert month.notna().sum() >= math.ceil(0.8 * len(days))
        vals.append(float(np.prod(1 + month.dropna()) - 1))
    assert ch.loc[d, "seas_2_5an"] == pytest.approx(np.mean(vals))
    early = IDX[400]  # fewer than 5 years of history: missing
    assert np.isnan(ch.loc[early, "seas_2_5an"])


def test_fundamental_ratios_use_me_on_the_day(one):
    daily, _, ch = one
    d = IDX[T]
    me = daily.loc[d, "me"]
    assert ch.loc[d, "be_me"] == pytest.approx(900.0 / me)
    assert ch.loc[d, "ni_me"] == pytest.approx(80.0 / me)
    assert ch.loc[d, "netdebt_me"] == pytest.approx(300.0 / me)
    for c in ("niq_be", "ocf_at", "gp_at", "ni_inc8q", "at_gr1", "sale_gr1", "oaccruals_at",
              "taccruals_at", "debt_gr3", "noa_at", "cash_at", "niq_su", "saleq_su"):
        assert ch.loc[d, c] == daily.loc[d, c]
    neg = daily.assign(be=-5.0)
    assert stock_characteristics(neg, one[1], FS)["be_me"].isna().all()  # BE <= 0


def test_earnings_timing_uses_only_past_report_dates(one):
    daily, _, ch = one
    last = RDQS[-1]  # 2020-04-30, a Thursday
    i = IDX.get_loc(last)
    assert np.isnan(ch["days_since_earn"].iloc[IDX.get_loc(RDQS[0]) - 1])  # none known yet
    # on the report day itself the event is not usable yet (after the close):
    # the count runs from the previous report
    prev_event = IDX.searchsorted(RDQS[-2], side="right")
    assert ch["days_since_earn"].iloc[i] == i - prev_event
    assert ch["days_since_earn"].iloc[i + 1] == 0  # the next trading day
    assert ch["days_since_earn"].iloc[i + 7] == 6
    # expected next report: earliest d + 364 after t over known d
    t = IDX[IDX.get_loc(pd.Timestamp("2020-04-24"))]  # 2019-05-02 + 364 = 2020-04-30
    assert ch.loc[t, "earn_next5"] == 1.0  # within 5 business days
    far = pd.Timestamp("2020-03-02")  # next expected 2020-04-30: too far
    assert ch.loc[far, "earn_next5"] == 0.0
    # a report date in the future of t never counts
    future_only = daily.copy()
    future_only["rdq_date"] = pd.NaT
    future_only.loc[pd.Timestamp("2020-05-01"), "rdq_date"] = pd.Timestamp("2019-05-03")
    got = stock_characteristics(future_only, one[1], FS)
    assert np.isnan(got.loc[pd.Timestamp("2020-04-30"), "earn_next5"])
    assert got.loc[pd.Timestamp("2020-05-01"), "earn_next5"] == 0.0  # known from that day


def test_minimum_count_windows(one):
    daily, m, _ = one
    thin = daily.copy()
    pos = np.arange(T - 19, T + 1)
    assert daily["ret"].iloc[pos].isna().sum() == 1  # the planted gap at T - 3
    thin.iloc[pos[:4], thin.columns.get_loc("ret")] = np.nan  # 15 of 20 left: < 16
    ch = stock_characteristics(thin, m, FS)
    assert np.isnan(ch.loc[IDX[T], "ret_20d"])
    thin.iloc[pos[3], thin.columns.get_loc("ret")] = daily["ret"].iloc[pos[3]]  # 16 of 20
    ch = stock_characteristics(thin, m, FS)
    assert not np.isnan(ch.loc[IDX[T], "ret_20d"])
    assert FS.min_obs(20) == 16 and FS.min_obs(21) == 17 and FS.min_obs(126) == 101


def test_missing_columns_leave_their_characteristics_missing():
    daily, m = _stock(3)
    bare = daily[["ret", "volume", "dollar_volume"]]
    ch = stock_characteristics(bare, m, FS)
    for c in ("prc_highprc_252d", "ma50_gap", "qspread_21d", "overnight_20d", "log_me",
              "be_me", "niq_su", "days_since_earn", "earn_next5"):
        assert ch[c].isna().all(), c
    assert ch["ret_20d"].notna().any()


def test_no_lookahead_for_every_characteristic(one):
    """The value at t from data truncated at t equals the value from all of it."""
    daily, m, full = one
    for t in (IDX[T - 300], IDX[T - 3], IDX[T], IDX[N - 1]):
        trunc = stock_characteristics(daily.loc[:t], m.loc[:t], FS)
        a = full.loc[t].to_numpy(dtype=float)
        b = trunc.loc[t].to_numpy(dtype=float)
        bad = [c for c, x, y in zip(full.columns, a, b, strict=True)
               if not (np.isnan(x) and np.isnan(y)) and not np.isclose(x, y, rtol=1e-9)]
        assert not bad, f"future data changed {bad} at {t.date()}"
    # through stock_features as well (the sequence channels and market joins)
    spec = StageBSpec(seq_len=10)
    f_full = stock_features(daily, market_frame(m, spec), spec)
    cols = [c for c in f_full.columns if c not in (spec.stock_target, "mkt_fwd_ret")]
    t = IDX[T]
    f_cut = stock_features(daily.loc[:t], market_frame(m.loc[:t], spec), spec)
    assert np.allclose(f_full.loc[t, cols].to_numpy(float), f_cut.loc[t, cols].to_numpy(float),
                       equal_nan=True)


# --------------------------------------------------------------------------- #
# The cross-section: ranks, fill, flags, industry
# --------------------------------------------------------------------------- #


def _rows(n_dates: int = 3, n: int = 30, seed: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    frames = []
    for d in range(n_dates):
        df = pd.DataFrame(g.normal(size=(n, len(JKP_FEATURES) + len(MARKET_FEATURES))),
                          columns=[*JKP_FEATURES, *MARKET_FEATURES])
        df["earn_next5"] = (g.uniform(size=n) < 0.2).astype(float)
        df["sector"] = np.where(np.arange(n) < 12, 45.0, np.where(np.arange(n) < 15, 10.0,
                                                                np.nan))
        df["me"] = g.uniform(1, 100, n)
        df["date"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_ranks_lie_in_minus_one_one_and_average_zero():
    rows = _rows()
    out, _ = cross_section(rows, FS)
    ranked = [c for c in Q26_FEATURES if c not in UNRANKED]
    for d in range(3):
        block = out.loc[rows["date"] == d, ranked]
        assert block.min().min() >= -1.0 and block.max().max() <= 1.0
        for c in (*JKP_FEATURES, "ivol_capm_21d"):  # no ties, nothing missing
            assert block[c].mean() == pytest.approx(0.0, abs=1e-12)
            assert block[c].min() == -1.0 and block[c].max() == 1.0
    # the mapping: 2 (rank - 1) / (n - 1) - 1, ties averaged, n = 1 gives 0
    v = pd.Series([3.0, 1.0, 2.0, 2.0, np.nan])
    got = rank_to_unit(v, pd.Series([0] * 5))
    assert got.tolist()[:4] == pytest.approx([1.0, -1.0, 0.0, 0.0])
    assert np.isnan(got.iloc[4])
    assert rank_to_unit(pd.Series([5.0]), pd.Series([0])).iloc[0] == 0.0


def test_zero_fill_and_flags():
    rows = _rows(n_dates=1)
    rows.loc[0, "ret_20d"] = np.nan  # a CRSP characteristic
    rows.loc[1, "be_me"] = np.nan  # a fundamental
    rows.loc[2, "earn_next5"] = np.nan  # Compustat report dates: in neither flag
    out, rep = cross_section(rows, FS)
    assert out.loc[0, "ret_20d"] == 0.0 and out.loc[0, "flag_price_missing"] == 1.0
    assert out.loc[0, "flag_fund_missing"] == 0.0
    assert out.loc[1, "be_me"] == 0.0 and out.loc[1, "flag_fund_missing"] == 1.0
    assert out.loc[1, "flag_price_missing"] == 0.0
    assert out.loc[2, "earn_next5"] == 0.0
    assert out.loc[2, ["flag_price_missing", "flag_fund_missing"]].sum() == 0.0
    assert out.loc[3:, ["flag_price_missing", "flag_fund_missing"]].to_numpy().sum() == 0.0
    assert not out.isna().any().any()
    assert "earn_next5" not in CRSP_FLAGGED and "days_since_earn" not in CRSP_FLAGGED
    assert set(FUND_FLAGGED) == set(JKP_FEATURES[10:])
    assert rep["present_share"]["ret_20d"] == pytest.approx(29 / 30)


def test_sector_dummies_and_within_sector_ranks_with_fallback():
    rows = _rows(n_dates=1)
    out, rep = cross_section(rows, FS)
    # dummies: sector 45 for 12 rows, 10 for 3, none for 15 (all zero)
    assert out.loc[:11, "gics_45"].eq(1).all() and out.loc[12:14, "gics_10"].eq(1).all()
    assert out.loc[15:, list(SECTOR_FEATURES)].to_numpy().sum() == 0
    # sector 45 has 12 >= 5 values: ranked inside it
    inside = rank_to_unit(rows.loc[:11, "be_me"], pd.Series([0] * 12, index=range(12)))
    assert np.allclose(out.loc[:11, "be_me_sector"], inside)
    # sector 10 has 3 < 5: falls back to the universe rank; so do rows without one
    assert np.allclose(out.loc[12:14, "be_me_sector"], out.loc[12:14, "be_me"])
    assert np.allclose(out.loc[15:, "be_me_sector"], out.loc[15:, "be_me"])
    assert rep["within_sector_fallback_share"]["be_me_sector"] == pytest.approx(18 / 30)


def test_industry_signals_by_hand():
    rows = _rows(n_dates=1)
    fs_eq = dataclasses.replace(FS, industry_weighting="equal")
    for fs in (FS, fs_eq):
        sig = pd.Series(np.nan, index=rows.index)
        rel = pd.Series(np.nan, index=rows.index)
        for code in (45.0, 10.0):
            s = rows[rows["sector"] == code]
            w = s["me"] if fs is FS else pd.Series(1.0, index=s.index)
            sig[s.index] = np.sum(s["mom_12_1"] * w) / w.sum()  # own value included
            rel[s.index] = s["ret_20d"] - np.sum(s["ret_20d"] * w) / w.sum()
        out, _ = cross_section(rows, fs)
        assert np.allclose(out["ind_mom_12_1"], rank_to_unit(sig, rows["date"]).fillna(0.0))
        assert np.allclose(out["ret_20d_ind_rel"], rank_to_unit(rel, rows["date"]).fillna(0.0))
    out, _ = cross_section(rows, FS)
    assert out.loc[:11, "ind_mom_12_1"].nunique() == 1  # one sector, one value
    assert (out.loc[15:, "ind_mom_12_1"] == 0.0).all()  # no sector: missing, filled


# --------------------------------------------------------------------------- #
# Rows: never dropped for a missing characteristic
# --------------------------------------------------------------------------- #


def test_rows_are_kept_without_characteristics_and_dropped_without_a_target():
    daily = {str(10001 + i): _stock(i)[0] for i in range(6)}
    m = _stock(0)[1]
    spec = StageBSpec(seq_len=10)
    daily["10003"].iloc[1500:1520, daily["10003"].columns.get_loc("volume")] = np.nan
    daily["10004"].iloc[1550, daily["10004"].columns.get_loc("ret")] = np.nan
    panel = assemble_panel(daily, market_frame(m, spec), spec)
    assert panel.date_labels is not None
    on = {panel.date_labels[int(c)] for c in panel.date[panel.entity == 2]}
    assert str(IDX[1510].date()) in on  # missing volume: kept (filled and flagged)
    on4 = {panel.date_labels[int(c)] for c in panel.date[panel.entity == 3]}
    for k in range(spec.horizon):  # the targets whose window holds the gap: dropped
        assert str(IDX[1550 - 1 - k].date()) not in on4
    assert str(IDX[1550].date()) in on4  # the day itself has its own target
    # the first days of history are rows too (q26), with zero-filled windows
    assert panel.metadata["features"]["seq_window_incomplete_rows"] > 0
    assert not torch.isnan(panel.x_seq).any()


def test_snapshot_plus_hidden_requires_the_sequence_window():
    daily = {str(10001 + i): _stock(i)[0] for i in range(6)}
    m = _stock(0)[1]
    plain = StageBSpec(seq_len=10)
    hidden = StageBSpec(seq_len=10, input_mode="snapshot_plus_hidden")
    a = assemble_panel(daily, market_frame(m, plain), plain)
    b = assemble_panel(daily, market_frame(m, hidden), hidden)
    assert len(b) < len(a)
    assert b.metadata["features"]["seq_window_incomplete_rows"] == 0
    check_panel_input_mode(b, "snapshot_plus_hidden")
    with pytest.raises(ValueError, match="input_mode"):
        check_panel_input_mode(a, "snapshot_plus_hidden")
    cfg = dataclasses.replace(
        small_config(),
        data=data_config_from_panel(a),
        experts=ExpertConfig(hidden_dims=(8,), input_mode="snapshot_plus_hidden"),
    )
    from nec_moe import NECModel, Trainer

    with pytest.raises(ValueError, match="input_mode"):
        Trainer(NECModel(cfg)).fit(a, steps=1)


# --------------------------------------------------------------------------- #
# Fundamentals from the Compustat fixture, point in time
# --------------------------------------------------------------------------- #

CAL = pd.bdate_range("2009-01-01", "2022-12-31")


@pytest.fixture(scope="module")
def fund(tmp_path_factory):
    f = cfz.write_fixture(tmp_path_factory.mktemp("cfzq") / "Data")
    spec = CompustatSpec(crsp_dir=str(f.root), release=cfz.RELEASE)
    cs.extract_compustat(spec, [*range(10001, 10011)], verbose=False)
    ex = cs.load_compustat_extract(spec)
    q, _ = cs.quarterly_fundamentals(ex, spec, CAL)
    return f, q


def test_fundamental_inputs_by_hand(fund):
    f, q = fund
    g = q[q["KYGVKEY"] == "001001"].reset_index(drop=True)
    daily = cs.fundamental_inputs(g, CAL, FS)
    row = g.set_index(["FYEARQ", "FQTR"]).loc[(2019, 2)]
    t = row["avail"] + pd.offsets.BDay(2)  # this quarter is the newest known
    nxt = g.set_index(["FYEARQ", "FQTR"]).loc[(2019, 3)]
    assert t < nxt["avail"]
    by = g.set_index("fq_index")
    L = int(row["fq_index"])
    v = lambda item, lag=0: by.loc[L - lag, item]  # noqa: E731
    be = lambda lag=0: v("SEQQ", lag) + v("TXDITCQ", lag) - v("PSTKQ", lag)  # noqa: E731
    got = daily.loc[t]
    assert got["be"] == pytest.approx(be())
    assert got["ni_ttm"] == pytest.approx(sum(v("IBQ", k) for k in range(4)))
    assert got["niq_be"] == pytest.approx(v("IBQ") / be(1))
    ocf = sum(v("OANCFQ", k) for k in range(4))
    assert got["ocf_at"] == pytest.approx(ocf / v("ATQ"))
    assert got["gp_at"] == pytest.approx(
        sum(v("SALEQ", k) - v("COGSQ", k) for k in range(4)) / v("ATQ"))
    assert got["at_gr1"] == pytest.approx(v("ATQ") / v("ATQ", 4) - 1)
    assert got["sale_gr1"] == pytest.approx(
        sum(v("SALEQ", k) for k in range(4)) / sum(v("SALEQ", k) for k in range(4, 8)) - 1)
    assert got["oaccruals_at"] == pytest.approx(
        (sum(v("IBQ", k) for k in range(4)) - ocf) / v("ATQ"))
    debt = lambda lag: v("DLTTQ", lag) + v("DLCQ", lag)  # noqa: E731
    assert got["debt_gr3"] == pytest.approx(debt(0) / debt(12) - 1)
    assert got["netdebt"] == pytest.approx(debt(0) - v("CHEQ"))
    assert got["cash_at"] == pytest.approx(v("CHEQ") / v("ATQ"))
    noa = (v("ATQ") - v("CHEQ")) - (v("ATQ") - v("DLCQ") - v("DLTTQ") - v("MIBQ")
                                   - v("PSTKQ") - v("CEQQ"))
    assert got["noa_at"] == pytest.approx(noa / v("ATQ", 4))

    def tab(lag):
        wc = (v("ACTQ", lag) - v("CHEQ", lag)) - (v("LCTQ", lag) - v("DLCQ", lag))
        nco = (v("ATQ", lag) - v("ACTQ", lag) - v("IVAOQ", lag)) - (
            v("LTQ", lag) - v("LCTQ", lag) - v("DLTTQ", lag))
        fin = (v("IVSTQ", lag) + v("IVAOQ", lag)) - (
            v("DLTTQ", lag) + v("DLCQ", lag) + v("PSTKQ", lag))
        return wc + nco + fin
    assert got["taccruals_at"] == pytest.approx(
        (tab(0) - tab(4)) / ((v("ATQ") + v("ATQ", 4)) / 2))
    streak = 0
    for j in range(8):
        if v("IBQ", j) > v("IBQ", j + 4):
            streak += 1
        else:
            break
    assert got["ni_inc8q"] == streak
    # the surprise runs on the RDQ clock, from the quarters reported by then
    ts = row["avail_rdq"] + pd.offsets.BDay(1)
    diffs = [v("IBQ", j) - v("IBQ", j + 4) for j in range(8)]
    assert daily.loc[ts, "niq_su"] == pytest.approx(diffs[0] / np.std(diffs, ddof=1))
    sdiffs = [v("SALEQ", j) - v("SALEQ", j + 4) for j in range(8)]
    assert daily.loc[ts, "saleq_su"] == pytest.approx(sdiffs[0] / np.std(sdiffs, ddof=1))


def test_book_equity_fallbacks():
    base = {"fq_index": 1, "avail": pd.Timestamp("2020-01-02")}
    k = cs._Known(pd.DataFrame([{**base, "SEQQ": 10.0, "TXDITCQ": np.nan, "PSTKQ": 2.0}]))
    assert k.book_equity() == 8.0  # TXDITCQ missing counts as 0
    k = cs._Known(pd.DataFrame([{**base, "SEQQ": np.nan, "CEQQ": 7.0, "PSTKQ": 2.0,
                                 "TXDITCQ": 1.0}]))
    assert k.book_equity() == 7.0 + 2.0 + 1.0 - 2.0
    k = cs._Known(pd.DataFrame([{**base, "SEQQ": np.nan, "CEQQ": np.nan, "PSTKQ": np.nan,
                                 "ATQ": 30.0, "LTQ": 25.0, "TXDITCQ": np.nan}]))
    assert k.book_equity() == 5.0


def test_zero_if_missing_items_in_accruals_and_noa(fund):
    """IVAOQ, IVSTQ, MIBQ and PSTKQ count as 0 when blank (Tom, 2026-10-05);
    a core item (ACTQ) left blank still makes total accruals missing."""
    _, q = fund
    g = q[q["KYGVKEY"] == "001001"].copy()
    t = g.set_index(["FYEARQ", "FQTR"]).loc[(2019, 2), "avail"] + pd.offsets.BDay(2)
    full = cs.fundamental_inputs(g, CAL, FS).loc[t]
    blank = g.copy()
    for item in ("IVAOQ", "IVSTQ", "MIBQ", "PSTKQ"):
        blank[item] = np.nan
    got = cs.fundamental_inputs(blank, CAL, FS).loc[t]
    assert not np.isnan(got["taccruals_at"]) and not np.isnan(got["noa_at"])
    zeroed = g.copy()
    for item in ("IVAOQ", "IVSTQ", "MIBQ", "PSTKQ"):
        zeroed[item] = 0.0
    want = cs.fundamental_inputs(zeroed, CAL, FS).loc[t]
    assert got["taccruals_at"] == pytest.approx(want["taccruals_at"])
    assert got["noa_at"] == pytest.approx(want["noa_at"])
    assert got["taccruals_at"] != pytest.approx(full["taccruals_at"])  # the items mattered
    core = g.copy()
    core["ACTQ"] = np.nan
    assert np.isnan(cs.fundamental_inputs(core, CAL, FS).loc[t, "taccruals_at"])
    strict = dataclasses.replace(FS, zero_if_missing_items=())
    assert np.isnan(cs.fundamental_inputs(blank, CAL, strict).loc[t, "taccruals_at"])


def test_trailing_flows_need_four_consecutive_quarters(fund):
    _, q = fund
    g = q[q["KYGVKEY"] == "001003"].copy()
    # drop one quarter: every trailing sum that spans it becomes missing
    gap = g.set_index(["FYEARQ", "FQTR"]).loc[(2019, 2), "fq_index"]
    g = g[g["fq_index"] != gap]
    daily = cs.fundamental_inputs(g, CAL, FS)
    after = g[g["fq_index"] == gap + 1].iloc[0]
    t = after["avail"] + pd.offsets.BDay(1)
    assert np.isnan(daily.loc[t, "ni_ttm"]) and np.isnan(daily.loc[t, "ocf_at"])
    assert not np.isnan(daily.loc[t, "cash_at"])  # a point-in-time level is unaffected


def test_fundamentals_have_no_lookahead(fund):
    """Truncate the Compustat availability dates at t: the value at t is the same."""
    _, q = fund
    g = q[q["KYGVKEY"] == "001004"].reset_index(drop=True)
    full = cs.fundamental_inputs(g, CAL, FS)
    for t in (pd.Timestamp("2016-03-01"), pd.Timestamp("2019-11-15"),
              pd.Timestamp("2020-08-03")):
        cut = g.copy()
        cut["avail"] = cut["avail"].where(cut["avail"] <= t)
        cut["avail_rdq"] = cut["avail_rdq"].where(cut["avail_rdq"] <= t)
        trunc = cs.fundamental_inputs(cut, CAL[CAL <= t], FS)
        assert np.allclose(full.loc[t].to_numpy(float), trunc.loc[t].to_numpy(float),
                           equal_nan=True), t


# --------------------------------------------------------------------------- #
# The full build: CRSP fixture + Compustat fixture
# --------------------------------------------------------------------------- #

#: shorter price windows: the CRSP fixture holds about 19 months before its window
SMALL = FeatureSpec(beta_corr_window=120, beta_corr_min_obs=80, seas_first_year=1,
                    seas_last_year=1)


@pytest.fixture(scope="module")
def q26_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("both") / "Data"
    fx.write_fixture(root)
    cfz.write_fixture(root)
    crsp_spec = CRSPSpec(crsp_dir=str(root), release=fx.RELEASE, start=fx.START, end=fx.END,
                         chunk_bytes=20_000, member_count_min=5, member_count_max=8)
    extract_crsp(crsp_spec, verbose=False)
    cspec = CompustatSpec(crsp_dir=str(root), release=cfz.RELEASE, start=fx.START,
                          end=fx.END)
    cs.extract_compustat(cspec, [*range(10001, 10011)], verbose=False)
    stage_b = StageBSpec(seq_len=10, horizon=5, min_names_per_date=4, features=SMALL)
    build = build_crsp_panel(crsp_spec, stage_b, extract=load_extract(crsp_spec),
                             compustat=cs.load_compustat_extract(cspec),
                             compustat_spec=cspec, verbose=False)
    return build, load_extract(crsp_spec), crsp_spec


def test_q26_build_on_both_fixtures(q26_build):
    build, ex, _ = q26_build
    panel = build.panel
    assert panel.x_snap.shape[1] == 57
    meta = panel.metadata
    assert meta["feature_set"] == "q26" and meta["sector_source"] == "gics"
    assert meta["compustat_release"] == cfz.RELEASE
    assert meta["crsp_stock_file"] == "StkDlySecurityData"
    assert meta["features"]["rows_restricted_to_universe"] is True
    # every row is a member that day (10007 only from JOIN, 10008 until LEAVE)
    assert panel.date_labels is not None and panel.entity_labels is not None
    code7 = panel.entity_labels.index("10007")
    days7 = pd.to_datetime([panel.date_labels[int(d)] for d in panel.date[panel.entity == code7]])
    assert days7.min() >= fx.JOIN
    # the market-neutral target averages to zero over each date's rows
    for d in torch.unique(panel.date)[:15]:
        assert abs(float(panel.y[panel.date == d].double().mean())) < 1e-6
    rep = build.report["q26_inputs"]
    assert rep["permno_days"] > 0 and "quarters" in rep


def test_q26_build_attaches_sector_me_and_fundamentals(q26_build):
    build, ex, spec = q26_build
    panel = build.panel
    idx = {c: i for i, c in enumerate(Q26_FEATURES)}
    assert panel.date_labels is not None and panel.entity_labels is not None
    lab = np.asarray(panel.entity_labels)[panel.entity.numpy()]
    day = pd.to_datetime(np.asarray(panel.date_labels)[panel.date.numpy()])
    x = panel.x_snap.numpy()
    # 10003's sector switches from 45 to 50 on the fixture's INDFROM date
    m3 = lab == "10003"
    assert (x[m3 & (day < cfz.SECTOR_SWITCH), idx["gics_45"]] == 1).all()
    assert (x[m3 & (day >= cfz.SECTOR_SWITCH), idx["gics_50"]] == 1).all()
    # 10008 has no GICS row: every dummy 0
    m8 = lab == "10008"
    assert x[m8][:, [idx[c] for c in SECTOR_FEATURES]].sum() == 0
    # fundamentals: 10001 is linked, its flag is off once enough history is known;
    # 10007 has no link before MOVE (2019-07-01) but the window starts after it
    assert (x[lab == "10001", idx["flag_fund_missing"]] == 0).any()
    # ME sums every share class: 10003's company includes 10010
    from nec_moe.compustat import company_market_equity

    me = company_market_equity(ex, spec.cap_unit_dollars)
    t = pd.Timestamp("2020-03-02")
    cap = ex.stock.set_index(["PERMNO", "DlyCalDt"])["DlyCap"]
    assert me.loc[t, 10003] == pytest.approx(
        (cap[(10003, t)] + cap[(10010, t)]) * 1000 / 1e6)
    assert me.loc[t, 10004] == pytest.approx(cap[(10004, t)] * 1000 / 1e6)


def test_registry_records_feature_set(q26_build, tmp_path: Path):
    panel = q26_build[0].panel
    cfg = dataclasses.replace(small_config(), data=data_config_from_panel(panel))
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(panel, [nec_arm("soft", cfg)], seeds=(0,), registry=reg, tag="q26",
                  steps=3, n_folds=2, test_dates_per_fold=10, purge_dates=5, verbose=False)
    rows = reg.trials("q26")
    assert rows and all(r.config["feature_set"] == "q26" for r in rows)
