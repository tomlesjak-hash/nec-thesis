"""The Q26 characteristics: per-stock definitions and the per-date cross-section.

Brief 08 part D; decision Q26 (``Master Thesis/Advisor_Questions.md``). Every
window, threshold and cap is a :class:`~nec_moe.features.FeatureSpec` field.

Two stages:

1. :func:`stock_characteristics` computes the 40 characteristics of one
   stock from its daily frame (:data:`nec_moe.features.DAILY_COLUMNS`), all
   trailing through the close of t. The CRSP layer adds to that frame the
   company market equity ``me`` ($ millions), the GICS ``sector``, the
   report dates ``rdq_date`` and the point-in-time fundamental inputs
   (:data:`FUNDAMENTAL_INPUTS`, from :mod:`nec_moe.compustat`); a column the
   frame lacks leaves its characteristics missing, never zero.
2. :func:`cross_section` turns a date's universe rows into the 57 inputs:
   the industry signals, the per-date ranks mapped to [-1, 1], the
   within-sector ranks, the 0-fill and the two missing-value flags.

Conventions (brief 08 D): ``r`` is the daily log return, ``R`` the simple
return, ``m`` the market's daily log return; ``P`` is a split- and
distribution-consistent price index, the cumulative product of
``1 + DlyRetx`` (raw price levels are never compared across days); windows are
in trading days. A rolling statistic needs at least ``min_obs_frac`` of its
window's days (:meth:`FeatureSpec.min_obs`), otherwise it is missing. A
window's sum of log returns is the sum over the days that have one: CRSP's
next return after a gap spans the missing days (``DlyRetDurFlg`` P-codes),
so the available returns already carry the window's move.

Two date rules use the calendar without looking ahead: the seasonality's
"month of t+1" and the earnings window's "next 5 trading days" are read off
``t`` plus business days (``pandas.offsets.BDay``), not off later rows,
so a value at t never depends on what the data holds after t. Only an
exchange holiday at a month end or inside those 5 days makes them differ
from the true trading calendar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - features imports this module
    from .features import FeatureSpec

__all__ = [
    "JKP_FEATURES",
    "MARKET_FEATURES",
    "SECTOR_FEATURES",
    "INDUSTRY_SIGNALS",
    "WITHIN_SECTOR_FEATURES",
    "FLAG_FEATURES",
    "Q26_FEATURES",
    "Q26_CHARACTERISTICS",
    "UNRANKED",
    "CRSP_FLAGGED",
    "FUND_FLAGGED",
    "FUNDAMENTAL_INPUTS",
    "stock_characteristics",
    "cross_section",
    "rank_to_unit",
]

#: D.1, 13 JKP themes x 2, in the brief's order.
JKP_FEATURES: tuple[str, ...] = (
    "ret_5d", "ret_20d", "mom_12_1", "prc_highprc_252d", "rvol_21d", "beta_bab",
    "log_me", "ami_126d", "seas_2_5an", "coskew_21d", "be_me", "ni_me", "niq_be",
    "ocf_at", "gp_at", "ni_inc8q", "at_gr1", "sale_gr1", "oaccruals_at", "taccruals_at",
    "debt_gr3", "noa_at", "netdebt_me", "cash_at", "niq_su", "saleq_su",
)
#: D.2, 6 short-horizon market themes x 2.
MARKET_FEATURES: tuple[str, ...] = (
    "ivol_capm_21d", "vol_shock", "rskew_21d", "rmax1_21d", "volume_z", "qspread_21d",
    "overnight_20d", "ma50_gap", "earn_next5", "days_since_earn", "var_ratio_60d",
    "ret_vol_corr_60d",
)
#: D.3: 11 GICS sector dummies, 2 industry return signals, 4 within-sector ranks.
SECTOR_FEATURES: tuple[str, ...] = tuple(
    f"gics_{c}" for c in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
)
INDUSTRY_SIGNALS: tuple[str, ...] = ("ind_mom_12_1", "ret_20d_ind_rel")
#: (input name, the characteristic ranked within its sector)
WITHIN_SECTOR: tuple[tuple[str, str], ...] = (
    ("be_me_sector", "be_me"), ("ni_me_sector", "ni_me"),
    ("niq_be_sector", "niq_be"), ("ocf_at_sector", "ocf_at"),
)
WITHIN_SECTOR_FEATURES: tuple[str, ...] = tuple(name for name, _ in WITHIN_SECTOR)
#: D.4: the two missing-value flags.
FLAG_FEATURES: tuple[str, ...] = ("flag_price_missing", "flag_fund_missing")
#: The 57 inputs, in channel order.
Q26_FEATURES: tuple[str, ...] = (
    *JKP_FEATURES, *MARKET_FEATURES, *SECTOR_FEATURES, *INDUSTRY_SIGNALS,
    *WITHIN_SECTOR_FEATURES, *FLAG_FEATURES,
)
#: The 40 characteristics (26 + 12 + the 2 industry signals).
Q26_CHARACTERISTICS: tuple[str, ...] = (*JKP_FEATURES, *MARKET_FEATURES, *INDUSTRY_SIGNALS)
#: 0/1 inputs, never ranked.
UNRANKED: tuple[str, ...] = ("earn_next5", *SECTOR_FEATURES, *FLAG_FEATURES)
#: Characteristics 1-10 and 27-38 that come from CRSP (D.4.5): all but the
#: two earnings-timing ones, which come from Compustat report dates.
CRSP_FLAGGED: tuple[str, ...] = tuple(
    c for c in (*JKP_FEATURES[:10], *MARKET_FEATURES)
    if c not in ("earn_next5", "days_since_earn")
)
#: Characteristics 11-26.
FUND_FLAGGED: tuple[str, ...] = JKP_FEATURES[10:]

#: Point-in-time fundamental inputs the CRSP layer attaches to a daily frame
#: (:func:`nec_moe.compustat.fundamental_inputs`): the ratios that need no
#: market equity, and the three numerators that do (``be``, ``ni_ttm``,
#: ``netdebt``; divided by ``me`` on the day).
FUNDAMENTAL_INPUTS: tuple[str, ...] = (
    "be", "ni_ttm", "netdebt", "niq_be", "ocf_at", "gp_at", "ni_inc8q", "at_gr1",
    "sale_gr1", "oaccruals_at", "taccruals_at", "debt_gr3", "noa_at", "cash_at",
    "niq_su", "saleq_su",
)


# --------------------------------------------------------------------------- #
# Rolling helpers (pairwise-complete, minimum count)
# --------------------------------------------------------------------------- #


def _roll(x: pd.Series, window: int, min_obs: int):
    return x.rolling(window, min_periods=min_obs)


def _rolling_moments(
    x: pd.Series, y: pd.Series, window: int, min_obs: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Over each window's days where both are present: count, means, and the
    centred sums of squares and cross products ``Sxx, Syy, Sxy``."""
    both = x.notna() & y.notna()
    xv, yv = x.where(both), y.where(both)
    n = both.astype(float).rolling(window, min_periods=1).sum()
    sx = xv.rolling(window, min_periods=1).sum()
    sy = yv.rolling(window, min_periods=1).sum()
    sxx = (xv * xv).rolling(window, min_periods=1).sum()
    syy = (yv * yv).rolling(window, min_periods=1).sum()
    sxy = (xv * yv).rolling(window, min_periods=1).sum()
    enough = n >= min_obs
    mx, my = sx / n, sy / n
    cxx = (sxx - n * mx * mx).where(enough)
    cyy = (syy - n * my * my).where(enough)
    cxy = (sxy - n * mx * my).where(enough)
    return n.where(enough), mx, my, cxx, cyy, cxy


def _rolling_corr(x: pd.Series, y: pd.Series, window: int, min_obs: int) -> pd.Series:
    _, _, _, cxx, cyy, cxy = _rolling_moments(x, y, window, min_obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cxy / np.sqrt(cxx * cyy)


def _price_index(retx: pd.Series) -> pd.Series:
    """``P``: the cumulative product of ``1 + DlyRetx`` over the days that
    have one, missing on a day without. Only ratios of ``P`` inside a window
    are used, so its anchor (the first row) never matters."""
    lr = np.log1p(retx)
    p = np.exp(lr.fillna(0.0).cumsum())
    return p.where(retx.notna())


def _volume_z(volume: pd.Series, factor: pd.Series, window: int, min_obs: int) -> pd.Series:
    """The split-invariant volume z-score (``volume_z_20d``) with a minimum
    count: every volume in the window on today's share basis, ``v_s F_s /
    F_t``, against the window's mean and std over the days that have one;
    missing without today's volume."""
    from numpy.lib.stride_tricks import sliding_window_view

    v = volume.to_numpy(dtype=float)
    f = factor.to_numpy(dtype=float)
    pad = np.full(window - 1, np.nan)
    vw = sliding_window_view(np.concatenate([pad, v]), window)
    fw = sliding_window_view(np.concatenate([pad, f]), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = vw * fw / f[:, None]
    count = np.sum(~np.isnan(adj), axis=1)
    out = np.full(len(v), np.nan)
    ok = (count >= min_obs) & ~np.isnan(v) & (f != 0) & ~np.isnan(f)
    if ok.any():
        a = adj[ok]
        mean = np.nanmean(a, axis=1)
        sd = np.nanstd(a, axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (v[ok] - mean) / sd
        out[ok] = np.where(sd > 0, z, np.nan)
    return pd.Series(out, index=volume.index)


def _seasonality(simple: pd.Series, fs: FeatureSpec) -> pd.Series:
    """``seas_2_5an``: mean of the stock's simple returns in the calendar month
    of t+1, over the years ``seas_first_year``..``seas_last_year`` back.

    A month's return compounds its daily simple returns; it counts when at
    least ``min_obs_frac`` of the month's trading days (on this stock's
    calendar) have one. The years needed must all have that month's return
    (the minimum count of ``seas_last_year - seas_first_year + 1`` values,
    rounded up from ``min_obs_frac``).
    """
    idx = pd.DatetimeIndex(simple.index)
    key = idx.year * 12 + (idx.month - 1)
    df = pd.DataFrame({"key": key, "lr": np.log1p(simple.to_numpy()),
                       "ok": simple.notna().to_numpy()})
    grp = df.groupby("key")
    days = grp.size()
    have = grp["ok"].sum()
    month_ret = np.expm1(grp["lr"].sum(min_count=1))
    enough = have >= np.ceil(fs.min_obs_frac * days - 1e-9)
    month_ret = month_ret.where(enough)
    nxt = idx + pd.offsets.BDay(1)  # the month of t+1, from t alone
    years = range(fs.seas_first_year, fs.seas_last_year + 1)
    vals = np.column_stack([
        month_ret.reindex((nxt.year - k) * 12 + (nxt.month - 1)).to_numpy()
        for k in years
    ])
    need = fs.min_obs(len(years))
    count = np.sum(~np.isnan(vals), axis=1)
    total = np.nansum(vals, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return pd.Series(np.where(count >= need, mean, np.nan), index=simple.index)


def _earnings_timing(
    index: pd.DatetimeIndex, rdq: pd.Series, fs: FeatureSpec
) -> tuple[pd.Series, pd.Series]:
    """``earn_next5`` and ``days_since_earn`` from report dates known at t.

    ``rdq`` holds, on the first trading day on or after each report date,
    that report date (``NaT`` elsewhere); a report date is known from that
    row on, so nothing after t is read. ``days_since_earn`` counts trading
    days since the first trading day **after** the latest known report date
    (announcements are often after the close). ``earn_next5`` is 1 when the
    earliest ``d + earn_cycle_days`` over known report dates d that falls
    after t is within ``earn_ahead_days`` business days of t, else 0;
    missing with no report date known.
    """
    n = len(index)
    since = np.full(n, np.nan)
    nxt5 = np.full(n, np.nan)
    rows = np.flatnonzero(rdq.notna().to_numpy())
    if len(rows) == 0:
        return pd.Series(since, index=index), pd.Series(nxt5, index=index)
    days = index.to_numpy()
    d = pd.DatetimeIndex(rdq.iloc[rows]).to_numpy()
    # event day: the first trading day strictly after each report date
    ev_pos = np.searchsorted(days, d, side="right")
    cycle = np.timedelta64(fs.earn_cycle_days, "D")
    horizon = (index + pd.offsets.BDay(fs.earn_ahead_days)).to_numpy()
    for t in range(rows[0], n):
        known = rows <= t
        usable = ev_pos[known] <= t
        if usable.any():
            since[t] = t - ev_pos[known][usable].max()
        expected = d[known] + cycle
        ahead = expected[expected > days[t]]
        nxt5[t] = float(ahead.size > 0 and ahead.min() <= horizon[t])
    return pd.Series(since, index=index), pd.Series(nxt5, index=index)


# --------------------------------------------------------------------------- #
# One stock
# --------------------------------------------------------------------------- #


def _col(f: pd.DataFrame, name: str) -> pd.Series:
    return f[name].astype(float) if name in f.columns else pd.Series(np.nan, index=f.index)


def stock_characteristics(
    daily: pd.DataFrame, m: pd.Series, fs: FeatureSpec
) -> pd.DataFrame:
    """The 40 per-stock characteristics (the industry signals and the ranks
    come later, from the cross-section), plus ``sector`` and ``me``.

    ``daily`` is one stock's frame; ``m`` the market's daily log return on
    the same dates. Everything at t uses rows through t only.
    """
    r = _col(daily, "ret")
    big_r = np.expm1(r)
    retx = _col(daily, "retx")
    p = _price_index(retx)
    me = _col(daily, "me")
    mo = fs.min_obs
    out = pd.DataFrame(index=daily.index)

    # D.1, CRSP part
    out["ret_5d"] = _roll(r, fs.ret_5d_window, mo(fs.ret_5d_window)).sum()
    out["ret_20d"] = _roll(r, fs.ret_20d_window, mo(fs.ret_20d_window)).sum()
    mom_len = fs.mom_start_lag - fs.mom_end_lag + 1
    out["mom_12_1"] = _roll(r, mom_len, mo(mom_len)).sum().shift(fs.mom_end_lag)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["prc_highprc_252d"] = p / _roll(p, fs.high_window, mo(fs.high_window)).max()
    out["rvol_21d"] = _roll(r, fs.rvol_window, mo(fs.rvol_window)).std()
    o = fs.beta_overlap_days
    r3 = r.rolling(o, min_periods=o).sum()
    m3 = m.rolling(o, min_periods=o).sum()
    corr = _rolling_corr(r3, m3, fs.beta_corr_window, fs.beta_corr_min_obs)
    vw = fs.beta_vol_window
    with np.errstate(invalid="ignore", divide="ignore"):
        out["beta_bab"] = corr * _roll(r, vw, mo(vw)).std() / _roll(m, vw, mo(vw)).std()
        out["log_me"] = np.log(me.where(me > 0))
    vol = _col(daily, "volume")
    dollar_m = _col(daily, "dollar_volume") / 1e6
    with np.errstate(invalid="ignore", divide="ignore"):
        illiq = (big_r.abs() / dollar_m).where((vol > 0) & (dollar_m > 0))
    out["ami_126d"] = _roll(illiq, fs.amihud_window, mo(fs.amihud_window)).mean()
    out["seas_2_5an"] = _seasonality(big_r, fs)
    w = fs.coskew_window
    n, mx, my, cxx, cyy, _ = _rolling_moments(r, m, w, mo(w))
    both = r.notna() & m.notna()
    # mean(x y^2) with x, y centred inside each window, expanded by hand
    xv, yv = r.where(both), m.where(both)
    s = lambda z: z.rolling(w, min_periods=1).sum()  # noqa: E731
    sxyy = s(xv * yv * yv) - 2 * my * s(xv * yv) + my * my * s(xv) \
        - mx * s(yv * yv) + 2 * mx * my * s(yv) - n * mx * my * my
    with np.errstate(invalid="ignore", divide="ignore"):
        out["coskew_21d"] = (sxyy / n) / (np.sqrt(cxx / n) * (cyy / n))

    # D.1, fundamentals (point in time; ratios to ME use ME on the day)
    be = _col(daily, "be")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["be_me"] = (be / me).where((be > 0) & (me > 0))
        out["ni_me"] = (_col(daily, "ni_ttm") / me).where(me > 0)
        out["netdebt_me"] = (_col(daily, "netdebt") / me).where(me > 0)
    for name in ("niq_be", "ocf_at", "gp_at", "ni_inc8q", "at_gr1", "sale_gr1",
                 "oaccruals_at", "taccruals_at", "debt_gr3", "noa_at", "cash_at",
                 "niq_su", "saleq_su"):
        out[name] = _col(daily, name)

    # D.2, the short-horizon market block
    w = fs.ivol_window
    n, _, _, cxx, cyy, cxy = _rolling_moments(r, m, w, mo(w))
    with np.errstate(invalid="ignore", divide="ignore"):
        ssr = (cxx - cxy * cxy / cyy).clip(lower=0.0)
        out["ivol_capm_21d"] = np.sqrt(ssr / (n - 1))
        out["vol_shock"] = (
            _roll(r, fs.vol_shock_short, mo(fs.vol_shock_short)).std()
            / _roll(r, fs.vol_shock_long, mo(fs.vol_shock_long)).std()
        )
    out["rskew_21d"] = _roll(r, fs.rskew_window, mo(fs.rskew_window)).skew()
    out["rmax1_21d"] = _roll(big_r, fs.rmax_window, mo(fs.rmax_window)).max()
    factor = _col(daily, "share_factor") if "share_factor" in daily.columns \
        else pd.Series(1.0, index=daily.index)
    out["volume_z"] = _volume_z(vol, factor, fs.volume_z_window, mo(fs.volume_z_window))
    bid, ask = _col(daily, "bid"), _col(daily, "ask")
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = ((ask - bid) / ((ask + bid) / 2)).where((bid > 0) & (bid <= ask))
        intraday = np.log(_col(daily, "close") / _col(daily, "open"))
    out["qspread_21d"] = _roll(spread, fs.spread_window, mo(fs.spread_window)).mean()
    overnight = (r - intraday.where(np.isfinite(intraday)))
    out["overnight_20d"] = _roll(overnight, fs.overnight_window, mo(fs.overnight_window)).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ma50_gap"] = p / _roll(p, fs.ma_window, mo(fs.ma_window)).mean() - 1.0
    rdq = daily["rdq_date"] if "rdq_date" in daily.columns else pd.Series(pd.NaT, index=daily.index)
    out["days_since_earn"], out["earn_next5"] = _earnings_timing(
        pd.DatetimeIndex(daily.index), pd.to_datetime(rdq), fs
    )
    q, w = fs.var_ratio_q, fs.var_ratio_window
    rq = r.rolling(q, min_periods=q).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["var_ratio_60d"] = _roll(rq, w, mo(w)).var() / (q * _roll(r, w, mo(w)).var())
        log_vol = np.log((vol * factor).where(vol > 0))
    w = fs.ret_vol_corr_window
    out["ret_vol_corr_60d"] = _rolling_corr(r, log_vol, w, mo(w))

    out["sector"] = _col(daily, "sector")
    out["me"] = me
    out = out[[*JKP_FEATURES, *MARKET_FEATURES, "sector", "me"]]
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# The cross-section of a date
# --------------------------------------------------------------------------- #


def rank_to_unit(values: pd.Series, groups: list[pd.Series] | pd.Series) -> pd.Series:
    """Rank within each group (ties: the average rank) over the non-missing
    values, mapped to ``2 (rank - 1) / (n - 1) - 1`` in [-1, 1]; 0 when the
    group has one value; missing stays missing."""
    g = values.groupby(groups)
    rank = g.rank(method="average")
    n = g.transform("count")
    with np.errstate(invalid="ignore", divide="ignore"):
        x = 2.0 * (rank - 1.0) / (n - 1.0) - 1.0
    return x.where(n > 1, 0.0).where(values.notna())


def _sector_mean(
    values: pd.Series, weights: pd.Series, date: pd.Series, sector: pd.Series
) -> pd.Series:
    ok = values.notna() & weights.notna() & sector.notna()
    v, w = values.where(ok, 0.0), weights.where(ok, 0.0)
    keys = [date, sector]
    num = (v * w).groupby(keys).transform("sum")
    den = w.groupby(keys).transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return out.where(sector.notna() & (den > 0))


def cross_section(rows: pd.DataFrame, fs: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    """The 57 inputs of a set of rows, computed per date over those rows.

    ``rows`` holds a ``date`` column, the 40 characteristics of
    :func:`stock_characteristics` (the industry signals excepted), ``sector``
    and ``me``; it must be the dates' universe rows, nothing else, since every
    rank and sector aggregate is taken over it. Steps (D.3, D.4):

    1. the industry signals: the sector's ``mom_12_1`` and ``ret_20d``,
       weighted by ``me`` (``industry_weighting="value"``) or equally, over
       the date's rows in the sector, each stock's own value included;
       ``ret_20d_ind_rel`` is the stock's ``ret_20d`` minus its sector's;
    2. each characteristic ranked per date over the rows that have it, ties
       averaged, mapped to [-1, 1] (n = 1 gives 0); ``earn_next5`` is not
       ranked;
    3. ``be_me``, ``ni_me``, ``niq_be``, ``ocf_at`` also ranked within the
       date's sector; a sector with fewer than ``min_sector_names`` valid
       values that date (or a stock without a sector) falls back to the
       universe rank;
    4. the 0-fill, after ranking (0 is the median), and the flags:
       ``flag_price_missing`` when any CRSP characteristic was filled,
       ``flag_fund_missing`` when any of characteristics 11-26 was;
    5. the 11 sector dummies, all 0 without a sector.

    Returns the inputs in :data:`Q26_FEATURES` order and aggregate counts.
    """
    date, sector = rows["date"], rows["sector"]
    weights = rows["me"] if fs.industry_weighting == "value" else \
        pd.Series(1.0, index=rows.index)
    raw = rows[[*JKP_FEATURES, *MARKET_FEATURES]].copy()
    raw["ind_mom_12_1"] = _sector_mean(rows["mom_12_1"], weights, date, sector)
    raw["ret_20d_ind_rel"] = rows["ret_20d"] - _sector_mean(rows["ret_20d"], weights,
                                                            date, sector)

    out = pd.DataFrame(index=rows.index)
    for c in (*JKP_FEATURES, *MARKET_FEATURES):
        out[c] = raw[c] if c == "earn_next5" else rank_to_unit(raw[c], date)
    for code, name in zip((10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60), SECTOR_FEATURES,
                          strict=True):
        out[name] = (sector == code).astype(float)
    for c in INDUSTRY_SIGNALS:
        out[c] = rank_to_unit(raw[c], date)
    sector_key = sector.fillna(-1.0)
    for name, base in WITHIN_SECTOR:
        valid = raw[base].notna()
        n_sector = valid.groupby([date, sector_key]).transform("sum")
        inside = rank_to_unit(raw[base], [date, sector_key])
        use_sector = sector.notna() & (n_sector >= fs.min_sector_names)
        out[name] = inside.where(use_sector, out[base])

    filled = out[[*Q26_CHARACTERISTICS, *WITHIN_SECTOR_FEATURES]].isna()
    out["flag_price_missing"] = filled[list(CRSP_FLAGGED)].any(axis=1).astype(float)
    out["flag_fund_missing"] = filled[list(FUND_FLAGGED)].any(axis=1).astype(float)
    report = {
        "rows": int(len(rows)),
        "present_share": {c: float(1.0 - filled[c].mean()) for c in Q26_CHARACTERISTICS},
        "flag_price_rate": float(out["flag_price_missing"].mean()) if len(out) else 0.0,
        "flag_fund_rate": float(out["flag_fund_missing"].mean()) if len(out) else 0.0,
        "sector_missing_share": float(sector.isna().mean()) if len(out) else 0.0,
        "within_sector_fallback_share": {
            name: float((out[name].notna() & ~(sector.notna() & (
                raw[base].notna().groupby([date, sector_key]).transform("sum")
                >= fs.min_sector_names))).mean()) if len(out) else 0.0
            for name, base in WITHIN_SECTOR
        },
    }
    if "year" in rows.columns:  # aggregate coverage by calendar year (E.3)
        by_year: dict[str, dict] = {}
        for year, idx in rows.groupby("year").groups.items():
            fy = filled.loc[idx]
            by_year[str(year)] = {
                "rows": int(len(idx)),
                "present_share": {c: float(1.0 - fy[c].mean()) for c in Q26_CHARACTERISTICS},
                "flag_price_rate": float(out.loc[idx, "flag_price_missing"].mean()),
                "flag_fund_rate": float(out.loc[idx, "flag_fund_missing"].mean()),
                "sector_missing_share": float(sector.loc[idx].isna().mean()),
            }
        report["by_year"] = by_year
    out = out.fillna(0.0)
    return out[list(Q26_FEATURES)], report
