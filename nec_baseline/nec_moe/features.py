"""Stage B feature engineering: daily returns and volume -> the Panel contract (design doc §6).

Input: one **daily frame** per entity (:data:`DAILY_COLUMNS`): the daily log
return, raw share volume, raw dollar volume, a cumulative share-adjustment
factor, whether the day is a tradable row, and the post-delisting fill; for
the Q26 features also (:data:`OPTIONAL_DAILY_COLUMNS`) the return without
distributions, the day's open/high/low/close and closing bid/ask, the
company's market equity, the GICS sector, the report dates and the
point-in-time fundamental inputs. The CRSP layer (:mod:`nec_moe.crsp`, with
:mod:`nec_moe.compustat`) builds these frames from ``StkDlySecurityData``,
``StkDlyCumulativeAdjFactor`` and the CRSP/Compustat Merged files; nothing
here knows the source.

Timing convention (the anti-leakage rule, stated once and enforced by test)
---------------------------------------------------------------------------
A row (date t, entity i) represents a prediction made **at the close of t**:

- every feature at t is computed from information through the close of t
  (trailing rolling windows over past daily returns and volumes; pandas
  rolling windows are trailing by construction). Price levels are never an
  input: the drawdown compounds returns inside its own trailing window, and
  share volumes are put on one share basis with split factors dated inside
  the window (brief 06 A.4.4 and A.4.6);
- the target is the *forward* return — the only column allowed to touch the
  future. Three kinds (``StageBSpec.target_kind``):
  ``"market_neutral"`` (the default; decision Q25): ``fwd_mn_ret_{h}d`` =
  ``fwd_ret_{h}d`` minus the equal-weighted mean of ``fwd_ret_{h}d`` over the
  date's universe rows with a valid target. The mean is cross-sectional, so
  it is taken in :func:`assemble_panel`, never per stock; it transforms the
  label only, never a feature;
  ``"raw"``: ``fwd_ret_{h}d`` = the sum of the next ``h`` daily log returns;
  ``"residual"``: ``fwd_resid_ret_{h}d = fwd_ret − β_t · mkt_fwd_ret``, a
  trailing-beta (CAPM-style) residual. ``β_t`` is a **trailing** rolling OLS
  beta (:func:`rolling_beta`, window ``beta_window``) — estimated from past
  data only, so the residualization itself introduces no lookahead. It is
  *not* the Q25 decision and is kept as an option;
- cross-sectional rank normalization uses only date-t's own cross-section;
- fundamentals enter only once available (one trading day after the later
  of the report date and the filing), carried forward at most a year
  (:mod:`nec_moe.compustat`).

``test_no_lookahead`` (here, in ``test_crsp.py`` and, for every Q26
characteristic and the fundamentals, in ``test_q26_features.py``) verifies
this mechanically: features at t computed from data truncated at t equal
those computed from all of it.

Feature sets (``StageBSpec.features``, a :class:`FeatureSpec`)
------------------------------------------------------------
``"q26"`` (the default; decision Q26, brief 08 D): 57 inputs
(:data:`nec_moe.characteristics.Q26_FEATURES`), every window a
:class:`FeatureSpec` field:

- 26 JKP characteristics, two per theme of Jensen, Kelly & Pedersen (2023):
  ``ret_5d``, ``ret_20d`` (short-term reversal); ``mom_12_1``,
  ``prc_highprc_252d`` (momentum); ``rvol_21d``, ``beta_bab`` (low risk);
  ``log_me``, ``ami_126d`` (size); ``seas_2_5an``, ``coskew_21d``
  (seasonality); ``be_me``, ``ni_me`` (value); ``niq_be``, ``ocf_at``
  (profitability); ``gp_at``, ``ni_inc8q`` (quality); ``at_gr1``,
  ``sale_gr1`` (investment); ``oaccruals_at``, ``taccruals_at`` (accruals);
  ``debt_gr3``, ``noa_at`` (debt issuance); ``netdebt_me``, ``cash_at``
  (low leverage); ``niq_su``, ``saleq_su`` (profit growth);
- 12 short-horizon market characteristics: ``ivol_capm_21d``,
  ``vol_shock``; ``rskew_21d``, ``rmax1_21d``; ``volume_z``,
  ``qspread_21d``; ``overnight_20d``, ``ma50_gap``; ``earn_next5``,
  ``days_since_earn``; ``var_ratio_60d``, ``ret_vol_corr_60d``;
- the industry block: 11 GICS sector dummies, ``ind_mom_12_1`` and
  ``ret_20d_ind_rel`` (value-weighted by default), and within-sector ranks
  of ``be_me``, ``ni_me``, ``niq_be``, ``ocf_at``;
- ``flag_price_missing`` and ``flag_fund_missing``.

Each characteristic is ranked per date over the date's universe rows that
have it (ties averaged) and mapped to [-1, 1]; a missing value is then 0 (the
median) and sets its family's flag; rows are no longer dropped for a missing
characteristic. Definitions: :mod:`nec_moe.characteristics`.

``"legacy14"`` (kept for comparison and the older tests): returns
1/5/20/60d, momentum 120d, realized vol 5/20/60d, downside vol 20d, drawdown
vs 60d high, log dollar volume 20d, volume z-score 20d, market-relative 20d
return and vol ratio, ranked to [-0.5, 0.5]; a row with any of them missing
is dropped.

Sequence channels (both sets; trailing ``seq_len`` window per row, natural
units): daily log return, market-relative daily return, trailing 20d vol,
volume z-score, the observable regime signals the gate's encoder reads.

The result is a plain :class:`~nec_moe.data.Panel` — everything downstream
(Trainer, walk-forward harness, backtest) works unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .characteristics import (
    FUNDAMENTAL_INPUTS,
    JKP_FEATURES,
    MARKET_FEATURES,
    Q26_FEATURES,
    cross_section,
    stock_characteristics,
)
from .config import DataConfig
from .data import FeatureSchema, Panel

__all__ = [
    "DAILY_COLUMNS",
    "OPTIONAL_DAILY_COLUMNS",
    "TARGET_KINDS",
    "CALENDAR_DAYS_PER_YEAR",
    "FeatureSpec",
    "price_history_trading_days",
    "fundamental_history_quarters",
    "SEQUENCE_FEATURES",
    "SNAPSHOT_FEATURES",
    "StageBSpec",
    "rolling_beta",
    "market_frame",
    "stock_features",
    "post_fill_used",
    "forward_daily_returns",
    "feature_warmup",
    "assemble_panel",
    "check_panel_input_mode",
    "data_config_from_panel",
]

#: Columns of the per-entity daily frame the feature layer consumes.
#:
#: - ``ret``: daily log return; NaN where the return is missing (never 0);
#: - ``volume``: raw share volume of the day;
#: - ``dollar_volume``: raw price times raw share volume of the same day;
#: - ``share_factor``: cumulative factor that puts share counts on one basis
#:   (CRSP ``DlyCumFacShr``); only its ratios inside a trailing window are used;
#: - ``tradable``: the day is an ordinary trading row, so a panel row may sit
#:   on it (False on a delisting-return row and on post-delisting fill days);
#: - ``fill_ret``: the post-delisting return on the days after an entity's
#:   final return, NaN everywhere else. It only completes forward windows.
DAILY_COLUMNS: tuple[str, ...] = (
    "ret",
    "volume",
    "dollar_volume",
    "share_factor",
    "tradable",
    "fill_ret",
)

#: Optional daily-frame columns the Q26 characteristics read (a column a
#: frame lacks leaves its characteristics missing, never zero):
#:
#: - ``retx``: the simple return without distributions (``DlyRetx``), whose
#:   cumulative product is the price index ``P``;
#: - ``open``, ``high``, ``low``, ``close``, ``bid``, ``ask``: the day's
#:   prices and closing quotes (``StkDlySecurityData``);
#: - ``cap``: the PERMNO's ``DlyCap``; ``me``: its company's market equity in
#:   $ millions (every share class), on the day;
#: - ``sector``: the GICS sector code in force that day (NaN without one);
#: - ``rdq_date``: a report date, on the first trading day on or after it;
#: - the point-in-time fundamental inputs
#:   (:data:`nec_moe.characteristics.FUNDAMENTAL_INPUTS`).
OPTIONAL_DAILY_COLUMNS: tuple[str, ...] = (
    "retx", "open", "high", "low", "close", "bid", "ask", "cap", "me", "sector",
    "rdq_date", *FUNDAMENTAL_INPUTS,
)

SEQUENCE_FEATURES: tuple[str, ...] = (
    "ret_1d",
    "rel_ret_1d",
    "vol_20d",
    "volume_z_20d",
)

SNAPSHOT_FEATURES: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "mom_120d",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "downside_vol_20d",
    "drawdown_60d",
    "dollar_vol_20d",
    "volume_z_20d",
    "rel_ret_20d",
    "rel_vol_20d",
)


#: Calendar days per year, for turning trading-day windows into calendar
#: spans (a property of the calendar, not a modelling choice).
CALENDAR_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class FeatureSpec:
    """Every window, threshold and cap of the Q26 feature set (brief 08 D).

    ``feature_set="q26"`` is the decision (Q26, 2026-10-05): 26 JKP
    characteristics, 12 short-horizon market characteristics, 17 industry
    inputs and 2 missing-value flags, 57 inputs. ``"legacy14"`` is the
    earlier 14-feature price/volume set, kept for comparison and the older
    tests only. Windows are in trading days and trailing through the close
    of t; fundamentals are point in time (:mod:`nec_moe.compustat`).

    Defaults are the brief's definitions; the field comments name the
    characteristic each one belongs to. ``trading_days_per_year`` turns the
    year-based windows (seasonality, and the extract's lookback) into trading
    days; a month is a twelfth of it.
    """

    feature_set: Literal["q26", "legacy14"] = "q26"
    # D.4: a rolling window needs at least this share of its days, rounded up
    min_obs_frac: float = 0.8
    trading_days_per_year: int = 252
    # ---- D.1, CRSP part of the JKP block
    ret_5d_window: int = 5  # ret_5d
    ret_20d_window: int = 20  # ret_20d
    mom_start_lag: int = 251  # mom_12_1: sum of r from t-251 ...
    mom_end_lag: int = 21  # ... to t-21 (the last 21 days skipped)
    high_window: int = 252  # prc_highprc_252d
    rvol_window: int = 21  # rvol_21d
    beta_corr_window: int = 1260  # beta_bab: correlation window ...
    beta_corr_min_obs: int = 750  # ... and its minimum count
    beta_overlap_days: int = 3  # ... of overlapping 3-day summed returns
    beta_vol_window: int = 252  # ... times std(r_i, 252) / std(m, 252)
    amihud_window: int = 126  # ami_126d
    seas_first_year: int = 2  # seas_2_5an: years 2 ...
    seas_last_year: int = 5  # ... to 5 back
    coskew_window: int = 21  # coskew_21d
    # ---- D.2, the short-horizon market block
    ivol_window: int = 21  # ivol_capm_21d
    vol_shock_short: int = 5  # vol_shock = std(r, 5) ...
    vol_shock_long: int = 60  # ... / std(r, 60)
    rskew_window: int = 21  # rskew_21d
    rmax_window: int = 21  # rmax1_21d
    volume_z_window: int = 20  # volume_z (the split-invariant z-score)
    spread_window: int = 21  # qspread_21d
    overnight_window: int = 20  # overnight_20d
    ma_window: int = 50  # ma50_gap
    earn_ahead_days: int = 5  # earn_next5: announcement within 5 trading days
    earn_cycle_days: int = 364  # ... expected at a past RDQ + 364 calendar days
    var_ratio_window: int = 60  # var_ratio_60d
    var_ratio_q: int = 5  # ... of 5-day against 1-day variance
    ret_vol_corr_window: int = 60  # ret_vol_corr_60d
    # ---- D.3, the industry block
    industry_weighting: Literal["value", "equal"] = "value"
    min_sector_names: int = 5  # within-sector rank falls back below this
    # ---- fundamentals (C.4, D.1)
    fundamental_max_staleness_days: int = 365  # calendar days after availability
    trailing_quarters: int = 4  # trailing-4-quarter flows
    growth_lag_quarters: int = 4  # at_gr1, sale_gr1, taccruals_at, noa_at, ni_inc8q
    debt_growth_lag_quarters: int = 12  # debt_gr3
    surprise_std_quarters: int = 8  # niq_su / saleq_su: std over 8 quarters ...
    surprise_min_quarters: int = 6  # ... with at least 6
    earnings_streak_max: int = 8  # ni_inc8q

    def validate(self) -> FeatureSpec:
        if self.feature_set not in ("q26", "legacy14"):
            raise ValueError(f"unknown feature_set {self.feature_set!r}; use 'q26' or 'legacy14'")
        if not 0.0 < self.min_obs_frac <= 1.0:
            raise ValueError(f"min_obs_frac must be in (0, 1], got {self.min_obs_frac}")
        if self.industry_weighting not in ("value", "equal"):
            raise ValueError(f"unknown industry_weighting {self.industry_weighting!r}")
        windows = {
            k: v for k, v in dataclasses.asdict(self).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        bad = {k: v for k, v in windows.items() if v < 1}
        if bad:
            raise ValueError(f"FeatureSpec windows must be >= 1, got {bad}")
        if self.mom_end_lag >= self.mom_start_lag:
            raise ValueError("mom_end_lag must be smaller than mom_start_lag")
        if self.seas_first_year > self.seas_last_year:
            raise ValueError("seas_first_year must be <= seas_last_year")
        if self.beta_corr_min_obs > self.beta_corr_window:
            raise ValueError("beta_corr_min_obs must be <= beta_corr_window")
        if self.surprise_min_quarters > self.surprise_std_quarters:
            raise ValueError("surprise_min_quarters must be <= surprise_std_quarters")
        return self

    @property
    def trading_days_per_month(self) -> int:
        return self.trading_days_per_year // 12

    def min_obs(self, window: int) -> int:
        """Observations a ``window``-day statistic needs: ``min_obs_frac`` of
        the window, rounded up (D.4)."""
        return max(1, math.ceil(self.min_obs_frac * window - 1e-9))


def price_history_trading_days(fs: FeatureSpec) -> int:
    """Trading days of CRSP history the longest Q26 price window reaches back.

    The longest of: the beta correlation window over overlapping sums, the
    seasonality years (``seas_last_year`` years plus the month itself), the
    momentum lag, the 52-week high, and every shorter window. Feeds the
    extract's lookback (:attr:`nec_moe.crsp.CRSPSpec.lookback_days`); the
    training window itself does not move.
    """
    return max(
        fs.beta_corr_window + fs.beta_overlap_days - 1,
        fs.beta_vol_window,
        fs.seas_last_year * fs.trading_days_per_year + fs.trading_days_per_month,
        fs.mom_start_lag + 1,
        fs.high_window,
        fs.amihud_window,
        fs.ma_window,
        fs.vol_shock_long,
        fs.var_ratio_window,
        fs.ret_vol_corr_window,
    )


def fundamental_history_quarters(fs: FeatureSpec) -> int:
    """Fiscal quarters before the latest one the Q26 fundamentals reach back.

    The longest of: debt growth over ``debt_growth_lag_quarters``; the
    surprise standard deviation over ``surprise_std_quarters`` differences,
    each ``growth_lag_quarters`` long; the earnings streak's comparisons; and
    a trailing sum a growth lag earlier (sales growth). Feeds the Compustat
    extract's start (:func:`nec_moe.compustat.fundamentals_start`).
    """
    return max(
        fs.debt_growth_lag_quarters,
        fs.surprise_std_quarters - 1 + fs.growth_lag_quarters,
        fs.earnings_streak_max - 1 + fs.growth_lag_quarters,
        fs.trailing_quarters - 1 + fs.growth_lag_quarters,
    )


#: The target kinds :class:`StageBSpec` accepts; ``"market_neutral"`` is the
#: decision (Q25), the other two are kept as options.
TARGET_KINDS: tuple[str, ...] = ("market_neutral", "raw", "residual")


@dataclass(frozen=True)
class StageBSpec:
    """Knobs of the real-data panel build.

    ``target_kind`` (Q25, decided 2026-10-01; default ``"market_neutral"``):

    - ``"market_neutral"``: ``fwd_mn_ret_{h}d``, the raw forward return minus
      the date's equal-weighted mean of it over the universe rows with a
      valid target (:func:`assemble_panel`). It sums to zero on every date
      over those rows, and a return common to every stock cancels from it;
    - ``"raw"``: ``fwd_ret_{h}d``, the sum of the next ``h`` daily log
      returns, completed by the post-delisting fill;
    - ``"residual"``: ``fwd − β_t·mkt_fwd`` with ``β_t`` a trailing
      ``beta_window``-day rolling OLS beta. The beta warm-up consumes
      ``beta_window`` leading days per ticker — rows without a converged beta
      have a NaN target and are dropped by validity. Not the decision.
    """

    seq_len: int = 20
    horizon: int = 5  # forward-return target horizon in trading days
    cs_rank: bool = True  # cross-sectional rank-normalize snapshot features
    min_names_per_date: int = 5  # drop dates with too thin a cross-section
    target_kind: Literal["market_neutral", "raw", "residual"] = "market_neutral"
    beta_window: int = 250  # trailing window for the market beta (residual only)
    # the snapshot features (Q26): FeatureSpec.feature_set picks "q26" (the
    # decision, 57 inputs) or "legacy14"; every window is a field there
    features: FeatureSpec = field(default_factory=FeatureSpec)
    # the experts' input mode the panel is built for (brief 08 D.4.6): only
    # "snapshot_plus_hidden" drops a q26 row whose sequence window is
    # incomplete; must match ExpertConfig.input_mode (checked when training)
    input_mode: Literal["snapshot", "snapshot_plus_hidden"] = "snapshot"

    def validate(self) -> StageBSpec:
        if self.seq_len < 2 or self.horizon < 1 or self.min_names_per_date < 2:
            raise ValueError(
                f"invalid StageBSpec: seq_len={self.seq_len}, "
                f"horizon={self.horizon}, min_names={self.min_names_per_date}"
            )
        if self.target_kind not in TARGET_KINDS:
            raise ValueError(
                f"unknown target_kind {self.target_kind!r}; use one of {TARGET_KINDS}"
            )
        if self.target_kind == "residual" and self.beta_window < 20:
            raise ValueError(
                f"beta_window={self.beta_window} is too short to estimate a "
                "market beta (need >= 20 trailing days)"
            )
        if self.input_mode not in ("snapshot", "snapshot_plus_hidden"):
            raise ValueError(f"unknown input_mode {self.input_mode!r}")
        self.features.validate()
        return self

    @property
    def feature_set(self) -> str:
        return self.features.feature_set

    @property
    def snapshot_features(self) -> tuple[str, ...]:
        """The panel's snapshot columns: the 57 Q26 inputs or the legacy 14."""
        return Q26_FEATURES if self.feature_set == "q26" else SNAPSHOT_FEATURES

    @property
    def target(self) -> str:
        """The panel's target column: ``fwd_mn_ret_{h}d``, ``fwd_ret_{h}d`` or
        ``fwd_resid_ret_{h}d``; :func:`nec_moe.config.target_horizon` reads
        ``h`` back from every one of them."""
        prefix = {
            "market_neutral": "fwd_mn_ret", "raw": "fwd_ret", "residual": "fwd_resid_ret",
        }[self.target_kind]
        return f"{prefix}_{self.horizon}d"

    @property
    def raw_target(self) -> str:
        """The raw forward return's column, ``fwd_ret_{h}d``."""
        return f"fwd_ret_{self.horizon}d"

    @property
    def stock_target(self) -> str:
        """The target column :func:`stock_features` writes for one stock.

        The market-neutral target needs the whole date's cross-section, so a
        single stock's frame carries the raw forward return it is built from;
        the other kinds are complete per stock."""
        return self.raw_target if self.target_kind == "market_neutral" else self.target


# --------------------------------------------------------------------------- #
# Per-series features (all trailing; only the target looks forward)
# --------------------------------------------------------------------------- #


def rolling_beta(r: pd.Series, mkt_r: pd.Series, window: int) -> pd.Series:
    """Trailing rolling OLS beta of ``r`` on ``mkt_r``: cov/var over ``window``.

    Both inputs are daily returns aligned on the same index; pandas rolling
    windows are trailing by construction, so ``beta_t`` uses days ≤ t only —
    the residual target's no-lookahead property rests on exactly this.
    """
    return r.rolling(window).cov(mkt_r) / mkt_r.rolling(window).var()


#: The longest trailing window of the feature list (``mom_120d``) and the
#: window of the sequence channels that need one (``vol_20d``,
#: ``volume_z_20d``), in trading days. Properties of the named features, not
#: knobs: changing them renames a feature.
_LONGEST_SNAPSHOT_WINDOW = 120
_SEQUENCE_CHANNEL_WINDOW = 20


def feature_warmup(spec: StageBSpec) -> int:
    """Trading days of history the features need before the window's first date.

    The longest of: the snapshot windows (for ``q26``, the longest price
    window, :func:`price_history_trading_days`; rows are not dropped for a
    missing ``q26`` characteristic, but the extract must reach this far back
    for the early rows to have their values), the sequence window of
    channels that are themselves 20-day statistics, and the residual
    target's beta window.
    """
    snapshot = (
        price_history_trading_days(spec.features)
        if spec.feature_set == "q26" else _LONGEST_SNAPSHOT_WINDOW
    )
    need = max(snapshot, spec.seq_len - 1 + _SEQUENCE_CHANNEL_WINDOW)
    if spec.target_kind == "residual":
        need = max(need, spec.beta_window)
    return need


def _forward_sum(r: pd.Series, horizon: int) -> pd.Series:
    """Sum of the next ``horizon`` values, t+1..t+h; NaN if any is missing."""
    return r.rolling(horizon).sum().shift(-horizon)


def market_frame(mkt_ret: pd.Series, spec: StageBSpec | None = None) -> pd.DataFrame:
    """Market series needed for relative features: ret_1d/ret_20d/vol_20d.

    ``mkt_ret`` is the market's daily log return on its own calendar (on the
    CRSP panel, ``log(1 + DlyTotRet)`` of ``CRSPSpec.market_indno``). With
    ``spec`` given, also emits ``mkt_fwd_ret``, the market's forward
    ``horizon``-day log return on the **market's own calendar** (so an entity
    with missing days still gets the correctly aligned market move), used only
    inside the residual target. It is forward-looking by definition, exactly
    like the target it feeds; it is never a feature.
    """
    out = pd.DataFrame(
        {
            "mkt_ret_1d": mkt_ret,
            "mkt_ret_20d": mkt_ret.rolling(20).sum(),
            "mkt_vol_20d": mkt_ret.rolling(20).std(),
        }
    )
    if spec is not None:
        out["mkt_fwd_ret"] = _forward_sum(mkt_ret, spec.horizon)
    return out


def _window_drawdown(r: pd.Series, window: int) -> pd.Series:
    """Drawdown against the trailing ``window``-date high of a total-return index.

    The index is compounded from the daily log returns **inside the window
    only** (brief 06 A.4.4): the window's first date is the base, level 0 in
    logs, and the ``window - 1`` returns after it build the rest. The value at
    t therefore depends on returns through t alone, and a missing return
    anywhere in the window makes it missing rather than silently zero.
    """
    x = r.to_numpy(dtype=float)
    n = window - 1  # returns inside a window of `window` dates
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cum = np.cumsum(sliding_window_view(x, n), axis=1)
        peak = np.maximum(cum.max(axis=1), 0.0)  # 0 is the window's first date
        out[n - 1 :] = np.exp(cum[:, -1] - peak) - 1.0
    return pd.Series(out, index=r.index)


def _split_invariant_volume_z(
    volume: pd.Series, share_factor: pd.Series, window: int
) -> pd.Series:
    """Z-score of today's share volume against its trailing ``window``.

    Every volume in the window is put on today's share basis with the ratio of
    cumulative share factors, ``v_s * F_s / F_t`` (brief 06 A.4.6). The ratio
    depends only on splits dated inside ``(s, t]``, so no factor anchored to
    the end of the sample reaches the value, and a 2-for-1 split inside the
    window leaves the z-score unchanged. ``F`` is CRSP's ``DlyCumFacShr``:
    shares times ``F`` is continuous across a split.
    """
    v = volume.to_numpy(dtype=float)
    fac = share_factor.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= window:
        fw = sliding_window_view(fac, window)
        adj = sliding_window_view(v, window) * (fw / fw[:, -1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[window - 1 :] = (adj[:, -1] - adj.mean(axis=1)) / adj.std(axis=1, ddof=1)
    return pd.Series(out, index=volume.index)


def _target_returns(daily: pd.DataFrame) -> pd.Series:
    """The daily returns the forward target compounds.

    The entity's own returns (a delisting-return row included), completed
    after its final return by the post-delisting fill. ``fill_ret`` is NaN
    everywhere else, so a missing return inside the entity's life stays
    missing.
    """
    r = daily["ret"]
    if "fill_ret" not in daily.columns:
        return r
    return r.fillna(daily["fill_ret"])


def post_fill_used(daily: pd.DataFrame, spec: StageBSpec) -> pd.Series:
    """True where the row's forward window uses at least one post-delisting fill day."""
    if "fill_ret" not in daily.columns:
        return pd.Series(False, index=daily.index)
    filled = daily["fill_ret"].notna().astype(float)
    return _forward_sum(filled, spec.horizon).fillna(0.0) > 0


def forward_daily_returns(
    daily: pd.DataFrame, f: pd.DataFrame, spec: StageBSpec
) -> np.ndarray:
    """``(P, h)``: each row's ``h`` daily forward returns, the target's pieces.

    Column ``k`` is the return on the ``k+1``-th day after the row, from the
    same series as the target (the post-delisting fill included). For the
    residual target each day's market part is removed with the row's
    trailing beta, so the columns sum to the per-stock target. For the
    market-neutral target they are the raw daily returns; :func:`assemble_panel`
    removes each day's cross-sectional mean over the same rows as the
    target's, so on the panel they sum to the target for every kind. Forward
    by construction, like the target; never a feature.
    """
    tr = _target_returns(daily).to_numpy(dtype=float)
    n, h = len(tr), spec.horizon
    out = np.full((n, h), np.nan)
    for k in range(1, h + 1):
        out[: n - k, k - 1] = tr[k:]
    if spec.target_kind == "residual":
        beta = rolling_beta(f["ret_1d"], f["mkt_ret_1d"], spec.beta_window).to_numpy()
        m = f["mkt_ret_1d"].to_numpy(dtype=float)
        for k in range(1, h + 1):
            mk = np.full(n, np.nan)
            mk[: n - k] = m[k:]
            out[:, k - 1] -= beta * mk
    return out


def stock_features(
    daily: pd.DataFrame, mkt: pd.DataFrame, spec: StageBSpec
) -> pd.DataFrame:
    """All named feature columns + the forward-return target, for one entity.

    ``daily`` holds :data:`DAILY_COLUMNS` (``share_factor``, ``tradable`` and
    ``fill_ret`` are optional) on the entity's own dates; market columns are
    joined on date, and dates the market lacks end up NaN and are dropped by
    panel validity.

    Returns, drawdown and the target come from the daily log return; dollar
    volume is the raw price times raw volume of each day, the dollars actually
    traded (brief 06 A.4.5; audit finding D-2).
    """
    r = daily["ret"]
    v = daily["volume"]
    factor = (
        daily["share_factor"]
        if "share_factor" in daily.columns
        else pd.Series(1.0, index=daily.index)
    )
    f = pd.DataFrame(index=daily.index)
    f["ret_1d"] = r
    f["ret_5d"] = r.rolling(5).sum()
    f["ret_20d"] = r.rolling(20).sum()
    f["ret_60d"] = r.rolling(60).sum()
    f["mom_120d"] = r.rolling(120).sum()
    f["vol_5d"] = r.rolling(5).std()
    f["vol_20d"] = r.rolling(20).std()
    f["vol_60d"] = r.rolling(60).std()
    f["downside_vol_20d"] = r.clip(upper=0.0).rolling(20).std()
    f["drawdown_60d"] = _window_drawdown(r, 60)
    with np.errstate(divide="ignore"):
        f["dollar_vol_20d"] = np.log(daily["dollar_volume"].rolling(20).mean())
    f["volume_z_20d"] = _split_invariant_volume_z(v, factor, 20)

    f = f.join(mkt, how="left")
    f["rel_ret_1d"] = f["ret_1d"] - f["mkt_ret_1d"]
    f["rel_ret_20d"] = f["ret_20d"] - f["mkt_ret_20d"]
    f["rel_vol_20d"] = f["vol_20d"] / f["mkt_vol_20d"]
    if spec.feature_set == "q26":
        # the Q26 characteristics (ret_5d and ret_20d replace the legacy
        # full-window ones with their minimum-count versions), plus sector
        # and market equity for the cross-section
        ch = stock_characteristics(daily, f["mkt_ret_1d"], spec.features)
        for c in ch.columns:
            f[c] = ch[c]

    # the target — the ONLY forward-looking column(s). The market-neutral
    # target is the raw one here; assemble_panel demeans it per date
    fwd = _forward_sum(_target_returns(daily), spec.horizon)
    if spec.target_kind in ("raw", "market_neutral"):
        f[spec.stock_target] = fwd
    else:  # residual: trailing-beta residual forward return
        if "mkt_fwd_ret" not in f.columns:
            raise ValueError(
                "residual target needs the market's forward return: call "
                "market_frame(mkt_ret, spec) with the same spec"
            )
        beta = rolling_beta(f["ret_1d"], f["mkt_ret_1d"], spec.beta_window)
        f[spec.target] = fwd - beta * f["mkt_fwd_ret"]
    return f.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #


def _window_ok(f: pd.DataFrame, spec: StageBSpec) -> pd.Series:
    """Every sequence channel present on each of the row's last ``seq_len`` rows."""
    seq_ok_today = f[list(SEQUENCE_FEATURES)].notna().all(axis=1)
    return seq_ok_today.astype(float).rolling(spec.seq_len).sum() == spec.seq_len


def _valid_rows(
    f: pd.DataFrame, spec: StageBSpec, tradable: pd.Series | None = None
) -> pd.Series:
    """Rows usable as samples.

    ``legacy14``: a tradable day with every snapshot feature, the target and
    a full trailing sequence window. ``q26`` (D.4.6): a tradable day with
    the target; the sequence window is required only for
    ``input_mode="snapshot_plus_hidden"``, and a missing characteristic never
    drops a row (it is filled and flagged in the cross-section).
    """
    target_ok = f[spec.stock_target].notna()
    if spec.feature_set == "q26":
        ok = target_ok.copy()
        if spec.input_mode == "snapshot_plus_hidden":
            ok &= _window_ok(f, spec)
    else:
        snap_ok = f[list(SNAPSHOT_FEATURES)].notna().all(axis=1)
        ok = snap_ok & target_ok & _window_ok(f, spec)
    if tradable is not None:
        ok &= tradable.reindex(f.index, fill_value=False).astype(bool)
    return ok


class _MemberSource(Protocol):
    """Anything that says which entities were index members on a date
    (:class:`nec_moe.universe.Universe`; not imported, to keep the modules
    acyclic)."""

    def members_asof(self, date: str | pd.Timestamp) -> frozenset[str]: ...


def _target_eligible(
    f: pd.DataFrame,
    spec: StageBSpec,
    tradable: pd.Series | None,
    calendar: pd.DatetimeIndex,
    member: np.ndarray | None,
) -> np.ndarray:
    """On the calendar: a tradable row with a valid raw forward target, and a
    member of the universe that day when ``member`` is given.

    These are the rows the market-neutral target's per-date mean runs over
    (Q25): taken before any row is dropped for a missing characteristic or an
    incomplete sequence window, so one stock's target never depends on
    whether another stock's features are complete."""
    ok = f[spec.raw_target].reindex(calendar).notna().to_numpy().copy()
    if tradable is not None:
        ok &= tradable.reindex(calendar, fill_value=False).astype(bool).to_numpy()
    if member is not None:
        ok &= member
    return ok


def assemble_panel(
    daily: dict[str, pd.DataFrame],
    mkt: pd.DataFrame,
    spec: StageBSpec | None = None,
    *,
    data_source: str = "unspecified",
    metadata: dict[str, Any] | None = None,
    window: tuple[str, str] | None = None,
    universe: _MemberSource | None = None,
) -> Panel:
    """Assemble the contract-shaped :class:`Panel` from per-entity daily frames.

    ``mkt`` is :func:`market_frame` output for the same ``spec``; its index is
    the calendar the per-date name counts are taken on. Entity labels are the
    keys of ``daily`` (PERMNOs on the CRSP panel), sorted. ``window`` keeps
    only rows dated inside ``[start, end]``; history before it still feeds
    the features, and returns after it still feed the targets.

    **The market-neutral target** (``spec.target_kind="market_neutral"``,
    Q25). On each date t,
    ``y[i,t] = fwd_ret_h[i,t] - mean_j fwd_ret_h[j,t]``, equal-weighted over
    the date's rows with a valid raw target on a tradable day
    (:func:`_target_eligible`). With ``universe`` given, only that date's
    members count (the point-in-time universe; the caller then keeps only
    member rows, as :func:`nec_moe.crsp.build_crsp_panel` does with
    :func:`nec_moe.universe.filter_point_in_time`); without it, every entity
    in ``daily`` counts, so ``daily`` must then be the universe itself. The
    daily forward returns ``y_daily`` are demeaned the same way, day by day
    over the same rows, so they still sum to the target. The panel's
    metadata records ``target_kind`` and, for this kind,
    ``target_demeaned_over`` (``"universe"`` or ``"entities"``).

    **The Q26 inputs** (``spec.features.feature_set="q26"``, brief 08 D). Rows
    are the date's rows with a target on a tradable day (and, for
    ``input_mode="snapshot_plus_hidden"``, a complete sequence window); with
    ``universe`` given, only that date's members are rows at all, because
    every rank and sector aggregate is taken over the date's universe rows
    (:func:`nec_moe.characteristics.cross_section`), and the point-in-time
    filter afterwards finds nothing to drop. A row's sequence window that
    reaches before the stock's history, or holds a missing value, is
    zero-filled (only possible in ``"snapshot"`` mode, where the experts do
    not read it); the metadata counts such rows. The metadata also records
    ``feature_set``, ``input_mode`` and the aggregate coverage of the
    characteristics before the fill.
    """
    spec = (spec if spec is not None else StageBSpec()).validate()
    calendar = pd.DatetimeIndex(mkt.index)
    q26 = spec.feature_set == "q26"
    if q26 and not spec.cs_rank:
        raise ValueError("feature_set='q26' inputs are per-date ranks by definition: cs_rank=True")
    member_sets = (
        [universe.members_asof(d) for d in calendar] if universe is not None else None
    )

    def member_mask(t: str) -> np.ndarray | None:
        if member_sets is None:
            return None
        return np.fromiter((t in m for m in member_sets), dtype=bool, count=len(calendar))

    frames: dict[str, pd.DataFrame] = {}
    valid: dict[str, pd.Series] = {}
    for t, d in daily.items():
        f = stock_features(d, mkt, spec)
        frames[t] = f
        valid[t] = _valid_rows(f, spec, d["tradable"] if "tradable" in d.columns else None)
        if q26 and member_sets is not None:
            on = pd.Series(member_mask(t), index=calendar).reindex(f.index, fill_value=False)
            valid[t] &= on.astype(bool)

    # every entity's forward daily returns, aligned to its own rows
    fwd_daily_all = {t: forward_daily_returns(daily[t], frames[t], spec) for t in frames}

    # market-neutral target: per-date equal-weighted means over eligible rows
    mean_y = mean_daily = None
    if spec.target_kind == "market_neutral":
        sum_y = np.zeros(len(calendar))
        sum_d = np.zeros((len(calendar), spec.horizon))
        cnt = np.zeros(len(calendar))
        for t, f in frames.items():
            member = member_mask(t)
            d = daily[t]
            ok = _target_eligible(
                f, spec, d["tradable"] if "tradable" in d.columns else None, calendar, member
            )
            raw = f[spec.raw_target].reindex(calendar).to_numpy(dtype=float)
            fd = pd.DataFrame(fwd_daily_all[t], index=f.index).reindex(calendar).to_numpy()
            sum_y[ok] += raw[ok]
            sum_d[ok] += fd[ok]
            cnt[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_y = pd.Series(sum_y / cnt, index=calendar)
            mean_daily = pd.DataFrame(sum_d / cnt[:, None], index=calendar)

    # dates with a thick enough valid cross-section
    counts: pd.Series = sum(
        (v.reindex(mkt.index, fill_value=False).astype(int) for v in valid.values()),
        start=pd.Series(0, index=mkt.index),
    )
    kept_dates = counts.index[counts >= spec.min_names_per_date]
    if window is not None:
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        kept_dates = kept_dates[(kept_dates >= lo) & (kept_dates <= hi)]
    if len(kept_dates) < 10:
        raise ValueError(
            f"only {len(kept_dates)} dates have >= {spec.min_names_per_date} "
            "valid names — widen the date range or universe"
        )

    tickers = sorted(frames)
    date_codes = {d: i for i, d in enumerate(kept_dates)}

    rows_snap, rows_seq, rows_y, rows_date, rows_ent = [], [], [], [], []
    rows_daily: list[np.ndarray] = []
    rows_char: list[pd.DataFrame] = []
    char_cols = [*_Q26_RAW, "sector", "me"]
    seq_incomplete = 0
    for ent_code, t in enumerate(tickers):
        f = frames[t]
        keep = valid[t] & f.index.isin(kept_dates)
        if not keep.any():
            continue
        fwd_daily = fwd_daily_all[t][keep.to_numpy()]
        y_rows = f.loc[keep, spec.stock_target].to_numpy(dtype=float)
        if mean_y is not None and mean_daily is not None:
            days = f.index[keep]
            y_rows = y_rows - mean_y.reindex(days).to_numpy()
            fwd_daily = fwd_daily - mean_daily.reindex(days).to_numpy()
        rows_daily.append(fwd_daily.astype(np.float32))
        # trailing windows over the entity's own rows (positions, not calendar)
        seq_mat = f[list(SEQUENCE_FEATURES)].to_numpy(dtype=np.float32)
        pos = np.flatnonzero(keep.to_numpy())
        if q26:
            # rows are kept without a full window (D.4.6): pad the front, and
            # zero-fill what is missing (never read by snapshot-mode experts)
            pad = np.full((spec.seq_len - 1, seq_mat.shape[1]), np.nan, dtype=np.float32)
            windows = sliding_window_view(np.concatenate([pad, seq_mat]), spec.seq_len, axis=0)
            seq = np.swapaxes(windows[pos], 1, 2)
            seq_incomplete += int(np.isnan(seq).any(axis=(1, 2)).sum())
            rows_seq.append(np.nan_to_num(seq, nan=0.0))
            chars = f.loc[keep, char_cols].copy()
            chars["date"] = [date_codes[d] for d in f.index[keep]]
            chars["year"] = pd.DatetimeIndex(f.index[keep]).year
            rows_char.append(chars)
        else:
            windows = sliding_window_view(seq_mat, spec.seq_len, axis=0)  # (P, d, T)
            win_idx = pos - (spec.seq_len - 1)
            assert (win_idx >= 0).all()  # guaranteed by _valid_rows' window check
            rows_seq.append(np.swapaxes(windows[win_idx], 1, 2))  # (n, T, d_seq)
            rows_snap.append(f.loc[keep, list(SNAPSHOT_FEATURES)].to_numpy(np.float32))
        rows_y.append(y_rows.astype(np.float32))
        rows_date.append(np.array([date_codes[d] for d in f.index[keep]]))
        rows_ent.append(np.full(int(keep.sum()), ent_code))

    x_seq = torch.from_numpy(np.concatenate(rows_seq))
    y = torch.from_numpy(np.concatenate(rows_y))
    y_daily = torch.from_numpy(np.concatenate(rows_daily))
    date = torch.from_numpy(np.concatenate(rows_date)).long()
    entity = torch.from_numpy(np.concatenate(rows_ent)).long()

    feature_report: dict[str, Any] = {}
    if q26:
        rows = pd.concat(rows_char, ignore_index=True)
        inputs, feature_report = cross_section(rows, spec.features)
        x_snap = torch.from_numpy(inputs.to_numpy(dtype=np.float32))
    else:
        x_snap = torch.from_numpy(np.concatenate(rows_snap))
        if spec.cs_rank:
            x_snap = _cross_sectional_rank(x_snap, date)

    order = torch.argsort(date * (entity.max() + 1) + entity)  # date-major
    schema = FeatureSchema(
        sequence_features=SEQUENCE_FEATURES,
        snapshot_features=spec.snapshot_features,
        target=spec.target,
        rank_normalized=spec.cs_rank,
    )
    meta = dict(metadata or {})
    meta["target_kind"] = spec.target_kind
    meta["feature_set"] = spec.feature_set
    meta["input_mode"] = spec.input_mode
    if q26:
        meta["features"] = {**feature_report, "seq_window_incomplete_rows": seq_incomplete,
                            "rows_restricted_to_universe": universe is not None}
    if spec.target_kind == "market_neutral":
        meta["target_demeaned_over"] = "universe" if universe is not None else "entities"
    return Panel(
        x_seq=x_seq[order],
        x_snap=x_snap[order],
        y=y[order],
        date=date[order],
        entity=entity[order],
        schema=schema,
        date_labels=tuple(str(d.date()) for d in kept_dates),
        entity_labels=tuple(tickers),
        data_source=data_source,
        metadata=meta,
        y_daily=y_daily[order],
    )


def _cross_sectional_rank(x_snap: torch.Tensor, date: torch.Tensor) -> torch.Tensor:
    """Per-date percentile rank of each snapshot column, centered to [-0.5, 0.5].

    Uses only the rows it is given for each date. That is point-in-time safe
    only if those rows are the date's actual universe: ranks computed over a
    candidate list that includes names outside the index that day depend on
    those names (audit finding D-1). ``filter_point_in_time`` therefore
    re-ranks after filtering a rank-normalized panel.
    """
    out = torch.empty_like(x_snap)
    # one stable sort by date, then each date is a contiguous block: O(N log N)
    # rather than one full scan of the rows per date
    order = torch.argsort(date, stable=True)
    _, counts = torch.unique_consecutive(date[order], return_counts=True)
    start = 0
    for n in counts.tolist():
        rows = order[start : start + n]
        ranks = x_snap[rows].argsort(dim=0).argsort(dim=0).float()
        out[rows] = ranks / max(n - 1, 1) - 0.5
        start += n
    return out


#: The per-stock columns :func:`nec_moe.characteristics.cross_section` reads:
#: the 40 characteristics except the two industry signals it computes itself.
_Q26_RAW: tuple[str, ...] = (*JKP_FEATURES, *MARKET_FEATURES)


def check_panel_input_mode(panel: Panel, input_mode: str) -> None:
    """Refuse a Q26 panel built for another expert input mode (D.4.6).

    A panel built for ``"snapshot"`` keeps rows whose sequence window is
    incomplete (zero-filled); experts in ``"snapshot_plus_hidden"`` mode read
    that window, so they must be trained on a panel that dropped those rows.
    """
    meta = panel.metadata or {}
    if meta.get("feature_set") == "q26" and meta.get("input_mode") != input_mode:
        raise ValueError(
            f"this panel was built for input_mode={meta.get('input_mode')!r}, the "
            f"experts use {input_mode!r}: rebuild it with StageBSpec(input_mode="
            f"{input_mode!r}) (brief 08 D.4.6)"
        )


def data_config_from_panel(panel: Panel) -> DataConfig:
    """The model-side DataConfig implied by a built panel's schema/shapes."""
    return DataConfig(
        d_seq=len(panel.schema.sequence_features),
        d_snap=len(panel.schema.snapshot_features),
        seq_len=int(panel.x_seq.shape[1]),
        sequence_features=panel.schema.sequence_features,
        snapshot_features=panel.schema.snapshot_features,
        target=panel.schema.target,
    )
