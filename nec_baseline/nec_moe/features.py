"""Stage B feature engineering: daily returns and volume -> the Panel contract (design doc §6).

Input: one **daily frame** per entity (:data:`DAILY_COLUMNS`): the daily log
return, raw share volume, raw dollar volume, a cumulative share-adjustment
factor, whether the day is a tradable row, and the post-delisting fill. The
CRSP layer (:mod:`nec_moe.crsp`) builds these frames from ``DlyRet``,
``DlyPrc``, ``DlyVol`` and ``DlyCumFacShr``; nothing here knows the source.

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
  future. Two kinds (``StageBSpec.target_kind``):
  ``"raw"``: ``fwd_ret_{h}d`` = the sum of the next ``h`` daily log returns;
  ``"residual"``: ``fwd_resid_ret_{h}d = fwd_ret − β_t · mkt_fwd_ret`` — the
  market-neutralized target the syllabus prescribes so the model cannot score
  by just learning market direction. ``β_t`` is a **trailing** rolling OLS
  beta (:func:`rolling_beta`, window ``beta_window``) — estimated from past
  data only, so the residualization itself introduces no lookahead;
- cross-sectional rank normalization uses only date-t's own cross-section.

``test_no_lookahead`` verifies this mechanically: features at t computed from
a series truncated at t equal those computed from the full series.

Feature set (syllabus §3 "initial feature set", every column named):

- snapshot (cross-sectionally rank-normalized to [-0.5, 0.5] per date by
  default): returns 1/5/20/60d, momentum 120d, realized vol 5/20/60d,
  downside vol 20d, drawdown vs 60d high, log dollar volume 20d, volume
  z-score 20d, market-relative 20d return and vol ratio;
- sequence channels (trailing ``seq_len`` window per row, natural units):
  daily log return, market-relative daily return, trailing 20d vol, volume
  z-score — the observable regime signals the gate's encoder reads.

The result is a plain :class:`~nec_moe.data.Panel` — everything downstream
(Trainer, walk-forward harness, backtest) works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .config import DataConfig
from .data import FeatureSchema, Panel

__all__ = [
    "DAILY_COLUMNS",
    "SEQUENCE_FEATURES",
    "SNAPSHOT_FEATURES",
    "StageBSpec",
    "rolling_beta",
    "market_frame",
    "stock_features",
    "post_fill_used",
    "feature_warmup",
    "assemble_panel",
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


@dataclass(frozen=True)
class StageBSpec:
    """Knobs of the real-data panel build.

    ``target_kind="residual"`` switches the target to the market-neutralized
    forward return ``fwd − β_t·mkt_fwd`` with ``β_t`` a trailing
    ``beta_window``-day rolling OLS beta (syllabus: "later, use residual
    returns … this prevents the model from just learning market direction").
    The beta warm-up consumes ``beta_window`` leading days per ticker — rows
    without a converged beta have a NaN target and are dropped by validity.
    """

    seq_len: int = 20
    horizon: int = 5  # forward-return target horizon in trading days
    cs_rank: bool = True  # cross-sectional rank-normalize snapshot features
    min_names_per_date: int = 5  # drop dates with too thin a cross-section
    target_kind: Literal["raw", "residual"] = "raw"
    beta_window: int = 250  # trailing window for the market beta (residual only)

    def validate(self) -> StageBSpec:
        if self.seq_len < 2 or self.horizon < 1 or self.min_names_per_date < 2:
            raise ValueError(
                f"invalid StageBSpec: seq_len={self.seq_len}, "
                f"horizon={self.horizon}, min_names={self.min_names_per_date}"
            )
        if self.target_kind not in ("raw", "residual"):
            raise ValueError(f"unknown target_kind {self.target_kind!r}")
        if self.target_kind == "residual" and self.beta_window < 20:
            raise ValueError(
                f"beta_window={self.beta_window} is too short to estimate a "
                "market beta (need >= 20 trailing days)"
            )
        return self

    @property
    def target(self) -> str:
        prefix = "fwd_resid_ret" if self.target_kind == "residual" else "fwd_ret"
        return f"{prefix}_{self.horizon}d"


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
    """Trading days of history a row needs before its first valid date.

    The longest of: the snapshot windows, the sequence window of channels that
    are themselves 20-day statistics, and the residual target's beta window.
    """
    need = max(_LONGEST_SNAPSHOT_WINDOW, spec.seq_len - 1 + _SEQUENCE_CHANNEL_WINDOW)
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

    # the target — the ONLY forward-looking column(s)
    fwd = _forward_sum(_target_returns(daily), spec.horizon)
    if spec.target_kind == "raw":
        f[spec.target] = fwd
    else:  # residual: market-neutralized forward return
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


def _valid_rows(
    f: pd.DataFrame, spec: StageBSpec, tradable: pd.Series | None = None
) -> pd.Series:
    """Rows usable as samples: a tradable day with snapshot + target + a full
    trailing seq window."""
    snap_ok = f[list(SNAPSHOT_FEATURES)].notna().all(axis=1)
    target_ok = f[spec.target].notna()
    seq_ok_today = f[list(SEQUENCE_FEATURES)].notna().all(axis=1)
    window_ok = (
        seq_ok_today.astype(float).rolling(spec.seq_len).sum() == spec.seq_len
    )
    ok = snap_ok & target_ok & window_ok
    if tradable is not None:
        ok &= tradable.reindex(f.index, fill_value=False).astype(bool)
    return ok


def assemble_panel(
    daily: dict[str, pd.DataFrame],
    mkt: pd.DataFrame,
    spec: StageBSpec | None = None,
    *,
    data_source: str = "unspecified",
    metadata: dict[str, Any] | None = None,
    window: tuple[str, str] | None = None,
) -> Panel:
    """Assemble the contract-shaped :class:`Panel` from per-entity daily frames.

    ``mkt`` is :func:`market_frame` output for the same ``spec``; its index is
    the calendar the per-date name counts are taken on. Entity labels are the
    keys of ``daily`` (PERMNOs on the CRSP panel), sorted. ``window`` keeps
    only rows dated inside ``[start, end]``; history before it still feeds
    the features, and returns after it still feed the targets.
    """
    spec = (spec if spec is not None else StageBSpec()).validate()

    frames: dict[str, pd.DataFrame] = {}
    valid: dict[str, pd.Series] = {}
    for t, d in daily.items():
        f = stock_features(d, mkt, spec)
        frames[t] = f
        valid[t] = _valid_rows(f, spec, d["tradable"] if "tradable" in d.columns else None)

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
    for ent_code, t in enumerate(tickers):
        f = frames[t]
        keep = valid[t] & f.index.isin(kept_dates)
        if not keep.any():
            continue
        # trailing windows over the entity's own rows (positions, not calendar)
        seq_mat = f[list(SEQUENCE_FEATURES)].to_numpy(dtype=np.float32)
        windows = sliding_window_view(seq_mat, spec.seq_len, axis=0)  # (P, d, T)
        pos = np.flatnonzero(keep.to_numpy())
        win_idx = pos - (spec.seq_len - 1)
        assert (win_idx >= 0).all()  # guaranteed by _valid_rows' window check
        rows_seq.append(np.swapaxes(windows[win_idx], 1, 2))  # (n, T, d_seq)
        rows_snap.append(f.loc[keep, list(SNAPSHOT_FEATURES)].to_numpy(np.float32))
        rows_y.append(f.loc[keep, spec.target].to_numpy(np.float32))
        rows_date.append(np.array([date_codes[d] for d in f.index[keep]]))
        rows_ent.append(np.full(int(keep.sum()), ent_code))

    x_snap = torch.from_numpy(np.concatenate(rows_snap))
    x_seq = torch.from_numpy(np.concatenate(rows_seq))
    y = torch.from_numpy(np.concatenate(rows_y))
    date = torch.from_numpy(np.concatenate(rows_date)).long()
    entity = torch.from_numpy(np.concatenate(rows_ent)).long()

    if spec.cs_rank:
        x_snap = _cross_sectional_rank(x_snap, date)

    order = torch.argsort(date * (entity.max() + 1) + entity)  # date-major
    schema = FeatureSchema(
        sequence_features=SEQUENCE_FEATURES,
        snapshot_features=SNAPSHOT_FEATURES,
        target=spec.target,
        rank_normalized=spec.cs_rank,
    )
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
        metadata=dict(metadata or {}),
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
