"""Stage B feature engineering: prices -> the Panel contract (design doc §6).

Timing convention (the anti-leakage rule, stated once and enforced by test)
---------------------------------------------------------------------------
A row (date t, ticker i) represents a prediction made **at the close of t**:

- every feature at t is computed from information through the close of t
  (trailing rolling windows, diffs of past closes — pandas rolling/diff are
  trailing by construction);
- the target is the *forward* return — the only column allowed to touch the
  future. Two kinds (``StageBSpec.target_kind``):
  ``"raw"``: ``fwd_ret_{h}d = log(C_{t+h} / C_t)``;
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
from typing import Literal

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .config import DataConfig
from .data import FeatureSchema, Panel
from .market_data import DEFAULT_UNIVERSE, MARKET_SYMBOL, load_universe

__all__ = [
    "SEQUENCE_FEATURES",
    "SNAPSHOT_FEATURES",
    "StageBSpec",
    "rolling_beta",
    "market_features",
    "ticker_features",
    "build_panel",
    "build_stage_b_panel",
    "data_config_from_panel",
]

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

    def validate(self) -> "StageBSpec":
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


def market_features(
    market_px: pd.DataFrame, spec: StageBSpec | None = None
) -> pd.DataFrame:
    """Market-symbol series needed for relative features: ret_1d/ret_20d/vol_20d.

    With ``spec`` given, also emits ``mkt_fwd_ret`` — the market's forward
    ``horizon``-day log return, computed on the **market's own calendar** (so a
    ticker with missing days still gets the correctly aligned market move) and
    used only inside the residual target. It is forward-looking by definition,
    exactly like the target it feeds; it is never a feature.
    """
    logc = np.log(market_px["close"])
    r1 = logc.diff()
    out = pd.DataFrame(
        {
            "mkt_ret_1d": r1,
            "mkt_ret_20d": logc.diff(20),
            "mkt_vol_20d": r1.rolling(20).std(),
        }
    )
    if spec is not None:
        out["mkt_fwd_ret"] = logc.shift(-spec.horizon) - logc
    return out


def ticker_features(
    px: pd.DataFrame, mkt: pd.DataFrame, spec: StageBSpec
) -> pd.DataFrame:
    """All named feature columns + the forward-return target, for one ticker.

    Indexed by the ticker's own trading dates (market columns joined on date;
    dates the market lacks end up NaN and are dropped by panel validity).
    """
    c, v = px["close"], px["volume"]
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=px.index)
    f["ret_1d"] = r1
    f["ret_5d"] = logc.diff(5)
    f["ret_20d"] = logc.diff(20)
    f["ret_60d"] = logc.diff(60)
    f["mom_120d"] = logc.diff(120)
    f["vol_5d"] = r1.rolling(5).std()
    f["vol_20d"] = r1.rolling(20).std()
    f["vol_60d"] = r1.rolling(60).std()
    f["downside_vol_20d"] = r1.clip(upper=0.0).rolling(20).std()
    f["drawdown_60d"] = c / c.rolling(60).max() - 1.0
    f["dollar_vol_20d"] = np.log((c * v).rolling(20).mean())
    vol_roll = v.rolling(20)
    f["volume_z_20d"] = (v - vol_roll.mean()) / vol_roll.std()

    f = f.join(mkt, how="left")
    f["rel_ret_1d"] = f["ret_1d"] - f["mkt_ret_1d"]
    f["rel_ret_20d"] = f["ret_20d"] - f["mkt_ret_20d"]
    f["rel_vol_20d"] = f["vol_20d"] / f["mkt_vol_20d"]

    # the target — the ONLY forward-looking column(s)
    fwd = logc.shift(-spec.horizon) - logc
    if spec.target_kind == "raw":
        f[spec.target] = fwd
    else:  # residual: market-neutralized forward return
        if "mkt_fwd_ret" not in f.columns:
            raise ValueError(
                "residual target needs the market's forward return: call "
                "market_features(market_px, spec) with the same spec"
            )
        beta = rolling_beta(f["ret_1d"], f["mkt_ret_1d"], spec.beta_window)
        f[spec.target] = fwd - beta * f["mkt_fwd_ret"]
    return f.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #


def _valid_rows(f: pd.DataFrame, spec: StageBSpec) -> pd.Series:
    """Rows usable as samples: snapshot + target + full trailing seq window."""
    snap_ok = f[list(SNAPSHOT_FEATURES)].notna().all(axis=1)
    target_ok = f[spec.target].notna()
    seq_ok_today = f[list(SEQUENCE_FEATURES)].notna().all(axis=1)
    window_ok = (
        seq_ok_today.astype(float).rolling(spec.seq_len).sum() == spec.seq_len
    )
    return snap_ok & target_ok & window_ok


def build_panel(
    prices: dict[str, pd.DataFrame],
    market_px: pd.DataFrame,
    spec: StageBSpec = StageBSpec(),
) -> Panel:
    """Assemble the contract-shaped :class:`Panel` from per-ticker OHLCV."""
    spec = spec.validate()
    mkt = market_features(market_px, spec)

    frames: dict[str, pd.DataFrame] = {}
    valid: dict[str, pd.Series] = {}
    for t, px in prices.items():
        f = ticker_features(px, mkt, spec)
        frames[t] = f
        valid[t] = _valid_rows(f, spec)

    # dates with a thick enough valid cross-section
    counts: pd.Series = sum(
        (v.reindex(mkt.index, fill_value=False).astype(int) for v in valid.values()),
        start=pd.Series(0, index=mkt.index),
    )
    kept_dates = counts.index[counts >= spec.min_names_per_date]
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
        # trailing windows over the ticker's own rows (positions, not calendar)
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
    )


def _cross_sectional_rank(x_snap: torch.Tensor, date: torch.Tensor) -> torch.Tensor:
    """Per-date percentile rank of each snapshot column, centered to [-0.5, 0.5].

    Uses only the date's own cross-section — point-in-time safe by
    construction, and the natural normalization for rank-IC prediction.
    """
    out = torch.empty_like(x_snap)
    for d in torch.unique(date):
        m = date == d
        block = x_snap[m]
        n = block.shape[0]
        ranks = block.argsort(dim=0).argsort(dim=0).float()
        out[m] = ranks / max(n - 1, 1) - 0.5
    return out


# --------------------------------------------------------------------------- #
# End-to-end convenience
# --------------------------------------------------------------------------- #


def build_stage_b_panel(
    start: str,
    end: str,
    cache_dir: str,
    *,
    tickers: tuple[str, ...] = DEFAULT_UNIVERSE,
    market_symbol: str = MARKET_SYMBOL,
    source: str = "stooq",
    spec: StageBSpec = StageBSpec(),
    refresh: bool = False,
) -> Panel:
    """Download (or read cached) daily data and build the Stage-B panel.

    ``source``: ``"stooq"`` (primary; may require the manual browser-download
    workflow — see :mod:`nec_moe.market_data`) or ``"yfinance"`` (the
    syllabus's prototyping fallback). Survivorship warning: the default
    universe is pipeline-verification grade, not thesis-claim grade.
    """
    prices, market_px = load_universe(
        tickers,
        start,
        end,
        cache_dir,
        market_symbol=market_symbol,
        source=source,  # type: ignore[arg-type]
        refresh=refresh,
    )
    return build_panel(prices, market_px, spec)


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
