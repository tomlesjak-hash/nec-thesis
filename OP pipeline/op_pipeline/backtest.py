"""
Decile long-short backtest.

The simplest cross-sectional backtest that's not embarrassing:
- Each day, sort assets by predicted return.
- Long the top decile, short the bottom decile, equal-weight within deciles.
- Hold one day, then rebalance.
- Subtract flat transaction costs (bps per leg, applied to turnover).

What this gives you:
* daily PnL series
* cumulative equity curve
* annualised return / vol / Sharpe
* turnover and cost drag

What it does NOT give you:
* realistic execution (no slippage curve, no participation cap)
* borrow / financing costs
* capacity constraints
* sector or risk-factor neutralization

The OP pipeline did not have a backtest at all — predictions stopped at IC.
This is the first end-of-pipe sanity check that the predictions actually
translate to PnL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class BacktestResult:
    daily_pnl: pd.Series       # net daily PnL (decimal return units)
    equity_curve: pd.Series    # (1 + daily_pnl).cumprod()
    daily_turnover: pd.Series  # |Δw|_1, summed across assets, per day
    summary: dict


def _decile_weights(preds: pd.Series, n_q: int) -> pd.Series:
    """
    Compute target weights for one cross-section:
    +1/k for top decile, -1/k for bottom decile, 0 otherwise.
    Net-zero, gross = 2.
    """
    if len(preds) < n_q:
        return pd.Series(0.0, index=preds.index)
    ranks = preds.rank(method="first")
    n = len(preds)
    top_thr = n - n / n_q
    bot_thr = n / n_q
    w = pd.Series(0.0, index=preds.index)
    top_mask = ranks > top_thr
    bot_mask = ranks <= bot_thr
    n_top = int(top_mask.sum())
    n_bot = int(bot_mask.sum())
    if n_top > 0:
        w[top_mask] = 1.0 / n_top
    if n_bot > 0:
        w[bot_mask] = -1.0 / n_bot
    return w


def run_decile_backtest(
    panel: pd.DataFrame,
    pred_col: str = "y_pred",
    realised_col: str = "y_realised_1d",
    date_col: str = "date",
    asset_col: str = "asset",
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Parameters
    ----------
    panel : DataFrame with columns (date, asset, pred_col, realised_col).
        `realised_col` is the realised next-period return for the asset (one
        period = the forecast horizon used to train).

    Returns
    -------
    BacktestResult
    """
    cfg = cfg or BacktestConfig()
    df = panel.dropna(subset=[pred_col, realised_col]).copy()
    df = df.sort_values([date_col, asset_col])

    daily_pnl_list = []
    daily_turn_list = []
    prev_w: pd.Series | None = None

    for date, g in df.groupby(date_col):
        preds = g.set_index(asset_col)[pred_col]
        rets = g.set_index(asset_col)[realised_col]
        w = _decile_weights(preds, cfg.n_quantiles)

        # turnover: |Δw|_1 vs previous holdings (treat first day as full entry)
        if prev_w is None:
            turn = float(w.abs().sum())
        else:
            all_assets = w.index.union(prev_w.index)
            w_full = w.reindex(all_assets, fill_value=0.0)
            p_full = prev_w.reindex(all_assets, fill_value=0.0)
            turn = float((w_full - p_full).abs().sum())

        cost = turn * (cfg.cost_bps / 1e4)
        gross_pnl = float((w * rets).sum())
        net_pnl = gross_pnl - cost

        daily_pnl_list.append((date, net_pnl))
        daily_turn_list.append((date, turn))
        prev_w = w

    pnl = pd.Series(dict(daily_pnl_list), name="pnl").sort_index()
    turn = pd.Series(dict(daily_turn_list), name="turnover").sort_index()
    equity = (1.0 + pnl).cumprod()

    ann = cfg.annualization
    mean_ann = float(pnl.mean() * ann)
    vol_ann = float(pnl.std(ddof=1) * np.sqrt(ann)) if pnl.std(ddof=1) > 0 else float("nan")
    sharpe = mean_ann / vol_ann if vol_ann and not np.isnan(vol_ann) else float("nan")
    avg_turn = float(turn.mean())
    cost_drag_ann = float(avg_turn * (cfg.cost_bps / 1e4) * ann)

    summary = {
        "days": int(len(pnl)),
        "ann_return": mean_ann,
        "ann_vol": vol_ann,
        "sharpe": sharpe,
        "avg_daily_turnover": avg_turn,
        "ann_cost_drag": cost_drag_ann,
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
    }
    return BacktestResult(
        daily_pnl=pnl,
        equity_curve=equity,
        daily_turnover=turn,
        summary=summary,
    )
