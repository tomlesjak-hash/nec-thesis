"""
End-to-end demo:
data → features → labels → (LGBM and MLP-OP) → eval → backtest.

Run from the repo root:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time

# allow `python scripts/run_pipeline.py` from repo root without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from op_pipeline.config import PipelineConfig
from op_pipeline.data import generate_panel, split_by_date
from op_pipeline.train import (
    prepare_panel,
    train_tabular_model,
    train_sequence_model,
    score_panel,
    score_panel_sequence,
)
from op_pipeline.models import LGBMModel, MLPOPModel
from op_pipeline.evaluate import evaluate_predictions, daily_ic_series
from op_pipeline.backtest import run_decile_backtest


def add_realised_next(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Realised forward return at the model horizon, used by the backtest."""
    panel = panel.sort_values(["asset", "date"]).reset_index(drop=True).copy()
    log_close = np.log(panel["close"])
    panel["y_realised_1d"] = log_close.groupby(panel["asset"]).shift(-horizon) - log_close
    return panel


def print_metrics(label: str, m) -> None:
    print(
        f"  {label:12s}  n={m.n:6d}  rmse={m.rmse:.5f}  "
        f"IC={m.ic:+.4f}  rankIC={m.rank_ic:+.4f}  hit={m.hit_rate:.3f}"
    )


def main() -> int:
    cfg = PipelineConfig()
    t0 = time.time()

    print("=" * 70)
    print("OP pipeline — end-to-end demo")
    print("=" * 70)

    print("\n[1/5] Generating synthetic panel...")
    panel = generate_panel(cfg.data)
    panel = add_realised_next(panel, cfg.labels.horizon)
    print(
        f"  panel rows: {len(panel):,}, "
        f"dates: {panel['date'].nunique()}, "
        f"assets: {panel['asset'].nunique()}"
    )

    print("\n[2/5] Building features & labels...")
    feat_df, feature_cols = prepare_panel(cfg, panel)
    print(f"  feature cols: {len(feature_cols)}")
    print(f"  sample: {feature_cols[:6]} ... {feature_cols[-3:]}")

    # =============================================================
    # LGBM
    # =============================================================
    print("\n[3/5] Training LGBM (continuous target, incremental rounds)...")
    lgbm = LGBMModel(
        rounds=cfg.train.rounds,
        num_boost_round=200,
        early_stopping=30,
    )
    assets_lgbm = train_tabular_model(cfg, panel, lgbm, use_bucket_label=False)

    scored = score_panel(assets_lgbm, feat_df)
    # split for eval
    _, _, test_df = split_by_date(scored, cfg.train.train_frac, cfg.train.valid_frac)
    test_clean = test_df.dropna(subset=["y_pred", "y_ret"])
    m = evaluate_predictions(test_clean["y_ret"].to_numpy(), test_clean["y_pred"].to_numpy())
    print("  test-set metrics:")
    print_metrics("LGBM(cont)", m)

    # cross-sectional daily rank-IC
    daily_ic = daily_ic_series(test_clean, "y_ret", "y_pred")
    daily_ic = daily_ic.dropna()
    print(f"  cross-sectional rank-IC: mean={daily_ic.mean():+.4f}  "
          f"std={daily_ic.std():.4f}  IR={daily_ic.mean()/daily_ic.std():+.2f}")

    # backtest
    bt_input = test_df.merge(
        panel[["date", "asset", "y_realised_1d"]], on=["date", "asset"], how="left"
    )
    bt_lgbm = run_decile_backtest(bt_input, cfg=cfg.backtest)
    print(f"  backtest: days={bt_lgbm.summary['days']}  "
          f"annRet={bt_lgbm.summary['ann_return']:+.3%}  "
          f"annVol={bt_lgbm.summary['ann_vol']:.3%}  "
          f"Sharpe={bt_lgbm.summary['sharpe']:+.2f}  "
          f"avgTurn={bt_lgbm.summary['avg_daily_turnover']:.2f}")

    # =============================================================
    # MLP-OP
    # =============================================================
    print("\n[4/5] Training MLP-OP (PyTorch sequence model)...")
    mlp = MLPOPModel(
        in_features=len(feature_cols),
        seq_len=cfg.features.seq_len,
        window=10,
        hidden=64,
        epochs=15,
        batch_size=2048,
        early_stopping=4,
    )
    assets_mlp = train_sequence_model(cfg, panel, mlp)

    scored_mlp = score_panel_sequence(assets_mlp, feat_df, cfg.features.seq_len)
    _, _, test_df_mlp = split_by_date(scored_mlp, cfg.train.train_frac, cfg.train.valid_frac)
    test_clean_mlp = test_df_mlp.dropna(subset=["y_pred", "y_ret"])
    m_mlp = evaluate_predictions(
        test_clean_mlp["y_ret"].to_numpy(),
        test_clean_mlp["y_pred"].to_numpy(),
    )
    print("  test-set metrics:")
    print_metrics("MLP-OP", m_mlp)

    daily_ic_mlp = daily_ic_series(test_clean_mlp, "y_ret", "y_pred").dropna()
    print(f"  cross-sectional rank-IC: mean={daily_ic_mlp.mean():+.4f}  "
          f"std={daily_ic_mlp.std():.4f}  "
          f"IR={daily_ic_mlp.mean()/daily_ic_mlp.std():+.2f}")

    bt_input_mlp = test_df_mlp.merge(
        panel[["date", "asset", "y_realised_1d"]], on=["date", "asset"], how="left"
    )
    bt_mlp = run_decile_backtest(bt_input_mlp, cfg=cfg.backtest)
    print(f"  backtest: days={bt_mlp.summary['days']}  "
          f"annRet={bt_mlp.summary['ann_return']:+.3%}  "
          f"annVol={bt_mlp.summary['ann_vol']:.3%}  "
          f"Sharpe={bt_mlp.summary['sharpe']:+.2f}  "
          f"avgTurn={bt_mlp.summary['avg_daily_turnover']:.2f}")

    # =============================================================
    # Summary
    # =============================================================
    print("\n[5/5] Summary")
    print(f"  total runtime: {time.time() - t0:.1f}s")
    print(f"  LGBM    rank-IC IR: {daily_ic.mean()/daily_ic.std():+.2f}  "
          f"Sharpe: {bt_lgbm.summary['sharpe']:+.2f}")
    print(f"  MLP-OP  rank-IC IR: {daily_ic_mlp.mean()/daily_ic_mlp.std():+.2f}  "
          f"Sharpe: {bt_mlp.summary['sharpe']:+.2f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
