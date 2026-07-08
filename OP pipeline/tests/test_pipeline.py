"""End-to-end smoke test for the pipeline."""

import numpy as np
import pandas as pd
import pytest

from op_pipeline.data import generate_panel, split_by_date
from op_pipeline.train import (
    prepare_panel,
    train_tabular_model,
    score_panel,
)
from op_pipeline.evaluate import evaluate_predictions, daily_ic_series
from op_pipeline.backtest import run_decile_backtest


def test_end_to_end_lgbm(small_cfg):
    pytest.importorskip("lightgbm")
    from op_pipeline.models import LGBMModel

    panel = generate_panel(small_cfg.data)
    feat_df, cols = prepare_panel(small_cfg, panel)

    assets = train_tabular_model(
        small_cfg, panel,
        LGBMModel(rounds=1, num_boost_round=60, early_stopping=15),
        use_bucket_label=False,
    )
    assert len(assets.feature_cols) > 5
    scored = score_panel(assets, feat_df)
    _, _, test_df = split_by_date(scored, small_cfg.train.train_frac, small_cfg.train.valid_frac)
    test_clean = test_df.dropna(subset=["y_pred", "y_ret"])
    m = evaluate_predictions(test_clean["y_ret"].to_numpy(), test_clean["y_pred"].to_numpy())
    # synthetic data has a planted signal; pipeline should recover SOMETHING
    assert m.n > 100
    assert not np.isnan(m.rank_ic), "rank IC came back NaN"

    # backtest plumbing runs and produces a daily PnL series
    panel_with_real = panel.sort_values(["asset", "date"]).reset_index(drop=True).copy()
    log_close = np.log(panel_with_real["close"])
    panel_with_real["y_realised_1d"] = (
        log_close.groupby(panel_with_real["asset"]).shift(-small_cfg.labels.horizon)
        - log_close
    )
    bt_input = test_df.merge(
        panel_with_real[["date", "asset", "y_realised_1d"]],
        on=["date", "asset"], how="left",
    )
    bt = run_decile_backtest(bt_input, cfg=small_cfg.backtest)
    assert bt.summary["days"] > 0
    assert len(bt.daily_pnl) == bt.summary["days"]
