import numpy as np
import pandas as pd

from op_pipeline.data import generate_panel, split_by_date
from op_pipeline.config import DataConfig


def test_panel_shape_and_columns():
    cfg = DataConfig(n_assets=10, n_days=120, seed=1)
    df = generate_panel(cfg)
    assert len(df) == cfg.n_assets * cfg.n_days
    for col in ("date", "asset", "open", "high", "low", "close", "volume", "f_planted"):
        assert col in df.columns
    # High >= max(open, close) and Low <= min(open, close)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    # close strictly positive
    assert (df["close"] > 0).all()


def test_determinism():
    a = generate_panel(DataConfig(n_assets=5, n_days=40, seed=42))
    b = generate_panel(DataConfig(n_assets=5, n_days=40, seed=42))
    pd.testing.assert_frame_equal(a, b)


def test_split_by_date_is_disjoint_and_ordered():
    df = generate_panel(DataConfig(n_assets=4, n_days=100, seed=2))
    tr, va, te = split_by_date(df, 0.6, 0.2)
    tr_d = set(tr["date"]); va_d = set(va["date"]); te_d = set(te["date"])
    assert tr_d.isdisjoint(va_d) and va_d.isdisjoint(te_d) and tr_d.isdisjoint(te_d)
    assert max(tr["date"]) < min(va["date"]) < max(va["date"]) < min(te["date"])


def test_planted_signal_predicts_next_day_return():
    """Sanity: the data has a *learnable* planted signal."""
    df = generate_panel(DataConfig(n_assets=20, n_days=300, seed=3))
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    log_close = np.log(df["close"])
    df["next_ret"] = log_close.groupby(df["asset"]).shift(-1) - log_close
    clean = df.dropna(subset=["next_ret", "f_planted"])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(clean["f_planted"], clean["next_ret"])
    # planted signal should be a modest positive predictor in expectation
    assert rho > 0.02, f"expected planted signal to be positively predictive, got {rho:.4f}"
