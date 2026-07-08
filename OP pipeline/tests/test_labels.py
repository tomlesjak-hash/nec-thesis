import numpy as np
import pandas as pd

from op_pipeline.config import LabelConfig
from op_pipeline.labels import (
    add_forward_return,
    signed_bucket_labels,
    downsample_zero,
    balance_signed,
    build_balanced_sample,
)


def test_signed_bucket_labels_basic():
    s = pd.Series([0.0, 0.003, -0.003, 0.012, -0.05, 0.0003])
    out = signed_bucket_labels(s, thresholds=[0.005, 0.01, 0.02])
    expected = pd.Series([0.0, 0.0, 0.0, 2.0, -3.0, 0.0])
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_downsample_zero_respects_multiple():
    n_zero, n_nonzero = 1000, 200
    vals = [0] * n_zero + [1] * n_nonzero
    s = pd.Series(vals)
    kept = downsample_zero(s, multiple=2.0)
    # target = 2 * 200 = 400
    assert len(kept) == 400


def test_balance_signed_is_symmetric():
    vals = [1] * 50 + [-1] * 30 + [2] * 20 + [-2] * 80
    s = pd.Series(vals)
    kept = balance_signed(s)
    sub = s.loc[kept]
    # for |k|=1, min is 30 -> 30+30. For |k|=2, min is 20 -> 20+20. Total 100.
    assert (sub == 1).sum() == (sub == -1).sum() == 30
    assert (sub == 2).sum() == (sub == -2).sum() == 20
    assert len(sub) == 100


def test_build_balanced_sample_drops_extreme_labels(small_panel):
    cfg = LabelConfig(horizon=5, thresholds=[0.005, 0.01, 0.02])
    df = add_forward_return(small_panel.copy(), cfg.horizon)
    out = build_balanced_sample(df, cfg)
    # OP file drops |y_ret| > 0.2
    assert (out["y_ret"].abs() < 0.2).all()
    assert "y_bucket" in out.columns
