import numpy as np

from op_pipeline.config import FeatureConfig
from op_pipeline.features import build_features, feature_columns, build_sequence_tensor


def test_features_have_expected_columns(small_panel):
    cfg = FeatureConfig(windows=[5, 10])
    feats = build_features(small_panel, cfg)
    expected = {"ret_1d_mean_5", "ret_1d_std_10", "log_vol_mean_5", "ret_resid_5"}
    assert expected.issubset(feats.columns), \
        f"missing: {expected - set(feats.columns)}"


def test_no_inf_in_features(small_panel):
    feats = build_features(small_panel)
    cols = feature_columns(feats)
    arr = feats[cols].to_numpy()
    assert not np.isinf(arr).any(), "feature matrix contains inf"


def test_sequence_tensor_shape(small_panel):
    cfg = FeatureConfig(windows=[5, 10], seq_len=12)
    feats = build_features(small_panel, cfg).dropna(subset=feature_columns(build_features(small_panel, cfg)))
    cols = feature_columns(feats)
    X, row_idx, asset_idx = build_sequence_tensor(feats, cols, seq_len=cfg.seq_len)
    assert X.ndim == 3
    assert X.shape[1] == cfg.seq_len
    assert X.shape[2] == len(cols)
    assert len(row_idx) == X.shape[0] == len(asset_idx)
