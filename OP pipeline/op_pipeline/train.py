"""
Glue: panel → feature matrix → labels → fitted model.

Two training entry points:

* `train_tabular_model(...)` — for any model that consumes a 2-D (N, F)
  feature matrix and a 1-D target. Used for LGBM.

* `train_sequence_model(...)` — for sequence models that need (N, T, F).
  Used for MLP-OP.

Both honor the OP-style cross-section + time discipline:
- features are built on the full panel (we let the test set's *features*
  be computed, since features depend only on past prices),
- labels with horizon=H mean rows in [test_start - H, test_start) leak
  forward returns into training; we drop those (the "purge" step in
  combinatorial cross-validation),
- the train/valid/test split is over unique dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig, LabelConfig
from .data import split_by_date
from .features import build_features, feature_columns, build_sequence_tensor
from .labels import add_forward_return, build_balanced_sample
from .models.base import BaseModel


@dataclass
class TrainedAssets:
    """What you need to keep around after training to score and backtest."""
    model: BaseModel
    feature_cols: list[str]
    horizon: int


def _purge_overlap(df: pd.DataFrame, train_end_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """
    Drop the last `horizon` trading days of the training window so labels
    don't overlap into the validation window. Standard purge step.
    """
    purge_cutoff = train_end_date - pd.tseries.offsets.BDay(horizon)
    return df[df["date"] <= purge_cutoff]


def prepare_panel(
    cfg: PipelineConfig,
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    panel -> panel-with-features-and-labels, plus feature column list.
    """
    feat_df = build_features(panel, cfg.features)
    feat_df = add_forward_return(feat_df, cfg.labels.horizon)
    cols = feature_columns(feat_df)
    return feat_df, cols


def train_tabular_model(
    cfg: PipelineConfig,
    panel: pd.DataFrame,
    model: BaseModel,
    use_bucket_label: bool = False,
) -> TrainedAssets:
    """
    panel -> features -> labels -> tabular train/valid -> fit(model).

    Parameters
    ----------
    use_bucket_label : if True, train on signed-bucket labels with OP-style
        balancing/downsampling; if False, train on continuous y_ret.

    Tabular here means features are flattened to (N, F).
    """
    feat_df, cols = prepare_panel(cfg, panel)
    train_df, valid_df, _ = split_by_date(
        feat_df, cfg.train.train_frac, cfg.train.valid_frac
    )
    # purge label overlap at the train/valid boundary
    if len(train_df) > 0 and len(valid_df) > 0:
        train_end = train_df["date"].max()
        train_df = _purge_overlap(train_df, train_end, cfg.labels.horizon)

    if use_bucket_label:
        train_proc = build_balanced_sample(train_df, cfg.labels)
        y_train = train_proc["y_bucket"].to_numpy()
        train_proc_X = train_proc
    else:
        train_proc = train_df.dropna(subset=["y_ret"] + cols).copy()
        y_train = train_proc["y_ret"].to_numpy()
        train_proc_X = train_proc

    X_train = train_proc_X[cols].to_numpy(dtype=np.float32)
    valid_clean = valid_df.dropna(subset=["y_ret"] + cols).copy()
    X_valid = valid_clean[cols].to_numpy(dtype=np.float32)
    y_valid = valid_clean["y_ret"].to_numpy()

    # if training on buckets, the validation target should still be continuous —
    # but we can't compare directly. Use bucket on valid too for honesty.
    if use_bucket_label:
        from .labels import signed_bucket_labels
        y_valid_for_fit = signed_bucket_labels(
            valid_clean["y_ret"], cfg.labels.thresholds
        ).to_numpy().astype(np.float32)
    else:
        y_valid_for_fit = y_valid

    model.fit(X_train, y_train, X_valid, y_valid_for_fit)
    return TrainedAssets(model=model, feature_cols=cols, horizon=cfg.labels.horizon)


def train_sequence_model(
    cfg: PipelineConfig,
    panel: pd.DataFrame,
    model: BaseModel,
) -> TrainedAssets:
    """
    Build (N, T, F) tensors from the long panel and fit a sequence model.
    """
    feat_df, cols = prepare_panel(cfg, panel)

    # Build tensors on the whole panel, then split by date using row_idx → date.
    # Drop rows with NaNs in features so the tensor doesn't carry them.
    feat_clean = feat_df.dropna(subset=cols).copy()
    feat_clean = feat_clean.reset_index(drop=True)

    X, row_idx, _ = build_sequence_tensor(
        feat_clean, cols, seq_len=cfg.features.seq_len
    )
    y = feat_clean.loc[row_idx, "y_ret"].to_numpy()
    dates_for_rows = feat_clean.loc[row_idx, "date"].to_numpy()

    # split tensor rows by date
    unique_dates = np.sort(np.unique(dates_for_rows))
    n_tr = int(len(unique_dates) * cfg.train.train_frac)
    n_va = int(len(unique_dates) * cfg.train.valid_frac)
    train_dates = set(unique_dates[:n_tr])
    valid_dates = set(unique_dates[n_tr:n_tr + n_va])

    train_mask = np.isin(dates_for_rows, list(train_dates))
    valid_mask = np.isin(dates_for_rows, list(valid_dates))

    # drop NaN labels too
    valid_label_mask_tr = ~np.isnan(y) & train_mask
    valid_label_mask_va = ~np.isnan(y) & valid_mask

    X_tr, y_tr = X[valid_label_mask_tr], y[valid_label_mask_tr]
    X_va, y_va = X[valid_label_mask_va], y[valid_label_mask_va]

    model.fit(X_tr, y_tr, X_va, y_va)
    return TrainedAssets(model=model, feature_cols=cols, horizon=cfg.labels.horizon)


def score_panel(
    assets: TrainedAssets,
    feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score a featurised panel with a fitted tabular model. Returns the panel
    with an added `y_pred` column.
    """
    cols = assets.feature_cols
    out = feat_df.copy()
    # only score rows where all features are present
    mask = out[cols].notna().all(axis=1)
    X = out.loc[mask, cols].to_numpy(dtype=np.float32)
    preds = assets.model.predict(X)
    out.loc[:, "y_pred"] = np.nan
    out.loc[mask, "y_pred"] = preds
    return out


def score_panel_sequence(
    assets: TrainedAssets,
    feat_df: pd.DataFrame,
    seq_len: int,
) -> pd.DataFrame:
    """
    Score a featurised panel with a fitted sequence model. Returns the panel
    with an added `y_pred` column aligned by row index.
    """
    cols = assets.feature_cols
    feat_clean = feat_df.dropna(subset=cols).copy().reset_index(drop=True)
    X, row_idx, _ = build_sequence_tensor(feat_clean, cols, seq_len)
    preds = assets.model.predict(X)
    out = feat_df.copy()
    out["y_pred"] = np.nan
    # map back via the original index of feat_df, which we lost when dropping NaNs;
    # use a (date, asset) merge instead.
    pred_map = (
        feat_clean.loc[row_idx, ["date", "asset"]]
        .assign(y_pred=preds)
        .drop_duplicates(["date", "asset"])
    )
    out = out.drop(columns=["y_pred"]).merge(pred_map, on=["date", "asset"], how="left")
    return out
