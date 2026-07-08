"""
Prediction-quality metrics.

The OP files printed RMSE and Pearson IC. We add Spearman rank-IC (more
robust to the heavy tails financial returns produce) and hit-rate.

These metrics are *not* sufficient to declare a model "good" — that requires
the backtest. They are the first-pass diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


@dataclass
class PredictionMetrics:
    n: int
    rmse: float
    ic: float            # Pearson IC
    rank_ic: float       # Spearman rank IC
    hit_rate: float      # P(sign(pred) == sign(actual))

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "rmse": self.rmse,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "hit_rate": self.hit_rate,
        }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> PredictionMetrics:
    """
    Standard regression diagnostics for return prediction.

    Notes
    -----
    Pearson IC is sensitive to a few large moves; rank IC is what most
    quant shops report.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be same length")
    if len(y_true) < 2:
        return PredictionMetrics(len(y_true), float("nan"), float("nan"), float("nan"), float("nan"))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ic_val = float(pearsonr(y_true, y_pred)[0])
    rank_ic_val = float(spearmanr(y_true, y_pred)[0])

    # hit rate ignores observations with sign==0 to avoid degenerate ties
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() > 0:
        hit = float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())
    else:
        hit = float("nan")

    return PredictionMetrics(
        n=int(len(y_true)),
        rmse=rmse,
        ic=ic_val,
        rank_ic=rank_ic_val,
        hit_rate=hit,
    )


def daily_ic_series(df: "pd.DataFrame", y_col: str, pred_col: str, date_col: str = "date"):
    """
    Compute the per-date cross-sectional rank-IC series.

    This is the metric quants actually look at: for each day, take the rank
    correlation between the cross-section of predictions and the cross-section
    of realised forward returns. Then look at the mean / std / IR of that
    series, not just one number across the whole sample.

    Requires pandas at call time.
    """
    import pandas as pd  # local import to keep evaluate.py importable without pandas

    def _rank_ic(g):
        if len(g) < 5:
            return np.nan
        rho, _ = spearmanr(g[y_col].to_numpy(), g[pred_col].to_numpy())
        return rho

    return df.groupby(date_col).apply(_rank_ic).rename("rank_ic")
