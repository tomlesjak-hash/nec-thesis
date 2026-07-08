"""
LightGBM wrapper — distilled from `lgbm_quant_pipeline_explained.py`.

Differences from the original (deliberate cleanups):
- No global state. The OP file leaned on `feature_ls`, `label`, `thresholds`
  as module-level globals; we pass everything explicitly.
- No proprietary parquet I/O / `dw_data.fastpai` dependency. The wrapper
  consumes plain numpy arrays.
- Incremental-round training collapsed into a single `fit_incremental`
  method to keep the example self-contained, but the pattern (multiple
  rounds, 20% replay buffer, learning-rate decay) is preserved.
- No on-disk model checkpoints (the OP file dumped `.pkl` + `.txt` + JSON
  to two redundant paths). We hold the booster in memory; saving is the
  user's call.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from .base import BaseModel


_DEFAULT_PARAMS = dict(
    boosting_type="gbdt",
    objective="regression",
    metric="rmse",
    num_leaves=63,
    max_depth=-1,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
)


class LGBMModel(BaseModel):
    name = "lgbm"

    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 200,
        early_stopping: int = 50,
        rounds: int = 1,
        replay_frac: float = 0.2,
        lr_decay_per_round: float = 0.005,
        min_lr: float = 0.01,
        random_state: int = 42,
    ):
        if not _HAS_LGB:
            raise ImportError(
                "lightgbm not installed; `pip install lightgbm`."
            )
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.rounds = rounds
        self.replay_frac = replay_frac
        self.lr_decay_per_round = lr_decay_per_round
        self.min_lr = min_lr
        self.random_state = random_state
        self.booster: Optional["lgb.Booster"] = None

    def _replay_subset(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        k = int(len(X) * self.replay_frac)
        if k <= 0:
            return X[:0], y[:0]
        idx = rng.choice(len(X), size=k, replace=False)
        return X[idx], y[idx]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "LGBMModel":
        """
        Train with the OP-style incremental-rounds pattern.

        If `self.rounds == 1`, behaves like a vanilla single-shot fit.
        If > 1, we partition the training set into `rounds` chunks, train
        round 0 from scratch, then continue training each subsequent round
        with `init_model=previous` and a small learning-rate decay, mixing
        in a 20% replay buffer from the prior round's data.
        """
        rng = np.random.default_rng(self.random_state)
        n = len(X_train)
        perm = rng.permutation(n)
        X_train = X_train[perm].astype(np.float32, copy=False)
        y_train = y_train[perm].astype(np.float64, copy=False)

        chunks = np.array_split(np.arange(n), self.rounds)
        booster: Optional["lgb.Booster"] = None
        replay_X, replay_y = None, None

        callbacks = [lgb.log_evaluation(period=0)]
        if X_valid is not None and self.early_stopping > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping, verbose=False))

        for r, idx in enumerate(chunks):
            Xr, yr = X_train[idx], y_train[idx]
            if replay_X is not None:
                Xr = np.vstack([replay_X, Xr])
                yr = np.concatenate([replay_y, yr])

            params = dict(self.params)
            # decay LR per round
            params["learning_rate"] = max(
                self.params["learning_rate"] - r * self.lr_decay_per_round,
                self.min_lr,
            )

            dtrain = lgb.Dataset(Xr, yr, free_raw_data=False)
            valid_sets = [dtrain]
            valid_names = ["train"]
            if X_valid is not None:
                dvalid = lgb.Dataset(
                    X_valid.astype(np.float32, copy=False),
                    y_valid.astype(np.float64, copy=False),
                    reference=dtrain,
                    free_raw_data=False,
                )
                valid_sets.append(dvalid)
                valid_names.append("valid")

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
                init_model=booster,
                keep_training_booster=True,
            )

            # build replay buffer for next round
            replay_X, replay_y = self._replay_subset(X_train[idx], y_train[idx])

        # finalize booster as a regular (non-training) booster so predict() is stable
        if booster is None:
            raise RuntimeError("LGBM training produced no booster.")
        self.booster = lgb.Booster(model_str=booster.model_to_string())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not fitted.")
        return self.booster.predict(X.astype(np.float32, copy=False))

    def report(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Quick diagnostic — RMSE and Pearson IC."""
        yhat = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        ic = float(pearsonr(y, yhat)[0]) if len(y) > 1 else float("nan")
        return {"rmse": rmse, "ic": ic, "n": int(len(y))}
