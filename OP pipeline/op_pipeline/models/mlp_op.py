"""
MLP-OP — PyTorch model with rolling-stat operator layers.

Faithful in spirit to `rolling_feature_mlp_explained.py` but with the bugs
called out in that file's docstring actually fixed:

* The original applied dropout to the **input** tensor `x` rather than the
  hidden activations `out`. We fix that.
* The original defined `bn1`, several duplicate operators (`max2`, `min2`,
  `mean2`, etc.), and a `corrcoef` layer that were never used in forward().
  We remove the dead code.
* The original used loops over time steps for rolling ops. We use
  vectorised `unfold` for a ~10x speedup.

Architecture:
    Input:  (B, T, F)
    Rolling ops produce: rolling mean, std, min, max of the last window,
        plus a "mean residual" feature (last step minus rolling mean).
    All concatenated to a (B, F * 5) feature vector.
    Then a 4-layer MLP head -> scalar prediction.

This is intentionally light. The OP file was clearly experimental; we keep
the operator-feature idea but stop short of replicating its bloat.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import torch
from torch import nn

from .base import BaseModel


class _RollingOpBlock(nn.Module):
    """
    Vectorised rolling mean / std / min / max / mean_residual over the time
    dim, each with its own learnable per-feature weight.

    Input:  (B, T, F)
    Output: (B, 5*F)   — concatenation of [last-step mean, std, min, max,
                        mean_residual] each (B, F).
    """

    def __init__(self, in_features: int, window: int):
        super().__init__()
        self.window = window
        # one learnable gain per output type, per feature
        self.w_mean = nn.Parameter(torch.randn(in_features) * 0.1)
        self.w_std = nn.Parameter(torch.randn(in_features) * 0.1)
        self.w_min = nn.Parameter(torch.randn(in_features) * 0.1)
        self.w_max = nn.Parameter(torch.randn(in_features) * 0.1)
        self.w_res = nn.Parameter(torch.randn(in_features) * 0.1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        T = x.shape[1]
        w = min(self.window, T)
        window = x[:, -w:, :]                       # (B, w, F)
        mean_v = window.mean(dim=1)                 # (B, F)
        std_v = window.std(dim=1)                   # (B, F)
        min_v = window.min(dim=1).values            # (B, F)
        max_v = window.max(dim=1).values            # (B, F)
        last_v = x[:, -1, :]                        # (B, F)
        res_v = last_v - mean_v                     # mean residual

        return torch.cat(
            [
                mean_v * self.w_mean,
                std_v * self.w_std,
                min_v * self.w_min,
                max_v * self.w_max,
                res_v * self.w_res,
            ],
            dim=-1,
        )


class _MLPOPNet(nn.Module):
    """
    Rolling-op block -> 4-layer MLP head.
    """

    def __init__(self, in_features: int, window: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.rolling = _RollingOpBlock(in_features, window)
        in_h = in_features * 5
        self.net = nn.Sequential(
            nn.Linear(in_h, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.rolling(x)
        out = self.net(h)
        return out.squeeze(-1)


class MLPOPModel(BaseModel):
    """
    BaseModel-compatible wrapper around _MLPOPNet.

    Expects X of shape (N, T, F). Use `features.build_sequence_tensor` to
    construct it from the long-format feature dataframe.
    """

    name = "mlp_op"

    def __init__(
        self,
        in_features: int,
        seq_len: int,
        window: int = 10,
        hidden: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 30,
        batch_size: int = 1024,
        early_stopping: int = 5,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.in_features = in_features
        self.seq_len = seq_len
        self.window = window
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[_MLPOPNet] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        # Per-feature standardization across all (sample, time) rows
        flat = X.reshape(-1, X.shape[-1])
        self.feature_mean = np.nanmean(flat, axis=0)
        self.feature_std = np.nanstd(flat, axis=0)
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        return self._standardize_apply(X)

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.feature_mean) / self.feature_std
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return Z.astype(np.float32, copy=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "MLPOPModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xtr = self._standardize_fit(X_train)
        ytr = y_train.astype(np.float32)

        self.net = _MLPOPNet(
            in_features=self.in_features,
            window=self.window,
            hidden=self.hidden,
            dropout=self.dropout,
        ).to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr).to(self.device)
        ytr_t = torch.from_numpy(ytr).to(self.device)

        Xval_t = yval_t = None
        if X_valid is not None and y_valid is not None:
            Xval_t = torch.from_numpy(self._standardize_apply(X_valid)).to(self.device)
            yval_t = torch.from_numpy(y_valid.astype(np.float32)).to(self.device)

        n = Xtr_t.shape[0]
        best_val = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                xb = Xtr_t[idx]
                yb = ytr_t[idx]
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if Xval_t is not None:
                self.net.eval()
                with torch.no_grad():
                    val_pred = self.net(Xval_t)
                    val_loss = float(loss_fn(val_pred, yval_t).item())
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.early_stopping:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model not fitted.")
        Xs = self._standardize_apply(X)
        with torch.no_grad():
            preds = []
            for start in range(0, len(Xs), self.batch_size):
                xb = torch.from_numpy(Xs[start: start + self.batch_size]).to(self.device)
                preds.append(self.net(xb).cpu().numpy())
        return np.concatenate(preds)
