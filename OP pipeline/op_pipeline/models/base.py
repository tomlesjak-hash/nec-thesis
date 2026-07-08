"""
Common model interface so train/eval/backtest don't need to know which model
they're talking to. The OP folder had nine different files with nine different
calling conventions — this fixes that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Minimal predictor interface."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "BaseModel":
        """Train. Returns self for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1-D predictions of length len(X)."""
