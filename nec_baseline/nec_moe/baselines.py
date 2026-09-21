"""Single-model baselines (syllabus candidate models 1 and 3).

The thesis's primary research question is comparative: *do MoE routing methods
improve out-of-sample cross-sectional prediction relative to single-model
baselines and classical regime-switching baselines?* The classical side is the
Hamilton corner (§7); this module supplies the single-model side:

- :class:`RidgeBaseline` — closed-form ridge on the snapshot features
  (Module 4 conventions: intercept not penalized). The "many weak correlated
  signals" default.
- :class:`MLPBaseline` — one :class:`~nec_moe.networks.MLPBlock` (the *same*
  network class as a single NEC expert, deliberately: capacity-matched to one
  expert) plus a scalar learnable noise sigma, trained on the same Gaussian
  NLL family as the NEC — so held-out NLLs are directly comparable.

Both consume ``x_snap`` only, mirroring Decision A's snapshot-only experts: the
comparison isolates *routing/regime structure*, not input sets. LightGBM
(candidate 2) is deliberately not wired in — heavy optional dependency, and the
repo carries a separate LGBM pipeline; add it later behind an import guard if
the thesis wants the row.

On the planted sign-flip regime panel these baselines are *structurally* unable
to score: the pooled cross-sectional relationship averages to zero — which is
exactly the point of the comparison (and pinned as a test).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .data import Panel
from .likelihood import expert_log_likelihood
from .networks import MLPBlock

__all__ = ["BaselineModel", "RidgeBaseline", "MLPBaseline"]


class BaselineModel(ABC):
    """fit / predict / nll on panels — the minimal harness contract."""

    @abstractmethod
    def fit(self, train: Panel) -> None: ...

    @abstractmethod
    def predict(self, panel: Panel) -> Tensor:
        """Per-sample point predictions ``(N,)``."""

    @abstractmethod
    def nll(self, panel: Panel) -> float:
        """Held-out mean Gaussian NLL (comparable with the NEC's mixture NLL)."""


class RidgeBaseline(BaselineModel):
    """Closed-form ridge on ``x_snap`` with an unpenalized intercept.

    ``beta = (X'X + lam*I_p)^-1 X'y`` with the identity zeroed on the intercept
    coordinate (Module 4: standardize-and-don't-penalize-the-intercept
    convention; snapshot features arrive rank-normalized from the pipeline).
    The noise scale is the training residual std, giving a proper Gaussian
    predictive NLL.
    """

    def __init__(self, l2: float = 1.0) -> None:
        if l2 < 0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.l2 = l2
        self._beta: Tensor | None = None  # (d+1,), intercept last
        self._log_sigma: float | None = None

    @staticmethod
    def _design(x_snap: Tensor) -> Tensor:
        return torch.cat([x_snap, torch.ones(len(x_snap), 1)], dim=1)

    def fit(self, train: Panel) -> None:
        x = self._design(train.x_snap.double())
        y = train.y.double()
        d = x.shape[1]
        penalty = self.l2 * torch.eye(d, dtype=torch.float64)
        penalty[d - 1, d - 1] = 0.0  # never penalize the intercept
        self._beta = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
        resid = y - x @ self._beta
        sigma = float(resid.pow(2).mean().sqrt())
        self._log_sigma = math.log(max(sigma, 1e-8))

    def predict(self, panel: Panel) -> Tensor:
        if self._beta is None:
            raise RuntimeError("fit() before predict()")
        return (self._design(panel.x_snap.double()) @ self._beta).float()

    def nll(self, panel: Panel) -> float:
        assert self._log_sigma is not None, "fit() before nll()"
        mu = self.predict(panel).unsqueeze(-1)  # (N, 1)
        log_sigma = torch.tensor([self._log_sigma])
        return float(-expert_log_likelihood(mu, log_sigma, panel.y).mean())


class MLPBaseline(BaselineModel):
    """One expert-class MLP + scalar noise, trained on the Gaussian NLL.

    Uses the identical network class and loss family as a single NEC expert,
    so 'MoE vs single model' compares routing structure at matched capacity —
    not architecture families. Deterministic given ``seed``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.05,
        activation: str = "relu",
        lr: float = 1e-3,
        steps: int = 400,
        batch_size: int = 256,
        sigma_init: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.steps = steps
        self.batch_size = batch_size
        self.sigma_init = sigma_init
        self.seed = seed
        self._net: MLPBlock | None = None
        self._log_sigma: Tensor | None = None

    def fit(self, train: Panel) -> None:
        torch.manual_seed(self.seed)
        net = MLPBlock(
            self.input_dim, self.hidden_dims,
            dropout=self.dropout, activation=self.activation,
        )
        log_sigma = torch.tensor([math.log(self.sigma_init)], requires_grad=True)
        opt = torch.optim.AdamW(
            [*net.parameters(), log_sigma], lr=self.lr, weight_decay=1e-4
        )
        g = torch.Generator().manual_seed(self.seed)
        n = len(train.y)
        net.train()
        done = 0
        while done < self.steps:
            for idx in torch.randperm(n, generator=g).split(self.batch_size):
                mu = net(train.x_snap[idx]).unsqueeze(-1)  # (B, 1)
                loss = -expert_log_likelihood(mu, log_sigma, train.y[idx]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                done += 1
                if done >= self.steps:
                    break
        self._net = net.eval()
        self._log_sigma = log_sigma.detach()

    @torch.no_grad()
    def predict(self, panel: Panel) -> Tensor:
        if self._net is None:
            raise RuntimeError("fit() before predict()")
        return self._net(panel.x_snap)

    @torch.no_grad()
    def nll(self, panel: Panel) -> float:
        assert self._net is not None and self._log_sigma is not None
        mu = self._net(panel.x_snap).unsqueeze(-1)
        return float(-expert_log_likelihood(mu, self._log_sigma, panel.y).mean())
