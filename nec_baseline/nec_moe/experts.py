"""Emission models: the expert side of the mixture.

The emission is a pluggable axis (design doc §7 design goal): an
:class:`Emission` maps expert input to per-expert Gaussian parameters
``(mu (B, K), log_sigma (K,))``. Two implementations:

- :class:`ExpertBank` — K structurally identical small MLP regressors (the
  baseline's neural experts), each with its own scalar learnable ``log_sigma``
  (homoscedastic per expert — the minimal noise model the mixture NLL needs).
- :class:`ClassicalGaussianEmission` — per-regime constant Gaussians (learned
  ``mu_k, sigma_k`` scalars, no network). Combined with the HMM prior this is
  the classical Hamilton / Gaussian-HMM corner of the design's two plug axes;
  selected via ``ExpertConfig.kind = "classical"``.

Like the priors, emissions are registry-built (``EMISSION_REGISTRY``): the
design's two orthogonal plug axes — emission (neural | classical) × prior
(memoryless | Markov) — are both config strings, and each emission owns its
symmetry-breaking warm-start (:meth:`Emission.warmstart_slices`).

Experts are indexed, never named ("extreme"/"normal" do not appear): regime
semantics are assigned post hoc by diagnostics, because label switching makes
fixed names meaningless across runs and folds (Module 8 failure mode 2).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from .config import ExpertConfig
from .utils import assert_shape

if TYPE_CHECKING:  # pragma: no cover
    from .config import NECConfig

__all__ = [
    "Emission",
    "ExpertMLP",
    "ExpertBank",
    "ClassicalGaussianEmission",
    "EMISSION_REGISTRY",
    "build_emission",
]


class Emission(nn.Module, ABC):
    """``x (B, d) -> (mu (B, K), log_sigma (K,))``."""

    n_experts: int
    #: part of the contract: every emission owns per-expert noise scales —
    #: the sigma freeze schedule and the canonical (sigma-sorted) expert
    #: ordering both depend on this attribute existing
    log_sigma: Tensor

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def warmstart_slices(
        self,
        x: Tensor,
        y: Tensor,
        slices: Sequence[Tensor],
        *,
        steps: int = 150,
        lr: float = 1e-2,
    ) -> None:
        """Symmetry-breaking warm-start: fit component k to sample slice k.

        The slices come from sorting by an observable regime proxy (design doc
        §4.3); each emission breaks symmetry its own way — gradient pre-training
        for neural experts, closed-form moments for classical Gaussians. Only
        the emission is touched: the gate stays unsupervised (Decision B).
        """


def _seeded_linear_init(linear: nn.Linear, g: torch.Generator) -> None:
    """Default nn.Linear init (uniform ±1/sqrt(fan_in)) from a private generator."""
    fan_in = linear.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=g)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound, generator=g)


class ExpertMLP(nn.Module):
    """One expert: ``d -> h -> h//2 -> 1`` ReLU funnel with dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim // 2, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def reset_parameters_seeded(self, seed: int) -> None:
        """Per-expert seeded re-init: the symmetry-breaking device (§4.3)."""
        g = torch.Generator().manual_seed(seed)
        for lin in (self.l1, self.l2, self.l3):
            _seeded_linear_init(lin, g)

    def forward(self, x: Tensor) -> Tensor:
        assert_shape(x, (None, self.input_dim), "expert input")
        h = self.drop(self.act(self.l1(x)))
        h = self.drop(self.act(self.l2(h)))
        return self.l3(h).squeeze(-1)  # (B,)


class ExpertBank(Emission):
    """K diversified :class:`ExpertMLP`s + per-expert learnable ``log_sigma``."""

    def __init__(
        self,
        cfg: ExpertConfig,
        input_dim: int,
        sigma_init: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.input_dim = input_dim
        experts = [
            ExpertMLP(input_dim, cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.n_experts)
        ]
        for k, expert in enumerate(experts):
            expert.reset_parameters_seeded(seed + 7919 * (k + 1))
        self.experts = nn.ModuleList(experts)
        self.log_sigma = nn.Parameter(
            torch.full((cfg.n_experts,), math.log(sigma_init))
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert_shape(x, (None, self.input_dim), "expert bank input")
        mu = torch.stack([e(x) for e in self.experts], dim=-1)  # (B, K)
        return mu, self.log_sigma

    def warmstart_slices(
        self,
        x: Tensor,
        y: Tensor,
        slices: Sequence[Tensor],
        *,
        steps: int = 150,
        lr: float = 1e-2,
    ) -> None:
        """Briefly pre-train expert k as a plain MSE regressor on slice k."""
        opt = torch.optim.Adam(self.experts.parameters(), lr=lr)
        for _ in range(steps):
            for k, idx in enumerate(slices):
                mu, _ = self(x[idx])
                loss = torch.mean((mu[:, k] - y[idx]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()


class ClassicalGaussianEmission(Emission):
    """Per-regime constant Gaussians: the Hamilton baseline's emission model.

    Ignores the input features entirely — ``mu_k`` and ``sigma_k`` are learned
    scalars. Paired with :class:`~nec_moe.priors.HMMRegimePrior`
    (``prior.kind="hmm"``) this yields a classical Gaussian-HMM / Hamilton
    regime-switching model from the same skeleton and training loop (design
    doc §7 design goal); paired with a memoryless prior it is a
    feature-gated mixture with classical emissions.
    """

    def __init__(self, n_experts: int, sigma_init: float = 1.0) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.mu = nn.Parameter(torch.zeros(n_experts))
        self.log_sigma = nn.Parameter(
            torch.full((n_experts,), math.log(sigma_init))
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b = x.shape[0]
        return self.mu.unsqueeze(0).expand(b, -1), self.log_sigma

    def warmstart_slices(
        self,
        x: Tensor,
        y: Tensor,
        slices: Sequence[Tensor],
        *,
        steps: int = 150,
        lr: float = 1e-2,
    ) -> None:
        """Closed-form moment init: ``mu_k, sigma_k`` from slice k's mean/std.

        The classical analogue of quantile initialization for a GMM (``steps``
        and ``lr`` are irrelevant and ignored).
        """
        del steps, lr
        with torch.no_grad():
            for k, idx in enumerate(slices):
                self.mu[k] = y[idx].mean()
                self.log_sigma[k] = y[idx].std(unbiased=True).clamp_min(1e-3).log()


# --------------------------------------------------------------------------- #
# Registry: the emission axis mirrors the prior axis — a config string selects
# the corner of the (emission x prior) grid.
# --------------------------------------------------------------------------- #

EMISSION_REGISTRY: dict[str, Callable[[NECConfig], Emission]] = {
    "mlp": lambda cfg: ExpertBank(
        cfg.experts,
        input_dim=cfg.expert_input_dim,
        sigma_init=cfg.train.sigma_init,
        seed=cfg.train.seed,
    ),
    "classical": lambda cfg: ClassicalGaussianEmission(
        cfg.experts.n_experts, sigma_init=cfg.train.sigma_init
    ),
}


def build_emission(cfg: NECConfig) -> Emission:
    try:
        return EMISSION_REGISTRY[cfg.experts.kind](cfg)
    except KeyError:
        raise ValueError(
            f"unknown experts.kind {cfg.experts.kind!r}; registered: "
            f"{sorted(EMISSION_REGISTRY)}"
        ) from None
