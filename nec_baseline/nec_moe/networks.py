"""The shared MLP block, and the activation registry.

One block class builds *every* network in the package — the frozen base ``f0``
and each expert head — which is the arrangement of Ye & Borde
(arXiv:2608.12251): a single ``Block(q; theta)`` of fixed depth and width
reused for base and experts, so a parameter-count comparison between "one
network" and "a mixture of K networks" is a comparison of counts rather than
of architectures.

Nothing here hardcodes a shape or a nonlinearity. Widths arrive as a
``hidden_dims`` tuple (depth and width in one field; see
:func:`nec_moe.config.pyramid_dims`) and the activation arrives as a registry
key, because none of these values is chosen yet: they are swept from the
settings block and recorded per trial.

``zero_init_head`` is the residual design's initialization: with the final
layer's weight *and* bias zeroed, the block outputs identically zero, so a
mixture of such blocks added to a base starts exactly at the base and the
first optimizer step moves away from it rather than toward it. Ye & Borde's
ablation replaces this with random initialization and reports a lower IC and
markedly less stable training.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .utils import assert_shape

__all__ = ["ACTIVATION_REGISTRY", "build_activation", "MLPBlock"]

#: Activation by config string — selected, never hardcoded. ``relu`` is the
#: default (Gu, Kelly & Xiu's choice for the benchmark network class on a
#: characteristic panel); ``gelu`` is what Ye & Borde use.
ACTIVATION_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def build_activation(kind: str) -> nn.Module:
    try:
        return ACTIVATION_REGISTRY[kind]()
    except KeyError:
        raise ValueError(
            f"unknown activation {kind!r}; registered: "
            f"{sorted(ACTIVATION_REGISTRY)}"
        ) from None


def _seeded_linear_init(linear: nn.Linear, g: torch.Generator) -> None:
    """Default ``nn.Linear`` init (uniform ±1/sqrt(fan_in)) from a private RNG."""
    bound = 1.0 / math.sqrt(linear.weight.shape[1])
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=g)
        if linear.bias is not None:
            linear.bias.uniform_(-bound, bound, generator=g)


class MLPBlock(nn.Module):
    """``input_dim -> hidden_dims... -> 1``, scalar output ``(B,)``.

    Dropout is applied after each hidden activation; the output layer is
    linear. With ``zero_init_head`` the output layer starts at exactly zero
    (see the module docstring).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        dropout: float = 0.0,
        activation: str = "relu",
        zero_init_head: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        layers: list[nn.Module] = []
        widths = [input_dim, *self.hidden_dims]
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            layers += [nn.Linear(a, b), build_activation(activation), nn.Dropout(dropout)]
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_dims[-1], 1)
        if zero_init_head:
            self.zero_head_()

    def zero_head_(self) -> None:
        """Zero the output layer: the block becomes identically zero."""
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def reset_parameters_seeded(self, seed: int, *, zero_head: bool = False) -> None:
        """Per-network seeded re-init — the expert-diversification device (§4.3).

        ``zero_head`` re-applies the residual initialization afterwards, so
        seeding the hidden layers never resurrects a nonzero correction.
        """
        g = torch.Generator().manual_seed(seed)
        for module in [*self.hidden, self.head]:
            if isinstance(module, nn.Linear):
                _seeded_linear_init(module, g)
        if zero_head:
            self.zero_head_()

    def forward(self, x: Tensor) -> Tensor:
        assert_shape(x, (None, self.input_dim), "MLP block input")
        return self.head(self.hidden(x)).squeeze(-1)  # (B,)
