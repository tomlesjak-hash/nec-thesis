"""Small shared utilities: shape assertions and seeding.

The ``assert_shape`` helper is the runtime half of the defect-8 fix (the design
doc's ledger: an encoder/head hidden-dim mismatch must fail loudly with names
and numbers, never silently produce garbage). The config-validation half lives
in :mod:`nec_moe.config`.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor

__all__ = ["assert_shape", "set_seed"]


def assert_shape(t: Tensor, expected: tuple[int | None, ...], name: str) -> None:
    """Assert ``t`` has shape ``expected``; ``None`` entries are wildcards.

    Raises
    ------
    ValueError
        With the tensor's *name*, the expected shape, and the actual shape —
        so a mis-wired module dies with a readable message instead of a
        broadcast surprise (ledger defect 8).
    """
    actual = tuple(t.shape)
    ok = len(actual) == len(expected) and all(
        e is None or a == e for a, e in zip(actual, expected, strict=True)
    )
    if not ok:
        exp_str = tuple("*" if e is None else e for e in expected)
        raise ValueError(
            f"shape mismatch for '{name}': expected {exp_str}, got {actual}"
        )


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch global RNGs (single seed function)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
