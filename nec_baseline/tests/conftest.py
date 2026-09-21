"""Shared test fixtures: import path + small standard configs."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nec_moe import (  # noqa: E402
    BaseConfig,
    DataConfig,
    EncoderConfig,
    ExpertConfig,
    NECConfig,
    PriorConfig,
    TrainConfig,
)

# Reference dims used across tests
D_SEQ, D_SNAP, SEQ_LEN = 2, 3, 8


def small_config(
    *,
    n_experts: int = 2,
    input_mode: str = "snapshot",
    prior_kind: str = "soft",
    expert_kind: str = "mlp",
    expert_dropout: float = 0.0,
    expert_hidden_dims: tuple[int, ...] = (16, 8),
    correction_mode: bool = False,
    zero_init_head: bool = True,
    base: BaseConfig | None = None,
    sigma_init: float = 1.0,
    seed: int = 0,
    **train_overrides,
) -> NECConfig:
    """A tiny, deterministic config for unit tests (dropout off by default)."""
    sequence_ordered = train_overrides.pop(
        "sequence_ordered", prior_kind == "hmm"
    )
    train_overrides.setdefault("sigma_freeze_steps", 0)
    return NECConfig(
        data=DataConfig(d_seq=D_SEQ, d_snap=D_SNAP, seq_len=SEQ_LEN),
        encoder=EncoderConfig(hidden_dim=16, num_layers=1),
        experts=ExpertConfig(
            n_experts=n_experts,
            hidden_dims=expert_hidden_dims,
            dropout=expert_dropout,
            input_mode=input_mode,
            kind=expert_kind,
            correction_mode=correction_mode,
            zero_init_head=zero_init_head,
        ),
        prior=PriorConfig(kind=prior_kind),
        base=base if base is not None else BaseConfig(),
        train=TrainConfig(
            sigma_init=sigma_init,
            seed=seed,
            sequence_ordered=sequence_ordered,
            **train_overrides,
        ),
    )


def small_base_config(**overrides) -> BaseConfig:
    """A tiny, fast frozen-base config for residual-mode tests."""
    kw = dict(enabled=True, hidden_dims=(16, 8), steps=60, batch_size=64, lr=1e-2)
    kw.update(overrides)
    return BaseConfig(**kw)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    yield


def random_inputs(b: int = 16, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x_seq = torch.randn(b, SEQ_LEN, D_SEQ, generator=g)
    x_snap = torch.randn(b, D_SNAP, generator=g)
    y = torch.randn(b, generator=g)
    return x_seq, x_snap, y
