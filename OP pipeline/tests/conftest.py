"""Shared pytest fixtures."""

import os
import sys

import pytest

# make repo importable without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from op_pipeline.config import DataConfig, PipelineConfig
from op_pipeline.data import generate_panel


@pytest.fixture(scope="session")
def small_cfg() -> PipelineConfig:
    """Smaller version of the config for fast tests."""
    cfg = PipelineConfig()
    cfg.data = DataConfig(n_assets=15, n_days=180, seed=11)
    cfg.train.rounds = 1
    return cfg


@pytest.fixture(scope="session")
def small_panel(small_cfg):
    return generate_panel(small_cfg.data)
