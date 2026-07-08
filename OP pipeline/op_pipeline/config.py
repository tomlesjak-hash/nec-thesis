"""
Central config — kept tiny on purpose.

The OP files had configuration scattered across globals and notebook cells.
We collect the knobs here so train/eval/backtest stay readable.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    n_assets: int = 50
    n_days: int = 750            # ~3 years of trading days
    seed: int = 7
    base_vol: float = 0.015      # daily vol of innovations
    factor_signal_strength: float = 0.35  # how much the planted signal matters


@dataclass
class FeatureConfig:
    # Rolling windows for feature construction — match the spirit of the
    # MLP-OP rolling operators (Mean/Std/Min/Max/Rate).
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    seq_len: int = 20             # window used by the MLP-OP sequence model


@dataclass
class LabelConfig:
    horizon: int = 5              # forward-return horizon (days)
    # OP-file signed-bucket thresholds (on absolute return). Used when
    # `bucketed=True` in label construction.
    thresholds: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.04])
    # OP downsampling parameters
    zero_multiple: float = 1.0    # cap on |class 0| relative to |non-zero|


@dataclass
class TrainConfig:
    # Time-ordered split. Fractions of unique trading days.
    train_frac: float = 0.7
    valid_frac: float = 0.15
    # test_frac is the remainder.
    rounds: int = 3               # OP-style incremental rounds
    replay_frac: float = 0.2      # carry-forward fraction of last round's data
    lr_decay_per_round: float = 0.005
    min_lr: float = 0.01


@dataclass
class BacktestConfig:
    n_quantiles: int = 10         # decile sort
    cost_bps: float = 5.0         # round-trip per-leg cost in bps
    annualization: int = 252


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
