"""Data contract (§6): schema, batches, the general Panel, and the synthetic generator.

What lives here — named feature schema, explicit tensors, no reserved columns,
the model never slices feature tensors (ledger defect 7 fix):

- :class:`Panel` — the general supervised panel every data source produces.
  The Stage-B real pipeline (:mod:`nec_moe.features`) builds plain ``Panel``
  instances (with ``date_labels``/``entity_labels`` mapping integer codes back
  to calendar dates and tickers).
- :class:`SyntheticPanel` — a ``Panel`` plus the ground truth needed to score
  against it (latent regime path, true betas/transition, noiseless target).
  ``PanelData`` is kept as an alias for backward compatibility.

Synthetic regime-process modes (design doc §6.4): ``"iid"`` (regime drawn per
date from a fixed marginal — memoryless-favourable) and ``"markov"`` (sticky
first-order chain over dates — HMM-favourable; ground-truth transition matrix
exposed). Regime is a *date-level* variable (shared by all entities on a
date), matching the premise that makes load-balancing batch scope dangerous
(§4.3).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .utils import assert_shape

__all__ = [
    "FeatureSchema",
    "Batch",
    "Panel",
    "SyntheticSpec",
    "SyntheticPanel",
    "PanelData",
    "SyntheticRegimePanel",
]


@dataclass(frozen=True)
class FeatureSchema:
    """Named feature columns; order == channel order of the tensors."""

    sequence_features: tuple[str, ...]
    snapshot_features: tuple[str, ...]
    target: str


@dataclass
class Batch:
    """What the model consumes. Shapes checked at construction.

    - ``x_seq``  float32 ``(B, T, d_seq)`` — trailing window, oldest → newest;
      row t uses information available at or before t.
    - ``x_snap`` float32 ``(B, d_snap)``  — prediction-date snapshot features.
    - ``y``      float32 ``(B,)``          — forward return target.
    - ``regime`` optional int64 ``(B,)``   — ground-truth latent regime;
      synthetic data / diagnostics only, never a model input.
    - ``date``   optional int64 ``(B,)``   — the row's date code. Not a
      feature and never reaches a network: it is the key a **precomputed**
      prior looks its date-level regime probabilities up by (brief 03 §1).
    """

    x_seq: Tensor
    x_snap: Tensor
    y: Tensor
    regime: Tensor | None = None
    date: Tensor | None = None

    def __post_init__(self) -> None:
        if self.x_seq.ndim != 3:
            raise ValueError(f"x_seq must be (B, T, d_seq), got {tuple(self.x_seq.shape)}")
        b = self.x_seq.shape[0]
        assert_shape(self.x_snap, (b, None), "x_snap")
        assert_shape(self.y, (b,), "y")
        if self.regime is not None:
            assert_shape(self.regime, (b,), "regime")
        if self.date is not None:
            assert_shape(self.date, (b,), "date")

    def __len__(self) -> int:
        return self.x_seq.shape[0]

    def validate_schema(self, schema: FeatureSchema) -> Batch:
        """Width check against the named schema (defect-7 contract clause).

        A target column smuggled into a feature tensor changes its width and
        fails here — there is no reserved column to silently slice off.
        """
        if self.x_seq.shape[2] != len(schema.sequence_features):
            raise ValueError(
                f"x_seq width {self.x_seq.shape[2]} != "
                f"len(sequence_features)={len(schema.sequence_features)}"
            )
        if self.x_snap.shape[1] != len(schema.snapshot_features):
            raise ValueError(
                f"x_snap width {self.x_snap.shape[1]} != "
                f"len(snapshot_features)={len(schema.snapshot_features)}"
            )
        return self


# --------------------------------------------------------------------------- #
# The general panel
# --------------------------------------------------------------------------- #


@dataclass(kw_only=True)
class Panel:
    """A supervised panel in the contract's tensor layout (design doc §6).

    Rows are (date, entity) samples, **date-major ordered** (all of date 0's
    entities, then date 1's, …) — the ordering the stateful trainer's
    per-date threading relies on. ``date`` and ``entity`` are integer codes;
    ``date_labels``/``entity_labels`` (optional) map codes back to calendar
    dates and tickers for reporting.
    """

    x_seq: Tensor  # (N, T, d_seq) float32
    x_snap: Tensor  # (N, d_snap) float32
    y: Tensor  # (N,) float32 forward-return target
    date: Tensor  # (N,) int64 date codes
    entity: Tensor  # (N,) int64 stable name ids (weight alignment across dates)
    schema: FeatureSchema
    date_labels: tuple[str, ...] | None = None
    entity_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        n = self.x_seq.shape[0]
        assert_shape(self.x_snap, (n, len(self.schema.snapshot_features)), "x_snap")
        if self.x_seq.shape[2] != len(self.schema.sequence_features):
            raise ValueError(
                f"x_seq width {self.x_seq.shape[2]} != "
                f"len(sequence_features)={len(self.schema.sequence_features)}"
            )
        assert_shape(self.y, (n,), "y")
        assert_shape(self.date, (n,), "date")
        assert_shape(self.entity, (n,), "entity")

    def __len__(self) -> int:
        return self.x_seq.shape[0]

    # -------------------------------------------------------- row selection
    def _row_kwargs(self, idx: Tensor) -> dict:
        """Row-aligned fields to re-index on subsetting; subclasses extend."""
        return dict(
            x_seq=self.x_seq[idx],
            x_snap=self.x_snap[idx],
            y=self.y[idx],
            date=self.date[idx],
            entity=self.entity[idx],
        )

    def _take(self, idx: Tensor) -> Panel:
        return dataclasses.replace(self, **self._row_kwargs(idx))

    def _batch_regime(self, idx: Tensor) -> Tensor | None:
        return None  # ground-truth regime exists only on synthetic panels

    def _batch(self, idx: Tensor) -> Batch:
        return Batch(
            self.x_seq[idx], self.x_snap[idx], self.y[idx],
            self._batch_regime(idx), self.date[idx],
        )

    # ------------------------------------------------------------- batching
    def full_batch(self) -> Batch:
        return self._batch(torch.arange(len(self)))

    def date_batches(self) -> Iterator[Batch]:
        """One cross-sectional batch per date, in chronological order."""
        for d in torch.unique(self.date, sorted=True):
            yield self._batch(self.date == d)

    def time_sequence(self) -> list[Batch]:
        """Chronological list of per-date batches (the stateful trainer's input)."""
        return list(self.date_batches())

    def minibatches(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> Iterator[Batch]:
        """Shuffled minibatches (memoryless training only)."""
        perm = torch.randperm(len(self), generator=generator)
        for i in range(0, len(self), batch_size):
            yield self._batch(perm[i : i + batch_size])

    def subset_dates(self, dates: Tensor) -> Panel:
        """Rows whose date is in ``dates`` (walk-forward fold construction)."""
        return self._take(torch.isin(self.date, dates))

    def split_by_date(self, train_frac: float) -> tuple[Panel, Panel]:
        """Chronological split (leakage discipline holds even on synthetic data)."""
        dates = torch.unique(self.date, sorted=True)
        cut = dates[int(math.floor(train_frac * len(dates))) - 1]
        m = self.date <= cut
        return self._take(m), self._take(~m)


# --------------------------------------------------------------------------- #
# Synthetic regime panel
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyntheticSpec:
    """Knobs of the synthetic generator.

    ``y = x_snap · beta[z] + eps`` with per-regime coefficients; for K=2 the
    betas are sign-flipped (momentum in calm, reversal in stress). Regime is
    inferable from the window: channel 0 of ``x_seq`` has per-regime volatility
    ``vol_levels[z]``.
    """

    n_regimes: int = 2
    d_seq: int = 2
    d_snap: int = 3
    seq_len: int = 8
    regime_process: Literal["iid", "markov"] = "iid"
    marginal: tuple[float, ...] | None = None  # iid mode; default uniform
    transition_stay: float = 0.95  # markov mode: diagonal of the true chain
    vol_levels: tuple[float, ...] = (0.5, 2.0)  # channel-0 vol per regime
    beta_scale: float = 2.0
    noise_std: float = 0.5
    seed: int = 0

    def validate(self) -> SyntheticSpec:
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        if len(self.vol_levels) != self.n_regimes:
            raise ValueError(
                f"len(vol_levels)={len(self.vol_levels)} != n_regimes={self.n_regimes}"
            )
        if self.marginal is not None and (
            len(self.marginal) != self.n_regimes
            or abs(sum(self.marginal) - 1.0) > 1e-6
        ):
            raise ValueError("marginal must have n_regimes entries summing to 1")
        if not 0.0 < self.transition_stay < 1.0:
            raise ValueError("transition_stay must be in (0, 1)")
        return self


@dataclass(kw_only=True)
class SyntheticPanel(Panel):
    """A :class:`Panel` plus the ground truth needed to score against it."""

    y_clean: Tensor  # (N,) noiseless conditional mean beta[z]·x_snap
    regime: Tensor  # (N,) int64 latent regime path
    betas: Tensor  # (K, d_snap)
    transition: Tensor | None  # (K, K) true chain (markov mode) else None
    spec: SyntheticSpec

    def _row_kwargs(self, idx: Tensor) -> dict:
        return super()._row_kwargs(idx) | dict(
            y_clean=self.y_clean[idx], regime=self.regime[idx]
        )

    def _batch_regime(self, idx: Tensor) -> Tensor | None:
        return self.regime[idx]


#: Backward-compatible alias (the pre-Stage-B name).
PanelData = SyntheticPanel


class SyntheticRegimePanel:
    """Generator of regime-labelled synthetic panels of the contract shape."""

    def __init__(self, spec: SyntheticSpec) -> None:
        self.spec = spec.validate()

    # ------------------------------------------------------------ internals
    def _regime_path(self, n_dates: int, g: torch.Generator) -> tuple[Tensor, Tensor | None]:
        k = self.spec.n_regimes
        if self.spec.regime_process == "iid":
            probs = torch.tensor(
                self.spec.marginal if self.spec.marginal else [1.0 / k] * k
            )
            path = torch.multinomial(
                probs.expand(n_dates, k), num_samples=1, replacement=True, generator=g
            ).squeeze(1)
            return path, None
        # markov: sticky chain, off-diagonal mass spread evenly
        stay = self.spec.transition_stay
        off = (1.0 - stay) / (k - 1)
        trans = torch.full((k, k), off)
        trans.fill_diagonal_(stay)
        path = torch.empty(n_dates, dtype=torch.long)
        path[0] = torch.randint(k, (1,), generator=g)
        for t in range(1, n_dates):
            path[t] = torch.multinomial(trans[path[t - 1]], 1, generator=g)
        return path, trans

    def _betas(self, g: torch.Generator) -> Tensor:
        k, d = self.spec.n_regimes, self.spec.d_snap
        if k == 2:
            v = torch.randn(d, generator=g)
            v = self.spec.beta_scale * v / v.norm()
            return torch.stack([v, -v])  # sign-flipped: the ledger's story
        b = torch.randn(k, d, generator=g)
        return self.spec.beta_scale * b / b.norm(dim=1, keepdim=True)

    # ------------------------------------------------------------ interface
    def generate(self, n_dates: int, n_entities: int) -> SyntheticPanel:
        s = self.spec
        g = torch.Generator().manual_seed(s.seed)
        regime_by_date, trans = self._regime_path(n_dates, g)
        betas = self._betas(g)
        vols = torch.tensor(s.vol_levels)

        n = n_dates * n_entities
        date = torch.arange(n_dates).repeat_interleave(n_entities)
        entity = torch.arange(n_entities).repeat(n_dates)
        regime = regime_by_date[date]

        x_seq = torch.randn(n, s.seq_len, s.d_seq, generator=g)
        x_seq[:, :, 0] *= vols[regime].view(n, 1)  # regime-scaled vol channel
        x_snap = torch.randn(n, s.d_snap, generator=g)
        y_clean = (x_snap * betas[regime]).sum(dim=1)
        y = y_clean + s.noise_std * torch.randn(n, generator=g)

        schema = FeatureSchema(
            sequence_features=tuple(f"seq_{i}" for i in range(s.d_seq)),
            snapshot_features=tuple(f"snap_{i}" for i in range(s.d_snap)),
            target="y_synth",
        )
        return SyntheticPanel(
            x_seq=x_seq.float(),
            x_snap=x_snap.float(),
            y=y.float(),
            date=date,
            entity=entity,
            schema=schema,
            y_clean=y_clean.float(),
            regime=regime,
            betas=betas,
            transition=trans,
            spec=s,
        )
