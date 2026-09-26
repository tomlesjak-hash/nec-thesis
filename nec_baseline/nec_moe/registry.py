"""Trial registry: every fitted/evaluated configuration is a logged trial (Module 5).

"Best of 20 initializations" is a selection event like any other (design doc,
Module 8 failure-mode 3; Module 5's multiple-testing discipline): the moment a
result is *chosen*, the number of candidates it was chosen from becomes part of
its statistical meaning. This registry makes that number un-losable:

- **append-only JSONL on disk** — one line per trial (timestamp, tag, config,
  seed, metrics, notes); restarts, seeds, K values, prior/floor settings all
  land here, as the design doc's reporting clauses require;
- ``n_trials(tag)`` is the ``N`` that feeds the deflated Sharpe ratio
  (:mod:`nec_moe.multiple_testing`) and the Bonferroni/BH families;
- ``best(...)`` does not silently pick — it **records the selection event**
  (as a trial row tagged ``<tag>#selection``) with the candidate count baked in.

Tags group trials into the families over which corrections are computed, e.g.
one tag per experiment sweep ("routing_sweep_v1").

Every row carries its **provenance** in ``config`` (:data:`PROVENANCE_KEYS`):
the panel's ``data_source`` and the construction switches a result depends on
(brief 06 A.6). :meth:`TrialRegistry.log` refuses a row without them, so no
result can be mistaken for one from another source or built under another
setting. ``None`` means "not applicable", e.g. ``post_delisting_return`` on a
synthetic panel that has no delistings.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["PROVENANCE_KEYS", "TrialRecord", "TrialRegistry", "trial_provenance"]

_SELECTION_SUFFIX = "#selection"

#: Config keys every logged trial must carry (see the module docstring).
PROVENANCE_KEYS: tuple[str, ...] = ("data_source", "post_delisting_return", "hidden_init")


def trial_provenance(panel: Any, cfg: Any = None) -> dict[str, Any]:
    """The provenance keys of a trial run on ``panel`` with model config ``cfg``.

    ``data_source`` and, from the panel's build metadata, the post-delisting
    fill (``None`` on panels without one, such as synthetic panels); and the
    experts' ``hidden_init`` from ``cfg`` (``None`` without a config, e.g. for
    a baseline model, which has no experts).
    """
    metadata = getattr(panel, "metadata", None) or {}
    experts = getattr(cfg, "experts", None)
    return {
        "data_source": getattr(panel, "data_source", "unspecified"),
        "post_delisting_return": metadata.get("post_delisting_return"),
        "hidden_init": getattr(experts, "hidden_init", None),
    }


@dataclass(frozen=True)
class TrialRecord:
    """One logged trial."""

    trial_id: int
    timestamp: str  # UTC ISO-8601
    tag: str  # experiment family; corrections run per tag
    metrics: dict[str, float]
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    notes: str = ""


class TrialRegistry:
    """Append-only JSONL trial log; safe to reopen across sessions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ io
    def _read_all(self) -> list[TrialRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                records.append(TrialRecord(**json.loads(line)))
        return records

    def log(
        self,
        tag: str,
        metrics: dict[str, float],
        *,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        notes: str = "",
    ) -> TrialRecord:
        """Append one trial. ``metrics`` values must be finite floats or None-free,
        and ``config`` must carry every key of :data:`PROVENANCE_KEYS`."""
        missing = [k for k in PROVENANCE_KEYS if k not in (config or {})]
        if missing:
            raise ValueError(
                f"trial config lacks the provenance keys {missing}; add "
                "trial_provenance(panel) (None where a key does not apply)"
            )
        if _SELECTION_SUFFIX in tag:
            raise ValueError(
                f"tag must not contain {_SELECTION_SUFFIX!r} (reserved for "
                "selection events recorded by best())"
            )
        clean = {k: float(v) for k, v in metrics.items()}
        for k, v in clean.items():
            if not math.isfinite(v):
                raise ValueError(f"metric {k!r} is not finite: {v}")
        rec = TrialRecord(
            trial_id=len(self._read_all()),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            tag=tag,
            metrics=clean,
            config=config or {},
            seed=seed,
            notes=notes,
        )
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    # --------------------------------------------------------------- queries
    def trials(self, tag: str | None = None) -> list[TrialRecord]:
        """All trials, or those of one family (selection events excluded)."""
        recs = [r for r in self._read_all() if not r.tag.endswith(_SELECTION_SUFFIX)]
        return recs if tag is None else [r for r in recs if r.tag == tag]

    def n_trials(self, tag: str | None = None) -> int:
        """The selection multiplicity: the N for deflated-Sharpe / corrections."""
        return len(self.trials(tag))

    def metric_values(self, tag: str, metric: str) -> list[float]:
        vals = [r.metrics[metric] for r in self.trials(tag) if metric in r.metrics]
        if not vals:
            raise KeyError(f"no trials under tag {tag!r} carry metric {metric!r}")
        return vals

    # -------------------------------------------------------------- selection
    def best(self, tag: str, metric: str, *, mode: str = "max") -> TrialRecord:
        """Pick the best trial of a family — and log that a pick happened.

        The returned record is the winner; a ``<tag>#selection`` row is
        appended recording the candidate count and the winner's id, so the
        multiplicity of the selection survives into any later analysis.
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        candidates = [r for r in self.trials(tag) if metric in r.metrics]
        if not candidates:
            raise KeyError(f"no trials under tag {tag!r} carry metric {metric!r}")
        pick = (max if mode == "max" else min)(
            candidates, key=lambda r: r.metrics[metric]
        )
        rec = TrialRecord(
            trial_id=len(self._read_all()),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            tag=f"{tag}{_SELECTION_SUFFIX}",
            metrics={metric: pick.metrics[metric]},
            config={
                "selected_trial_id": pick.trial_id,
                "n_candidates": len(candidates),
                "mode": mode,
                # the winner's objective travels with the selection event, so
                # the selection row is interpretable without a join
                "objective": pick.config.get("objective"),
                "correction_penalty_weight": pick.config.get(
                    "correction_penalty_weight"
                ),
                # and so does its provenance: a selection row is a registry
                # entry like any other
                **{k: pick.config.get(k) for k in PROVENANCE_KEYS},
            },
            notes=f"selection event: best {metric!r} of {len(candidates)}",
        )
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return pick

    def selection_events(self, tag: str) -> list[TrialRecord]:
        return [
            r for r in self._read_all() if r.tag == f"{tag}{_SELECTION_SUFFIX}"
        ]
