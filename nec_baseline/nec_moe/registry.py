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

__all__ = [
    "PROVENANCE_KEYS", "TrialRecord", "TrialRegistry", "gate_weight_of", "panel_feature_set",
    "panel_target_kind", "trial_provenance", "scheme_provenance", "fold_provenance",
    "SCHEME_KEYS",
]

_SELECTION_SUFFIX = "#selection"

#: Config keys every logged trial must carry (see the module docstring).
PROVENANCE_KEYS: tuple[str, ...] = (
    "data_source", "post_delisting_return", "hidden_init", "portfolio_scheme",
    "target_kind", "gate_weight", "feature_set", "crsp_stock_file", "compustat_release",
    "sector_source",
    # brief 09 I.1, the training scheme: the folds (scheme, pilot or main,
    # the test years), the sample's first and last date, the three memories,
    # the gate's estimator, and the pilot selection the run was locked to
    "fold_scheme", "protocol", "test_years", "sample_start", "sample_end",
    "base_decay_half_life_days", "expert_decay", "expert_decay_half_life",
    "gate_memory", "gate_half_life", "gate_fit_backend", "pilot_selection_hash",
)

#: The model-config part of the brief 09 keys (:func:`scheme_provenance`),
#: named as the ``Experiment`` fields and the pilot's selection file name them.
SCHEME_KEYS: tuple[str, ...] = (
    "base_decay_half_life_days", "expert_decay", "expert_decay_half_life",
    "gate_memory", "gate_half_life", "gate_fit_backend",
)

#: Target-name prefixes of the Stage B target kinds (``StageBSpec.target``),
#: longest first so ``fwd_ret_`` cannot shadow the others.
_TARGET_PREFIXES: tuple[tuple[str, str], ...] = (
    ("fwd_resid_ret_", "residual"),
    ("fwd_mn_ret_", "market_neutral"),
    ("fwd_ret_", "raw"),
)


def panel_target_kind(panel: Any) -> str | None:
    """The panel's ``target_kind`` (Q25): from its build metadata, else from
    its target's name; ``None`` for a target that is neither (synthetic)."""
    metadata = getattr(panel, "metadata", None) or {}
    if metadata.get("target_kind") is not None:
        return str(metadata["target_kind"])
    schema = getattr(panel, "schema", None)
    target = str(getattr(schema, "target", ""))
    for prefix, kind in _TARGET_PREFIXES:
        if target.startswith(prefix):
            return kind
    return None


def panel_feature_set(panel: Any) -> str | None:
    """The panel's ``feature_set`` (Q26): from its build metadata, else from
    its snapshot columns (the legacy 14 or the Q26 57); ``None`` otherwise
    (synthetic panels)."""
    from .characteristics import Q26_FEATURES
    from .features import SNAPSHOT_FEATURES

    metadata = getattr(panel, "metadata", None) or {}
    if metadata.get("feature_set") is not None:
        return str(metadata["feature_set"])
    names = tuple(getattr(getattr(panel, "schema", None), "snapshot_features", ()))
    if names == SNAPSHOT_FEATURES:
        return "legacy14"
    if names == Q26_FEATURES:
        return "q26"
    return None


def gate_weight_of(cfg: Any) -> str | None:
    """The fitted gate's ``gate_weight`` (Q27) for a model config, or ``None``
    when the model has no Hamilton gate (other priors, baselines)."""
    prior = getattr(cfg, "prior", None)
    if getattr(prior, "kind", None) != "markov":
        return None
    return str(cfg.markov_gate.gate_weight)


def scheme_provenance(cfg: Any) -> dict[str, Any]:
    """The memories and the gate estimator of a model config (brief 09 I.1):
    ``None`` where a key does not apply (no base, no regime-clock decay, no
    Hamilton gate, or no config at all, e.g. a baseline model)."""
    out: dict[str, Any] = dict.fromkeys(SCHEME_KEYS)
    if cfg is None:
        return out
    base, train = getattr(cfg, "base", None), getattr(cfg, "train", None)
    if base is not None and getattr(base, "enabled", False):
        out["base_decay_half_life_days"] = base.decay_half_life_days
    if train is not None:
        out["expert_decay"] = train.expert_decay
        if train.expert_decay != "none":
            out["expert_decay_half_life"] = train.expert_decay_half_life
    if getattr(getattr(cfg, "prior", None), "kind", None) == "markov":
        mg = cfg.markov_gate
        out["gate_memory"] = mg.memory
        out["gate_half_life"] = mg.gate_half_life
        out["gate_fit_backend"] = mg.fit_backend
    return out


def fold_provenance(folds: Any) -> dict[str, Any]:
    """The folds' scheme, protocol and test years (brief 09 I.1). Main and
    pilot folds are calendar years; count folds have no test year."""
    folds = list(folds or ())
    kinds = {getattr(f, "kind", "count") for f in folds}
    years = sorted({int(f.test_year) for f in folds if getattr(f, "test_year", None) is not None})
    return {
        "fold_scheme": None if not folds else (
            "count" if kinds == {"count"} else "calendar_year"),
        "protocol": None if not folds else ("pilot" if "pilot" in kinds else "main"),
        "test_years": years or None,
    }


def _sample_window(panel: Any) -> tuple[str | None, str | None]:
    labels = getattr(panel, "date_labels", None)
    date = getattr(panel, "date", None)
    if not labels or date is None or len(date) == 0:
        return None, None
    return str(labels[int(date.min())]), str(labels[int(date.max())])


def trial_provenance(
    panel: Any, cfg: Any = None, portfolio_scheme: str | None = None, folds: Any = None
) -> dict[str, Any]:
    """The provenance keys of a trial run on ``panel`` with model config ``cfg``.

    ``data_source`` and, from the panel's build metadata, the post-delisting
    fill (``None`` on panels without one, such as synthetic panels); the
    experts' ``hidden_init`` from ``cfg`` (``None`` without a config, e.g. for
    a baseline model, which has no experts); the long-short book's
    ``portfolio_scheme`` the run was configured with (audit E-3); the
    panel's ``target_kind`` (:func:`panel_target_kind`, brief 08 A); the
    gate's ``gate_weight`` from ``cfg`` (:func:`gate_weight_of`, brief 08 B);
    the panel's ``feature_set`` (:func:`panel_feature_set`, brief 08 D); and,
    from the build metadata, the CRSP stock file, the Compustat release and
    the sector source (``"gics"``) (brief 08 E.1); the folds' scheme,
    protocol and test years (:func:`fold_provenance` of ``folds``), the
    panel's first and last date, and the model's memories and gate estimator
    (:func:`scheme_provenance`) (brief 09 I.1). ``pilot_selection_hash`` is
    ``None`` here; a run under the pre-registration lock merges its own in.
    """
    metadata = getattr(panel, "metadata", None) or {}
    experts = getattr(cfg, "experts", None)
    start, end = _sample_window(panel)
    return {
        "data_source": getattr(panel, "data_source", "unspecified"),
        "post_delisting_return": metadata.get("post_delisting_return"),
        "hidden_init": getattr(experts, "hidden_init", None),
        "portfolio_scheme": portfolio_scheme,
        "target_kind": panel_target_kind(panel),
        "gate_weight": gate_weight_of(cfg),
        "feature_set": panel_feature_set(panel),
        # the data layer (brief 08 E.1): which CRSP stock file, which
        # CRSP/Compustat Merged release, which sector scheme; None where the
        # panel's build did not record one (synthetic, pre-brief-08 files)
        "crsp_stock_file": metadata.get("crsp_stock_file"),
        "compustat_release": metadata.get("compustat_release"),
        "sector_source": metadata.get("sector_source"),
        # brief 09 I.1: the folds (scheme, pilot or main, test years), the
        # panel's first and last date, the memories and the gate estimator;
        # the pilot-selection hash is the run's, merged in by the caller
        **fold_provenance(folds),
        "sample_start": start,
        "sample_end": end,
        **scheme_provenance(cfg),
        "pilot_selection_hash": None,
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
        self._append(rec)
        return rec

    def _append(self, rec: TrialRecord) -> None:
        """Append one row by rewriting the file through a temporary file and a
        rename, so a crash mid-write never leaves a half-written row (brief 07
        C.4); the registry is small enough for that to be cheap."""
        from .runstore import atomic_write_text

        existing = self.path.read_text() if self.path.exists() else ""
        atomic_write_text(self.path, existing + json.dumps(asdict(rec)) + "\n")

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
        self._append(rec)
        return pick

    def selection_events(self, tag: str) -> list[TrialRecord]:
        return [
            r for r in self._read_all() if r.tag == f"{tag}{_SELECTION_SUFFIX}"
        ]
