"""Hyperparameter discipline (handbook III.3 item 4): tune once, freeze, log the pick.

The protocol
------------
Hand-picked hyperparameters are researcher degrees of freedom; tuning on the
test window is leakage. The honest middle path, implemented here:

1. carve a **validation tail** off the *training* window — the last
   ``val_dates`` dates, separated from the inner-train dates by the same
   label-overlap purge as the outer split (a validation tail *is* a one-fold
   walk-forward on the training panel, so it reuses that exact tested code);
2. score every candidate (× seeds) on the tail through the identical harness
   (:func:`~nec_moe.sweep.run_sweep` with ``n_folds=1``);
3. pick the winner by the **seed-mean** of the chosen metric and **record the
   selection event in the registry** (arm-level aggregate rows under
   ``<tag>.arms``, then ``registry.best`` on them) — tuning is a selection
   with multiplicity, and the multiplicity survives into any later deflation;
4. **freeze**: retrain the winner on the full training window and never touch
   the candidate list again. Tune on the *first* fold's training window (or a
   dedicated pre-test window), pre-register the candidate list, and reuse the
   frozen choice for every fold — per-fold re-tuning multiplies degrees of
   freedom and is deliberately not offered.

Nothing here ever sees the outer test block: the caller passes the *training*
panel (e.g. ``panel.subset_dates(folds[0].train_dates)``), and the tail is
purged inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from .data import Panel
from .evaluation import WalkForwardFold, walk_forward_folds
from .registry import TrialRegistry, trial_provenance
from .sweep import SweepArm, SweepReport, run_sweep

__all__ = ["validation_tail", "TuneResult", "tune"]


def validation_tail(
    date: Tensor,
    *,
    val_dates: int,
    purge_dates: int,
    min_train_dates: int = 1,
) -> WalkForwardFold:
    """The inner split: last ``val_dates`` dates as the (purged) validation tail.

    Returns a :class:`WalkForwardFold` whose ``test_dates`` *are* the
    validation tail — the arithmetic (and its tests) are the outer split's,
    applied inside the training window.
    """
    [fold] = walk_forward_folds(
        date,
        n_folds=1,
        test_dates_per_fold=val_dates,
        purge_dates=purge_dates,
        min_train_dates=min_train_dates,
    )
    return fold


@dataclass(frozen=True)
class TuneResult:
    """The frozen outcome of one tuning pass."""

    winner: str  # arm name to carry forward, frozen
    metric: str
    mode: str  # "max" | "min"
    scores: dict[str, float]  # arm -> seed-mean of the metric on the tail
    n_candidates: int  # the selection multiplicity (also logged)
    report: SweepReport  # the full per-arm summary, for the appendix table


def tune(
    train_panel: Panel,
    candidates: Sequence[SweepArm],
    *,
    seeds: Sequence[int],
    registry: TrialRegistry,
    tag: str,
    steps: int,
    val_dates: int,
    purge_dates: int,
    metric: str = "mean_ic",
    mode: str = "max",
    verbose: bool = True,
    hac_lags: int | None = None,
    hac_kernel: str = "uniform",
) -> TuneResult:
    """Score ``candidates`` × ``seeds`` on the training window's validation tail.

    ``train_panel`` must be a *training* window (never the full panel with the
    outer test blocks still inside — that would tune on the test period).
    ``purge_dates`` must equal the label horizon, exactly as in the outer
    split. The pick is by seed-mean of ``metric`` (``mode="min"`` for NLL);
    per-run trials land under ``tag``, arm-level aggregates under
    ``<tag>.arms``, and the selection event under ``<tag>.arms#selection``.
    """
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
    if len(candidates) < 2:
        raise ValueError(
            "tuning needs >= 2 candidates — with one there is no selection "
            "(and nothing to log)"
        )

    report = run_sweep(
        train_panel,
        candidates,
        seeds=seeds,
        registry=registry,
        tag=tag,
        steps=steps,
        n_folds=1,  # the single "test block" IS the validation tail
        test_dates_per_fold=val_dates,
        purge_dates=purge_dates,
        verbose=verbose,
        hac_lags=hac_lags,
        hac_kernel=hac_kernel,
    )

    missing = [a.arm for a in report.arms if metric not in a.mean]
    if missing:
        raise ValueError(
            f"metric {metric!r} missing for candidates {missing}; available: "
            f"{sorted(report.arms[0].mean)}"
        )
    # arm-level aggregate rows -> the registry, so best() records the pick
    # with the candidate count attached (selection provenance)
    arms_tag = f"{tag}.arms"
    records = {c.name: (c.config_record or {}) for c in candidates}
    for a in report.arms:
        rec = records.get(a.arm, {})
        registry.log(
            arms_tag,
            {metric: a.mean[metric], f"{metric}_std": a.std[metric]},
            config={
                "arm": a.arm,
                "n_seeds": a.n_seeds,
                "objective": rec.get("objective"),
                "correction_penalty_weight": rec.get("correction_penalty_weight"),
                **trial_provenance(train_panel),
            },
            notes=f"tuning aggregate over {a.n_seeds} seeds",
        )
    pick = registry.best(arms_tag, metric, mode=mode)

    scores = {a.arm: a.mean[metric] for a in report.arms}
    winner = pick.config["arm"]
    if verbose:
        print(
            f"[tune:{tag}] winner={winner!r} by {mode} {metric} "
            f"over {len(candidates)} candidates x {len(list(seeds))} seeds: "
            f"{ {k: round(v, 4) for k, v in scores.items()} }"
        )
    return TuneResult(
        winner=winner,
        metric=metric,
        mode=mode,
        scores=scores,
        n_candidates=len(candidates),
        report=report,
    )
