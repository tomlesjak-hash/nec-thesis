"""Multi-seed sweep runner: the missing protocol layer (handbook III.3 item 1).

"Nothing is a result until it has a seed std." This module turns that rule into
the default workflow: a sweep is a set of named **arms** (NEC configurations
and/or baselines) crossed with a **seed grid**, every (arm, seed) run scored by
the identical walk-forward harness, every trial logged to the registry, and the
summary reported as mean ± std over seeds per arm.

Design points:

- **One grading path.** NEC arms go through ``walk_forward_evaluate``, baseline
  arms through ``walk_forward_evaluate_baseline`` — which share the fold
  accumulator, so every row of the resulting table is graded by the same code.
- **Seeds vary everything they should.** For NEC arms the seed is planted both
  in the global RNG (encoder init) and in ``TrainConfig.seed`` (per-expert
  diversification, minibatch shuffling) via ``dataclasses.replace`` — two seeds
  give genuinely different initializations, deterministic per seed.
- **The registry is the ledger.** Each (arm, seed) run is one trial row whose
  ``config`` records the arm (and the full ``NECConfig`` dict for NEC arms), so
  any number in the report is reconstructible. ``n_trials(tag)`` afterwards is
  the selection multiplicity for the deflated Sharpe.
- **The claim family is corrected across arms, not seeds.** Seed replicates of
  one arm are *not* independent hypotheses — they are repeated measurements of
  the same one. Each arm's p-values are combined by their **median** (a robust
  replicate summary; pre-register the choice), and Benjamini–Hochberg runs over
  the per-arm medians (:func:`corrected_claims`).

Deterministic baselines (e.g. ridge) simply show a seed std of 0 — itself
informative in the table.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from .baselines import BaselineModel
from .config import NECConfig
from .data import Panel
from .evaluation import WalkForwardResult, walk_forward_evaluate, walk_forward_evaluate_baseline
from .model import NECModel
from .multiple_testing import benjamini_hochberg, ic_pvalue
from .registry import TrialRegistry
from .train import Trainer

__all__ = [
    "SweepArm",
    "nec_arm",
    "baseline_arm",
    "ArmSummary",
    "SweepReport",
    "run_sweep",
    "corrected_claims",
]


@dataclass(frozen=True)
class SweepArm:
    """One named configuration in a sweep.

    Exactly one of ``build_trainer`` / ``build_baseline`` is set; both builders
    take a seed and must return a **fresh** object on every call (the harness
    calls them once per fold — fit-once-per-window).
    """

    name: str
    build_trainer: Callable[[int], Trainer] | None = None
    build_baseline: Callable[[int], BaselineModel] | None = None
    warmstart_key: Callable[[Panel], Tensor] | None = None  # NEC arms only
    config_record: dict | None = None  # logged to the registry per trial

    def __post_init__(self) -> None:
        if (self.build_trainer is None) == (self.build_baseline is None):
            raise ValueError(
                f"arm {self.name!r}: set exactly one of build_trainer / "
                "build_baseline"
            )
        if self.build_baseline is not None and self.warmstart_key is not None:
            raise ValueError(
                f"arm {self.name!r}: warmstart_key applies to NEC arms only"
            )


def nec_arm(
    name: str,
    cfg: NECConfig,
    *,
    warmstart_key: Callable[[Panel], Tensor] | None = None,
) -> SweepArm:
    """Standard NEC arm: fresh, fully re-seeded model per (fold, seed).

    The seed is planted in the global RNG (encoder/gate init draws) *and* in
    ``TrainConfig.seed`` (expert diversification, shuffling), so seeds produce
    genuinely independent initializations while each seed stays deterministic.
    """

    def build(seed: int) -> Trainer:
        torch.manual_seed(seed)
        seeded = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, seed=seed)
        )
        return Trainer(NECModel(seeded))

    return SweepArm(
        name=name,
        build_trainer=build,
        warmstart_key=warmstart_key,
        config_record={"arm": name, "nec_config": cfg.to_dict()},
    )


def baseline_arm(
    name: str, build: Callable[[int], BaselineModel]
) -> SweepArm:
    """Baseline arm; ``build(seed)`` returns a fresh model (seed may be unused
    by deterministic baselines — their seed std will simply be 0)."""
    return SweepArm(
        name=name, build_baseline=build, config_record={"arm": name}
    )


# --------------------------------------------------------------------------- #
# Running and summarizing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    n_seeds: int
    mean: dict[str, float]  # metric -> mean over seeds
    std: dict[str, float]  # metric -> sample std over seeds (0.0 when n=1)
    median_p: float  # replicate-median of the per-run IC p-values


@dataclass(frozen=True)
class SweepReport:
    tag: str
    arms: tuple[ArmSummary, ...]

    def to_frame(self) -> pd.DataFrame:
        """Arms x metrics table: ``<metric>`` (mean) and ``<metric>_std`` columns."""
        rows = []
        for a in self.arms:
            row: dict[str, float] = {}
            for k in sorted(a.mean):
                row[k] = a.mean[k]
                row[f"{k}_std"] = a.std[k]
            row["median_p"] = a.median_p
            row["n_seeds"] = a.n_seeds
            rows.append(row)
        return pd.DataFrame(rows, index=[a.arm for a in self.arms])


def _metrics_from_result(res: WalkForwardResult) -> dict[str, float]:
    m = {
        "mean_ic": res.pooled_ic.mean_ic,
        "icir": res.pooled_ic.icir,
        "t_stat": res.pooled_ic.t_stat,
        "p": ic_pvalue(res.pooled_ic),
        "nll": res.mean_fold_nll,
    }
    if res.pooled_portfolio is not None:
        m["net_ir"] = res.pooled_portfolio.ir_net
        m["mean_net"] = res.pooled_portfolio.mean_net
        m["mean_turnover"] = res.pooled_portfolio.mean_turnover
    return m


def run_sweep(
    panel: Panel,
    arms: Sequence[SweepArm],
    *,
    seeds: Sequence[int],
    registry: TrialRegistry,
    tag: str,
    steps: int,
    n_folds: int,
    test_dates_per_fold: int,
    purge_dates: int,
    min_train_dates: int = 1,
    backtest_quantiles: int | None = None,
    cost_rate: float = 0.0,
    verbose: bool = True,
    resume_dir: str | Path | None = None,
) -> SweepReport:
    """Run every arm over every seed through the shared harness; log; summarize.

    Split parameters are shared across all arms by construction — the folds are
    byte-identical, which is half of what makes the table a fair comparison
    (the shared fold accumulator is the other half).

    **Resume** (``resume_dir``): a completed (arm, seed) run already has a
    registry row under this tag — it is skipped and its logged metrics reused
    (JSON round-trips floats exactly, so the summary is unchanged). An
    in-progress NEC run resumes at fold granularity via
    :func:`walk_forward_evaluate`'s per-run subdirectory
    (``<resume_dir>/<arm>_seed<seed>/``), which composes with the trainer's
    own mid-fit checkpoints. Baseline arms are seconds-fast and simply rerun.
    Resume with the same settings — the registry row is matched on
    (arm, seed) only.
    """
    if not arms or not seeds:
        raise ValueError("need at least one arm and one seed")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names: {sorted(names)}")

    resume = Path(resume_dir) if resume_dir is not None else None
    completed: dict[tuple[str, int], dict[str, float]] = {}
    if resume is not None:
        resume.mkdir(parents=True, exist_ok=True)
        for rec in registry.trials(tag):
            arm_name = rec.config.get("arm")
            if arm_name is not None and rec.seed is not None:
                completed[(arm_name, rec.seed)] = rec.metrics

    per_arm: dict[str, list[dict[str, float]]] = {a.name: [] for a in arms}
    for arm in arms:
        for seed in seeds:
            if (arm.name, seed) in completed:
                per_arm[arm.name].append(completed[(arm.name, seed)])
                if verbose:
                    print(
                        f"[sweep:{tag}] {arm.name} seed={seed}: already in the "
                        "registry — skipped (resume)"
                    )
                continue
            build_trainer, build_baseline = arm.build_trainer, arm.build_baseline
            if build_trainer is not None:
                res = walk_forward_evaluate(
                    panel,
                    lambda: build_trainer(seed),  # noqa: B023 — consumed this iteration
                    steps=steps,
                    warmstart_key=arm.warmstart_key,
                    n_folds=n_folds,
                    test_dates_per_fold=test_dates_per_fold,
                    purge_dates=purge_dates,
                    min_train_dates=min_train_dates,
                    backtest_quantiles=backtest_quantiles,
                    cost_rate=cost_rate,
                    resume_dir=(
                        resume / f"{arm.name}_seed{seed}"
                        if resume is not None
                        else None
                    ),
                )
            else:
                assert build_baseline is not None
                res = walk_forward_evaluate_baseline(
                    panel,
                    lambda: build_baseline(seed),  # noqa: B023 — consumed this iteration
                    n_folds=n_folds,
                    test_dates_per_fold=test_dates_per_fold,
                    purge_dates=purge_dates,
                    min_train_dates=min_train_dates,
                    backtest_quantiles=backtest_quantiles,
                    cost_rate=cost_rate,
                )
            metrics = _metrics_from_result(res)
            registry.log(tag, metrics, config=arm.config_record, seed=seed)
            per_arm[arm.name].append(metrics)
            if verbose:
                print(
                    f"[sweep:{tag}] {arm.name} seed={seed}: "
                    f"IC={metrics['mean_ic']:+.3f} p={metrics['p']:.3f} "
                    f"NLL={metrics['nll']:.3f}"
                )

    summaries = []
    for arm in arms:
        runs = per_arm[arm.name]
        keys = sorted(runs[0])
        mean = {k: statistics.fmean(r[k] for r in runs) for k in keys}
        std = {
            k: statistics.stdev([r[k] for r in runs]) if len(runs) > 1 else 0.0
            for k in keys
        }
        summaries.append(
            ArmSummary(
                arm=arm.name,
                n_seeds=len(runs),
                mean=mean,
                std=std,
                median_p=statistics.median(r["p"] for r in runs),
            )
        )
    return SweepReport(tag=tag, arms=tuple(summaries))


def corrected_claims(
    report: SweepReport, alpha: float = 0.10
) -> Mapping[str, tuple[bool, float]]:
    """BH-FDR over the arms' replicate-median p-values.

    -> ``{arm: (reject, q_value)}``. The family is the *arms* (the hypotheses);
    seeds are replicates and enter only through the median. Pre-register the
    median choice along with alpha.
    """
    arms = [a.arm for a in report.arms]
    reject, qvals = benjamini_hochberg(
        [a.median_p for a in report.arms], alpha=alpha
    )
    return {arm: (r, q) for arm, r, q in zip(arms, reject, qvals, strict=True)}
