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
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from .base import BaseCache
from .baselines import BaselineModel
from .config import NECConfig
from .data import Panel
from .evaluation import (
    WalkForwardFold,
    WalkForwardResult,
    fold_metrics_frame,
    walk_forward_evaluate,
    walk_forward_evaluate_baseline,
    walk_forward_folds,
)
from .model import NECModel
from .multiple_testing import benjamini_hochberg, ic_pvalue
from .registry import TrialRegistry, gate_weight_of, scheme_provenance, trial_provenance
from .runstore import Run
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
        config_record={
            "arm": name,
            # beside each other, deliberately: the penalty weight is not
            # comparable across objectives, so neither means anything alone
            "objective": cfg.train.objective,
            "correction_penalty_weight": cfg.train.correction_penalty_weight,
            # how the experts started (brief 06 D): not decided, so on record
            "hidden_init": cfg.experts.hidden_init,
            # the Hamilton gate's weight (Q27); None for every other prior
            "gate_weight": gate_weight_of(cfg),
            # the memories and the gate's estimator (brief 09 I.1)
            **scheme_provenance(cfg),
            "nec_config": cfg.to_dict(),
        },
    )


def baseline_arm(
    name: str, build: Callable[[int], BaselineModel]
) -> SweepArm:
    """Baseline arm; ``build(seed)`` returns a fresh model (seed may be unused
    by deterministic baselines — their seed std will simply be 0)."""
    # A baseline is not trained against the NEC objective at all, so the keys
    # are present (every row carries them) but explicitly not applicable.
    return SweepArm(
        name=name,
        build_baseline=build,
        config_record={
            "arm": name, "objective": None, "correction_penalty_weight": None,
            "hidden_init": None,
        },
    )


# --------------------------------------------------------------------------- #
# Running and summarizing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    n_seeds: int
    mean: dict[str, float]  # metric -> mean over the seeds that report it
    std: dict[str, float]  # metric -> sample std over those seeds (0.0 when n=1)
    median_p: float  # replicate-median of the per-run IC p-values
    # metrics that only some seeds report (e.g. an empty quantile bin of the
    # Q24 diagnostic, whose expert labels follow each seed's canonical order):
    # metric -> how many seeds the mean and std are over. Every key reported
    # by any seed is summarised; none is dropped
    n_reporting: dict[str, int] = field(default_factory=dict)


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
                if k in a.n_reporting:
                    row[f"{k}_n_seeds"] = a.n_reporting[k]
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
        # which standard error t_stat and p rest on (audit E-2)
        "ic_hac_lags": float(res.pooled_ic.hac_lags),
    }
    if res.pooled_portfolio is not None:
        m["net_ir"] = res.pooled_portfolio.ir_net
        m["mean_net"] = res.pooled_portfolio.mean_net
        m["mean_turnover"] = res.pooled_portfolio.mean_turnover
    # The parameter-count confound, logged beside every result it could
    # explain: under the HMM prior the encoder and gate head are dead
    # parameters, under the soft prior they are live, so prior kinds are only
    # comparable with this number in view. Every fold shares one architecture;
    # the first audited fold speaks for the arm.
    live = next(
        (f.live_param_count for f in res.folds if f.live_param_count is not None), None
    )
    if live is not None:
        m["live_param_count"] = float(live)
    # Residual design (brief 02 §5): the base's own out-of-sample score beside
    # every result, the improvement over it as the primary quantity, and the
    # correction magnitude actually used.
    scored = [f for f in res.folds if f.base_ic is not None]
    if scored:
        m["base_mean_ic"] = statistics.fmean(f.base_ic.mean_ic for f in scored)  # type: ignore[union-attr]
        m["base_icir"] = statistics.fmean(f.base_ic.icir for f in scored)  # type: ignore[union-attr]
        m["ic_improvement"] = statistics.fmean(
            f.ic_improvement for f in scored if f.ic_improvement is not None
        )
        with_nll = [f for f in scored if f.base_nll is not None]
        if with_nll:
            m["base_nll"] = statistics.fmean(f.base_nll for f in with_nll)  # type: ignore[misc]
            m["nll_improvement"] = statistics.fmean(
                f.nll_improvement for f in with_nll if f.nll_improvement is not None
            )
        # brief 06 C: the old improvement split into the variance gain (the
        # experts' noise scales under the gate) and the corrections' part;
        # all three beside each other, neither privileged
        with_single = [f for f in with_nll if f.base_single_nll is not None]
        if with_single:
            m["base_single_nll"] = statistics.fmean(
                f.base_single_nll for f in with_single  # type: ignore[misc]
            )
            m["nll_variance_gain"] = statistics.fmean(
                f.nll_variance_gain for f in with_single  # type: ignore[misc]
            )
            m["nll_total_gain"] = statistics.fmean(
                f.nll_total_gain for f in with_single  # type: ignore[misc]
            )
        with_pf = [f for f in scored if f.base_portfolio is not None]
        if with_pf:
            m["base_net_ir"] = statistics.fmean(f.base_portfolio.ir_net for f in with_pf)  # type: ignore[union-attr]
    # The fitted gate's own summary (brief 03 §4/§5): expected durations, stay
    # probabilities, the multi-start counts. Averaged over folds in canonical
    # regime order, which is what makes the average meaningful at all.
    # training dates the gate could not cover (autoregressive gate, G-2)
    m["gate_excluded_dates"] = statistics.fmean(f.gate_excluded_dates for f in res.folds)
    per_gate: dict[str, list[float]] = {}
    for fold in res.folds:
        for key, value in fold.gate_metrics:
            per_gate.setdefault(key, []).append(value)
    for key, values in per_gate.items():
        m[key] = statistics.fmean(values)
    per_corr: dict[str, list[float]] = {}
    for fold in res.folds:
        for key, value in fold.correction:
            per_corr.setdefault(key, []).append(value)
    for key, values in per_corr.items():
        m[key] = statistics.fmean(values)
    # Q24 expert-weight diagnostic, averaged over the folds that report each
    # key (canonical expert order, like the persistence keys below)
    per_ew: dict[str, list[float]] = {}
    for fold in res.folds:
        for key, value in fold.expert_weights:
            per_ew.setdefault(key, []).append(value)
    for key, values in per_ew.items():
        m[key] = statistics.fmean(values)
    # Chain persistence, averaged over folds in canonical state order (each
    # fold is an independent refit, so this is a mean of per-window estimates).
    per_key: dict[str, list[float]] = {}
    for fold in res.folds:
        for key, value in fold.persistence:
            per_key.setdefault(key, []).append(value)
    for key, values in per_key.items():
        m[key] = statistics.fmean(values)
    return m


def run_sweep(
    panel: Panel,
    arms: Sequence[SweepArm],
    *,
    seeds: Sequence[int],
    registry: TrialRegistry,
    tag: str,
    steps: int,
    n_folds: int | None = None,
    test_dates_per_fold: int | None = None,
    purge_dates: int,
    folds: list[WalkForwardFold] | None = None,
    min_train_dates: int = 1,
    backtest_quantiles: int | None = None,
    cost_rate: float = 0.0,
    verbose: bool = True,
    resume_dir: str | Path | None = None,
    base_cache: BaseCache | None = None,
    hac_lags: int | None = None,
    hac_kernel: str = "uniform",
    portfolio_scheme: str = "nonoverlapping",
    run: Run | None = None,
    provenance_extra: dict | None = None,
) -> SweepReport:
    """Run every arm over every seed through the shared harness; log; summarize.

    Split parameters are shared across all arms by construction — the folds are
    byte-identical, which is half of what makes the table a fair comparison
    (the shared fold accumulator is the other half). ``folds`` (e.g. the main
    study's calendar-year folds, brief 09 B) replaces the count-based split.

    **Resume** (``resume_dir``): a completed (arm, seed) run already has a
    registry row under this tag — it is skipped and its logged metrics reused
    (JSON round-trips floats exactly, so the summary is unchanged). An
    in-progress NEC run resumes at fold granularity via
    :func:`walk_forward_evaluate`'s per-run subdirectory
    (``<resume_dir>/<arm>_seed<seed>/``), which composes with the trainer's
    own mid-fit checkpoints. Baseline arms are seconds-fast and simply rerun.
    Resume with the same settings — the registry row is matched on
    (arm, seed) only.

    ``hac_lags`` and ``hac_kernel`` set the autocorrelation-consistent standard
    error of every IC t-statistic and hence of every ``p`` the corrected
    claims use; ``hac_lags=None`` means ``horizon - 1`` from the panel's
    target (audit E-2). ``portfolio_scheme`` sets the long-short book
    (audit E-3, :func:`~nec_moe.evaluation.long_short_book`) and is recorded on
    every row.

    **Run store** (brief 07 B): with ``run`` given, ``status.json`` follows
    the current arm and seed, each (arm, seed)'s per-fold and pooled
    summaries go to ``metrics/<arm>_seed<seed>_{folds.csv,pooled.json}``,
    and its per-security predictions to the run's ``Data/derived/runs/``
    folder.

    ``provenance_extra`` is merged into every trial row's config (the run's
    own provenance, e.g. the pilot-selection hash of brief 09 F.3).
    """
    if not arms or not seeds:
        raise ValueError("need at least one arm and one seed")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names: {sorted(names)}")

    # One cache for the whole sweep: the frozen base depends on (base config,
    # window, seed) and *not* on the gate, so every arm of a window and seed
    # shares one fitted base — the amortisation the compute budget assumes,
    # and the stricter comparison (identical floor, not merely a similar one).
    base_cache = base_cache if base_cache is not None else BaseCache()
    resume = Path(resume_dir) if resume_dir is not None else None
    completed: dict[tuple[str, int], dict[str, float]] = {}
    if resume is not None:
        resume.mkdir(parents=True, exist_ok=True)
        for rec in registry.trials(tag):
            arm_name = rec.config.get("arm")
            if arm_name is not None and rec.seed is not None:
                completed[(arm_name, rec.seed)] = rec.metrics

    # the folds every arm runs on, for the trial rows' fold provenance
    # (brief 09 I.1): the given ones, or the count-based split
    run_folds = folds if folds is not None else walk_forward_folds(
        panel.date, n_folds=n_folds or 0, test_dates_per_fold=test_dates_per_fold or 0,
        purge_dates=purge_dates, min_train_dates=min_train_dates,
    )
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
            per_security = (
                run.per_security_dir / f"{arm.name}_seed{seed}" if run is not None else None
            )
            if run is not None:
                run.status(arm=arm.name, seed=seed, fold=None, step=None)
            if build_trainer is not None:
                res = walk_forward_evaluate(
                    panel,
                    lambda: build_trainer(seed),  # noqa: B023 — consumed this iteration
                    steps=steps,
                    warmstart_key=arm.warmstart_key,
                    n_folds=n_folds,
                    test_dates_per_fold=test_dates_per_fold,
                    purge_dates=purge_dates,
                    folds=folds,
                    min_train_dates=min_train_dates,
                    backtest_quantiles=backtest_quantiles,
                    cost_rate=cost_rate,
                    resume_dir=(
                        resume / f"{arm.name}_seed{seed}"
                        if resume is not None
                        else None
                    ),
                    base_cache=base_cache,
                    seed=seed,
                    registry=registry,
                    hac_lags=hac_lags,
                    hac_kernel=hac_kernel,
                    portfolio_scheme=portfolio_scheme,
                    run=run,
                    predictions_dir=per_security,
                    provenance_extra=provenance_extra,
                )
            else:
                assert build_baseline is not None
                res = walk_forward_evaluate_baseline(
                    panel,
                    lambda: build_baseline(seed),  # noqa: B023 — consumed this iteration
                    n_folds=n_folds,
                    test_dates_per_fold=test_dates_per_fold,
                    purge_dates=purge_dates,
                    folds=folds,
                    min_train_dates=min_train_dates,
                    backtest_quantiles=backtest_quantiles,
                    cost_rate=cost_rate,
                    hac_lags=hac_lags,
                    hac_kernel=hac_kernel,
                    portfolio_scheme=portfolio_scheme,
                    predictions_dir=per_security,
                )
            metrics = _metrics_from_result(res)
            if run is not None:  # aggregate summaries only
                run.write_metrics(f"{arm.name}_seed{seed}_folds", fold_metrics_frame(res))
                run.write_metrics(f"{arm.name}_seed{seed}_pooled", metrics)
            registry.log(
                tag, metrics,
                config={**trial_provenance(panel, None, portfolio_scheme, run_folds),
                        **(arm.config_record or {}), **(provenance_extra or {})},
                seed=seed,
            )
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
        keys = sorted(set().union(*runs))
        values = {k: [r[k] for r in runs if k in r] for k in keys}
        mean = {k: statistics.fmean(v) for k, v in values.items()}
        std = {k: statistics.stdev(v) if len(v) > 1 else 0.0 for k, v in values.items()}
        summaries.append(
            ArmSummary(
                arm=arm.name,
                n_seeds=len(runs),
                mean=mean,
                std=std,
                median_p=statistics.median(r["p"] for r in runs),
                n_reporting={k: len(v) for k, v in values.items() if len(v) < len(runs)},
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
