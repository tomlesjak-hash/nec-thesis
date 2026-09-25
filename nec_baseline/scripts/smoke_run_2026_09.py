"""Brief 04 section C: the first end-to-end run on real data.

An **integration run, not an experiment** (C.1). It checks whether the whole
pipeline behaves sensibly when the Hamilton gate meets real returns, and how
long it takes. Nothing it produces may be quoted.

    DIAGNOSTIC ONLY — free data, survivorship biased, not CRSP.

Every setting is fixed in ``SMOKE`` below **before** the run. None of them is
changed in response to what the run shows: record observations, do not tune.
``K = 2`` and ``objective = "mixture_nll"`` are plumbing placeholders, not
decisions (Q18 and Q20 are open).

Arms (C.3), all through the shared walk-forward harness:

1. base only: the frozen ``f0``, scored by the same code as every mixture;
2. residual mixture with the Hamilton gate (``informed_jitter`` starts);
3. residual mixture with a frozen uniform gate: the same capacity and no
   regime information.

One ``BaseCache`` serves every arm, so within a (seed, fold) all three are
measured against the same fitted base, and the script checks that they were.

The C.5 stop conditions are checked as soon as their inputs exist: the gate
conditions on a pre-flight fit of every fold's gate, before any expert trains;
the leakage condition after each arm's run. If one fires, the script writes
the report up to that point, marks it STOPPED, and exits without running the
rest.

Usage (from ``nec_baseline/``)::

    python3.14 scripts/smoke_run_2026_09.py

Outputs: ``Master Thesis/Smoke_Run_2026-09-25.md`` with figures in
``Master Thesis/Smoke_Run_2026-09-25_figures/``; the trial registry and the raw
numbers in ``results/smoke_2026_09/``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_experiment as rx
from nec_moe import (
    BaseCache,
    NECConfig,
    Panel,
    PriorContext,
    TrialRegistry,
    load_vix,
    nec_arm,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe.diagnostics import stationary_distribution
from nec_moe.evaluation import FoldResult, WalkForwardResult
from nec_moe.markov_gate import build_markov_gate

THESIS = ROOT.parent / "Master Thesis"


# =========================================================================== #
#  SETTINGS: fixed before the run (C.1). Not to be edited in response to it.
# =========================================================================== #


def _experiment() -> rx.Experiment:
    """The model/harness settings, through run_experiment's tested config path.

    Everything not listed keeps run_experiment's documented default: base
    (64, 32) MLP, 1000 steps, batch 128, lr 1e-3; experts (64, 32), dropout
    0.05, lr 1e-3, batch 256; sigma_init auto (the base residual's std on the
    first training block); Hamilton gate on French Mkt-RF with the
    ``informed_jitter`` defaults (2 x 2 x 2 centres x 3 draws = 24 starts).
    """
    return rx.Experiment(
        tag="smoke_2026_09",
        mode="evaluate",
        data="panel_file",               # the prebuilt PIT panel, 2015-2024, offline
        panel_file=str(ROOT / "data_cache" / "pit_panel_2015_2024.pt"),
        horizon=5,                       # fwd_ret_5d; also the purge length
        n_experts=2,                     # K = 2: a placeholder, Q18 is open
        prior="markov",                  # replaced per arm below
        objective="mixture_nll",         # the only one registered; Q20 is open
        base_enabled=True,
        correction_mode=True,
        zero_init_head=True,
        freeze_gate=True,
        gate_series="market_excess_return",
        gate_start_scheme="informed_jitter",
        n_folds=3,
        test_dates_per_fold=120,         # about six months per fold
        seeds=(0, 1),
        steps=300,                       # modest: enough to exercise every path
        backtest_quantiles=None,         # not asked for by C.4; fewer numbers to quote
        include_ridge=False,
        include_mlp=False,
        figures=False,
        calibration=False,
    )


@dataclass(frozen=True)
class SmokeSettings:
    experiment: rx.Experiment = field(default_factory=_experiment)
    tag: str = "smoke_2026_09"           # every trial row carries this tag (C.4)
    label: str = "DIAGNOSTIC ONLY — free data, survivorship biased, not CRSP"
    source: str = (
        "free: yfinance daily bars for the S&P 500 point-in-time membership "
        "union (Wikipedia), 2015-2024, prebuilt pit_panel_2015_2024.pt; "
        "departed names missing at the free source (survivorship biased); "
        "gate series Kenneth French daily Mkt-RF; VIX from CBOE. NOT CRSP"
    )
    # arm name -> prior kind; "base_only" is read off the shared base (C.3)
    arms: tuple[tuple[str, str], ...] = (("hamilton", "markov"), ("uniform", "uniform"))
    # C.5 stop conditions
    min_stationary_prob: float = 0.02
    min_expected_duration: float = 2.0   # trading days
    max_ic_improvement: float = 0.05     # mean rank IC over the base
    # reporting
    report_path: Path = THESIS / "Smoke_Run_2026-09-25.md"
    figures_dir: Path = THESIS / "Smoke_Run_2026-09-25_figures"
    results_dir: Path = ROOT / "results" / "smoke_2026_09"
    vix_dir: Path = ROOT / "data_cache"
    correction_quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    hist_bins: int = 80
    figure_dpi: int = 130


SMOKE = SmokeSettings()


# =========================================================================== #
#  Registry: tag + source on every row
# =========================================================================== #


class SmokeRegistry(TrialRegistry):
    """Writes every row under the smoke tag with ``source`` in its config.

    The harness logs the gate's multi-start trials under its own family tag;
    that tag is kept as ``config["family"]`` so the rows stay distinguishable,
    and the running arm/seed is attached from ``context``. The registry file
    format is unchanged: ``source`` lives in the config dict, because the
    registry-level ``source`` field belongs to brief 01 section 7 (the loader
    / CRSP seam), which is outside this brief.
    """

    def __init__(self, path: Path, tag: str, source: str) -> None:
        super().__init__(path)
        self.smoke_tag, self.source = tag, source
        self.context: dict[str, Any] = {}

    def log(self, tag, metrics, *, config=None, seed=None, notes=""):
        cfg = dict(config or {})
        cfg.setdefault("family", tag)
        cfg |= {"source": self.source, **self.context}
        return super().log(self.smoke_tag, metrics, config=cfg, seed=seed, notes=notes)


# =========================================================================== #
#  Stop handling
# =========================================================================== #


class StopRun(Exception):
    """A C.5 stop condition fired: report, do not work around."""


@dataclass
class RunState:
    started: float = field(default_factory=time.perf_counter)
    fold_spans: list[dict[str, Any]] = field(default_factory=list)
    gate: list[dict[str, Any]] = field(default_factory=list)
    gate_probs: dict[int, pd.DataFrame] = field(default_factory=dict)
    runs: dict[tuple[str, int], WalkForwardResult] = field(default_factory=dict)
    trainers: dict[tuple[str, int], list[Any]] = field(default_factory=dict)
    warnings: dict[tuple[str, int], list[str]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    stopped: str | None = None
    sigma_init: float | None = None
    probe_base_s: float | None = None
    base_params: int | None = None
    env: dict[str, str] = field(default_factory=dict)


# =========================================================================== #
#  Gate pre-flight (C.5 gate conditions, before any expert trains)
# =========================================================================== #


def _gate_probabilities(prior, codes: torch.Tensor, k: int) -> np.ndarray:
    """The gate's probabilities for ``codes`` via its public forward (date lookup)."""
    out = prior(torch.zeros(len(codes), k), PriorContext(date=codes))
    return out.log_prior.exp().double().numpy()


def gate_preflight(
    s: SmokeSettings, panel: Panel, folds, cfg: NECConfig, st: RunState
) -> None:
    k = cfg.experts.n_experts
    labels = pd.to_datetime(list(panel.date_labels or ()))
    for fold in folds:
        train = panel.subset_dates(fold.train_dates)
        prior = build_markov_gate(cfg)
        t0 = time.perf_counter()
        try:
            prior.fit(train)
        except RuntimeError as e:
            st.gate.append({"fold": fold.fold, "error": str(e)})
            raise StopRun(f"C.5: the gate failed to converge on fold {fold.fold}: {e}") from e
        fit_s = time.perf_counter() - t0
        # exactly the harness's application (brief 03 section 3.2: the model
        # constructed over train plus test), so the plotted gate is the gate
        # the mixture used
        codes = torch.cat([fold.train_dates, fold.test_dates])
        prior.apply_causal(panel.subset_dates(codes))
        fit = prior.fit_result
        assert fit is not None  # fit() either set it or raised
        stationary = stationary_distribution(torch.from_numpy(fit.transition)).double().numpy()
        probs = _gate_probabilities(prior, codes, k)
        # the same frozen fit applied over the CONTIGUOUS span (purge-gap
        # returns included), only to measure what skipping the gap does to
        # the test block; nothing downstream uses it
        span = build_markov_gate(cfg)
        span.fit(train)
        span.apply_causal(panel.subset_dates(torch.arange(0, int(fold.test_dates.max()) + 1)))
        n_test = len(fold.test_dates)
        gap_diff = np.abs(
            probs[-n_test:, k - 1]
            - _gate_probabilities(span, fold.test_dates, k)[:, k - 1]
        )
        train_max = int(fold.train_dates.max())
        frame = pd.DataFrame(
            {"date": labels[codes.numpy()], "p_high": probs[:, k - 1]}
        )
        frame["block"] = np.where(codes.numpy() <= train_max, "fitted", "test")
        frame.attrs["purge"] = (labels[train_max + 1], labels[int(fold.test_dates.min()) - 1])
        st.gate_probs[fold.fold] = frame
        row = {
            "fold": fold.fold,
            "fit_s": fit_s,
            "metrics": fit.metrics(),
            "means": fit.means.tolist(),
            "variances": fit.variances.tolist(),
            "transition": fit.transition.tolist(),
            "durations": fit.expected_durations.tolist(),
            "stationary": stationary.tolist(),
            "permutation": list(fit.permutation),
            "start_status": list(fit.start_status),
            "start_llfs": [x if math.isfinite(x) else None for x in fit.start_llfs],
            "table": probs,
            "codes": codes,
            # |P_high(harness, gap skipped) - P_high(contiguous)| on test dates
            "gap_skip_max_abs": float(gap_diff.max()),
            "gap_skip_first_date": float(gap_diff[0]),
            "gap_skip_dates_over_1e-3": int((gap_diff > 1e-3).sum()),
        }
        st.gate.append(row)
        print(f"[gate] fold {fold.fold}: llf {fit.llf:.2f}, var {fit.variances}, "
              f"durations {np.round(fit.expected_durations, 1)}, "
              f"stationary {np.round(stationary, 3)}, "
              f"{fit.n_converged}/{fit.n_starts} converged, {fit_s:.1f}s")
        low_pi = [i for i, p in enumerate(stationary) if p < s.min_stationary_prob]
        short = [i for i, d in enumerate(fit.expected_durations)
                 if d < s.min_expected_duration]
        if low_pi or short:
            raise StopRun(
                f"C.5: degenerate regimes on fold {fold.fold}: stationary "
                f"{np.round(stationary, 4).tolist()} (threshold "
                f"{s.min_stationary_prob}), expected durations "
                f"{np.round(fit.expected_durations, 2).tolist()} (threshold "
                f"{s.min_expected_duration} days)"
            )


# =========================================================================== #
#  Arms
# =========================================================================== #


def run_arm(
    s: SmokeSettings, name: str, cfg: NECConfig, seed: int, panel: Panel, purge: int,
    cache: BaseCache, registry: SmokeRegistry, st: RunState,
) -> None:
    exp = s.experiment
    arm = nec_arm(name, cfg)  # the standard seeded builder, as run_sweep uses
    trainers: list[Any] = []

    def make():
        t = arm.build_trainer(seed)  # type: ignore[misc]
        trainers.append(t)
        return t

    registry.context = {"run_arm": name, "run_seed": seed}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = walk_forward_evaluate(
            panel, make, n_folds=exp.n_folds,
            test_dates_per_fold=exp.test_dates_per_fold, purge_dates=purge,
            steps=exp.steps, base_cache=cache, seed=seed, registry=registry,
        )
    st.runs[(name, seed)] = res
    st.trainers[(name, seed)] = trainers
    msgs = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    st.warnings[(name, seed)] = msgs
    for f in res.folds:
        registry.context = {"run_arm": name, "run_seed": seed, "record": "fold"}
        registry.log(s.tag, _fold_metrics(f), config=_arm_config(name, cfg, exp),
                     seed=seed, notes=f"{name} seed {seed} fold {f.fold}")
    registry.context = {"run_arm": name, "run_seed": seed, "record": "pooled"}
    registry.log(s.tag, _pooled_metrics(res), config=_arm_config(name, cfg, exp),
                 seed=seed, notes=f"{name} seed {seed} pooled")
    base_ic = res.pooled_base_ic.mean_ic if res.pooled_base_ic else float("nan")
    print(f"[arm] {name} seed {seed}: pooled IC {res.pooled_ic.mean_ic:+.4f} "
          f"(base {base_ic:+.4f})")
    # C.5 leakage condition: pooled and every fold
    worst = [(f"fold {f.fold}", f.ic_improvement) for f in res.folds]
    if res.pooled_base_ic is not None:
        worst.append(("pooled", res.pooled_ic.mean_ic - res.pooled_base_ic.mean_ic))
    for where, d in worst:
        if d is not None and d > s.max_ic_improvement:
            raise StopRun(
                f"C.5: arm {name!r} seed {seed} improves on the base by {d:+.4f} "
                f"in mean rank IC ({where}), above the {s.max_ic_improvement} "
                "threshold. Almost certainly leakage; stopped to find where"
            )


def _arm_config(name: str, cfg: NECConfig, exp: rx.Experiment) -> dict[str, Any]:
    return {
        "arm": name,
        "objective": cfg.train.objective,
        "correction_penalty_weight": cfg.train.correction_penalty_weight,
        "nec_config": cfg.to_dict(),
        "n_folds": exp.n_folds,
        "test_dates_per_fold": exp.test_dates_per_fold,
        "steps": exp.steps,
    }


def _fold_metrics(f: FoldResult) -> dict[str, float]:
    m: dict[str, float | None] = {
        "mean_ic": f.ic.mean_ic, "icir": f.ic.icir, "nll": f.nll, "fold": float(f.fold),
    }
    if f.base_ic is not None:
        m |= {"base_mean_ic": f.base_ic.mean_ic, "base_icir": f.base_ic.icir,
              "ic_improvement": f.ic_improvement}
    if f.base_nll is not None:
        m |= {"base_nll": f.base_nll, "nll_improvement": f.nll_improvement}
    if f.live_param_count is not None:
        m["live_param_count"] = float(f.live_param_count)
    m |= dict(f.correction) | dict(f.gate_metrics) | dict(f.timing)
    return {k: float(v) for k, v in m.items() if v is not None and math.isfinite(v)}


def _pooled_nll(res: WalkForwardResult, attr: str) -> float | None:
    vals = [(getattr(f, attr), f.n_test) for f in res.folds]
    if any(v is None for v, _ in vals):
        return None
    return sum(v * n for v, n in vals) / sum(n for _, n in vals)


def _pooled_metrics(res: WalkForwardResult) -> dict[str, float]:
    m: dict[str, float | None] = {"mean_ic": res.pooled_ic.mean_ic, "icir": res.pooled_ic.icir,
         "nll": _pooled_nll(res, "nll")}
    if res.pooled_base_ic is not None:
        m |= {"base_mean_ic": res.pooled_base_ic.mean_ic,
              "base_icir": res.pooled_base_ic.icir,
              "ic_improvement": res.pooled_ic.mean_ic - res.pooled_base_ic.mean_ic}
    bn = _pooled_nll(res, "base_nll")
    if bn is not None:
        m |= {"base_nll": bn, "nll_improvement": bn - float(m["nll"] or 0.0)}
    return {k: float(v) for k, v in m.items() if v is not None}


# =========================================================================== #
#  Post-run checks and per-sample corrections
# =========================================================================== #


@torch.no_grad()
def corrections_on_test(trainer, test: Panel) -> np.ndarray:
    """``sum_k pi_k r_k`` on the fold's test rows, as the model applied it."""
    trainer.model.eval()
    out = trainer.model(test.x_seq, test.x_snap, PriorContext(date=test.date))
    return out.correction.double().numpy()


def _gated_arm(s: SmokeSettings) -> str:
    return next(name for name, kind in s.arms if kind == "markov")


def consistency_checks(s: SmokeSettings, panel: Panel, folds, st: RunState) -> None:
    """Things that must hold by construction; a failure is a flag, not a stop."""
    exp = s.experiment
    k = exp.n_experts
    for seed in exp.seeds:
        runs = {name: st.runs.get((name, seed)) for name, _ in s.arms}
        if any(r is None for r in runs.values()):
            continue
        # 1. the shared base: identical base IC in every arm, fold by fold
        ref = runs[s.arms[0][0]]
        for name, r in runs.items():
            for fa, fb in zip(ref.folds, r.folds, strict=True):  # type: ignore[union-attr]
                if fa.base_ic != fb.base_ic:
                    st.flags.append(
                        f"base IC differs between arms {s.arms[0][0]} and {name} "
                        f"(seed {seed}, fold {fa.fold}): the base was not shared"
                    )
        # 2. the harness's gate is the pre-flight gate (deterministic refit)
        gated = _gated_arm(s)
        for i, (f, trainer) in enumerate(
            zip(runs[gated].folds, st.trainers[(gated, seed)], strict=True)  # type: ignore[union-attr]
        ):
            pre = st.gate[i]
            if dict(f.gate_metrics)["gate_llf"] != pre["metrics"]["gate_llf"]:
                st.flags.append(
                    f"harness gate llf differs from the pre-flight fit (seed {seed}, "
                    f"fold {f.fold})"
                )
            if not np.array_equal(
                _gate_probabilities(trainer.model.prior, pre["codes"], k), pre["table"]
            ):
                st.flags.append(
                    f"harness gate table differs from the pre-flight table "
                    f"(seed {seed}, fold {f.fold})"
                )


# =========================================================================== #
#  Figures
# =========================================================================== #


def plot_gate(s: SmokeSettings, fold: int, frame: pd.DataFrame, vix: pd.Series,
              row: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.6))
    # one fill per block, so nothing is drawn across the purge gap (the gate
    # has no rows there)
    for i, block in enumerate(("fitted", "test")):
        seg = frame[frame["block"] == block]
        ax.fill_between(seg["date"], 0, seg["p_high"], step="post",
                        color="tab:red", alpha=0.35, lw=0,
                        label="filtered P(high-variance regime)" if i == 0 else None)
    lo, hi = frame.attrs["purge"]
    ax.axvspan(lo, hi, color="0.6", alpha=0.35,
               label="purge gap (not in the gate's series, brief 03 s3.2)")
    d = frame.loc[frame["block"] == "test", "date"]
    ax.axvspan(d.min(), d.max(), color="tab:blue", alpha=0.12, label="test block")
    ax.set_ylim(0, 1)
    ax.set_ylabel("filtered probability, regime 1 (higher variance)")
    ax2 = ax.twinx()
    v = vix.reindex(frame["date"])
    ax2.plot(frame["date"], v.to_numpy(), color="k", lw=0.7, label="VIX close")
    ax2.set_ylabel("VIX")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=8, frameon=False)
    sd = np.sqrt(row["variances"])
    ax.set_title(
        f"Fold {fold}: Hamilton gate on French Mkt-RF, K=2 (placeholder, Q18). "
        f"Regime sd {sd[0]:.4f} / {sd[1]:.4f}",
        fontsize=10,
    )
    fig.text(0.5, 0.005, s.label, ha="center", va="bottom", fontsize=9,
             color="darkred", weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = s.figures_dir / f"gate_fold{fold}.png"
    fig.savefig(path, dpi=s.figure_dpi)
    plt.close(fig)
    return path


def plot_corrections(s: SmokeSettings, per_arm: dict[str, list[np.ndarray]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_folds = len(next(iter(per_arm.values())))
    fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 3.4), squeeze=False)
    for j in range(n_folds):
        ax = axes[0, j]
        lo = min(float(np.quantile(v[j], 0.001)) for v in per_arm.values())
        hi = max(float(np.quantile(v[j], 0.999)) for v in per_arm.values())
        bins = np.linspace(lo, hi, s.hist_bins)
        for name, vals in per_arm.items():
            ax.hist(vals[j], bins=bins, histtype="step", density=True, label=name)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"fold {j}", fontsize=9)
        ax.set_xlabel("sum_k pi_k r_k (test rows)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Out-of-sample correction distribution, seed 0 (0.1-99.9% range)",
                 fontsize=10)
    fig.text(0.5, 0.005, s.label, ha="center", va="bottom", fontsize=9,
             color="darkred", weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = s.figures_dir / "corrections_seed0.png"
    fig.savefig(path, dpi=s.figure_dpi)
    plt.close(fig)
    return path


# =========================================================================== #
#  Report
# =========================================================================== #


def _f(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if nd <= 6 else f"{x:.{nd}f}"


def _e(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3e}"


def _table(header: list[str], rows: list[list[str]], label: str) -> str:
    out = [f"*{label}.*", "", "| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_report(s: SmokeSettings, panel: Panel, folds, purge: int, st: RunState,
                 figures: dict[str, Path]) -> None:
    exp = s.experiment
    L = s.label
    labels = list(panel.date_labels or ())
    rel = lambda p: f"{s.figures_dir.name}/{p.name}"  # noqa: E731
    md: list[str] = []
    w = md.append
    w("# Smoke run 2026-09-25: first end-to-end run on real data (brief 04 C)\n")
    w(f"> **{L}.** Nothing in this document may be quoted. This is an integration "
      "run, not an experiment: it checks that the pipeline behaves sensibly when "
      "the Hamilton gate meets real returns, and how long it takes. No setting was "
      "changed in response to anything below.\n")
    if st.stopped:
        w(f"> **STOPPED (C.5).** {st.stopped}\n>\n> The run was stopped at that "
          "point as the brief requires; everything below is what had been "
          "produced by then.\n")
    w("## 1. What was run\n")
    w("- **K = 2 is a placeholder for plumbing, not a decision; Q18 is open.**")
    w("- **Objective `mixture_nll` is the only one registered, not a decision; "
      "Q20 is open.**")
    w(f"- Data: `{Path(exp.panel_file).name}` (offline; nothing downloaded). "
      f"{len(panel):,} rows, {len(torch.unique(panel.date))} dates "
      f"{labels[int(panel.date.min())]} to {labels[int(panel.date.max())]}, "
      f"{len(torch.unique(panel.entity))} distinct entities; target `fwd_ret_5d` "
      "= log(C_{t+5}/C_t).")
    w(f"- Source recorded on every trial row: {s.source}.")
    reg = s.results_dir / "trials.jsonl"
    reg_shown = reg.relative_to(ROOT.parent) if reg.is_relative_to(ROOT.parent) else reg
    w(f"- Registry: `{reg_shown}`, every "
      f"row tagged `{s.tag}` with `source` in its config.")
    w("- Gate series: market excess return as registered (`market_excess_return`, "
      "French daily Mkt-RF aligned by date label). Hamilton `MarkovRegression`, "
      "switching mean and variance, fitted per fold on the training block only, "
      "frozen, applied causally with `filter(params)`; filtered, never smoothed. "
      f"Starts `{exp.gate_start_scheme}`: quantiles {exp.gate_start_vol_quantiles}, "
      f"windows {exp.gate_start_vol_windows}, persistences "
      f"{exp.gate_start_persistences}, {exp.gate_start_draws_per_centre} draws per "
      f"centre, relative jitter {exp.gate_start_jitter_rel}.")
    sigma = "n/a" if st.sigma_init is None else f"{st.sigma_init:.5f}"
    w("- Residual mixture: base on, `correction_mode=True`, `zero_init_head=True`, "
      "gate frozen (`freeze_gate=True`), so only the experts train. "
      f"Base {exp.base_hidden_dims}, {exp.base_steps} steps, batch "
      f"{exp.base_batch_size}, lr {exp.base_lr}; experts {exp.expert_hidden_dims}, "
      f"dropout {exp.expert_dropout}, {exp.steps} steps per fold, lr {exp.lr}, batch "
      f"{exp.batch_size}; sigma_init {sigma} (auto: std of the base residual on "
      "fold 0's training block, run_experiment's documented default).")
    w(f"- Walk-forward: {exp.n_folds} folds x {exp.test_dates_per_fold} test dates, "
      f"purge {purge} dates (= horizon), seeds {list(exp.seeds)}. One `BaseCache` "
      "shared by every arm.")
    w("- Arms: (1) base only; (2) residual mixture, Hamilton gate; (3) residual "
      "mixture, frozen uniform gate (same capacity, no regime information).")
    w(f"- C.5 stop thresholds: stationary probability < {s.min_stationary_prob}, "
      f"expected duration < {s.min_expected_duration} days, improvement over the "
      f"base > {s.max_ic_improvement} mean rank IC (checked pooled and per fold). "
      "Gate non-convergence read as: no start converged on a fold (the fit raises); "
      "individual non-converged starts are reported, not a stop.\n")
    w("Fold spans:\n")
    w(_table(["fold", "fitted (train) dates", "purge", "test dates"],
             [[str(sp["fold"]), f"{sp['train'][0]} to {sp['train'][1]} "
               f"({sp['n_train']})", f"{sp['purge']}",
               f"{sp['test'][0]} to {sp['test'][1]} ({sp['n_test']})"]
              for sp in st.fold_spans], L))

    # ---------------------------------------------------------------- gate
    w("## 2. The gate, per fold\n")
    w("Canonical order: regime 0 has the lower variance. The fit is deterministic "
      "(fixed `start_seed`), so the gate is the same for both seeds; the script "
      "fits it once before any expert trains (to check C.5 first) and then checks "
      "that the harness's own fit reproduced it exactly.\n")
    rows = []
    for g in st.gate:
        if "error" in g:
            rows.append([str(g["fold"])] + ["FAILED"] * 8)
            continue
        m = g["metrics"]
        sd = np.sqrt(g["variances"])
        rows.append([
            str(g["fold"]),
            f"{g['means'][0]:+.5f} / {g['means'][1]:+.5f}",
            f"{g['variances'][0]:.3e} / {g['variances'][1]:.3e}",
            f"{sd[0]:.4f} / {sd[1]:.4f}",
            f"{g['durations'][0]:.1f} / {g['durations'][1]:.1f}",
            f"{g['stationary'][0]:.3f} / {g['stationary'][1]:.3f}",
            f"{m['gate_llf']:.2f}",
            str(g["permutation"]),
        ])
    w(_table(["fold", "mean (daily, 0 / 1)", "variance (0 / 1)", "sd (0 / 1)",
              "expected duration, days (0 / 1)", "stationary prob (0 / 1)",
              "log-lik", "permutation applied"], rows, L))
    w("Transition matrices (row-stochastic, canonical order; row = from):\n")
    rows = []
    for g in st.gate:
        if "error" in g:
            continue
        p = g["transition"]
        rows.append([str(g["fold"]), f"{p[0][0]:.4f}", f"{p[0][1]:.4f}",
                     f"{p[1][0]:.4f}", f"{p[1][1]:.4f}"])
    w(_table(["fold", "p(0->0)", "p(0->1)", "p(1->0)", "p(1->1)"], rows, L))
    w("Multi-start record (brief 04 B.3):\n")
    rows = []
    for g in st.gate:
        if "error" in g:
            rows.append([str(g["fold"]), g["error"]] + [""] * 6)
            continue
        m = g["metrics"]
        bms = m.get("gate_best_minus_second")
        rows.append([
            str(g["fold"]), str(int(m["gate_n_starts"])),
            str(int(m["gate_n_converged"])), str(int(m["gate_n_failed"])),
            str(int(m["gate_n_nonconverged"])), str(int(m["gate_n_distinct_optima"])),
            "n/a (one optimum)" if bms is None else f"{bms:.3f}",
            f"{g['fit_s']:.1f}",
        ])
    w(_table(["fold", "starts", "converged", "failed (raised)", "non-converged",
              "distinct optima", "best minus second (nats)", "gate fit, s"], rows, L))
    if st.gate_probs:
        w("Filtered probability of the high-variance regime against VIX. Spearman "
          "rank correlation between the two, on the fitted dates and on the test "
          "dates separately (a sanity check on the split, not a performance "
          "number):\n")
        rows = []
        for fold, frame in st.gate_probs.items():
            corr = frame.attrs.get("vix_spearman", {})
            share = frame.attrs.get("share_high", {})
            rows.append([str(fold), _f(corr.get("fitted"), 3), _f(corr.get("test"), 3),
                         f"{share.get('fitted', float('nan')):.3f}",
                         f"{share.get('test', float('nan')):.3f}"])
        w(_table(["fold", "rho(P_high, VIX) fitted", "rho(P_high, VIX) test",
                  "share of fitted dates P_high > 0.5", "share of test dates P_high > 0.5"],
                 rows, L))
        w("The harness applies the frozen filter over the training dates followed "
          "directly by the test dates, as brief 03 section 3.2 specifies (\"construct "
          "the model over train plus test\"). The purge-gap market returns are "
          "therefore not in the filtered series, and one transition step bridges "
          "the gap. The same frozen fit run over the contiguous span, gap included, "
          "measures what that does to the test block (see Observations):\n")
        w(_table(["fold", "max abs dP_high over test dates", "abs dP_high on the first test date",
                  "test dates with abs dP_high > 1e-3"],
                 [[str(g["fold"]), f"{g['gap_skip_max_abs']:.2e}",
                   f"{g['gap_skip_first_date']:.2e}", str(g["gap_skip_dates_over_1e-3"])]
                  for g in st.gate if "error" not in g], L))
        for fold in st.gate_probs:
            if f"gate_fold{fold}" in figures:
                w(f"![Fold {fold} gate against VIX. {L}]"
                  f"({rel(figures[f'gate_fold{fold}'])})\n")

    # ---------------------------------------------------------------- arms
    w("## 3. The arms\n")
    w("The base's IC is identical in every arm of a (seed, fold) because the base is "
      "shared; the base's NLL is not, because it is evaluated with each mixture's own "
      "noise scale (brief 02 section 5), so it is listed per arm. "
      "Correction = `sum_k pi_k r_k` on the test rows; pairwise distance = RMS "
      "distance between the two experts' corrections `r_0, r_1` on the test rows.\n")
    for seed in exp.seeds:
        runs = {n: st.runs[(n, seed)] for n, _ in s.arms if (n, seed) in st.runs}
        if not runs:
            continue
        w(f"### Seed {seed}\n")
        ref = next(iter(runs.values()))
        pb = ref.pooled_base_ic
        rows = [["base only", "pooled", _f(pb.mean_ic), _f(pb.icir, 3),  # type: ignore[union-attr]
                 "—", "—", "—"]]
        for f in ref.folds:
            b = f.base_ic
            rows.append(["base only", str(f.fold), _f(b.mean_ic), _f(b.icir, 3),  # type: ignore[union-attr]
                         "—", "—", "—"])
        for name, r in runs.items():
            pm = _pooled_metrics(r)
            rows.append([name, "pooled", _f(pm["mean_ic"]), _f(pm["icir"], 3),
                         _f(pm["base_mean_ic"]), _f(pm["base_icir"], 3),
                         _f(pm["ic_improvement"])])
            for f in r.folds:
                b = f.base_ic
                rows.append([name, str(f.fold), _f(f.ic.mean_ic), _f(f.ic.icir, 3),
                             _f(b.mean_ic), _f(b.icir, 3),  # type: ignore[union-attr]
                             _f(f.ic_improvement)])
        w("Rank IC (mean over test dates) and ICIR (mean / sd of the daily IC):\n")
        w(_table(["arm", "fold", "mean rank IC", "ICIR", "base mean IC", "base ICIR",
                  "IC improvement over base"], rows, L))
        rows = []
        for name, r in runs.items():
            pm = _pooled_metrics(r)
            rows.append([name, "pooled", _f(pm["nll"]), _f(pm.get("base_nll")),
                         _f(pm.get("nll_improvement"), 5)])
            for f in r.folds:
                rows.append([name, str(f.fold), _f(f.nll), _f(f.base_nll),
                             _f(f.nll_improvement, 5)])
        w("Out-of-sample negative log likelihood per test row (pooled = row-weighted "
          "mean of the folds); improvement = base minus mixture, positive means the "
          "correction lowered it:\n")
        w(_table(["arm", "fold", "NLL", "base NLL (same sigma)", "NLL improvement"],
                 rows, L))
        rows = []
        for name, r in runs.items():
            for f in r.folds:
                c = dict(f.correction)
                rows.append([
                    name, str(f.fold), _e(c.get("correction_mean_abs")),
                    _e(c.get("correction_std")),
                    " / ".join(_e(c.get(f"correction_q{round(100 * q):02d}"))
                               for q in s.correction_quantiles),
                    _e(c.get("correction_max_abs")),
                    f"{c.get('correction_rel_base_std', float('nan')):.4f}",
                    _e(c.get("correction_min_pairwise_distance")),
                    str(f.live_param_count),
                ])
        qs = " / ".join(f"q{round(100 * q):02d}" for q in s.correction_quantiles)
        w("Correction applied out of sample, pairwise distance between the fitted "
          "corrections, and live parameter count:\n")
        w(_table(["arm", "fold", "mean abs", "sd", qs, "max abs",
                  "mean abs / base sd", "pairwise distance r_0 vs r_1", "live params"],
                 rows, L))
    if "corrections" in figures:
        w(f"![Correction distributions. {L}]({rel(figures['corrections'])})\n")
    if st.base_params is not None:
        w(f"Base-only arm: the base network has {st.base_params:,} parameters, all "
          "trained in its own fit and frozen before the experts train.\n")
    w("Live parameter count is the gradient audit's snapshot at the first backward "
      "pass: the encoder, gate and prior are frozen, and `log_sigma` is still "
      f"held by the {exp.sigma_freeze_steps}-step sigma freeze at that instant, so "
      "the count is the expert networks (brief 02 section 5 reads it this way).\n")

    # -------------------------------------------------------------- timing
    w("## 4. Timing\n")
    probe = "n/a" if st.probe_base_s is None else f"{st.probe_base_s:.1f} s"
    w("Wall clock per fold, seconds. The harness times gate fit, base fit and expert "
      "training separately. A base-fit time near zero is a `BaseCache` hit: the "
      "real base cost of a (seed, fold) is its one cache miss, in the second table "
      f"(fold 0 of seed 0 was fitted by the sigma probe before the arms: {probe}).\n")
    rows = []
    for (name, seed), r in st.runs.items():
        for f in r.folds:
            t = dict(f.timing)
            rows.append([name, str(seed), str(f.fold), f"{t['gate_fit_s']:.1f}",
                         f"{t['base_fit_s']:.1f}", f"{t['expert_train_s']:.1f}",
                         f"{sum(t.values()):.1f}"])
    w(_table(["arm", "seed", "fold", "gate fit", "base fit", "expert training",
              "total"], rows, L))
    miss: dict[tuple[int, int], float] = {}
    for (_, seed), r in st.runs.items():
        for f in r.folds:
            miss[(seed, f.fold)] = max(miss.get((seed, f.fold), 0.0),
                                       dict(f.timing)["base_fit_s"])
    if st.probe_base_s is not None and (exp.seeds[0], 0) in miss:
        miss[(exp.seeds[0], 0)] = st.probe_base_s
    if miss:
        w(_table(["seed", "fold", "base fit (cache miss), s"],
                 [[str(sd), str(fd), f"{v:.1f}"] for (sd, fd), v in sorted(miss.items())],
                 L))
    w(f"Total wall clock of the script: {time.perf_counter() - st.started:.0f} s. "
      f"Environment: {st.env.get('summary', '')}.\n")

    # -------------------------------------------------------- observations
    w("## 5. Observations\n")
    w("Anything that looks wrong or surprising. No interpretation of performance.\n")
    w("<!-- OBSERVATIONS: written by hand after reading the numbers above -->\n")
    w("Automatic checks (by construction these must hold):\n")
    if st.flags:
        w("\n".join(f"- **FLAG:** {x}" for x in st.flags) + "\n")
    else:
        w("- The base IC is identical across arms in every (seed, fold): the base "
          "was shared.\n- The harness's gate reproduced the pre-flight gate exactly "
          "(log-likelihood and the whole filtered-probability table) in every "
          "(seed, fold).\n")
    for (name, seed), msgs in st.warnings.items():
        if msgs:
            w(f"Warnings raised during {name}, seed {seed}:\n")
            w("\n".join(f"- `{m}`" for m in msgs) + "\n")
    w("---\n")
    w(f"*{L}.* Generated by `nec_baseline/scripts/smoke_run_2026_09.py` at commit "
      f"`{st.env.get('commit', '?')}`.\n")
    s.report_path.write_text("\n".join(md))


# =========================================================================== #
#  Main
# =========================================================================== #


def _env() -> dict[str, str]:
    import statsmodels

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "nec_moe"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        commit += " + uncommitted nec_moe changes" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "?"
    summary = (f"Python {platform.python_version()}, torch {torch.__version__}, "
               f"statsmodels {statsmodels.__version__}, {platform.machine()}, "
               f"{torch.get_num_threads()} torch threads, CPU")
    return {"commit": commit, "summary": summary}


def main(s: SmokeSettings = SMOKE) -> int:
    st = RunState(env=_env())
    s.figures_dir.mkdir(parents=True, exist_ok=True)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    trials = s.results_dir / "trials.jsonl"
    if trials.exists():
        raise SystemExit(f"{trials} exists: move it aside first (the registry is "
                         "append-only and this run must not mix with an earlier one)")
    registry = SmokeRegistry(trials, s.tag, s.source)
    exp = s.experiment
    figures: dict[str, Path] = {}

    panel, purge = rx._build_panel(exp)
    if purge != exp.horizon:
        raise AssertionError(f"purge {purge} != horizon {exp.horizon}")
    folds = walk_forward_folds(panel.date, n_folds=exp.n_folds,
                               test_dates_per_fold=exp.test_dates_per_fold,
                               purge_dates=purge)
    labels = list(panel.date_labels or ())
    for fold in folds:
        tr, te = fold.train_dates, fold.test_dates
        st.fold_spans.append({
            "fold": fold.fold,
            "train": (labels[int(tr.min())], labels[int(tr.max())]), "n_train": len(tr),
            "purge": int(te.min()) - int(tr.max()) - 1,
            "test": (labels[int(te.min())], labels[int(te.max())]), "n_test": len(te),
        })
    print(f"[data] {len(panel):,} rows, {len(labels)} dates; folds {st.fold_spans}")

    cache = BaseCache()
    first = panel.subset_dates(folds[0].train_dates)
    window = (int(folds[0].train_dates.min()), int(folds[0].train_dates.max()))
    t0 = time.perf_counter()
    st.sigma_init = rx._auto_sigma(exp, panel, first, cache, window)
    st.probe_base_s = time.perf_counter() - t0
    del first
    cfgs = {name: rx._nec_config(dataclasses.replace(exp, prior=kind), panel, st.sigma_init)
            for name, kind in s.arms}
    print(f"[base] sigma_init {st.sigma_init:.5f} ({st.probe_base_s:.1f}s)")

    vix = load_vix(s.vix_dir)
    try:
        gate_preflight(s, panel, folds, cfgs[_gated_arm(s)], st)
        for seed in exp.seeds:
            for name, _ in s.arms:
                t0 = time.perf_counter()
                run_arm(s, name, cfgs[name], seed, panel, purge, cache, registry, st)
                print(f"[arm] {name} seed {seed} done in {time.perf_counter() - t0:.0f}s")
    except StopRun as e:
        st.stopped = str(e)
        print(f"[STOP] {e}")

    gate_rows = {g["fold"]: g for g in st.gate if "error" not in g}
    for fold_i, frame in st.gate_probs.items():
        v = vix.reindex(frame["date"]).to_numpy()
        frame.attrs["vix_spearman"] = {
            blk: float(pd.Series(frame["p_high"].to_numpy()[m]).corr(
                pd.Series(v[m]), method="spearman"))
            for blk in ("fitted", "test")
            for m in [frame["block"].to_numpy() == blk]
        }
        frame.attrs["share_high"] = {
            blk: float((frame.loc[frame["block"] == blk, "p_high"] > 0.5).mean())
            for blk in ("fitted", "test")
        }
        figures[f"gate_fold{fold_i}"] = plot_gate(s, fold_i, frame, vix, gate_rows[fold_i])

    consistency_checks(s, panel, folds, st)
    built = [t for ts in st.trainers.values() for t in ts if t.model.base is not None]
    if built:
        st.base_params = sum(p.numel() for p in built[0].model.base.parameters())
    seed0 = exp.seeds[0]
    per_arm: dict[str, list[np.ndarray]] = {}
    for name, _ in s.arms:
        trainers = st.trainers.get((name, seed0))
        if trainers and len(trainers) == len(folds):
            per_arm[name] = [
                corrections_on_test(t, panel.subset_dates(f.test_dates))
                for t, f in zip(trainers, folds, strict=True)
            ]
    if per_arm:
        figures["corrections"] = plot_corrections(s, per_arm)

    write_report(s, panel, folds, purge, st, figures)
    raw = {
        "label": s.label, "source": s.source, "stopped": st.stopped,
        "settings": json.loads(json.dumps(dataclasses.asdict(exp))),
        "sigma_init": st.sigma_init, "fold_spans": st.fold_spans,
        "gate": [{k: v for k, v in g.items() if k not in ("table", "codes")}
                 for g in st.gate],
        "runs": {f"{n}_seed{sd}": {"pooled": _pooled_metrics(r),
                                    "folds": [_fold_metrics(f) for f in r.folds]}
                 for (n, sd), r in st.runs.items()},
        "flags": st.flags,
        "warnings": {f"{n}_seed{sd}": m for (n, sd), m in st.warnings.items()},
        "env": st.env,
    }
    (s.results_dir / "results.json").write_text(json.dumps(raw, indent=2, default=str))
    print(f"[report] {s.report_path}")
    return 1 if st.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
