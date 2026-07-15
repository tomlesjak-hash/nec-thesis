"""Reproducible thesis figures (handbook III.3 item 5).

Every figure the workflow needs, as functions of the package's own data
structures — so figures are artifacts of a run (save them next to the registry
rows they illustrate), not notebook one-offs. Each function returns the
``matplotlib`` Figure and optionally writes it (``path=``, dpi 150, parents
created). ``matplotlib`` is imported lazily inside the functions: it stays an
optional dependency and importing :mod:`nec_moe` never requires it.

The six figures and what they are for:

- :func:`plot_training_dashboard` — II.3's monitoring table as panels; the
  first thing to look at after any fit.
- :func:`plot_gate_utilization` — the gate's per-date regime belief (predictive
  ``pi`` for memoryless priors, the *filtered* posterior for the HMM — i.e. the
  classic Hamilton filtered-probability figure), optionally overlaid with a
  context series (VIX): the interpretability headline figure.
- :func:`plot_ic_series` — the honest per-date picture behind any pooled IC.
- :func:`plot_long_short_curve` — cumulative gross/net quantile long-short.
- :func:`plot_transition_matrix` — the learned Markov chain (Variation 3).
- :func:`plot_sweep_report` — mean ± seed-std per arm from a
  :class:`~nec_moe.sweep.SweepReport`: the comparison figure.

No function calls ``plt.show()`` and none touches the global style — safe in
scripts, notebooks, and headless CI (set ``MPLBACKEND=Agg`` there).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from torch import Tensor

from .alignment import gate_utilization_by_date
from .data import Panel
from .diagnostics import canonical_expert_order
from .evaluation import long_short_by_date, rank_ic_by_date
from .priors import HMMRegimePrior
from .sweep import SweepReport
from .train import Trainer

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

    from .calibration import ReliabilityReport

__all__ = [
    "plot_training_dashboard",
    "plot_gate_utilization",
    "plot_ic_series",
    "plot_long_short_curve",
    "plot_transition_matrix",
    "plot_sweep_report",
    "plot_reliability",
]


def _plt():
    import matplotlib.pyplot as plt  # lazy: optional dependency

    return plt


def _save(fig: Figure, path: str | Path | None) -> Figure:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    return fig


def _date_axis(panel: Panel, date_codes: Tensor):
    """Calendar dates when the panel has labels, integer codes otherwise."""
    if panel.date_labels is not None:
        return pd.to_datetime([panel.date_labels[int(d)] for d in date_codes])
    return [int(d) for d in date_codes]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_training_dashboard(
    history: Sequence[Mapping[str, float]],
    *,
    n_experts: int,
    path: str | Path | None = None,
) -> Figure:
    """Four panels over training steps: NLL, gate entropy (with the uniform
    ``log K`` reference), min running utilization (with the 0.05 alarm), and
    max off-diagonal expert correlation (the homogenization watch)."""
    if not history:
        raise ValueError("empty history — fit first")
    plt = _plt()
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2), tight_layout=True)

    ax[0].plot(steps, [h["nll"] for h in history], lw=1)
    ax[0].set_title("train NLL")
    ax[1].plot(steps, [h["gate_entropy"] for h in history], lw=1)
    ax[1].axhline(math.log(n_experts), ls="--", c="gray", lw=1, label="uniform")
    ax[1].set_title("gate entropy (nats)")
    ax[1].legend(loc="best", fontsize=8)
    ax[2].plot(steps, [h["min_running_utilization"] for h in history], lw=1)
    ax[2].axhline(0.05, ls="--", c="r", lw=1, label="alarm")
    ax[2].set_title("min running utilization")
    ax[2].legend(loc="best", fontsize=8)
    ax[3].plot(steps, [h["max_offdiag_expert_corr"] for h in history], lw=1)
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].set_title("max |expert output corr|")
    for a in ax:
        a.set_xlabel("step")
    return _save(fig, path)


def plot_gate_utilization(
    trainer: Trainer,
    panel: Panel,
    *,
    context: pd.DataFrame | None = None,
    context_col: str = "vix",
    expert: int | None = None,
    path: str | Path | None = None,
) -> Figure:
    """One expert's per-date gate share as a filled path, optionally overlaid
    with a context series on a twin axis.

    ``expert=None`` picks the highest-σ expert (the canonical "stress"
    candidate). Memoryless priors show the mean predictive ``pi``; the HMM
    shows the mean *filtered* posterior — the Hamilton filtered-probability
    figure. Context join: calendar labels for real panels, integer date codes
    for synthetic ones (index the frame accordingly).
    """
    dates, util = gate_utilization_by_date(trainer, panel)
    if expert is None:
        expert = int(canonical_expert_order(trainer.model.experts.log_sigma)[-1])
    x = _date_axis(panel, dates)

    plt = _plt()
    fig, ax1 = plt.subplots(figsize=(11, 3.5), tight_layout=True)
    ax1.fill_between(x, util[:, expert].numpy(), color="tab:red", alpha=0.35)
    ax1.plot(x, util[:, expert].numpy(), c="tab:red", lw=1,
             label=f"expert {expert} share")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("gate share")
    ax1.legend(loc="upper left", fontsize=8)
    if context is not None:
        if context_col not in context.columns:
            raise ValueError(
                f"context has no column {context_col!r}; got "
                f"{list(context.columns)}"
            )
        ax2 = ax1.twinx()
        ax2.plot(x, context[context_col].reindex(pd.Index(x)).to_numpy(),
                 c="gray", alpha=0.6, lw=1)
        ax2.set_ylabel(context_col)
    return _save(fig, path)


def plot_ic_series(
    pred: Tensor,
    y: Tensor,
    date: Tensor,
    *,
    roll: int = 20,
    path: str | Path | None = None,
) -> Figure:
    """Per-date rank-IC bars with a rolling mean — the honest picture behind a
    pooled IC (a pooled 0.03 can be a steady 0.03 or a lucky quarter)."""
    _, ics = rank_ic_by_date(pred, y, date)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 2.8), tight_layout=True)
    ax.bar(range(len(ics)), ics.numpy(), width=1.0, alpha=0.5)
    if len(ics) >= roll:
        ax.plot(pd.Series(ics.numpy()).rolling(roll).mean().to_numpy(),
                c="k", lw=1.5, label=f"{roll}d mean")
        ax.legend(loc="best", fontsize=8)
    ax.axhline(0.0, c="gray", lw=1)
    ax.set_ylabel("daily rank-IC")
    ax.set_xlabel("test date #")
    return _save(fig, path)


def plot_long_short_curve(
    pred: Tensor,
    y: Tensor,
    date: Tensor,
    entity: Tensor,
    *,
    n_quantiles: int,
    cost_rate: float = 0.0,
    path: str | Path | None = None,
) -> Figure:
    """Cumulative gross and net quantile long-short returns (per-period sums —
    no calendar, no annualization)."""
    _, gross, tno = long_short_by_date(pred, y, date, entity, n_quantiles=n_quantiles)
    net = gross - cost_rate * 2.0 * tno
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 2.8), tight_layout=True)
    ax.plot(gross.cumsum(0).numpy(), lw=1.2, label="gross")
    if cost_rate > 0:
        ax.plot(net.cumsum(0).numpy(), lw=1.2, label=f"net @ {cost_rate:.4f}/notional")
    ax.axhline(0.0, c="gray", lw=1)
    ax.set_ylabel("cum. L/S return (per-period)")
    ax.set_xlabel("test date #")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, path)


def plot_transition_matrix(
    source: object,
    *,
    path: str | Path | None = None,
) -> Figure:
    """Annotated heatmap of the learned transition matrix ``A`` (Variation 3).

    ``source`` may be an ``NECModel``/``Trainer`` (whose prior must be the
    HMM), an :class:`HMMRegimePrior`, or a ``(K, K)`` tensor.
    """
    obj = source
    if isinstance(obj, Trainer):
        obj = obj.model
    prior = getattr(obj, "prior", obj)
    if isinstance(prior, HMMRegimePrior):
        a = prior.transition_matrix.detach()
    elif isinstance(obj, Tensor):
        a = obj.detach()
    else:
        raise ValueError(
            "plot_transition_matrix needs an HMM prior (or a (K,K) tensor); "
            f"got {type(source).__name__} — memoryless priors have no "
            "transition matrix"
        )
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"transition matrix must be square, got {tuple(a.shape)}")

    plt = _plt()
    k = a.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * k, 1.0 + 0.9 * k), tight_layout=True)
    im = ax.imshow(a.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(k):
        for j in range(k):
            v = float(a[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xlabel("to regime")
    ax.set_ylabel("from regime")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_sweep_report(
    report: SweepReport,
    *,
    metric: str = "mean_ic",
    path: str | Path | None = None,
) -> Figure:
    """Mean ± seed-std of one metric per arm — the comparison-table figure.

    The error bars are the seed stds from the report: the visual form of
    "nothing is a result until it has a seed std". Deterministic baselines
    legitimately show none.
    """
    missing = [a.arm for a in report.arms if metric not in a.mean]
    if missing:
        raise ValueError(
            f"metric {metric!r} missing for arms {missing}; available: "
            f"{sorted(report.arms[0].mean)}"
        )
    arms = [a.arm for a in report.arms]
    means = [a.mean[metric] for a in report.arms]
    stds = [a.std[metric] for a in report.arms]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(arms), 3.2), tight_layout=True)
    ax.bar(arms, means, yerr=stds, capsize=4, alpha=0.75)
    ax.axhline(0.0, c="gray", lw=1)
    ax.set_ylabel(metric)
    ax.set_title(f"{report.tag}: {metric} (mean ± seed std, n={report.arms[0].n_seeds})")
    return _save(fig, path)


def plot_reliability(
    report: ReliabilityReport,
    *,
    path: str | Path | None = None,
) -> Figure:
    """Reliability diagram of the gate's mixture weights (M4 §4.9).

    Left: mean realized responsibility vs mean claimed probability per bin —
    a calibrated gate hugs the diagonal; sagging below at high confidence is
    the classic overconfidence signature. Right: where the claims live (bin
    weights). Title carries the ECE and the temperature the numbers were
    computed under (compare T=1 against a fitted T side by side).
    """
    plt = _plt()
    fig, (ax, axw) = plt.subplots(
        1, 2, figsize=(8.2, 3.4), tight_layout=True, width_ratios=[1.4, 1.0]
    )
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="perfect")
    conf = report.bin_confidence
    out = report.bin_outcome
    keep = [i for i, w in enumerate(report.bin_weight) if w > 0]
    ax.plot([conf[i] for i in keep], [out[i] for i in keep],
            marker="o", ms=4, lw=1.2, label="gate")
    ax.set_xlabel("claimed  π")
    ax.set_ylabel("realized responsibility  r")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"ECE = {report.ece:.3f}  (T = {report.temperature:.2f}, "
        f"{report.n_claims:,} claims)", fontsize=9,
    )
    edges = report.bin_edges
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)]
    axw.bar(centers, report.bin_weight, width=0.9 / len(centers), alpha=0.6)
    axw.set_xlabel("claimed  π")
    axw.set_ylabel("share of claims")
    return _save(fig, path)
