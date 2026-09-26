"""Gate–regime alignment diagnostics (the thesis's secondary research question).

"Do learned expert assignments correspond to recognizable market regimes?" is a
hypothesis to *test, not assume* (syllabus §2) — which is exactly why VIX and
factor data never enter training (Decision B) and only appear here, as the
yardstick. The report answers, per expert: how does its gate share move with
VIX, realized market volatility, market-move magnitude, and factor returns —
and how does its utilization differ between calm and stressed terciles?

Findings can go either way and both are results: strong VIX alignment supports
the regime interpretation; alignment with size/liquidity/nothing instead is the
"experts split on other dimensions" outcome the syllabus explicitly anticipates
(and arXiv:2604.09780's null hypothesis — routing may reflect geometry, not
regimes).

Label-switching note: expert indices here are *raw* model indices. Report them
together with the canonical σ-sorted order (Decision D) when aggregating across
fits — within one fitted model, raw indices are stable and that is all these
functions assume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .data import Panel
from .priors import PriorContext
from .train import Trainer

__all__ = [
    "gate_utilization_by_date",
    "ExpertAlignment",
    "RegimeAlignmentReport",
    "regime_alignment",
    "gate_regime_alignment",
]


@torch.no_grad()
def gate_utilization_by_date(trainer: Trainer, panel: Panel) -> tuple[Tensor, Tensor]:
    """Per-date mean gate share of each expert: ``(date_codes (D,), util (D, K))``.

    Memoryless priors: the mean predictive mixture weight ``pi(x)`` over the
    date's cross-section. Stateful (HMM) prior: the mean **prior** from the
    causal filtering pass, i.e. the regime probability the model actually
    uses at ``t``. Not the posterior: the posterior at ``t`` has been updated
    with the target dated ``t``, a forward return realised only after ``t``,
    so comparing it with VIX at ``t`` would let the future into the
    diagnostic (audit findings M-1, Q-1).
    """
    model = trainer.model
    model.eval()
    dates = torch.unique(panel.date, sorted=True)
    if model.prior.stateful:
        ev = trainer.evaluate_sequence(panel.time_sequence())
        util = ev.log_prior.exp().mean(dim=1)  # (L, K), L == len(dates)
        return dates, util
    out = model(panel.x_seq, panel.x_snap, PriorContext(date=panel.date))
    pi = out.prior.log_prior.exp()  # (N, K)
    util = torch.stack([pi[panel.date == d].mean(dim=0) for d in dates])
    return dates, util


@dataclass(frozen=True)
class ExpertAlignment:
    """One expert's alignment with the context series."""

    expert: int
    correlations: dict[str, float]  # context column -> Pearson corr(util_t, col_t)
    high_vix_utilization: float | None  # mean util, top VIX tercile of dates
    low_vix_utilization: float | None  # mean util, bottom tercile


@dataclass(frozen=True)
class RegimeAlignmentReport:
    per_expert: tuple[ExpertAlignment, ...]
    n_dates: int
    context_columns: tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        """Experts x diagnostics table for printing/thesis figures."""
        rows = []
        for e in self.per_expert:
            row: dict[str, float | None] = dict(e.correlations)
            row["high_vix_util"] = e.high_vix_utilization
            row["low_vix_util"] = e.low_vix_utilization
            rows.append(row)
        return pd.DataFrame(rows, index=[f"expert_{e.expert}" for e in self.per_expert])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regime_alignment(
    panel: Panel,
    date_codes: Tensor,
    util: Tensor,
    context: pd.DataFrame,
    *,
    vix_col: str = "vix",
) -> RegimeAlignmentReport:
    """Join a utilization series with context data and score the alignment.

    Panel dates are matched to the context's index via ``panel.date_labels``
    (real panels: calendar dates). Panels without labels (synthetic) join on
    the integer date codes directly — the context frame must then be indexed
    by those codes. Dates missing from the context are dropped; a NaN in a
    context column drops that date for that column only.
    """
    if panel.date_labels is not None:
        keys = pd.to_datetime([panel.date_labels[int(d)] for d in date_codes])
    else:
        keys = pd.Index([int(d) for d in date_codes])
    ctx = context.reindex(keys)
    if ctx.notna().values.sum() == 0:
        raise ValueError(
            "no overlap between panel dates and context index — check the "
            "date ranges (and that real panels carry date_labels)"
        )
    util_np = util.detach().cpu().numpy()  # (D, K)

    vix = ctx[vix_col].to_numpy(dtype=float) if vix_col in ctx.columns else None
    if vix is not None and np.isfinite(vix).sum() >= 9:
        finite = np.isfinite(vix)
        lo_cut, hi_cut = np.nanquantile(vix[finite], [1 / 3, 2 / 3])
        low_m = finite & (vix <= lo_cut)
        high_m = finite & (vix >= hi_cut)
    else:
        low_m = high_m = None

    experts = []
    for k in range(util_np.shape[1]):
        u = util_np[:, k]
        corrs: dict[str, float] = {}
        for col in ctx.columns:
            v = ctx[col].to_numpy(dtype=float)
            m = np.isfinite(v)
            corrs[col] = _corr(u[m], v[m]) if m.sum() >= 3 else float("nan")
        experts.append(
            ExpertAlignment(
                expert=k,
                correlations=corrs,
                high_vix_utilization=float(u[high_m].mean()) if high_m is not None else None,
                low_vix_utilization=float(u[low_m].mean()) if low_m is not None else None,
            )
        )
    return RegimeAlignmentReport(
        per_expert=tuple(experts),
        n_dates=int(len(date_codes)),
        context_columns=tuple(ctx.columns),
    )


def gate_regime_alignment(
    trainer: Trainer, panel: Panel, context: pd.DataFrame, *, vix_col: str = "vix"
) -> RegimeAlignmentReport:
    """Convenience: utilization series + alignment report in one call."""
    date_codes, util = gate_utilization_by_date(trainer, panel)
    return regime_alignment(panel, date_codes, util, context, vix_col=vix_col)
