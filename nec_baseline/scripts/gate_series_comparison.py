"""Brief 10 B.6: the gate's two series compared on the 2000-2008 training block.

A **gate-fit diagnostic, nothing is scored** (allowed on real data by brief 10).
The Hamilton gate is fitted, K = 2 and K = 3, on

- ``crsp_market_log_return``: the CRSP S&P 500 index's daily total return as
  a log return (INDNO 1000500; Q23, decided 2026-10-08, the default), read
  the way the real 2000-2024 panel reads it (that panel was built before
  brief 10, so from the extract's ``market.parquet``);
- ``market_excess_return``: French daily Mkt-RF (kept for the robustness
  check),

over the 2009 pilot fold's training block: every CRSP trading day from
2000-01-01 to 2008-12-31, minus the last ``purge_dates`` (the target
horizon), exactly the dates that fold fits the gate on. No date after 2008 is
read: the script refuses an ``end`` on or after 2009-01-01. Each fit is
reported with its convergence, regime variances, means and expected
durations; per K and backend, the two series' filtered stress probabilities
(the highest-variance regime) are correlated. There is no prediction, no
causal application to later dates and no score.

Writes ``results/benchmark/gate_series_comparison.json`` (aggregate only: fit
statistics, correlations, counts; no series values).

Usage (from nec_baseline/)::

    python3.14 scripts/gate_series_comparison.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nec_moe import FeatureSchema, MarkovGateConfig, MarkovSwitchingRegimePrior, Panel  # noqa: E402
from nec_moe.crsp import CRSPSpec  # noqa: E402
from nec_moe.markov_gate import date_level_series  # noqa: E402

#: brief 10 B.6: no date after 2008 (2009 is a pilot validation year)
LAST_ALLOWED = pd.Timestamp("2008-12-31")


@dataclass(frozen=True)
class Settings:
    """Every setting of the diagnostic, with its default."""

    start: str = "2000-01-01"
    end: str = "2008-12-31"
    # the target horizon: the 2009 pilot fold drops the last 5 dates of 2008
    purge_dates: int = 5
    k_values: tuple[int, ...] = (2, 3)
    # statsmodels is the harness's default estimator; native is the one the
    # pilot's gate-memory stage fits (brief 09 F.2)
    backends: tuple[str, ...] = ("statsmodels", "native")
    series: tuple[str, ...] = ("crsp_market_log_return", "market_excess_return")
    context_dir: str = "data_cache"  # French daily factors, cache only
    horizon: int = 5  # the panel target's horizon (the gate weight's h; unused by the fit)
    stress_threshold: float = 0.5  # share of days with P(stress) above this
    out: str = "results/benchmark/gate_series_comparison.json"

    def validate(self) -> Settings:
        if pd.Timestamp(self.end) > LAST_ALLOWED:
            raise ValueError(
                f"end={self.end}: this diagnostic reads no date after 2008 "
                "(brief 10 B.6); there is no override"
            )
        if self.purge_dates < 0 or not self.k_values:
            raise ValueError("purge_dates must be >= 0 and k_values non-empty")
        return self


def training_block(s: Settings, spec: CRSPSpec) -> pd.DatetimeIndex:
    """The CRSP trading days of ``[start, end]`` minus the last ``purge_dates``."""
    market = pd.read_parquet(spec.extract_dir / "market.parquet")
    days = pd.DatetimeIndex(market.index).normalize()
    days = days[(days >= pd.Timestamp(s.start)) & (days <= pd.Timestamp(s.end))]
    return days[: len(days) - s.purge_dates] if s.purge_dates else days


def date_panel(days: pd.DatetimeIndex, spec: CRSPSpec) -> Panel:
    """One row per date, carrying the build metadata of the real panel (built
    before brief 10, so the CRSP series comes from the extract it names)."""
    n = len(days)
    return Panel(
        x_seq=torch.zeros(n, 1, 1), x_snap=torch.zeros(n, 1), y=torch.zeros(n),
        date=torch.arange(n), entity=torch.zeros(n, dtype=torch.long),
        schema=FeatureSchema(sequence_features=("s",), snapshot_features=("x",),
                             target="fwd_ret_5d"),
        date_labels=tuple(str(d.date()) for d in days),
        data_source=spec.data_source,
        metadata={"crsp_release": spec.release, "crsp_stock_file": spec.stock_file,
                  "market_indno": spec.market_indno, "start": spec.start, "end": spec.end},
    )


def fit_one(
    panel: Panel, s: Settings, series: str, k: int, backend: str
) -> tuple[dict, np.ndarray]:
    cfg = MarkovGateConfig(series=series, k_regimes=k,
                           fit_backend=backend,  # type: ignore[arg-type]
                           context_dir=str(ROOT / s.context_dir))
    gate = MarkovSwitchingRegimePrior(k, cfg, horizon=s.horizon)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gate.fit(panel)
    fit = gate.fit_result
    assert fit is not None
    stress = gate.filtered_probabilities(torch.unique(panel.date, sorted=True))[:, k - 1]
    m = fit.metrics()
    row = {
        "series": series, "k": k, "backend": backend,
        "seconds": round(time.perf_counter() - t0, 1),
        "n_starts": fit.n_starts, "n_converged": fit.n_converged,
        "n_failed": int(m["gate_n_failed"]), "n_nonconverged": int(m["gate_n_nonconverged"]),
        "n_distinct_optima": int(m["gate_n_distinct_optima"]),
        "best_minus_second_optimum": m.get("gate_best_minus_second"),
        "llf": fit.llf,
        "variances": [float(v) for v in fit.variances],
        "sd_annualised": [float(np.sqrt(252 * v)) for v in fit.variances],
        "means": [float(v) for v in fit.means],
        "transition_diagonal": [float(fit.transition[i, i]) for i in range(k)],
        "expected_durations_days": [float(d) for d in fit.expected_durations],
        "stress_share": float((stress > s.stress_threshold).mean()),
        "warnings": sorted({type(w.message).__name__ for w in caught}),
    }
    return row, stress


def main(s: Settings) -> dict:
    s = s.validate()
    spec = CRSPSpec()
    days = training_block(s, spec)
    assert days.max() <= LAST_ALLOWED
    panel = date_panel(days, spec)
    print(f"[gate series] {len(days)} dates, {days[0].date()} .. {days[-1].date()}")

    described = {}
    for series in s.series:
        _, x = date_level_series(panel, MarkovGateConfig(series=series,
                                                         context_dir=str(ROOT / s.context_dir)))
        described[series] = x
    a, b = (described[k] for k in s.series[:2])

    fits, stress = [], {}
    for k in s.k_values:
        for backend in s.backends:
            for series in s.series:
                row, p = fit_one(panel, s, series, k, backend)
                fits.append(row)
                stress[(k, backend, series)] = p
                print(f"  K={k} {backend:11s} {series:22s} converged "
                      f"{row['n_converged']}/{row['n_starts']}, durations "
                      f"{[round(d, 1) for d in row['expected_durations_days']]}")
    agreement = []
    for k in s.k_values:
        for backend in s.backends:
            p, q = (stress[(k, backend, series)] for series in s.series[:2])
            agreement.append({
                "k": k, "backend": backend,
                "stress_probability_pearson": float(np.corrcoef(p, q)[0, 1]),
                "stress_probability_spearman": float(
                    pd.Series(p).corr(pd.Series(q), method="spearman")),
                "both_stressed_or_both_calm_share": float(
                    ((p > s.stress_threshold) == (q > s.stress_threshold)).mean()),
            })

    report = {
        "what": "brief 10 B.6: Hamilton gate fits on the two candidate series, "
                "2009 pilot fold's training block; a fit diagnostic, nothing scored",
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "settings": dataclasses.asdict(s),
        "window": {"first": str(days[0].date()), "last": str(days[-1].date()),
                   "dates": int(len(days))},
        "units": "decimals for both series (0.01 is about a 1% day)",
        "series_sd_daily": {k: float(v.std()) for k, v in described.items()},
        "series_correlation": float(np.corrcoef(a, b)[0, 1]),
        "fits": fits,
        "stress_probability_agreement": agreement,
        "stress_regime": "the highest-variance regime (canonical order: ascending variance)",
    }
    out = ROOT / s.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[gate series] wrote {out.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main(Settings())
