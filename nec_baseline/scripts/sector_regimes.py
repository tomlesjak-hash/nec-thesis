"""Brief 11 C: do sectors have their own regimes? (2000-2006, descriptive input to Q28)

**Tom runs this on real data.** Every date is inside 2000-01-01 to 2006-12-31:
the script refuses any end on or after 2007-01-01 (no override), and the CRSP
extract and the market series are cut at ``end`` before anything is computed.

1. Sector returns: for each GICS sector (point-in-time, from the panel's
   sector dummies) and date, the value-weighted mean of the members' daily
   simple returns (weights: the previous trading day's market equity,
   ``DlyCap``) over the date's universe rows (the panel's rows), as a log
   return; equal-weighted as robustness. A sector-date with fewer than
   ``min_sector_names`` names is skipped; a sector with any skipped date is
   reported as too thin and not fitted.
2. Two series per sector: sector-relative (sector minus the CRSP index log
   return) and raw.
3. A 2-state Hamilton gate per sector and series (full memory, the default
   estimator and multi-start, canonical order by variance; filtered
   probabilities only). The market's gate is part A's K = 2 fit.
4. Aggregate measures: share of days stressed, share stressed while the
   market is calm and P(sector stress | market calm), correlation with the
   market's stress probability, expected durations, stress episodes of at
   least ``min_episode_days``, and the sector x sector co-stress matrix.

It informs Q28's sub-questions 1 (relative or raw), 2 (granularity) and 6;
it does not choose the industry gate's form. Writes
``results/diagnostics/sector_regimes_2000_2006.csv`` and ``.json`` and one
figure, ``sector_regimes_2000_2006.png``: heat strips of each fitted sector's
stress probability, the market's on top. Nothing per security is written.

Usage (from nec_baseline/; reads the panel and the CRSP extract)::

    caffeinate -i python3.14 scripts/sector_regimes.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ic_decay as icd  # noqa: E402  (the guard, restrict, cut_at)
import ic_regime_split as irs  # noqa: E402  (part A: the market series and its gate)

from nec_moe import MarkovGateConfig  # noqa: E402
from nec_moe.characteristics import SECTOR_FEATURES  # noqa: E402
from nec_moe.features import FeatureSpec  # noqa: E402
from nec_moe.regime_split import (  # noqa: E402
    IN_SAMPLE_CAVEAT,
    SECTOR_NAMES,
    RegimeFit,
    co_stress,
    fit_regimes,
    regime_summary,
    sector_measures,
    sector_returns,
    sector_rows,
    series_panel,
    thin_sectors,
)

SERIES_KINDS: tuple[str, ...] = ("relative", "raw")


@dataclass(frozen=True)
class Settings:
    """Every setting of part C, with its default."""

    start: str = "2000-01-01"
    end: str = "2006-12-31"
    # states per sector gate; the market's gate is part A's K = 2 fit
    n_states: int = 2
    # the existing sector convention (FeatureSpec.min_sector_names)
    min_sector_names: int = FeatureSpec().min_sector_names
    # the first weighting is the primary one; equal weights are robustness
    weightings: tuple[str, ...] = ("value", "equal")
    stress_threshold: float = 0.5
    min_episode_days: int = 5
    fit_backend: str = "statsmodels"
    panel_file: str = icd.Settings().panel_file
    out_dir: str = "results/diagnostics"

    def validate(self) -> Settings:
        icd.check_end(self.end)
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError(f"start {self.start} is after end {self.end}")
        if self.n_states != 2:
            raise ValueError("part C fits 2-state sector gates (C.3)")
        if not self.weightings or not set(self.weightings) <= {"value", "equal"}:
            raise ValueError(f"weightings must be 'value' and/or 'equal', got {self.weightings}")
        if self.min_sector_names < 1 or self.min_episode_days < 1:
            raise ValueError("min_sector_names and min_episode_days must be >= 1")
        if not 0.0 < self.stress_threshold < 1.0:
            raise ValueError("stress_threshold must be in (0, 1)")
        return self

    @property
    def years(self) -> str:
        return f"{pd.Timestamp(self.start).year}_{pd.Timestamp(self.end).year}"

    def regime_settings(self) -> irs.Settings:
        return irs.Settings(start=self.start, end=self.end, k_list=(2,),
                            fit_backend=self.fit_backend)


# --------------------------------------------------------------------------- #
# Inputs (real data: Tom runs this part)
# --------------------------------------------------------------------------- #


def load_rows(s: Settings) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The universe rows ``(date, permno, sector, ret, me_prev)`` of
    ``[start, end]``: the panel's rows and sectors, the extract's simple
    returns and market equity, all cut to the window first. The previous
    day's market equity of the window's first date would lie before
    ``start``; it is not read, so sector returns begin on the second date."""
    from nec_moe.crsp import CRSPSpec, crsp_daily_frames, load_extract
    from nec_moe.features import StageBSpec

    panel = torch.load(ROOT / s.panel_file, weights_only=False, mmap=True)
    meta = panel.metadata or {}
    names = list(panel.schema.snapshot_features)
    labels = pd.to_datetime(np.asarray(panel.date_labels))
    codes = np.flatnonzero((labels >= pd.Timestamp(s.start)) & (labels <= pd.Timestamp(s.end)))
    rows = torch.isin(panel.date, torch.from_numpy(codes)).nonzero().squeeze(1)
    dummies = panel.x_snap[rows][:, [names.index(c) for c in SECTOR_FEATURES]].numpy()
    code_of = np.array([int(c.split("_")[1]) for c in SECTOR_FEATURES], dtype=float)
    sector = np.where(dummies.sum(axis=1) > 0, code_of[dummies.argmax(axis=1)], np.nan)
    index = pd.MultiIndex.from_arrays(
        [labels[panel.date[rows].numpy()],
         np.asarray(panel.entity_labels)[panel.entity[rows].numpy()]],
        names=["date", "permno"])
    sector_wide = window(pd.Series(sector, index=index).unstack("permno"), s)
    universe = window(pd.Series(True, index=index).unstack("permno", fill_value=False), s)
    del panel, dummies

    spec = CRSPSpec(release=meta["crsp_release"], stock_file=meta["crsp_stock_file"],
                    post_delisting_return=meta["post_delisting_return"])
    extract = icd.cut_at(load_extract(spec), s.end)
    frames, _, _ = crsp_daily_frames(extract, spec, StageBSpec(horizon=1),
                                     permnos=[int(p) for p in universe.columns])
    simple = window(np.expm1(pd.DataFrame({p: f["ret"] for p, f in frames.items()})), s)
    cap = window(pd.DataFrame({p: f["cap"] for p, f in frames.items()}), s)
    long = sector_rows(simple, cap, sector_wide, universe)
    provenance = {"panel_file": s.panel_file, "extract": spec.extract_dir.name,
                  "crsp_release": meta["crsp_release"], "dates": int(len(universe)),
                  "names": int(universe.shape[1])}
    return long, provenance


def window(frame: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """``frame``'s rows inside ``[start, end]`` (the ic_decay guard)."""
    return icd.restrict(frame, s)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The computation
# --------------------------------------------------------------------------- #


def sector_gate_config(s: Settings) -> MarkovGateConfig:
    """The gate's defaults, reading the series from sequence channel 0."""
    return MarkovGateConfig(series="sequence_channel", series_channel=0,
                            fit_backend=s.fit_backend)  # type: ignore[arg-type]


def fit_sectors(series: pd.DataFrame, s: Settings) -> dict[int, RegimeFit]:
    """A 2-state gate per column (a gap-free sector series)."""
    labels = [str(d.date()) for d in pd.DatetimeIndex(series.index)]
    out = {}
    for code in series.columns:
        x = series[code]
        if x.isna().any():
            raise ValueError(f"sector {code} has gaps: only gap-free sectors are fitted")
        out[int(code)] = fit_regimes(series_panel(labels, x.to_numpy()), s.n_states,
                                     sector_gate_config(s))
    return out


def run_sectors(s: Settings, root: Path = ROOT, rows: pd.DataFrame | None = None,
                log_ret: pd.Series | None = None) -> dict[str, Any]:
    """Part C. ``rows`` and ``log_ret`` (synthetic, in tests) skip the panel
    and the extract."""
    s = s.validate()
    provenance: dict[str, Any] = {"synthetic_inputs": True}
    if rows is None:
        rows, provenance = load_rows(s)
    rows = rows[(rows["date"] >= pd.Timestamp(s.start)) & (rows["date"] <= pd.Timestamp(s.end))]
    rs_s = s.regime_settings().validate()
    log_ret = irs.load_market(rs_s) if log_ret is None else window(log_ret.to_frame("r"), s)["r"]
    market = irs.fit_market(log_ret, rs_s)[2]
    # sector returns need the previous day's market equity inside the window
    dates = pd.DatetimeIndex(log_ret.index[1:])

    thin: dict[int, dict[str, Any]] = {}
    measures: list[dict[str, Any]] = []
    costress: dict[str, dict[str, Any]] = {}
    strips: dict[str, dict[int, RegimeFit]] = {}
    for weighting in s.weightings:
        ret, counts = sector_returns(rows, weighting, s.min_sector_names)
        thin = thin_sectors(counts, dates, s.min_sector_names)
        fitted = [c for c, t in thin.items() if t["fitted"]]
        ret = ret.reindex(index=dates, columns=fitted)
        series = {"relative": ret.sub(log_ret.reindex(dates), axis=0), "raw": ret}
        for kind in SERIES_KINDS:
            fits = fit_sectors(series[kind], s)
            for code, fit in fits.items():
                measures.append({"weighting": weighting, "series": kind, "sector": code,
                                 "name": SECTOR_NAMES[code]}
                                | sector_measures(fit, market, s.stress_threshold,
                                                  s.min_episode_days))
            co = co_stress(fits, s.stress_threshold) if fits else pd.DataFrame()
            costress.setdefault(weighting, {})[kind] = {
                "sectors": [int(c) for c in co.columns],
                "share_both_stressed": co.round(6).to_numpy().tolist(),
            }
            if weighting == s.weightings[0]:
                strips[kind] = fits

    table = pd.DataFrame(measures)
    thin_rows = pd.DataFrame([{"sector": c} | t for c, t in thin.items()])
    out = root / s.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"sector_regimes_{s.years}"
    table.merge(thin_rows[["sector", "thin_dates", "median_names"]], on="sector",
                how="left").to_csv(out / f"{stem}.csv", index=False)
    report = {
        "what": "brief 11 C: sector regimes on 2000-2006, descriptive input to Q28",
        "caveat": IN_SAMPLE_CAVEAT,
        "does_not_decide": "the industry gate's form (Q28), K (Q18)",
        "expectation_written_before_the_run": (
            "raw sector regimes largely copy the market's; sector-relative regimes should not"),
        "settings": dataclasses.asdict(s),
        "market_gate": regime_summary(market, s.stress_threshold),
        "sector_dates": {"first": str(dates[0].date()), "last": str(dates[-1].date()),
                         "dates": int(len(dates)),
                         "note": "starts on the window's second date: the previous day's "
                                 "market equity is never read from before `start`"},
        "thin_sectors": {str(c): t for c, t in thin.items()},
        "measures": table.astype(object).where(table.notna(), None).to_dict(orient="records"),
        "co_stress": costress,
        "provenance": provenance,
    }
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_strips(market, strips, s, out / f"{stem}.png")
    print(f"[sectors] fitted {sorted(c for c, t in thin.items() if t['fitted'])}; too thin "
          f"{sorted(c for c, t in thin.items() if not t['fitted'])}")
    print(f"[sectors] wrote {s.out_dir}/{stem}.csv, .json and .png")
    return report


def plot_strips(market: RegimeFit, strips: dict[str, dict[int, RegimeFit]], s: Settings,
                path: Path) -> None:
    """Heat strips of the filtered stress probability through the window:
    the market's on top, then each fitted sector; one panel per series kind
    (primary weighting)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(strips), 1, figsize=(12, 0.32 * 12 * len(strips) + 1),
                             squeeze=False)
    for ax, (kind, fits) in zip(axes[:, 0], strips.items(), strict=True):
        labels = ["Market (CRSP S&P 500)"] + [SECTOR_NAMES[c] for c in fits]
        pm = pd.Series(market.stress, index=pd.to_datetime(list(market.labels)))
        cols = [pm] + [pd.Series(f.stress, index=pd.to_datetime(list(f.labels)))
                       for f in fits.values()]
        grid = pd.concat(cols, axis=1).dropna()
        x0, x1 = mdates.date2num(grid.index[0]), mdates.date2num(grid.index[-1])
        im = ax.imshow(grid.T.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1,
                       interpolation="nearest", extent=(x0, x1, len(labels) - 0.5, -0.5))
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.axhline(0.5, color="black", lw=1.0)
        ax.xaxis_date()
        ax.set_title(f"P(stress), filtered, 2-state gate: sector {kind} log return "
                     f"({s.weightings[0]}-weighted)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.text(0.01, 0.003, IN_SAMPLE_CAVEAT, fontsize=6, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_sectors(Settings())
