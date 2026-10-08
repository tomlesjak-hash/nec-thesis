"""Brief 10 C (Q21 follow-up): how the short-horizon signals' IC decays with h.

**Tom runs this on the real panel; it is not run in development.** Dates are
2000-01-01 to 2006-12-31 by default, and the script **refuses** an ``end`` on
or after 2007-01-01 (the pilot's validation years and the test period). There
is no override.

For each signal and each horizon ``h`` in ``horizons``: on each date ``t``, the
Spearman rank correlation across the date's universe between the signal and
the **market-neutral** forward return over ``(t, t + h]``, i.e. the sum of the
next ``h`` daily log returns (the target's series, with the same
post-delisting fill, extended to ``h`` days) minus the date's equal-weighted
mean over the names with a valid value. Then the mean IC over dates and its
Hansen-Hodrick t-statistic with ``h - 1`` lags (:func:`nec_moe.evaluation.ic_summary`,
uniform kernel; it falls back to Bartlett only if the uniform long-run
variance comes out non-positive, and the output says which kernel was used).

Details that are choices of the code, stated so they can be checked:

- **No return after ``end`` is read.** The daily returns are cut at ``end``
  before any forward sum, so a forward window that would pass ``end`` is
  missing: each horizon loses its last ``h`` dates, as the pilot's 2007 fold
  purges the last dates of 2006.
- The universe of a date is the panel's rows on that date (its index
  members), and the signals are the panel's characteristics as ranked there
  (a missing characteristic is the panel's 0).
- The Spearman correlation uses **average ranks for ties**. Several signals
  are discrete (``earn_next5`` is 0/1, a missing value is 0), and
  ``rank_ic_by_date`` breaks ties by row order, which suits continuous
  predictions only; it is not used here.
- A Spearman correlation is unchanged by subtracting a per-date constant, so
  the market-neutral demeaning does not move the ICs; it is applied because
  the brief defines the target that way.

Outputs (aggregate only): ``results/diagnostics/ic_decay_2000_2006.csv`` and
``.json`` (signal, h, mean IC, t-statistic, number of dates, the HAC lags and
kernel) and ``.png`` (IC against h, one line per signal). Nothing per-security
is written anywhere.

Usage (from nec_baseline/; reads the 2000-2024 panel and the CRSP extract,
several GB of memory)::

    caffeinate -i python3.14 scripts/ic_decay.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nec_moe.characteristics import JKP_FEATURES, MARKET_FEATURES  # noqa: E402
from nec_moe.evaluation import ic_summary  # noqa: E402
from nec_moe.features import _forward_sum, _target_returns  # noqa: E402

#: The first date this script may never reach: the pilot's validation years
#: (2007-2009) and the test period (2010 on) start here.
FORBIDDEN_FROM = pd.Timestamp("2007-01-01")

#: JKP short-term reversal (ret_5d, ret_20d) and momentum (mom_12_1,
#: prc_highprc_252d), then the 12 short-horizon market characteristics.
DEFAULT_SIGNALS: tuple[str, ...] = (*JKP_FEATURES[:4], *MARKET_FEATURES)
assert JKP_FEATURES[:4] == ("ret_5d", "ret_20d", "mom_12_1", "prc_highprc_252d")


@dataclass(frozen=True)
class Settings:
    """Every setting of the script, with its default."""

    start: str = "2000-01-01"
    end: str = "2006-12-31"
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
    signals: tuple[str, ...] = DEFAULT_SIGNALS
    # dates with fewer names holding both a signal and a forward return are
    # skipped (rank_ic_by_date's default)
    min_names: int = 3
    # relative to nec_baseline/; the panel run_experiment.py reads by default
    panel_file: str = (
        "../Data/derived/pit_panel_crsp_2000-01-01_2024-12-31_q26_market_neutral.pt"
    )
    out_dir: str = "results/diagnostics"

    def validate(self) -> Settings:
        check_end(self.end)
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError(f"start {self.start} is after end {self.end}")
        if not self.horizons or any(int(h) != h or h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be positive integers, got {self.horizons}")
        if not self.signals:
            raise ValueError("no signals")
        if self.min_names < 3:
            raise ValueError(f"min_names must be >= 3, got {self.min_names}")
        return self

    @property
    def stem(self) -> str:
        return f"ic_decay_{pd.Timestamp(self.start).year}_{pd.Timestamp(self.end).year}"


def check_end(end: str | pd.Timestamp) -> None:
    """Refuse any end date on or after 2007-01-01. There is no override."""
    if pd.Timestamp(end) >= FORBIDDEN_FROM:
        raise ValueError(
            f"end={pd.Timestamp(end).date()}: the IC decay script reads nothing on or "
            "after 2007-01-01 (the pilot's validation years and the test period); "
            "there is no override"
        )


def restrict(frame: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """The rows of a dates x names frame inside ``[start, end]``; nothing later."""
    check_end(s.end)
    idx = pd.DatetimeIndex(frame.index)
    out = frame.loc[(idx >= pd.Timestamp(s.start)) & (idx <= pd.Timestamp(s.end))]
    assert out.empty or pd.Timestamp(out.index.max()) < FORBIDDEN_FROM
    return out


# --------------------------------------------------------------------------- #
# The computation (pure functions of dates x names frames)
# --------------------------------------------------------------------------- #


def forward_log_returns(daily: pd.DataFrame, h: int) -> pd.DataFrame:
    """Sum of the next ``h`` daily log returns, ``(t, t + h]``, per name; missing
    if any of them is, or if the window passes the frame's last date (the same
    rule as the panel's target, :func:`nec_moe.features._forward_sum`)."""
    out = _forward_sum(daily, h)
    assert isinstance(out, pd.DataFrame)
    return out


def market_neutral(fwd: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """``fwd`` minus each date's equal-weighted mean over the universe's names
    with a valid value (Q25's target, at any horizon); missing outside the
    universe."""
    x = fwd.where(_mask(universe, fwd))
    return x.sub(x.mean(axis=1), axis=0)


def _mask(universe: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """``universe`` on ``like``'s dates and names; absent means outside."""
    return universe.reindex(index=like.index, columns=like.columns, fill_value=False).astype(bool)


def spearman_ic_by_date(
    signal: pd.DataFrame, target: pd.DataFrame, min_names: int = 3
) -> pd.Series:
    """Per-date Spearman correlation across names, over the names where both
    are present, with **average ranks for ties**. Dates with fewer than
    ``min_names`` such names, or a constant signal or target, are skipped."""
    signal, target = signal.align(target, join="inner")
    both = signal.notna() & target.notna()
    rs = signal.where(both).rank(axis=1, method="average")
    rt = target.where(both).rank(axis=1, method="average")
    rs = rs.sub(rs.mean(axis=1), axis=0)
    rt = rt.sub(rt.mean(axis=1), axis=0)
    num = (rs * rt).sum(axis=1)
    den = np.sqrt((rs * rs).sum(axis=1) * (rt * rt).sum(axis=1))
    ok = (both.sum(axis=1) >= min_names) & (den > 0)
    return (num[ok] / den[ok]).astype(float)


def ic_decay(
    signals: Mapping[str, pd.DataFrame],
    daily: pd.DataFrame,
    universe: pd.DataFrame,
    horizons: Sequence[int],
    min_names: int = 3,
) -> pd.DataFrame:
    """One row per (signal, h): mean IC, Hansen-Hodrick t-statistic with
    ``h - 1`` lags, number of dates, the lags and the kernel used."""
    rows: list[dict[str, Any]] = []
    for h in horizons:
        target = market_neutral(forward_log_returns(daily, h), universe)
        for name, sig in signals.items():
            ic = spearman_ic_by_date(sig.where(_mask(universe, sig)), target, min_names)
            summary = ic_summary(torch.tensor(ic.to_numpy(), dtype=torch.float64),
                                 hac_lags=h - 1, hac_kernel="uniform")
            rows.append({
                "signal": name, "h": int(h),
                "mean_ic": summary.mean_ic, "t_stat": summary.t_stat,
                "n_dates": summary.n_dates, "hac_lags": summary.hac_lags,
                "hac_kernel": summary.hac_kernel,
                "first_date": str(pd.Timestamp(ic.index.min()).date()),
                "last_date": str(pd.Timestamp(ic.index.max()).date()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inputs (real data: Tom runs this part)
# --------------------------------------------------------------------------- #


def cut_at(extract: Any, end: str | pd.Timestamp) -> Any:
    """The CRSP extract with every stock row and market return after ``end``
    removed, before any return is computed: a forward window or a
    post-delisting fill cannot reach past ``end``."""
    check_end(end)
    end = pd.Timestamp(end)
    return dataclasses.replace(
        extract,
        stock=extract.stock[extract.stock["DlyCalDt"] <= end],
        market=extract.market[pd.DatetimeIndex(extract.market.index) <= end],
    )


def load_inputs(s: Settings) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame,
                                      dict[str, Any]]:
    """``(signals, daily target returns, universe, provenance)``, dates x PERMNOs,
    all inside ``[start, end]``."""
    from nec_moe.crsp import CRSPSpec, crsp_daily_frames, load_extract
    from nec_moe.features import StageBSpec

    panel = torch.load(ROOT / s.panel_file, weights_only=False, mmap=True)
    meta = panel.metadata or {}
    if meta.get("feature_set") != "q26":
        raise ValueError(f"{s.panel_file} is not a Q26 panel")
    names = list(panel.schema.snapshot_features)
    absent = [x for x in s.signals if x not in names]
    if absent:
        raise ValueError(f"signals {absent} are not inputs of the panel")
    labels = pd.to_datetime(np.asarray(panel.date_labels))
    codes = np.flatnonzero((labels >= pd.Timestamp(s.start)) & (labels <= pd.Timestamp(s.end)))
    rows = torch.isin(panel.date, torch.from_numpy(codes)).nonzero().squeeze(1)
    when = labels[panel.date[rows].numpy()]
    who = np.asarray(panel.entity_labels)[panel.entity[rows].numpy()]
    index = pd.MultiIndex.from_arrays([when, who], names=["date", "permno"])
    x = panel.x_snap[rows][:, [names.index(n) for n in s.signals]].numpy()
    signals = {
        n: restrict(pd.Series(x[:, j], index=index).unstack("permno"), s)
        for j, n in enumerate(s.signals)
    }
    universe = restrict(pd.Series(True, index=index).unstack("permno", fill_value=False), s)
    del panel, x

    # the target's daily returns, from the extract the panel was built from,
    # cut at `end` before anything else (no later return is ever read)
    spec = CRSPSpec(release=meta["crsp_release"], stock_file=meta["crsp_stock_file"],
                    post_delisting_return=meta["post_delisting_return"])
    extract = load_extract(spec)
    if int(extract.info.get("market_indno", -1)) != int(meta["market_indno"]):
        raise ValueError("the extract's market index is not the panel's")
    extract = cut_at(extract, s.end)
    frames, _, _ = crsp_daily_frames(
        extract, spec, StageBSpec(horizon=max(s.horizons)),
        permnos=[int(p) for p in universe.columns],
    )
    daily = pd.DataFrame({p: _target_returns(f) for p, f in frames.items()})
    daily = restrict(daily.reindex(columns=universe.columns), s)
    provenance = {
        "panel_file": s.panel_file, "crsp_release": meta["crsp_release"],
        "post_delisting_return": meta["post_delisting_return"],
        "extract": spec.extract_dir.name, "dates": int(len(universe)),
        "names": int(universe.shape[1]),
    }
    return signals, daily, universe, provenance


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def write_outputs(table: pd.DataFrame, s: Settings, provenance: Mapping[str, Any],
                  root: Path = ROOT) -> dict[str, Path]:
    """The CSV, the JSON and the figure; aggregate rows only."""
    out = root / s.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"{s.stem}.{k}" for k in ("csv", "json", "png")}
    table.to_csv(paths["csv"], index=False)
    paths["json"].write_text(json.dumps({
        "what": "brief 10 C: IC decay with the horizon, 2000-2006 only (Q21 follow-up)",
        "settings": dataclasses.asdict(s),
        "provenance": dict(provenance),
        "notes": [
            "Spearman with average ranks for ties, per date over the panel's rows",
            "target: sum of the next h daily log returns (post-delisting fill as the "
            "panel's), minus the date's equal-weighted mean over names with a value",
            "daily returns cut at `end`: each horizon loses its last h dates",
            "t-statistic: Hansen-Hodrick (uniform kernel), h - 1 lags; hac_kernel "
            "says if the Bartlett fallback was used",
        ],
        "rows": table.to_dict(orient="records"),
    }, indent=2) + "\n")
    plot_decay(table, paths["png"])
    return paths


def plot_decay(table: pd.DataFrame, path: Path) -> None:
    """Mean IC against h, one line per signal: the JKP signals and the
    short-horizon market block side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"JKP reversal and momentum": list(JKP_FEATURES[:4]),
              "short-horizon market block": list(MARKET_FEATURES)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (title, members) in zip(axes, groups.items(), strict=True):
        for name, g in table[table["signal"].isin(members)].groupby("signal", sort=False):
            g = g.sort_values("h")
            ax.plot(g["h"], g["mean_ic"], marker="o", label=str(name))
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("horizon h (trading days)")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("mean daily rank IC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(s: Settings) -> pd.DataFrame:
    s = s.validate()
    signals, daily, universe, provenance = load_inputs(s)
    table = ic_decay(signals, daily, universe, s.horizons, s.min_names)
    paths = write_outputs(table, s, provenance)
    print(table.to_string(index=False))
    for p in paths.values():
        print(f"[ic decay] wrote {p.relative_to(ROOT)}")
    return table


if __name__ == "__main__":
    main(Settings())
