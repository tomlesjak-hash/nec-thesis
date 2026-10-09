"""Brief 11 A and B: the market's regimes on 2000-2006, and the IC curve split by them.

**Descriptive diagnostics; Tom runs them on real data.** Every date is inside
2000-01-01 to 2006-12-31: the script refuses any end on or after 2007-01-01
(no override), the market series and the stock returns are cut at ``end``
before anything is computed, and forward windows end by ``end``, so the last
scored date moves back by h (as in brief 10 C).

Part A (``regimes``): the Hamilton gate (K = 2 primary, K = 3 secondary, full
memory, the default estimator and multi-start, canonical order by variance)
fitted on the CRSP S&P 500 index's daily log return (the gate's
``crsp_market_log_return`` series, decimals) over the whole span; its
parameters, durations, regime shares and multi-start trace, and a figure of
the market's cumulative log return with the filtered stress probability.
Writes ``results/diagnostics/market_regimes_2000_2006.json`` and ``.png``.

Part B (``ic``): the IC curve split by those regimes. For the 18 signals
(brief 10 C's 16 and the two industry signals) and h = 1, 2, 3, 5, 10, the
daily IC exactly as in brief 10 C; per (signal, h, K) the regime-weighted mean
IC with Kish's effective dates, and the pure-regime IC ``c_k`` from
``IC_t = sum_k c_k w_k(t) + e_t`` with Hansen-Hodrick (lag h - 1) and
Newey-West (Bartlett, lag max(h - 1, nw_min_lag)) standard errors and the
contrast ``c_stress - c_calm``. Robustness: the filtered weight, the hard
split ``1[wbar_stress > 0.5]`` (for K = 3, stress against the rest), and the
parameter-free ``stress_vol`` state. Families, declared before the run (in
the JSON): five one-sided primary tests (K = 2, h = 5, Hansen-Hodrick, the
soft regression on the ``weight_kind`` weights) with Holm; every other
(signal, h, K) contrast of that specification is exploratory, two-sided,
with Benjamini-Hochberg q-values; everything else is robustness with raw
p-values. Writes ``results/diagnostics/ic_regime_2000_2006.csv`` and
``.json``, ``ic_regime_2000_2006.png`` (one panel per signal) and
``ic_regime_contrast_2000_2006.png`` (the primary contrasts against h).

The gate's parameters are estimated on all of 2000-2006, so the probabilities
are causal in the data they filter but not in the parameters
(:data:`nec_moe.regime_split.IN_SAMPLE_CAVEAT`, written into every output).

Usage (from nec_baseline/)::

    python3.14 scripts/ic_regime_split.py regimes      # part A
    caffeinate -i python3.14 scripts/ic_regime_split.py ic   # part B
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ic_decay as icd  # noqa: E402  (the guard, restrict, the IC pieces)

from nec_moe import MarkovGateConfig, benjamini_hochberg, holm  # noqa: E402
from nec_moe.crsp import MARKET_LOG_RETURN_KEY, CRSPSpec, market_log_return_by_date  # noqa: E402
from nec_moe.regime_split import (  # noqa: E402
    HAC_CHOICES,
    IN_SAMPLE_CAVEAT,
    PRIMARY_HYPOTHESES,
    PRIMARY_SPEC,
    WEIGHT_KINDS,
    RegimeFit,
    fit_regimes,
    hac_choice,
    hard_split,
    p_value,
    regime_regression,
    regime_summary,
    regime_weights,
    series_panel,
    trailing_vol_state,
    weighted_regime_ic,
)

#: brief 10 C's 16 signals and the two industry signals (B.1)
SIGNALS: tuple[str, ...] = (*icd.DEFAULT_SIGNALS, "ret_20d_ind_rel", "ind_mom_12_1")


@dataclass(frozen=True)
class Settings:
    """Every setting of parts A and B, with its default."""

    start: str = "2000-01-01"
    end: str = "2006-12-31"
    # K = 2 primary, K = 3 secondary; both reported, Q18 is not decided here
    k_list: tuple[int, ...] = (2, 3)
    # the gate's estimator ("statsmodels", the harness default, or "native")
    fit_backend: str = "statsmodels"
    # the target horizon the gate's date panel is labelled with (the study's h)
    gate_horizon: int = 5
    # the regime weight the hypothesis families use (A.3); the other kind is
    # reported as robustness
    weight_kind: str = "window"
    # A.5: the trailing window (trading days) of the parameter-free state
    vol_window: int = 20
    # a date is "stress" in the hard split when its stress weight exceeds this
    stress_threshold: float = 0.5
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
    signals: tuple[str, ...] = SIGNALS
    min_names: int = 3  # as brief 10 C
    # Newey-West's lag is max(h - 1, nw_min_lag) (B.2)
    nw_min_lag: int = 20
    # Holm's family-wise level for the primary tests, Benjamini-Hochberg's
    # false-discovery level for the exploratory family (reject flags only;
    # the adjusted p- and q-values are reported whatever these are)
    primary_alpha: float = 0.05
    exploratory_fdr: float = 0.10
    panel_file: str = icd.Settings().panel_file
    out_dir: str = "results/diagnostics"

    def validate(self) -> Settings:
        icd.check_end(self.end)
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError(f"start {self.start} is after end {self.end}")
        if not self.k_list or any(k < 2 for k in self.k_list) or self.k_list[0] != 2:
            raise ValueError(f"k_list must start with the primary K = 2, got {self.k_list}")
        if self.weight_kind not in WEIGHT_KINDS:
            raise ValueError(f"weight_kind {self.weight_kind!r}: use one of {WEIGHT_KINDS}")
        if self.fit_backend not in ("statsmodels", "native"):
            raise ValueError(f"unknown fit_backend {self.fit_backend!r}")
        if not 0.0 < self.stress_threshold < 1.0 or self.vol_window < 2 or self.nw_min_lag < 0:
            raise ValueError("stress_threshold in (0, 1), vol_window >= 2, nw_min_lag >= 0")
        self.ic_settings().validate()
        primary = {x for hyp in PRIMARY_HYPOTHESES for x in hyp["signals"]}
        if not primary <= set(self.signals) or PRIMARY_SPEC["h"] not in self.horizons:
            raise ValueError(
                "the primary hypotheses are fixed (B.4): signals must include "
                f"{sorted(primary)} and horizons must include h = {PRIMARY_SPEC['h']}"
            )
        if not (0 < self.primary_alpha < 1 and 0 < self.exploratory_fdr < 1):
            raise ValueError("primary_alpha and exploratory_fdr must be in (0, 1)")
        return self

    def ic_settings(self) -> icd.Settings:
        """Brief 10 C's settings with these signals, horizons and dates."""
        return icd.Settings(start=self.start, end=self.end, horizons=self.horizons,
                            signals=self.signals, min_names=self.min_names,
                            panel_file=self.panel_file, out_dir=self.out_dir)

    @property
    def years(self) -> str:
        return f"{pd.Timestamp(self.start).year}_{pd.Timestamp(self.end).year}"


# --------------------------------------------------------------------------- #
# Part A: the market's regimes
# --------------------------------------------------------------------------- #


def load_market(s: Settings, spec: CRSPSpec | None = None) -> pd.Series:
    """The CRSP market index's daily log return on its trading days in
    ``[start, end]``, from the extract the 2000-2024 panel was built from;
    cut at ``end`` before the log is taken."""
    spec = spec if spec is not None else CRSPSpec()
    raw = pd.read_parquet(spec.extract_dir / "market.parquet")["DlyTotRet"].astype(float)
    raw.index = pd.DatetimeIndex(raw.index).normalize()
    cut = icd.restrict(raw.to_frame("ret"), s)["ret"]  # type: ignore[arg-type]
    if cut.isna().any():
        raise ValueError(f"{int(cut.isna().sum())} market return(s) missing in the window")
    return np.log1p(cut)


def market_panel(log_ret: pd.Series, s: Settings) -> Any:
    """The market series as the gate reads it: the ``crsp_market_log_return``
    key, stored with a one-row-per-date panel."""
    labels = [str(d.date()) for d in pd.DatetimeIndex(log_ret.index)]
    stored = market_log_return_by_date(log_ret, labels)
    return series_panel(labels, log_ret.to_numpy(), horizon=s.gate_horizon,
                        metadata={MARKET_LOG_RETURN_KEY: stored})


def gate_config(s: Settings) -> MarkovGateConfig:
    """The gate's default configuration on the CRSP series, with the
    settings' estimator; switching mean and variance, order 0 (the default)."""
    return MarkovGateConfig(series="crsp_market_log_return",
                            fit_backend=s.fit_backend)  # type: ignore[arg-type]


def fit_market(log_ret: pd.Series, s: Settings) -> dict[int, RegimeFit]:
    """One fit per K in ``k_list``, on the whole span."""
    panel = market_panel(log_ret, s)
    return {k: fit_regimes(panel, k, gate_config(s)) for k in s.k_list}


def regimes_report(log_ret: pd.Series, fits: dict[int, RegimeFit], s: Settings) -> dict:
    """Part A's aggregate output (no daily series)."""
    state, threshold = trailing_vol_state(log_ret, s.vol_window)
    return {
        "what": "brief 11 A: market regimes on 2000-2006 (descriptive)",
        "caveat": IN_SAMPLE_CAVEAT,
        "settings": dataclasses.asdict(s),
        "series": "crsp_market_log_return: log(1 + DlyTotRet) of INDNO 1000500, decimals",
        "window": {"first": str(log_ret.index[0].date()), "last": str(log_ret.index[-1].date()),
                   "dates": int(len(log_ret))},
        "fits": {str(k): regime_summary(f, s.stress_threshold) for k, f in fits.items()},
        "stress_vol": {
            "rule": f"trailing {s.vol_window}-day sd of the log return above its median",
            "median_sd": threshold,
            "share_of_days_stressed": float(state.mean()),
            "dates_without_value": int(state.isna().sum()),
        },
    }


def plot_regimes(log_ret: pd.Series, fits: dict[int, RegimeFit], path: Path) -> None:
    """The market's cumulative log return with each K's filtered stress
    probability shaded beneath it, one panel per K."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(fits), 1, figsize=(11, 3.2 * len(fits)), sharex=True,
                             squeeze=False)
    cum = log_ret.cumsum()
    for ax, (k, fit) in zip(axes[:, 0], fits.items(), strict=True):
        days = pd.DatetimeIndex(pd.to_datetime(list(fit.labels)))
        ax2 = ax.twinx()
        ax2.fill_between(days, 0.0, fit.stress, step="mid", color="#d62728", alpha=0.25,
                         linewidth=0, label="P(stress), filtered")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(stress)")
        ax.plot(cum.index, cum.to_numpy(), color="#1f1f1f", lw=1.0,
                label="cumulative log return")
        ax.set_ylabel("cumulative log return")
        ax.set_title(f"K = {k}: stress = highest-variance regime")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    fig.text(0.01, 0.005, IN_SAMPLE_CAVEAT, fontsize=7, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_regimes(s: Settings, root: Path = ROOT, log_ret: pd.Series | None = None) -> dict:
    """Part A: fit, report, figure. ``log_ret`` (synthetic, in tests) skips
    the extract."""
    s = s.validate()
    log_ret = load_market(s) if log_ret is None else icd.restrict(
        log_ret.to_frame("r"), s)["r"]  # type: ignore[arg-type]
    fits = fit_market(log_ret, s)
    report = regimes_report(log_ret, fits, s)
    out = root / s.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"market_regimes_{s.years}.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_regimes(log_ret, fits, out / f"market_regimes_{s.years}.png")
    for k, f in fits.items():
        print(f"[regimes] K={k}: converged {f.n_converged}/{f.n_starts}, sd "
              f"{np.round(np.sqrt(f.variances), 4).tolist()}, durations "
              f"{np.round(f.expected_durations, 1).tolist()}")
    print(f"[regimes] wrote {s.out_dir}/market_regimes_{s.years}.json and .png")
    return report


# --------------------------------------------------------------------------- #
# Part B: the IC curve split by regime
# --------------------------------------------------------------------------- #

CONTRAST = "contrast"  # the term name of c_stress - c_calm (c_stress - c_rest)


def split_rows(signal: str, h: int, k: int | None, kind: str, split: str, ic: pd.Series,
               weights: pd.DataFrame, s: Settings) -> list[dict[str, Any]]:
    """One row per regime (c_k, weighted IC, effective dates) and one for the
    contrast, under both HAC choices, on the dates with an IC and a weight."""
    both = ic.index.intersection(weights.dropna().index)
    y, x = ic.loc[both].to_numpy(dtype=float), weights.loc[both].to_numpy(dtype=float)
    wmean, eff = weighted_regime_ic(y, x)
    reg = {}
    for choice in HAC_CHOICES:
        lags, kernel = hac_choice(choice, h, s.nw_min_lag)
        reg[choice] = regime_regression(y, x, lags, kernel)
    hh, nw = reg["hansen_hodrick"], reg["newey_west"]
    base = {"signal": signal, "h": int(h), "K": k, "weight_kind": kind, "split": split,
            "n_dates": int(len(both)), "lag_hh": hh.lags, "lag_nw": nw.lags}
    rows = []
    for j, regime in enumerate(weights.columns):
        rows.append(base | {
            "term": str(regime), "weighted_ic": float(wmean[j]), "eff_dates": float(eff[j]),
            "estimate": float(hh.coef[j]), "se_hh": float(hh.se[j]), "t_hh": float(hh.t[j]),
            "kernel_hh": hh.kernels[j], "se_nw": float(nw.se[j]), "t_nw": float(nw.t[j]),
            "kernel_nw": nw.kernels[j],
        })
    rows.append(base | {
        "term": CONTRAST, "contrast_of": f"{weights.columns[-1]} - {weights.columns[0]}",
        "estimate": hh.contrast, "se_hh": hh.contrast_se, "t_hh": hh.contrast_t,
        "kernel_hh": hh.kernels[-1], "se_nw": nw.contrast_se, "t_nw": nw.contrast_t,
        "kernel_nw": nw.kernels[-1],
        "p_nw_two_sided": p_value(nw.contrast_t, "two-sided"),
    })
    return rows


def split_table(ics_by_h: dict[int, dict[str, pd.Series]], fits: dict[int, RegimeFit],
                vol_state: pd.Series, s: Settings) -> pd.DataFrame:
    """Every split of every (signal, h): soft weights of each kind and K, the
    hard split on the family's weight kind, and the stress_vol state."""
    kinds = (s.weight_kind, *(k for k in WEIGHT_KINDS if k != s.weight_kind))
    vol = pd.DataFrame(hard_split(vol_state.to_numpy(), 0.5), index=vol_state.index,
                       columns=["calm", "stress"])
    rows: list[dict[str, Any]] = []
    for h, ics in ics_by_h.items():
        weights = {(k, kind): regime_weights(fit, kind, h)
                   for k, fit in fits.items() for kind in kinds}
        for name, ic in ics.items():
            for k in fits:
                for kind in kinds:
                    rows += split_rows(name, h, k, kind, "soft", ic, weights[(k, kind)], s)
                w = weights[(k, s.weight_kind)]
                hard = pd.DataFrame(hard_split(w["stress"].to_numpy(), s.stress_threshold),
                                    index=w.index,
                                    columns=["calm", "stress"] if k == 2 else ["rest", "stress"])
                rows += split_rows(name, h, k, s.weight_kind, "hard", ic, hard, s)
            rows += split_rows(name, h, None, "none", "stress_vol", ic, vol, s)
    return pd.DataFrame(rows)


def assign_families(table: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """The B.4 families on the contrast rows. Primary: the five pre-specified
    one-sided tests, Holm. Exploratory: every other (signal, h, K) contrast of
    the same specification, two-sided, Benjamini-Hochberg. Robustness: the
    other contrasts, raw two-sided p-values."""
    t = table.copy()
    t["family"], t["alternative"] = "", ""
    t["p"], t["adjusted"], t["adjustment"] = np.nan, np.nan, ""
    t["reject"] = pd.Series(pd.NA, index=t.index, dtype="boolean")
    contrast = t["term"] == CONTRAST
    spec = contrast & (t["split"] == PRIMARY_SPEC["split"]) & (t["weight_kind"] == s.weight_kind)
    primary_signals = {x for hyp in PRIMARY_HYPOTHESES for x in hyp["signals"]}
    primary = spec & (t["K"] == PRIMARY_SPEC["k"]) & (t["h"] == PRIMARY_SPEC["h"]) & \
        t["signal"].isin(primary_signals)
    exploratory = spec & ~primary
    robustness = contrast & ~spec
    if int(primary.sum()) != len(primary_signals):
        raise AssertionError(f"{int(primary.sum())} primary tests, expected {len(primary_signals)}")
    t.loc[primary, "family"], t.loc[primary, "alternative"] = "primary", "less"
    t.loc[exploratory, "family"], t.loc[exploratory, "alternative"] = "exploratory", "two-sided"
    t.loc[robustness, "family"], t.loc[robustness, "alternative"] = "robustness", "two-sided"
    for mask in (primary, exploratory, robustness):
        t.loc[mask, "p"] = [p_value(x, a) for x, a in
                            zip(t.loc[mask, "t_hh"], t.loc[mask, "alternative"], strict=True)]
    for mask, name, fn, level in ((primary, "holm", holm, s.primary_alpha),
                                  (exploratory, "benjamini_hochberg", benjamini_hochberg,
                                   s.exploratory_fdr)):
        ok = mask & t["p"].notna()
        if ok.any():
            reject, adjusted = fn(t.loc[ok, "p"].tolist(), level)
            t.loc[ok, "adjusted"], t.loc[ok, "adjustment"] = adjusted, name
            t.loc[ok, "reject"] = reject
    return t


def families_declaration(s: Settings, table: pd.DataFrame) -> dict[str, Any]:
    """What the JSON states about the families (B.4), fixed before the run."""
    return {
        "primary": {
            "hypotheses": [dict(h, signals=list(h["signals"])) for h in PRIMARY_HYPOTHESES],
            "specification": PRIMARY_SPEC | {"weight_kind": s.weight_kind},
            "tests": int((table["family"] == "primary").sum()),
            "p_values": "one-sided (contrast < 0), normal approximation; Holm-adjusted",
        },
        "exploratory": {
            "definition": "every other (signal, h, K) contrast c_stress - c_calm of the primary "
                          "specification, two-sided; Benjamini-Hochberg q-values across the "
                          "family. Exploratory: not a test of a pre-specified hypothesis.",
            "tests": int((table["family"] == "exploratory").sum()),
        },
        "robustness": {
            "definition": "the filtered weight, the hard split (K = 3: stress against the "
                          "rest) and the stress_vol state; raw two-sided p-values, outside "
                          "both families. Newey-West t-statistics are on every row.",
            "tests": int((table["family"] == "robustness").sum()),
        },
    }


def plot_split(table: pd.DataFrame, s: Settings, path: Path) -> None:
    """One panel per signal: c_calm and c_stress against h (K = 2, the soft
    regression on the family's weights), with +-2 Hansen-Hodrick SE bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = table[(table["K"] == 2) & (table["split"] == "soft")
                & (table["weight_kind"] == s.weight_kind)]
    n = len(s.signals)
    cols = 6
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows), sharex=True,
                             squeeze=False)
    style = {"calm": ("#1f77b4", "-", "o"), "stress": ("#d62728", "--", "s")}
    for ax, name in zip(axes.flat, s.signals, strict=False):
        for regime, (colour, ls, marker) in style.items():
            g = sel[(sel["signal"] == name) & (sel["term"] == regime)].sort_values("h")
            ax.fill_between(g["h"], g["estimate"] - 2 * g["se_hh"],
                            g["estimate"] + 2 * g["se_hh"], color=colour, alpha=0.15, lw=0)
            ax.plot(g["h"], g["estimate"], color=colour, ls=ls, marker=marker, ms=3,
                    label=regime)
        ax.axhline(0.0, color="grey", lw=0.7)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("horizon h (trading days)", fontsize=9)
    fig.supylabel("pure-regime IC c_k (K = 2), +-2 HH s.e.", fontsize=9)
    fig.text(0.01, 0.002, IN_SAMPLE_CAVEAT, fontsize=6, wrap=True, va="bottom")
    fig.tight_layout(rect=(0.02, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


#: distinct colour, line style and marker per primary signal
PRIMARY_STYLES: tuple[tuple[str, str, str], ...] = (
    ("#1f77b4", "-", "o"), ("#ff7f0e", "--", "s"), ("#2ca02c", "-.", "^"),
    ("#d62728", ":", "D"), ("#9467bd", "-", "v"),
)


def plot_contrast(table: pd.DataFrame, s: Settings, path: Path) -> None:
    """The contrast c_stress - c_calm against h for the primary signals (K =
    2, the soft regression on the family's weights), +-2 HH s.e."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = table[(table["K"] == 2) & (table["split"] == "soft") & (table["term"] == CONTRAST)
                & (table["weight_kind"] == s.weight_kind)]
    names = [x for hyp in PRIMARY_HYPOTHESES for x in hyp["signals"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, (colour, ls, marker) in zip(names, PRIMARY_STYLES, strict=True):
        g = sel[sel["signal"] == name].sort_values("h")
        ax.fill_between(g["h"], g["estimate"] - 2 * g["se_hh"], g["estimate"] + 2 * g["se_hh"],
                        color=colour, alpha=0.10, lw=0)
        ax.plot(g["h"], g["estimate"], color=colour, ls=ls, marker=marker, label=name)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("horizon h (trading days)")
    ax.set_ylabel("c_stress - c_calm (K = 2), +-2 HH s.e.")
    ax.set_title("Regime contrast of the primary signals")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, IN_SAMPLE_CAVEAT, fontsize=6, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_ic(s: Settings, root: Path = ROOT, inputs: tuple | None = None,
           log_ret: pd.Series | None = None) -> pd.DataFrame:
    """Part B. ``inputs`` (signals, daily returns, universe, provenance) and
    ``log_ret`` (synthetic, in tests) skip the panel and the extract."""
    s = s.validate()
    if inputs is None:
        inputs = icd.load_inputs(s.ic_settings())
    signals, daily, universe, provenance = inputs
    ic_s = s.ic_settings()
    signals = {n: icd.restrict(f, ic_s) for n, f in signals.items()}
    daily, universe = icd.restrict(daily, ic_s), icd.restrict(universe, ic_s)
    log_ret = load_market(s) if log_ret is None else icd.restrict(
        log_ret.to_frame("r"), ic_s)["r"]
    fits = fit_market(log_ret, s)
    vol_state, vol_median = trailing_vol_state(log_ret, s.vol_window)
    ics_by_h = {h: icd.daily_ics(signals, daily, universe, h, s.min_names) for h in s.horizons}
    table = assign_families(split_table(ics_by_h, fits, vol_state, s), s)

    out = root / s.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"ic_regime_{s.years}"
    table.to_csv(out / f"{stem}.csv", index=False)
    report = {
        "what": "brief 11 B: the IC curve split by regime, 2000-2006 (descriptive)",
        "caveat": IN_SAMPLE_CAVEAT,
        "settings": dataclasses.asdict(s),
        "families": families_declaration(s, table),
        "gate_fits": {str(k): regime_summary(f, s.stress_threshold) for k, f in fits.items()},
        "stress_vol": {"window": s.vol_window, "median_sd": vol_median},
        "ic": "Spearman (average ranks) with the market-neutral forward log return over "
              "(t, t+h], as brief 10 C; returns cut at `end`",
        "provenance": dict(provenance),
        "primary_results": _records(table[table["family"] == "primary"][
            ["signal", "h", "K", "estimate", "se_hh", "t_hh", "p", "adjusted", "reject",
             "t_nw", "p_nw_two_sided"]]),
        "rows": _records(table),
    }
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    plot_split(table, s, out / f"{stem}.png")
    plot_contrast(table, s, out / f"ic_regime_contrast_{s.years}.png")
    print(table[table["family"] == "primary"][
        ["signal", "estimate", "t_hh", "p", "adjusted", "t_nw"]].to_string(index=False))
    print(f"[ic split] wrote {s.out_dir}/{stem}.csv, .json, .png and "
          f"ic_regime_contrast_{s.years}.png")
    return table


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON records, missing values as null (strict JSON, no NaN)."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _jsonable(x: Any) -> Any:
    """NaN-safe JSON for pandas' missing values."""
    if x is pd.NA:
        return None
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def main(argv: list[str]) -> None:
    if argv[:1] == ["regimes"]:
        run_regimes(Settings())
    elif argv[:1] == ["ic"]:
        run_ic(Settings())
    else:
        raise SystemExit("usage: python3.14 scripts/ic_regime_split.py regimes|ic")


if __name__ == "__main__":
    main(sys.argv[1:])
