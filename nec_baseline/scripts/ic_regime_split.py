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

The gate's parameters are estimated on all of 2000-2006, so the probabilities
are causal in the data they filter but not in the parameters
(:data:`nec_moe.regime_split.IN_SAMPLE_CAVEAT`, written into every output).

Usage (from nec_baseline/)::

    python3.14 scripts/ic_regime_split.py regimes      # part A
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

from nec_moe import MarkovGateConfig  # noqa: E402
from nec_moe.crsp import MARKET_LOG_RETURN_KEY, CRSPSpec, market_log_return_by_date  # noqa: E402
from nec_moe.regime_split import (  # noqa: E402
    IN_SAMPLE_CAVEAT,
    WEIGHT_KINDS,
    RegimeFit,
    fit_regimes,
    regime_summary,
    series_panel,
    trailing_vol_state,
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
        icd.Settings(start=self.start, end=self.end, horizons=self.horizons,
                     signals=self.signals, min_names=self.min_names).validate()
        return self

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


def main(argv: list[str]) -> None:
    if argv[:1] == ["regimes"]:
        run_regimes(Settings())
    else:
        raise SystemExit("usage: python3.14 scripts/ic_regime_split.py regimes")


if __name__ == "__main__":
    main(sys.argv[1:])
