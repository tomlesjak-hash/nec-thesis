"""Brief 12: the reporting additions for a finished run (Q9 update 2026-10-09).

**Reporting only, after the run.** It reads what the run saved: each arm's
per-fold predictions (``Data/derived/runs/<run_id>/<group>/``; per-security,
never copied out of ``Data/``), the panel (checked against the run's data
fingerprint) and, for the regime tables, the fitted gates. It changes no
file of the run, no setting, no hash, no resume state and no pilot lock;
the run's own metrics are recomputed by the same functions and come out
identical (tested). Its outputs are aggregate tables in the run's
``metrics/`` folder:

- ``reporting.csv``: one row per (group, predictor, execution lag, slice,
  metric) with mean, Hansen-Hodrick standard error, t, std, count and lag;
- ``reporting_settings.json``: the settings and what each metric means.

``group`` is an arm and seed (``soft_seed0``; ``soft_d2_seed0`` in a
campaign); ``predictor`` is ``model``, or ``base`` for the frozen base of a
residual arm, scored by the same code.

Metrics so far (brief 12 A): ``ic`` (the headline rank IC, the run's own
HAC lags and kernel, so it equals the run's number), ``ic_lower`` and
``ic_upper`` (half-sample ICs), ``spread``, ``long`` and ``short`` (the
book's gross spread and its legs, lag 0) and ``book_*`` (the existing book's
summary with the run's cost rate, as the run reports it).

Usage (from nec_baseline/), after a run has completed::

    python3.14 scripts/report_run.py <run_id>
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_experiment as rx  # noqa: E402
from nec_moe.data import Panel  # noqa: E402
from nec_moe.evaluation import portfolio_summary, rank_ic_by_date  # noqa: E402
from nec_moe.reporting import half_sample_ics, leg_returns, series_summary  # noqa: E402
from nec_moe.runstore import Run, RunStore, data_fingerprint  # noqa: E402


@dataclass(frozen=True)
class ReportSettings:
    """Every setting of the report, with its default. ``None`` means: what
    the run itself used."""

    n_quantiles: int | None = None  # the run's backtest_quantiles (the book's legs)
    portfolio_scheme: str | None = None  # the run's portfolio_scheme
    cost_rate: float | None = None  # the run's cost_rate (only for book_* rows)
    ic_hac_lags: int | None = None  # the run's ic_hac_lags, else h - 1
    ic_hac_kernel: str | None = None  # the run's ic_hac_kernel
    out_name: str = "reporting"


@dataclass
class Block:
    """One fold of one group: the saved predictions and the panel's labels."""

    fold: int
    date: torch.Tensor
    entity: torch.Tensor
    y: torch.Tensor
    y_daily: torch.Tensor | None
    preds: dict[str, torch.Tensor]  # predictor -> predictions
    date_labels: tuple[str, ...] | None
    gate_file: Path | None = None


@dataclass
class RunInputs:
    run: Run
    experiment: dict[str, Any]
    kind: str  # "evaluate" or "campaign"
    groups: dict[str, list[Block]] = field(default_factory=dict)
    horizon: int = 1


# --------------------------------------------------------------------------- #
# Loading what the run saved
# --------------------------------------------------------------------------- #


def open_run(run_id: str, root: str | Path | None = None) -> tuple[Run, dict, str]:
    """The run, its Experiment settings and its kind. Refuses a run that has
    not completed: the report reads its finished outputs."""
    store = RunStore(Path(root) if root is not None else rx._repo_path(rx.Experiment().out_dir))
    run = store.open(run_id)
    settings = run.settings()
    kind = "campaign" if "campaign_config" in settings else "evaluate"
    experiment = settings["experiment"] if kind == "campaign" else settings
    if experiment.get("per_security_dir") is not None:  # reopen where it wrote
        run = RunStore(store.root, rx._repo_path(experiment["per_security_dir"])).open(run_id)
    state = run.read_status().get("state")
    if state != "completed":
        raise ValueError(f"run {run_id} is {state!r}, not completed: report a finished run")
    if experiment.get("mode") == "quick":
        raise ValueError("quick-mode runs have one development split: nothing to report")
    return run, experiment, kind


def load_panel(experiment: dict[str, Any], run: Run) -> Panel:
    """The run's panel, rebuilt as the run built it (a panel file memory-
    mapped), and refused unless its data fingerprint is the run's."""
    exp = rx.Experiment(**{k: v for k, v in experiment.items()
                           if k in rx.Experiment.__dataclass_fields__})
    if exp.data == "panel_file":
        panel = torch.load(rx._repo_path(exp.panel_file), weights_only=False, mmap=True)
    else:
        panel = rx._build_panel(exp)[0]
    want = run.info().get("data_fingerprint")
    if want and data_fingerprint(panel) != want:
        raise ValueError("the panel's data fingerprint is not the run's: the data changed")
    return panel


def load_block(path: Path, panel: Panel, fold: int, gate_file: Path | None) -> Block:
    """A fold's saved predictions, with the test rows' daily forward returns
    from the panel (the saved rows must be the panel's, in the same order)."""
    saved = torch.load(path, weights_only=False)
    test = panel.subset_dates(torch.unique(saved["date"]))
    for name in ("date", "entity", "y"):
        if not torch.equal(getattr(test, name), saved[name]):
            raise ValueError(f"{path.name}: the saved {name} rows are not the panel's test rows")
    preds = {"model": saved["pred"]}
    if saved.get("base_pred") is not None:
        preds["base"] = saved["base_pred"]
    return Block(fold, saved["date"], saved["entity"], saved["y"], test.y_daily, preds,
                 saved.get("date_labels"), gate_file)


def load_inputs(run_id: str, root: str | Path | None = None,
                panel: Panel | None = None) -> RunInputs:
    run, experiment, kind = open_run(run_id, root)
    panel = load_panel(experiment, run) if panel is None else panel
    base = run.per_security_dir  # Data/derived/runs/<run_id>/
    gates = run.checkpoints_dir / ("jobs" if kind == "campaign" else "")
    inputs = RunInputs(run, experiment, kind, horizon=panel.horizon)
    for group_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(group_dir.glob("fold_*_predictions.pt"),
                       key=lambda p: int(p.name.split("_")[1]))
        if not files:
            continue
        blocks = []
        for f in files:
            k = int(f.name.split("_")[1])
            gate = gates / group_dir.name / f"fold_{k}_gate.pt"
            blocks.append(load_block(f, panel, k, gate if gate.exists() else None))
        inputs.groups[group_dir.name] = blocks
    if not inputs.groups:
        raise ValueError(f"no saved predictions under {base}")
    return inputs


# --------------------------------------------------------------------------- #
# Per-date series and their summaries
# --------------------------------------------------------------------------- #


def _labelled(codes: torch.Tensor, values: torch.Tensor, labels: tuple[str, ...] | None,
              name: str) -> pd.DataFrame:
    """A per-date (or per-period) series with its date code and label."""
    c = [int(x) for x in codes.tolist()]
    return pd.DataFrame({
        "code": c,
        "date": pd.to_datetime([labels[x] for x in c]) if labels else pd.NaT,
        name: values.double().numpy(),
    })


def predictor_series(blocks: list[Block], predictor: str, n_quantiles: int | None,
                     scheme: str, horizon: int) -> dict[str, pd.DataFrame]:
    """Every per-date series of one predictor, pooled over the folds in fold
    order (the harness's pooling order)."""
    out: dict[str, list[pd.DataFrame]] = {"ic": [], "ic_lower": [], "ic_upper": [], "book": []}
    for b in blocks:
        p = b.preds[predictor]
        d, ics = rank_ic_by_date(p, b.y, b.date)
        out["ic"].append(_labelled(d, ics, b.date_labels, "value"))
        (dl, il), (du, iu) = half_sample_ics(p, b.y, b.date)
        out["ic_lower"].append(_labelled(dl, il, b.date_labels, "value"))
        out["ic_upper"].append(_labelled(du, iu, b.date_labels, "value"))
        if n_quantiles is not None:
            dates, gross, tno, long, short = leg_returns(
                p, b.y, b.date, b.entity, n_quantiles=n_quantiles, horizon=horizon,
                scheme=scheme, y_daily=b.y_daily)
            frame = _labelled(dates, gross, b.date_labels, "spread")
            frame["turnover"] = tno.double().numpy()
            frame["long"] = long.double().numpy()
            frame["short"] = short.double().numpy()
            frame["gross_f32"] = list(gross)  # the book's own values, for book_* rows
            frame["turnover_f32"] = list(tno)
            out["book"].append(frame)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}


def summary_rows(series: dict[str, pd.DataFrame], *, horizon: int, ic_lags: int,
                 ic_kernel: str, cost_rate: float, scheme: str) -> list[dict[str, Any]]:
    """The slice "all" rows of one predictor."""
    from nec_moe.evaluation import ic_summary

    rows: list[dict[str, Any]] = []
    ic = series["ic"]["value"]
    head = ic_summary(torch.tensor(ic.to_numpy()), ic_lags, ic_kernel)
    rows.append({"metric": "ic", "mean": head.mean_ic, "se": head.mean_ic / head.t_stat
                 if head.t_stat else float("nan"), "t": head.t_stat, "std": head.ic_std,
                 "n": head.n_dates, "lags": head.hac_lags, "kernel": head.hac_kernel,
                 "icir": head.icir})
    for name in ("ic_lower", "ic_upper"):
        s = series_summary(series[name]["value"].to_numpy(), horizon - 1)
        rows.append({"metric": name} | s.as_dict() | {"icir": s.mean / s.std})
    if "book" in series:
        book = series["book"]
        for name in ("spread", "long", "short"):
            rows.append({"metric": name} | series_summary(book[name].to_numpy(), 0).as_dict())
        period = horizon if scheme == "nonoverlapping" else 1
        ps = portfolio_summary(torch.stack(list(book["gross_f32"])),
                               torch.stack(list(book["turnover_f32"])), cost_rate,
                               scheme=scheme, period_dates=period)
        for name in ("mean_gross", "mean_net", "ir_gross", "ir_net", "mean_turnover"):
            rows.append({"metric": f"book_{name}", "mean": getattr(ps, name), "n": ps.n_dates})
    return rows


def report(inputs: RunInputs, s: ReportSettings) -> pd.DataFrame:
    """Every row of the report (brief 12 A so far: slice "all", L = 0)."""
    exp = inputs.experiment
    h = inputs.horizon
    n_q = s.n_quantiles if s.n_quantiles is not None else exp.get("backtest_quantiles")
    scheme = s.portfolio_scheme or exp.get("portfolio_scheme", "nonoverlapping")
    cost = s.cost_rate if s.cost_rate is not None else float(exp.get("cost_rate", 0.0))
    lags = s.ic_hac_lags if s.ic_hac_lags is not None else exp.get("ic_hac_lags")
    ic_lags = h - 1 if lags is None else int(lags)
    kernel = s.ic_hac_kernel or exp.get("ic_hac_kernel", "uniform")
    rows = []
    for group, blocks in inputs.groups.items():
        for predictor in blocks[0].preds:
            series = predictor_series(blocks, predictor, n_q, scheme, h)
            for r in summary_rows(series, horizon=h, ic_lags=ic_lags, ic_kernel=kernel,
                                  cost_rate=cost, scheme=scheme):
                rows.append({"group": group, "predictor": predictor, "execution_lag": 0,
                             "slice_kind": "all", "slice": "all"} | r)
    return pd.DataFrame(rows)


def write_report(inputs: RunInputs, frame: pd.DataFrame, s: ReportSettings) -> None:
    """Aggregate tables into the run's metrics/ (new files only)."""
    inputs.run.write_metrics(s.out_name, frame)
    inputs.run.write_metrics(f"{s.out_name}_settings", {
        "what": "brief 12 reporting additions (Q9 update 2026-10-09): reporting only",
        "settings": dataclasses.asdict(s),
        "horizon": inputs.horizon,
        "groups": sorted(inputs.groups),
        "metrics": {
            "ic": "headline rank IC, the run's HAC lags and kernel (equals the run's number)",
            "ic_lower / ic_upper": "rank IC within the lower / upper half of each date's "
                                   "forecasts, Hansen-Hodrick lag h - 1",
            "spread / long / short": "the book's gross spread and its legs (long + short = "
                                     "spread), lag 0: the book's periods do not overlap",
            "book_*": "the existing book's pooled summary, as the run reports it",
        },
    })


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python3.14 scripts/report_run.py <run_id>")
    s = ReportSettings()
    inputs = load_inputs(argv[0])
    frame = report(inputs, s)
    write_report(inputs, frame, s)
    print(f"[report] {len(frame)} rows for {len(inputs.groups)} group(s) -> "
          f"{inputs.run.metrics_dir}/{s.out_name}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
