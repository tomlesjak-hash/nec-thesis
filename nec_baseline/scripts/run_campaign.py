"""Run a campaign: every (arm, depth, seed, fold) job in parallel (brief 09 G).

A campaign config (JSON; ``configs/campaign_example.json`` shows the format)
names the arms (each a set of ``Experiment`` fields over the fixed ones), the
depth scan (``pyramid_dims(first_width, depth)``), the seeds (paired across
arms) and the fixed ``Experiment`` fields, with the pilot's selection file.
The script

1. builds the panel once and saves its tensors to ``Data/derived/panel_cache/``
   (memory-mapped by every worker; never rebuilt per job);
2. checks the pre-registration lock for every arm (brief 09 F.3);
3. creates the campaign's run in the run store (``results/<campaign>/<run_id>/``)
   and runs every job in ``n_workers`` processes (default 10, one thread
   each), each job resumable in ``checkpoints/jobs/``;
4. when every job is done, combines each (arm, depth, seed)'s folds exactly
   as a sequential sweep does and writes its trial row and aggregate metrics.

Ctrl+C once: every worker checkpoints and stops, the run is marked
interrupted. Resume with ``--resume <run_id>``: finished jobs are skipped, an
interrupted one continues from its checkpoint. Keep the Mac awake for long
campaigns: ``caffeinate -i python3.14 scripts/run_campaign.py ...``.

Usage (from nec_baseline/):
    python3.14 scripts/run_campaign.py CONFIG.json [--workers N] [--resume RUN_ID]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_experiment as rx  # noqa: E402
from nec_moe import (  # noqa: E402
    COMPUTE_ENVELOPE,
    BaseCache,
    RunStore,
    TrialRegistry,
    fold_metrics_frame,
    graceful_interrupts,
    pyramid_dims,
    trial_provenance,
)
from nec_moe.campaign import (  # noqa: E402
    CampaignSpec,
    Job,
    aggregate,
    log_campaign_gate_starts,
    run_jobs,
    save_panel_cache,
)
from nec_moe.runstore import data_fingerprint  # noqa: E402
from nec_moe.sweep import _metrics_from_result  # noqa: E402

#: Campaign-config keys (anything else is refused).
KEYS = {"campaign", "tag", "purpose", "n_workers", "first_width", "depths", "seeds", "arms",
        "experiment", "pilot_selection_file"}


def load_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = sorted(set(cfg) - KEYS)
    if unknown:
        raise ValueError(f"unknown campaign key(s) {unknown}; known: {sorted(KEYS)}")
    missing = sorted({"campaign", "tag", "first_width", "depths", "seeds", "arms"} - set(cfg))
    if missing:
        raise ValueError(f"campaign config lacks {missing}")
    return cfg


def _experiment(cfg: dict[str, Any], out_dir: str | None) -> rx.Experiment:
    fixed = {k: tuple(v) if isinstance(v, list) else v
             for k, v in cfg.get("experiment", {}).items()}
    exp = dataclasses.replace(rx.Experiment(), **fixed, tag=cfg["tag"], campaign=cfg["campaign"],
                              purpose=cfg.get("purpose", ""), mode="evaluate",
                              pilot_selection_file=cfg.get("pilot_selection_file"))
    return dataclasses.replace(exp, out_dir=out_dir) if out_dir is not None else exp


def arm_experiment(exp: rx.Experiment, overrides: dict[str, Any], first_width: int,
                   depth: int) -> rx.Experiment:
    """An arm at one depth: the fixed settings, the arm's own, and pyramid
    widths from ``first_width`` (brief 09 H.1)."""
    over = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
    return dataclasses.replace(exp, **over,
                               expert_hidden_dims=pyramid_dims(first_width, depth))


def main(config: str | Path, *, n_workers: int | None = None, resume: str | None = None,
         out_dir: str | None = None, per_security_dir: str | None = None) -> dict[str, Any]:
    cfg = load_config(config)
    exp = _experiment(cfg, out_dir)
    store = RunStore(rx._repo_path(exp.out_dir),
                     rx._repo_path(per_security_dir) if per_security_dir is not None else None)
    panel, purge = rx._build_panel(exp)
    folds = rx.main_folds(exp, panel, purge)
    arms: dict[str, dict] = cfg["arms"]
    depths, seeds = list(cfg["depths"]), list(cfg["seeds"])
    # the compute envelope (brief 09 H.1): documented defaults, not limits
    for name, overrides in arms.items():
        k = int(overrides.get("n_experts", exp.n_experts))
        for note in COMPUTE_ENVELOPE.notes(first_width=cfg["first_width"], depths=depths,
                                           n_experts=k):
            print(f"[campaign] note: arm {name!r} is outside the compute envelope: {note}")
    lock: dict[str, Any] = {}
    for overrides in arms.values():  # the lock holds for every arm (F.3)
        for depth in depths:
            lock = rx.selection_lock(arm_experiment(exp, overrides, cfg["first_width"], depth),
                                     panel, purge)
    fingerprint = data_fingerprint(panel)
    settings = {"campaign_config": cfg, "experiment": rx._snapshot(exp)}
    if resume is None:
        run = store.create(
            campaign=exp.campaign, tag=exp.tag, mode="campaign", settings=settings,
            purpose=exp.purpose, data_source=panel.data_source, fingerprint=fingerprint,
            touches_test=rx._touches_test(panel), default_settings=None,
        )
    else:
        run = store.open(resume)
        if run.info().get("data_fingerprint") != fingerprint or run.settings() != settings:
            raise rx.ResumeRefused(
                f"refusing to resume {resume}: the campaign config or the data changed"
            )
        run.status(state="running", exit_reason="")
        store.update_index(resume, status="running", finished="", exit_reason="")
    run.record(**lock)
    cache_dir = save_panel_cache(panel, store.per_security_root.parent / "panel_cache"
                                 / fingerprint)
    first = panel.subset_dates(folds[0].train_dates)
    window = (int(folds[0].train_dates.min()), int(folds[0].train_dates.max()))
    sigma = rx._auto_sigma(exp, panel, first, BaseCache(), window)
    configs = {
        f"{name}_d{depth}": rx._nec_config(arm_experiment(exp, over, cfg["first_width"], depth),
                                           panel, sigma).to_dict()
        for name, over in arms.items() for depth in depths
    }
    warm = None
    if exp.warmstart and not (exp.correction_mode and exp.zero_init_head):
        warm = rx._proxy_channel(exp)
    provenance = rx.run_provenance(exp, run)
    spec = CampaignSpec(
        panel_cache=str(cache_dir), jobs_dir=str(run.checkpoints_dir / "jobs"),
        base_cache_dir=str(run.checkpoints_dir / "bases"),
        per_security_dir=str(run.per_security_dir), folds=folds, configs=configs,
        purge_dates=purge, warmstart_channel=warm, backtest_quantiles=exp.backtest_quantiles,
        cost_rate=exp.cost_rate, hac_lags=exp.ic_hac_lags, hac_kernel=exp.ic_hac_kernel,
        portfolio_scheme=exp.portfolio_scheme, provenance_extra=provenance,
    )
    jobs = [Job(name, depth, seed, f) for name in arms for depth in depths for seed in seeds
            for f in range(len(folds))]
    workers = n_workers if n_workers is not None else int(cfg.get("n_workers", 10))
    with run.logging(), graceful_interrupts():
        print(f"[campaign] {run.run_id}: {len(jobs)} jobs ({len(arms)} arms x {len(depths)} "
              f"depths x {len(seeds)} seeds x {len(folds)} folds), {workers} workers")

        def progress(rec: dict[str, Any]) -> None:
            print(f"[campaign] {rec['job']}: {rec['status']}"
                  + (f" ({rec['error']})" if rec.get("error") else ""))
            run.status(step=None, arm=rec["job"])

        records = run_jobs(spec, jobs, n_workers=workers, on_result=progress)
        bad = [r for r in records if r["status"] not in ("done", "already_done")]
        if bad:
            state = "failed" if any(r["status"] in ("failed", "lost") for r in bad) else (
                "interrupted")
            run.finish(state, f"{len(bad)} job(s) not finished: "
                              + ", ".join(f"{r['job']}={r['status']}" for r in bad[:10]))
            return {"run_id": run.run_id, "run_dir": run.dir, "records": records,
                    "status": state}
        results = aggregate(spec, jobs)
        registry = TrialRegistry(run.trials)
        log_campaign_gate_starts(
            spec, jobs, registry,
            {**trial_provenance(panel, None, exp.portfolio_scheme), **provenance},
        )
        rows = {}
        for group, res in results.items():
            arm, depth_s, seed_s = group.rsplit("_", 2)
            depth, seed = int(depth_s[1:]), int(seed_s[len("seed"):])
            metrics = _metrics_from_result(res)
            run.write_metrics(f"{group}_folds", fold_metrics_frame(res))
            run.write_metrics(f"{group}_pooled", metrics)
            registry.log(exp.tag, metrics, config={
                **trial_provenance(panel, spec.config(arm, depth), exp.portfolio_scheme),
                "arm": arm, "depth": depth, **provenance,
            }, seed=seed)
            rows[group] = metrics
        frame = pd.DataFrame(rows).T
        run.write_metrics("report", frame)
        run.finish("completed", metrics=frame)
    return {"run_id": run.run_id, "run_dir": run.dir, "records": records, "results": results,
            "status": "completed"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", default=None, metavar="RUN_ID")
    a = ap.parse_args()
    try:
        out = main(a.config, n_workers=a.workers, resume=a.resume)
    except KeyboardInterrupt:
        print("[campaign] interrupted: resume with --resume <run_id>", file=sys.stderr)
        sys.exit(130)
    sys.exit(0 if out["status"] == "completed" else 130)
