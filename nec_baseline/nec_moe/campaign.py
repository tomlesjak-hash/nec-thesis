"""The process-parallel campaign runner (brief 09 G; Q16 compute levers).

A campaign is a list of jobs, one per (arm, depth, seed, fold): one model,
trained once, for one fold and one seed. The jobs are independent (every fold
starts from fresh weights, brief 09 E.2), so they run in a pool of worker
processes and their results are combined afterwards exactly as a sequential
sweep would combine them.

- **Processes** (G.1). ``spawn`` start method (a clean interpreter per worker:
  no inherited locks or threads; PyTorch's multiprocessing notes), ``n_workers``
  of them (10, the M4 Pro's performance cores, by default), each calling
  ``torch.set_num_threads(1)`` and running one job at a time. The benchmark
  (Progress_Tracker 1c) found extra threads in one run give nothing and 10
  one-thread runs give 6.9x.
- **Data built once** (G.2). The panel's tensors are written once to
  ``tensors.pt`` (plus ``meta.pt``) in a cache folder, inside ``Data/derived/``
  for a real panel, and every worker memory-maps them read-only
  (``torch.load(mmap=True)``): one copy in the page cache, never a feature
  rebuild per job. Fitted bases are cached on disk per (fold, seed, base
  config) by :class:`DiskBaseCache`, behind a lock file, so two workers never
  fit or write the same base at once.
- **Resume** (G.3). Each job runs the walk-forward harness on its one fold
  with its own resume folder in the run's ``checkpoints/jobs/``: a finished
  job leaves ``fold_<i>.pt`` and is skipped on restart; an interrupted one
  continues from its trainer checkpoint. A first Ctrl+C (SIGINT/SIGTERM)
  sets a shared stop event: every worker checkpoints at its next step and
  stops; no new job starts. Results are written by the parent only (the
  workers never touch the trial registry).

Per-security predictions go only to the run's ``Data/derived/runs/<run_id>/``
folder, one file per job.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .base import BaseCache, BaseFit, base_cache_key, fit_base, load_base_fit, save_base_fit
from .config import BaseConfig, NECConfig
from .data import FeatureSchema, Panel
from .evaluation import (
    WalkForwardFold,
    WalkForwardResult,
    _FoldAccumulator,
    log_gate_starts,
    resolve_hac_lags,
    walk_forward_evaluate,
)
from .interrupt import RunInterrupted, graceful_interrupts, request_stop, stop_requested
from .model import NECModel
from .train import Trainer

__all__ = [
    "Job",
    "CampaignSpec",
    "DiskBaseCache",
    "save_panel_cache",
    "load_panel_cache",
    "run_job",
    "run_jobs",
    "job_done",
    "aggregate",
]

_PANEL_TENSORS = ("x_seq", "x_snap", "y", "date", "entity", "y_daily")


# --------------------------------------------------------------------------- #
# The panel, built once and memory-mapped (G.2)
# --------------------------------------------------------------------------- #


def save_panel_cache(panel: Panel, directory: str | Path) -> Path:
    """Write the panel's tensors (``tensors.pt``) and its schema, labels and
    provenance (``meta.pt``) to ``directory``, unless already there. The
    folder must be inside ``Data/derived/`` for a real panel (licensed)."""
    from .utils import atomic_torch_save

    out = Path(directory)
    if (out / "tensors.pt").exists() and (out / "meta.pt").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: getattr(panel, k) for k in _PANEL_TENSORS if getattr(panel, k) is not None}
    atomic_torch_save({k: v.contiguous() for k, v in tensors.items()}, out / "tensors.pt")
    atomic_torch_save({
        "schema": asdict(panel.schema), "date_labels": panel.date_labels,
        "entity_labels": panel.entity_labels, "data_source": panel.data_source,
        "metadata": panel.metadata,
    }, out / "meta.pt")
    return out


_PANELS: dict[str, Panel] = {}


def load_panel_cache(directory: str | Path) -> Panel:
    """The cached panel, its tensors memory-mapped (read-only, shared through
    the page cache), loaded once per process."""
    key = str(Path(directory).resolve())
    if key not in _PANELS:
        d = Path(directory)
        tensors = torch.load(d / "tensors.pt", mmap=True, weights_only=True)
        meta = torch.load(d / "meta.pt", weights_only=False)
        schema = meta["schema"]
        _PANELS[key] = Panel(
            **tensors,
            schema=FeatureSchema(
                sequence_features=tuple(schema["sequence_features"]),
                snapshot_features=tuple(schema["snapshot_features"]),
                target=schema["target"], rank_normalized=schema["rank_normalized"],
            ),
            date_labels=meta["date_labels"], entity_labels=meta["entity_labels"],
            data_source=meta["data_source"], metadata=meta["metadata"],
        )
    return _PANELS[key]


# --------------------------------------------------------------------------- #
# Fitted bases shared across processes, one writer at a time (G.2)
# --------------------------------------------------------------------------- #


class DiskBaseCache(BaseCache):
    """A :class:`~nec_moe.base.BaseCache` whose fits also live on disk, one
    file per cache key, shared by every worker of a campaign.

    ``get_or_fit`` takes an exclusive lock on ``<key>.lock`` before looking:
    the first worker fits and writes the base (atomically) while the others
    wait, then read the file. Fitting is deterministic, so a base is the same
    whichever worker fits it.
    """

    def __init__(self, directory: str | Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: tuple) -> Path:
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        return self.directory / f"base_{digest}.pt"

    def get_or_fit(
        self, panel: Panel, cfg: BaseConfig, *, window: object, seed: int
    ) -> BaseFit:
        key = base_cache_key(cfg, window, seed, panel.x_snap.shape[1])
        hit = self._fits.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        path = self._path(key)
        with open(path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                fit = load_base_fit(path, cfg, key) if path.exists() else None
                if fit is None:
                    self.misses += 1
                    fit = fit_base(panel, cfg, seed=seed + cfg.seed_offset)
                    save_base_fit(fit, key, path)
                else:
                    self.hits += 1
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        self._fits[key] = fit
        return fit


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Job:
    """One model, trained once, for one fold and one seed (G.1)."""

    arm: str
    depth: int
    seed: int
    fold: int  # index into CampaignSpec.folds

    @property
    def key(self) -> str:
        return f"{self.arm}_d{self.depth}_s{self.seed}_f{self.fold}"

    @property
    def group(self) -> str:
        """The (arm, depth, seed) a job's fold belongs to."""
        return f"{self.arm}_d{self.depth}_seed{self.seed}"


@dataclass
class CampaignSpec:
    """Everything a worker needs, picklable: the cached panel's folder, the
    folds, one model config per (arm, depth) as a dict, and the harness
    settings. Built once by the parent (``scripts/run_campaign.py``)."""

    panel_cache: str
    jobs_dir: str  # the run's checkpoints/jobs/
    base_cache_dir: str  # the run's checkpoints/bases/
    per_security_dir: str | None  # the run's Data/derived/runs/<run_id>/
    folds: list[WalkForwardFold]
    configs: dict[str, dict]  # f"{arm}_d{depth}" -> NECConfig.to_dict()
    purge_dates: int
    warmstart_channel: int | None = None  # None: no expert warm-start
    backtest_quantiles: int | None = None
    cost_rate: float = 0.0
    hac_lags: int | None = None
    hac_kernel: str = "uniform"
    portfolio_scheme: str = "nonoverlapping"
    provenance_extra: dict[str, Any] = field(default_factory=dict)

    def config(self, arm: str, depth: int) -> NECConfig:
        return NECConfig.from_dict(self.configs[f"{arm}_d{depth}"])

    def job_dir(self, job: Job) -> Path:
        return Path(self.jobs_dir) / job.group


def job_done(spec: CampaignSpec, job: Job) -> bool:
    """A job is done when its fold's scored payload exists."""
    return (spec.job_dir(job) / f"fold_{spec.folds[job.fold].fold}.pt").exists()


def run_job(spec: CampaignSpec, job: Job) -> dict[str, Any]:
    """Run one job in this process: the harness on the job's single fold,
    resumable in its own folder. Returns a status record."""
    panel = load_panel_cache(spec.panel_cache)
    cfg = spec.config(job.arm, job.depth)
    # the seed fixes the experts' initialisation and the data order, as in a
    # sweep's arm (nec_arm): paired across arms (brief 09 E.3)
    seeded = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=job.seed))

    def make_trainer() -> Trainer:
        torch.manual_seed(job.seed)
        return Trainer(NECModel(seeded))

    ch = spec.warmstart_channel
    warm: Callable[[Panel], Tensor] | None = (
        None if ch is None else (lambda p: p.x_seq[:, :, ch].std(dim=1))
    )
    per_security = (
        Path(spec.per_security_dir) / job.group if spec.per_security_dir is not None else None
    )
    walk_forward_evaluate(
        panel, make_trainer, folds=[spec.folds[job.fold]], purge_dates=spec.purge_dates,
        steps=cfg.train.steps, warmstart_key=warm,
        backtest_quantiles=spec.backtest_quantiles, cost_rate=spec.cost_rate,
        resume_dir=spec.job_dir(job), base_cache=DiskBaseCache(spec.base_cache_dir),
        seed=job.seed, hac_lags=spec.hac_lags, hac_kernel=spec.hac_kernel,
        portfolio_scheme=spec.portfolio_scheme, predictions_dir=per_security,
        provenance_extra=spec.provenance_extra,
    )
    return {"job": job.key, "status": "done", "threads": torch.get_num_threads(),
            "pid": os.getpid()}


def _worker(spec: CampaignSpec, tasks: Any, results: Any, stop: Any) -> None:
    """A worker process: one thread, one job at a time, until the sentinel.

    A shared ``stop`` event (set by the parent on Ctrl+C) is relayed to this
    process's stop flag by a watcher thread, so the running trainer
    checkpoints at its next step and raises :class:`RunInterrupted`. A
    SIGINT delivered to the worker itself (a terminal Ctrl+C reaches the
    whole process group) does the same through ``graceful_interrupts``."""
    torch.set_num_threads(1)

    def relay() -> None:
        stop.wait()
        request_stop()

    threading.Thread(target=relay, daemon=True).start()
    with graceful_interrupts():
        while True:
            job = tasks.get()
            if job is None:
                break
            if stop.is_set() or stop_requested():
                results.put({"job": job.key, "status": "skipped"})
                continue
            try:
                results.put(run_job(spec, job))
            except RunInterrupted as exc:
                results.put({"job": job.key, "status": "interrupted", "error": str(exc)})
            except KeyboardInterrupt as exc:
                results.put({"job": job.key, "status": "interrupted", "error": repr(exc)})
                break
            except Exception as exc:  # reported to the parent, never swallowed
                results.put({"job": job.key, "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})


def run_jobs(
    spec: CampaignSpec,
    jobs: list[Job],
    *,
    n_workers: int = 10,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every unfinished job; return one status record per job.

    Finished jobs (:func:`job_done`) are not rerun. ``n_workers = 0`` runs
    the jobs in this process, one after another (the same code path as a
    worker's). Otherwise a pool of ``n_workers`` spawned processes pulls jobs
    from a queue; the first stop request (Ctrl+C, SIGTERM) reaches every
    worker and no further job starts.
    """
    pending = [j for j in jobs if not job_done(spec, j)]
    records: list[dict[str, Any]] = [
        {"job": j.key, "status": "already_done"} for j in jobs if job_done(spec, j)
    ]
    if not pending:
        return records

    def note(rec: dict[str, Any]) -> None:
        records.append(rec)
        if on_result is not None:
            on_result(rec)

    if n_workers <= 0:
        for job in pending:
            if stop_requested():
                note({"job": job.key, "status": "skipped"})
                continue
            try:
                note(run_job(spec, job))
            except RunInterrupted as exc:
                note({"job": job.key, "status": "interrupted", "error": str(exc)})
        return records

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    tasks, results, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    for job in pending:
        tasks.put(job)
    workers = [ctx.Process(target=_worker, args=(spec, tasks, results, stop), daemon=False)
               for _ in range(min(n_workers, len(pending)))]
    for _ in workers:
        tasks.put(None)
    for w in workers:
        w.start()
    seen = 0
    try:
        while seen < len(pending):
            if stop_requested() and not stop.is_set():
                stop.set()  # every worker checkpoints and stops
            try:
                rec = results.get(timeout=0.5)
            except queue.Empty:
                if not any(w.is_alive() for w in workers):
                    break  # a worker died without reporting
                continue
            note(rec)
            seen += 1
    except KeyboardInterrupt:  # a second Ctrl+C: stop now
        stop.set()
        for w in workers:
            w.terminate()
        raise
    finally:
        for w in workers:
            w.join(timeout=60)
    reported = {r["job"] for r in records}
    for job in pending:
        if job.key not in reported:
            note({"job": job.key, "status": "lost", "error": "the worker exited without a result"})
    return records


# --------------------------------------------------------------------------- #
# Combining the jobs (the parent only)
# --------------------------------------------------------------------------- #


def aggregate(spec: CampaignSpec, jobs: list[Job]) -> dict[str, WalkForwardResult]:
    """Per (arm, depth, seed): the jobs' scored fold payloads, in fold order,
    pooled by the same accumulator a sequential walk-forward run uses, so the
    result is identical to running the folds in one process."""
    panel = load_panel_cache(spec.panel_cache)
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(job.group, []).append(job)
    out = {}
    for name, members in groups.items():
        missing = [j.key for j in members if not job_done(spec, j)]
        if missing:
            raise ValueError(f"cannot combine {name}: unfinished job(s) {missing}")
        acc = _FoldAccumulator(
            spec.backtest_quantiles, spec.cost_rate, resolve_hac_lags(panel, spec.hac_lags),
            spec.hac_kernel, spec.portfolio_scheme, panel.horizon,
        )
        for job in sorted(members, key=lambda j: spec.folds[j.fold].fold):
            fold = spec.folds[job.fold]
            acc.add_completed(torch.load(spec.job_dir(job) / f"fold_{fold.fold}.pt",
                                         weights_only=False))
        out[name] = acc.result()
    return out


def log_campaign_gate_starts(spec: CampaignSpec, jobs: list[Job], registry: Any,
                             provenance: dict[str, Any]) -> None:
    """The gate's multi-start trials of every job (from its saved gate
    state), logged by the parent, as the sequential harness logs them.
    ``provenance`` is the panel's and the run's (``trial_provenance`` and the
    run-level keys); each job's ``hidden_init`` and ``gate_weight`` are
    added from its own config."""
    from .registry import fold_provenance, gate_weight_of, scheme_provenance

    for job in jobs:
        gate_file = spec.job_dir(job) / f"fold_{spec.folds[job.fold].fold}_gate.pt"
        if not gate_file.exists():
            continue
        state = torch.load(gate_file, weights_only=False)
        fit = state.get("extra", {}).get("fit_result")
        if fit is None:
            continue
        cfg = spec.config(job.arm, job.depth)
        log_gate_starts(registry, fit, spec.folds[job.fold].fold, cfg,
                        cfg.markov_gate.registry_tag, job.seed,
                        {**provenance, "hidden_init": cfg.experts.hidden_init,
                         "gate_weight": gate_weight_of(cfg), **scheme_provenance(cfg),
                         **fold_provenance([spec.folds[job.fold]])})


def jobs_manifest(jobs: list[Job]) -> str:
    """The job list as JSON, for the run's records."""
    return json.dumps([asdict(j) for j in jobs])
