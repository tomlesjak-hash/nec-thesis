"""One timed run of the real pipeline: timing output only (brief 09 H.2).

On the real 2000-2024 panel, one job: the Hamilton gate, K = 3, expert widths
64-32, batch 4,096, 2,000 steps, the 2009 pilot fold (training 2000-2008,
minus the purge). It records

- seconds per training step in the real pipeline (the expert stage's wall
  clock over its steps);
- the wall clock of data loading, the gate fit, the base fit and the expert
  training;
- the peak memory of the process;

and the **overhead factor**: real seconds per step divided by the compute
benchmark's for the same configuration (one thread, K = 3, widths 64-32,
batch 4,096; ``results/benchmark/compute_benchmark_20261005_233042.json``).
One thread, as a campaign worker runs (brief 09 G).

**Nothing is scored.** No prediction is made on the validation year or any
other block: no IC, no NLL, no return, no portfolio. The gate is fitted on
the training block only (no causal application), and the trainer's per-step
training history is discarded unread. ``sigma_init`` is fixed (a scale for
the start of training; its value does not change the cost of a step), so the
base's residuals are never computed either. Writes
``results/benchmark/real_pipeline_timing.json`` (aggregate: timings, counts,
the machine).

Usage (from nec_baseline/):  caffeinate -i python3.14 scripts/time_real_pipeline.py
"""

from __future__ import annotations

import dataclasses
import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_experiment as rx  # noqa: E402
from nec_moe import BaseCache, NECModel, Trainer, pilot_folds, pilot_slice  # noqa: E402
from nec_moe.evaluation import (  # noqa: E402
    _attach_base,
    _check_protocol_base,
    _gate_training_block,
    expert_stage_seed,
)

#: the job brief 09 H.2 names (any gate: the default Hamilton gate)
JOB = {
    "prior": "markov", "n_experts": 3, "expert_hidden_dims": (64, 32), "batch_size": 4096,
    "steps": 2000, "base_enabled": True, "correction_mode": True, "freeze_gate": True,
    "data": "panel_file", "sigma_init": 0.03, "seeds": (0,),
}
VALIDATION_YEAR = 2009
BENCHMARK = ROOT / "results" / "benchmark" / "compute_benchmark_20261005_233042.json"
OUT = ROOT / "results" / "benchmark" / "real_pipeline_timing.json"


def benchmark_seconds_per_step(k: int, widths: tuple[int, ...], batch: int) -> float:
    rows = json.loads(BENCHMARK.read_text())["rows"]
    tag = "-".join(map(str, widths))
    match = [r for r in rows if r.get("part") == "A_grid" and r.get("k") == k
             and r.get("widths") == tag and r.get("batch") == batch and r.get("threads") == 1]
    if len(match) != 1:
        raise ValueError(f"expected one benchmark row for K={k}, widths {tag}, batch {batch}")
    return float(match[0]["ms_per_step"]) / 1000.0


def peak_memory_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9  # bytes on macOS


def main() -> dict:
    torch.set_num_threads(1)
    exp = dataclasses.replace(rx.Experiment(), **JOB)
    t0 = time.perf_counter()
    panel, purge = rx._build_panel(exp)
    t_load = time.perf_counter() - t0
    fc = rx.fold_config(exp)
    sliced = pilot_slice(panel, fc.first_test_year, purge)
    (fold,) = pilot_folds(sliced, validation_years=(VALIDATION_YEAR,),
                          first_test_year=fc.first_test_year, purge_dates=purge)
    train = sliced.subset_dates(fold.train_dates)
    del panel
    cfg = rx._nec_config(exp, sliced, sigma_init=float(exp.sigma_init or 0.03))
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    _check_protocol_base(trainer.cfg, [fold])

    t0 = time.perf_counter()
    trainer.model.prior.fit(train)  # the training block only; nothing applied or scored
    t_gate = time.perf_counter() - t0
    t0 = time.perf_counter()
    _attach_base(trainer, train, fold, BaseCache(), 0)
    t_base = time.perf_counter() - t0
    torch.manual_seed(expert_stage_seed(0, fold.fold))
    expert_train, _ = _gate_training_block(trainer, train)
    t0 = time.perf_counter()
    trainer.fit(expert_train, steps=exp.steps)
    t_train = time.perf_counter() - t0
    trainer.history.clear()  # the per-step training diagnostics are not kept

    per_step = t_train / exp.steps
    bench = benchmark_seconds_per_step(3, (64, 32), 4096)
    labels = sliced.date_labels or ()
    report = {
        "brief": "09 H.2: one timed real run, timing only; nothing scored",
        "job": {k: list(v) if isinstance(v, tuple) else v for k, v in JOB.items()},
        "fold": {"validation_year": VALIDATION_YEAR,
                 "train_first": labels[int(fold.train_dates.min())],
                 "train_last": labels[int(fold.train_dates.max())],
                 "train_rows": len(train), "train_dates": int(fold.train_dates.numel()),
                 "expert_train_rows": len(expert_train)},
        "gate": {"fit_backend": cfg.markov_gate.fit_backend,
                 "k_regimes": cfg.markov_gate.k_regimes},
        "base_steps": cfg.base.steps, "base_batch_size": cfg.base.batch_size,
        "threads": torch.get_num_threads(),
        "seconds": {"data_loading": t_load, "gate_fit": t_gate, "base_fit": t_base,
                    "expert_training": t_train},
        "seconds_per_step": per_step,
        "benchmark_seconds_per_step": bench,
        "benchmark_file": BENCHMARK.name,
        "overhead_factor": per_step / bench,
        "peak_memory_gb": peak_memory_gb(),
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__},
    }
    from nec_moe.runstore import atomic_write_text

    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(OUT, json.dumps(report, indent=2) + "\n")
    print(f"[timing] {per_step * 1000:.2f} ms per step (benchmark {bench * 1000:.2f} ms): "
          f"overhead factor {per_step / bench:.2f}")
    print(f"[timing] load {t_load:.1f}s, gate fit {t_gate:.1f}s, base fit {t_base:.1f}s, "
          f"expert training {t_train:.1f}s; peak memory {report['peak_memory_gb']:.1f} GB")
    print(f"[timing] wrote {OUT}")
    return report


if __name__ == "__main__":
    main()
