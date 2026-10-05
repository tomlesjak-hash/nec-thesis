"""Compute benchmark for the thesis model on this machine (synthetic data only).

What it measures
----------------
Training throughput of the residual mixture of experts: K MLP experts (pyramid widths, ReLU,
dropout) on top of a frozen base prediction, trained with the per-row mixture negative
log-likelihood (Q24), row weights (Q16 decay), a fixed gate prior per row, and Adam. The cost of
one training step is independent of how many rows the panel has (fixed step budget, Q16), so the
numbers here, steps per second for each (K, depth, first width, batch size), are what the compute
budget needs.

Five parts:
  A. single process, 1 thread: the full grid of K x depth x first width x batch size
  B. one process, 1/2/4/8 threads: does multithreading help one small run? (expected: little)
  C. N processes in parallel, 1 thread each: the aggregate throughput of running many runs at once
  D. Apple GPU (MPS), if available: a subset of the grid at larger batch sizes
  E. the same model with the experts batched into one matrix product (an engineering option)

No real data is read or written. All inputs are random numbers, so the script is safe to share.
Results go to results/benchmark/ (inside nec_baseline) as a CSV and a JSON summary.

Run it from the repository root with the project's virtual environment, plugged in to power, with
other heavy apps closed (takes about 10 to 15 minutes):

    cd ~/Desktop/"Quant Model"
    source .venv/bin/activate
    caffeinate -i python nec_baseline/scripts/benchmark_compute.py

Add --quick for a 2-minute smoke version.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

D_INPUT = 57  # Q26: 57 model inputs


# --------------------------------------------------------------------------- model
def pyramid(first_width: int, depth: int) -> tuple[int, ...]:
    """Same rule as nec_moe.config.pyramid_dims: (w, w/2, w/4, ...)."""
    return tuple(max(first_width // (2 ** i), 1) for i in range(depth))


def mlp(d_in: int, widths: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = d_in
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(dropout)]
        prev = w
    head = nn.Linear(prev, 1)
    nn.init.zeros_(head.weight)  # zero-initialised correction head, as in the code
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class LoopedExperts(nn.Module):
    """K separate MLPs in a Python loop: what nec_moe.experts.ExpertBank does."""

    def __init__(self, k: int, widths: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(mlp(D_INPUT, widths, dropout) for _ in range(k))
        self.log_sigma = nn.Parameter(torch.zeros(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([e(x) for e in self.experts], dim=-1)  # (B, K)


class BatchedExperts(nn.Module):
    """The same K MLPs stored as stacked weights and run with batched matrix products."""

    def __init__(self, k: int, widths: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        dims = (D_INPUT, *widths)
        self.ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            w = torch.empty(k, a, b)
            for j in range(k):
                nn.init.kaiming_uniform_(w[j].T, a=math.sqrt(5))
            self.ws.append(nn.Parameter(w))
            self.bs.append(nn.Parameter(torch.zeros(k, 1, b)))
        self.head_w = nn.Parameter(torch.zeros(k, dims[-1], 1))
        self.head_b = nn.Parameter(torch.zeros(k, 1, 1))
        self.drop = nn.Dropout(dropout)
        self.log_sigma = nn.Parameter(torch.zeros(k))
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(0).expand(self.k, -1, -1)  # (K, B, d)
        for w, b in zip(self.ws, self.bs):
            h = self.drop(torch.relu(torch.baddbmm(b, h, w)))
        out = torch.baddbmm(self.head_b, h, self.head_w)  # (K, B, 1)
        return out.squeeze(-1).T  # (B, K)


def n_weights(widths: tuple[int, ...]) -> tuple[int, int]:
    """(all weights incl. biases of one expert, weights of its first layer)."""
    dims = (D_INPUT, *widths, 1)
    total = sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
    return total, D_INPUT * widths[0]


def flops_per_step(k: int, widths: tuple[int, ...], batch: int) -> float:
    """6 FLOPs per weight per row, minus 2 per first-layer weight (no input gradient)."""
    p, first = n_weights(widths)
    return float(batch) * k * (6 * p - 2 * first)


@dataclass
class Config:
    k: int
    depth: int
    first_width: int
    batch: int
    device: str = "cpu"
    impl: str = "looped"  # "looped" | "batched"
    threads: int = 1
    dropout: float = 0.05


def _make_data(n_rows: int, k: int, device: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_rows, D_INPUT, generator=g) * 2 - 1  # ranks in [-1, 1]
    y = torch.randn(n_rows, generator=g) * 0.03  # 5-day market-neutral returns, ~3% sd
    base = torch.randn(n_rows, generator=g) * 0.002  # cached frozen-base prediction
    logprior = torch.log_softmax(torch.randn(n_rows, k, generator=g), dim=-1)  # gate weights
    w = torch.rand(n_rows, generator=g) + 0.5  # decay row weights
    return [t.to(device) for t in (x, y, base, logprior, w)]


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_config(cfg: Config, min_seconds: float, warmup: int, n_rows: int = 200_000) -> dict:
    """Steps per second for one configuration in this process."""
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(0)
    widths = pyramid(cfg.first_width, cfg.depth)
    model_cls = BatchedExperts if cfg.impl == "batched" else LoopedExperts
    model = model_cls(cfg.k, widths, cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, y, base, logprior, w = _make_data(n_rows, cfg.k, cfg.device)
    log2pi = math.log(2 * math.pi)

    def step() -> None:
        idx = torch.randint(0, n_rows, (cfg.batch,), device=cfg.device)
        corr = model(x[idx])  # (B, K)
        mu = base[idx].unsqueeze(-1) + corr
        ls = model.log_sigma
        loglik = -0.5 * ((y[idx].unsqueeze(-1) - mu) / ls.exp()) ** 2 - ls - 0.5 * log2pi
        nll = -torch.logsumexp(logprior[idx] + loglik, dim=-1)
        loss = (w[idx] * nll).sum() / w[idx].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    _sync(cfg.device)
    n, t0 = 0, time.perf_counter()
    while True:
        for _ in range(10):
            step()
        n += 10
        _sync(cfg.device)
        el = time.perf_counter() - t0
        if el >= min_seconds:
            break
    sps = n / el
    fl = flops_per_step(cfg.k, widths, cfg.batch)
    p, _ = n_weights(widths)
    return {
        **asdict(cfg),
        "widths": "-".join(map(str, widths)),
        "trainable_weights": cfg.k * p,
        "steps_per_s": round(sps, 2),
        "rows_per_s": round(sps * cfg.batch),
        "gflops_per_s": round(sps * fl / 1e9, 3),
        "ms_per_step": round(1000 / sps, 3),
    }


def _worker(args):
    cfg, seconds, warmup, barrier = args
    torch.set_num_threads(1)
    barrier.wait()
    return time_config(cfg, seconds, warmup)


def parallel_throughput(cfg: Config, n_proc: int, seconds: float, warmup: int) -> dict:
    """Aggregate steps/s with n_proc independent single-thread processes running at once."""
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        barrier = mgr.Barrier(n_proc)
        with ctx.Pool(n_proc) as pool:
            res = pool.map(_worker, [(cfg, seconds, warmup, barrier)] * n_proc)
    per = [r["steps_per_s"] for r in res]
    return {
        **asdict(cfg),
        "processes": n_proc,
        "aggregate_steps_per_s": round(sum(per), 2),
        "mean_steps_per_s_per_process": round(sum(per) / n_proc, 2),
        "min_steps_per_s_per_process": round(min(per), 2),
        "aggregate_gflops_per_s": round(sum(r["gflops_per_s"] for r in res), 3),
    }


# --------------------------------------------------------------------------- machine
def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count_logical": os.cpu_count(),
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
    }
    if platform.system() == "Darwin":
        info.update({
            "chip": _sysctl("machdep.cpu.brand_string"),
            "performance_cores": _sysctl("hw.perflevel0.physicalcpu"),
            "efficiency_cores": _sysctl("hw.perflevel1.physicalcpu"),
            "memory_gb": round(int(_sysctl("hw.memsize") or 0) / 2**30, 1),
        })
    return info


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="small grid, short timings")
    ap.add_argument("--out", default=None, help="output folder (default nec_baseline/results/benchmark)")
    args = ap.parse_args()

    secs, warm = (0.6, 5) if args.quick else (2.0, 20)
    par_secs = 3.0 if args.quick else 8.0
    ks = (2, 3) if args.quick else (2, 3, 4)
    depths = (1, 2) if args.quick else (1, 2, 3)
    widths = (32, 128) if args.quick else (32, 64, 128, 256)
    batches = (1024, 4096)

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else root / "results" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    info = machine_info()
    print("Machine:", json.dumps(info, indent=1))
    t_start = time.time()

    # A. single-thread grid
    grid = []
    total = len(ks) * len(depths) * len(widths) * len(batches)
    for i, (k, dpt, w1, b) in enumerate(
        (k, d, w, b) for k in ks for d in depths for w in widths for b in batches
    ):
        r = time_config(Config(k, dpt, w1, b), secs, warm)
        grid.append(r)
        print(f"A {i + 1:>3}/{total}  K={k} depth={dpt} w1={w1:>3} batch={b:>4}  "
              f"{r['steps_per_s']:>8.1f} steps/s  {r['gflops_per_s']:>6.2f} GFLOP/s")

    ref = Config(3, 2, 64, 4096)

    # B. threads within one process
    threads = []
    for th in (1, 2, 4, 8):
        r = time_config(Config(ref.k, ref.depth, ref.first_width, ref.batch, threads=th), secs, warm)
        threads.append(r)
        print(f"B threads={th}  {r['steps_per_s']:.1f} steps/s")

    # C. processes in parallel
    perf = int(info.get("performance_cores") or 0) or max((os.cpu_count() or 2) // 2, 1)
    eff = int(info.get("efficiency_cores") or 0)
    n_list = sorted({1, 2, 4, perf, perf + eff} - {0})
    if args.quick:
        n_list = sorted({1, min(4, perf)})
    parallel = []
    for n in n_list:
        r = parallel_throughput(ref, n, par_secs, warm)
        parallel.append(r)
        print(f"C processes={n:>2}  aggregate {r['aggregate_steps_per_s']:.1f} steps/s "
              f"(per process {r['mean_steps_per_s_per_process']:.1f})")
    # also the largest model of the grid, in parallel on the performance cores
    big = Config(max(ks), max(depths), max(widths), 4096)
    r = parallel_throughput(big, perf, par_secs, warm)
    parallel.append(r)
    print(f"C big model, processes={perf}: aggregate {r['aggregate_steps_per_s']:.1f} steps/s")

    # D. Apple GPU
    gpu = []
    if info["mps_available"]:
        for dpt in (depths if args.quick else (1, 3)):
            for w1 in (64, max(widths)):
                for b in (1024, 4096, 16384):
                    r = time_config(Config(3, dpt, w1, b, device="mps"), secs, warm)
                    gpu.append(r)
                    print(f"D mps depth={dpt} w1={w1} batch={b}  {r['steps_per_s']:.1f} steps/s")
    else:
        print("D skipped: MPS not available")

    # E. batched experts (CPU, 1 thread)
    batched = []
    for k in ks:
        for dpt in depths:
            r = time_config(Config(k, dpt, 64, 4096, impl="batched"), secs, warm)
            batched.append(r)
            print(f"E batched K={k} depth={dpt}  {r['steps_per_s']:.1f} steps/s")

    elapsed = round(time.time() - t_start, 1)
    rows = ([{"part": "A_grid", **r} for r in grid] + [{"part": "B_threads", **r} for r in threads]
            + [{"part": "C_parallel", **r} for r in parallel] + [{"part": "D_mps", **r} for r in gpu]
            + [{"part": "E_batched", **r} for r in batched])
    csv_path = out / f"compute_benchmark_{stamp}.csv"
    json_path = out / f"compute_benchmark_{stamp}.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"machine": info, "quick": args.quick,
                                     "elapsed_seconds": elapsed, "rows": rows}, indent=1))
    print(f"\nDone in {elapsed} s. Results:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
