#!/usr/bin/env python3
"""Run the genetic search. Deliverable: gp_results_topk.csv, decorrelated.

    cd "~/Desktop/Quant Model/Trexquant/pipeline"
    python3 -m pip install deap
    python3 tq_gp_run.py --seeds 12 --pop 500 --gen 40 --jobs 8

Three differences from your QIP run, in order of impact:

  1. DIMENSIONLESS TERMINALS. Raw price levels are gone, so there is no size
     attractor for the population to collapse onto.
  2. DIVERSITY BY CONSTRUCTION. Each seed gets a random SUBSET of terminals,
     so different runs are forced to explore different data. Survivors are
     then pruned greedily at <=0.5 PnL correlation — the platform's own rule,
     applied locally and for free.
  3. FITNESS = IR/sqrt(TVR), industry-neutral — the metric the competition's
     correlation-override clause actually uses.

Output columns mirror your gp_results_topk.csv, plus `trexsim` (paste-ready),
`valid_ir` (held-out 2018-2020) and `max_corr` (against accepted alphas).
"""

from __future__ import annotations

# THREAD PINNING — must run BEFORE numpy is imported, in the parent AND in every
# spawned worker (which re-imports this module, so top-level placement covers
# both). numpy on macOS uses Accelerate, which multithreads internally. With 10
# worker processes each spawning its own thread pool you get heavy
# oversubscription: threads fight for the same 10 cores and throughput drops.
# One thread per process is right when parallelism is already at the process
# level.
import os  # noqa: E402

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import operator  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import tq_gp_ops as O  # noqa: E402
from tq_gp import (TEST, TRAIN, VALID, build_terminals, fitness,  # noqa: E402
                   load_gp_data, report)

OUT = Path(__file__).resolve().parents[1] / "gp_results"

# ==========================================================================
#  CONFIG — edit here, or override any of it on the command line.
#  Just pressing Run in your editor uses exactly these values.
# ==========================================================================

#: Flip to True for a ~2 minute sanity run before committing hours.
SMOKE_TEST = False

#: Independent GP runs, each on a random 70% subset of terminals.
#: THE most valuable parameter — seeds buy structural diversity, which is what
#: the <=50% correlation gate rewards. Round to a multiple of JOBS: work
#: arrives in waves of JOBS, so 24 seeds on 10 workers wastes 6 cores for a
#: whole final wave, while 30 seeds is exactly 3 full waves for the same time.
SEEDS = 30

POP = 600            #: individuals per generation
GEN = 40             #: generations (eaSimple has usually converged by ~40)
JOBS = 10            #: concurrent seeds. Set to your PERFORMANCE core count:
#:                      sysctl -n hw.perflevel0.physicalcpu
#:                   M4 Pro = 10. The 4 efficiency cores keep macOS responsive.

HOF = 15             #: hall-of-fame kept per seed, before the correlation prune
CORR_LIMIT = 0.5     #: the platform's own qualification threshold
TOPK = 30            #: rows written to gp_results_topk.csv

#: ANTI-OVERFITTING (added after run 1's alphas failed Trexsim's 0.07 gate)
MAX_DEPTH = 5        #: was 8. Depth-8 trees had in-sample and held-out IR of
#:                      OPPOSITE SIGN. Node count is the best single predictor
#:                      of overfitting, and it drives runtime too.
PARSIMONY = 0.004    #: mild size penalty so a simpler expression wins ties

#: MEASURED cost per individual, run 2 config (depth<=5, 2006-2018 window):
#:     ~14 nodes, 3270 days  ->  337 ms
#: For contrast, run 1 (depth<=8, 1761 days) was ~40 nodes -> 416 ms. The
#: window is 1.9x longer but the trees are 3x smaller, so it is slightly
#: CHEAPER than before despite training on nearly twice the history.
#:
#: Wall clock = SEEDS x POP x (GEN+1) x 0.337 / JOBS seconds
#:   30 x 600 x 41 x 0.337 / 10  ~=  6.9 hours  (overnight)
#:   30 x 500 x 36 x 0.337 / 10  ~=  5.1 hours  (shorter overnight)

if SMOKE_TEST:
    SEEDS, POP, GEN, JOBS, HOF, TOPK = 4, 100, 8, 4, 5, 10


# ------------------------------------------------------------- primitives --
class Win(int):
    """Distinct type so windows can only appear in window slots."""


def build_pset(terminal_names: list[str]):
    from deap import gp

    n = len(terminal_names)
    pset = gp.PrimitiveSetTyped("MAIN", [pd.DataFrame] * n, pd.DataFrame)
    D = pd.DataFrame

    for f in (O.add, O.sub, O.mul, O.div):
        pset.addPrimitive(f, [D, D], D, name=f.__name__)
    for f in (O.neg, O.signlog, O.signsqrt, O.cs_rank, O.cs_zscore):
        pset.addPrimitive(f, [D], D, name=f.__name__)
    for f in (O.ts_mean, O.ts_std, O.ts_sum, O.ts_median, O.ts_max, O.ts_min,
              O.ts_skew, O.ts_delay, O.ts_delta, O.ts_zscore, O.ts_decay):
        pset.addPrimitive(f, [D, Win], D, name=f.__name__)
    pset.addPrimitive(O.ts_corr, [D, D, Win], D, name="ts_corr")

    # windows are terminals of their own type -> no ts_mean(x, -3) nonsense
    for w in O.WINDOWS:
        pset.addTerminal(Win(w), Win, name=f"w{w}")
    # DEAP requires at least one PRIMITIVE per type or tree generation fails;
    # see O.w_id. Harmless: it is an identity and to_trexsim unwraps it.
    pset.addPrimitive(O.w_id, [Win], Win, name="w_id")

    pset.renameArguments(**{f"ARG{i}": nm for i, nm in enumerate(terminal_names)})
    return pset


# ------------------------------------------------------------------- run ---
_CACHE: dict = {}


def _load_once() -> tuple[dict, dict]:
    """Per-process panel cache.

    macOS spawns rather than forks worker processes, so each one imports this
    module fresh and must load the panel itself. Caching in a module global
    means it happens once per worker, not once per seed.
    """
    if "d" not in _CACHE:
        d = load_gp_data(window=TRAIN)
        _CACHE["d"] = d
        _CACHE["T"] = build_terminals(d)
    return _CACHE["d"], _CACHE["T"]


def _seed_worker(job: tuple) -> list[tuple]:
    """Run one complete seed in this process. Returns only PICKLABLE results.

    Deliberately does NOT return the alpha DataFrames — each is ~4 MB and
    shipping them back would cost more than recomputing them in the parent
    from the formula string (~50 ms each).
    """
    seed, pop_n, gen_n, hof_n = job
    d, T = _load_once()
    res = run_seed(seed, T, d, pop_n, gen_n, hof_n)
    return [(f, fit, names) for f, fit, names, _ in res]


def run_seed(seed: int, terms: dict[str, pd.DataFrame], d_train: dict,
             pop_n: int, gen_n: int, hof_n: int) -> list[tuple]:
    from deap import algorithms, base, creator, gp, tools

    rng = random.Random(seed)
    # each seed explores a different slice of the data -> structural diversity
    k = max(5, int(len(terms) * 0.7))
    names = sorted(rng.sample(sorted(terms), k))
    cols = [terms[n] for n in names]
    pset = build_pset(names)

    if not hasattr(creator, "FitMax"):
        creator.create("FitMax", base.Fitness, weights=(1.0,))
        creator.create("Indiv", gp.PrimitiveTree, fitness=creator.FitMax)

    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    tb.register("individual", tools.initIterate, creator.Indiv, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    def evaluate(ind):
        # Terminals are float32 (max ~3.4e38), so a mul/div of two already-large
        # intermediates overflows to inf mid-expression, and a downstream
        # cs_zscore/ts_std over that inf emits
        #   RuntimeWarning: invalid value encountered in reduce / in subtract
        # These are harmless — sanitize() cleans before scoring and the
        # expression is rejected with -1.0 — but at ~800k evaluations they
        # flood the log. Suppress at the source rather than globally, so a
        # warning from anywhere else stays visible.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    f = tb.compile(ind)
                    out = f(*cols)
                except Exception:
                    return (-1.0,)
                if not isinstance(out, pd.DataFrame):
                    return (-1.0,)
                v = fitness(out, d_train)
        if not np.isfinite(v) or v <= 0:
            return (-1.0,)
        # PARSIMONY PRESSURE. Depth is capped, but within a depth budget the GP
        # still drifts toward bigger trees, and node count is the best single
        # predictor of overfitting: the first run's depth-8, 40-node winners
        # had in-sample IR 3.28 and held-out IR -2.40. A mild multiplicative
        # penalty makes a simpler expression win ties.
        v *= max(0.5, 1.0 - PARSIMONY * len(ind))
        return (v,)

    tb.register("evaluate", evaluate)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    # DEPTH CAP 5, down from 8. This is the highest-value anti-overfitting
    # lever available. At depth 8 the GP evolved ~40-node expressions whose
    # in-sample IR (3.28) and held-out IR (-2.40) had OPPOSITE signs, and whose
    # fitness-vs-held-out rank correlation across the whole run was -0.70.
    # Shallow trees also cost far less: node count drives evaluation time.
    tb.decorate("mate", gp.staticLimit(operator.attrgetter("height"), MAX_DEPTH))
    tb.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), MAX_DEPTH))

    random.seed(seed)
    np.random.seed(seed)
    pop = tb.population(n=pop_n)
    hof = tools.HallOfFame(hof_n)
    algorithms.eaSimple(pop, tb, cxpb=0.6, mutpb=0.25, ngen=gen_n,
                        halloffame=hof, verbose=False)

    out = []
    for ind in hof:
        try:
            arr = tb.compile(ind)(*cols)
        except Exception:
            continue
        if isinstance(arr, pd.DataFrame):
            out.append((str(ind), float(ind.fitness.values[0]), names, arr))
    return out


# ---------------------------------------------------------- decorrelation --
def pnl_series(alpha: pd.DataFrame, d: dict, delay: int = 1) -> pd.Series:
    """PnL used for the correlation gate.

    MUST industry-neutralize, exactly as `fitness` and `report` do. An earlier
    version did not, so the correlation prune compared raw PnL while every
    other metric used neutralized PnL — two alphas could pass the gate on raw
    correlation and then be near-duplicates on the platform, which is the one
    failure this gate exists to prevent.
    """
    from tq_gp import _industry_neutral, sanitize
    a = sanitize(alpha).where(d["universe"])
    if "industry" in d:
        a = _industry_neutral(a, d["industry"])
    a = a.sub(a.median(axis=1), axis=0)
    pos = a.div(a.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return (pos.shift(delay) * d["ret1"]).sum(axis=1, min_count=1)


def greedy_prune(cands: list[dict], limit: float = 0.5) -> list[dict]:
    """Trexquant's own qualification rule, applied locally and for free.

    Sort by IR/sqrt(TVR) descending, accept a candidate only if its PnL
    correlation against everything already accepted is <= `limit`. This is what
    stops twenty submissions from qualifying two.
    """
    kept: list[dict] = []
    for c in sorted(cands, key=lambda x: -x["fitness"]):
        if not np.isfinite(c["fitness"]) or c["fitness"] <= 0:
            continue
        mc = 0.0
        for k in kept:
            v = c["pnl"].corr(k["pnl"])
            if np.isfinite(v):
                mc = max(mc, abs(v))
        c["max_corr"] = round(mc, 4)
        if mc <= limit:
            kept.append(c)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Defaults come from the CONFIG block at the top of this "
                    "file, so pressing Run in an editor works.")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--gen", type=int, default=GEN)
    ap.add_argument("--hof", type=int, default=HOF)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--corr-limit", type=float, default=CORR_LIMIT)
    ap.add_argument("--topk", type=int, default=TOPK)
    a = ap.parse_args()

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    if SMOKE_TEST:
        print("*** SMOKE_TEST = True — tiny run. Set it to False in the "
              "CONFIG block for the real search. ***\n")
    est = a.seeds * a.pop * (a.gen + 1) * 0.099 / max(a.jobs, 1) / 60
    waves = -(-a.seeds // max(a.jobs, 1))
    print(f"config: seeds={a.seeds} pop={a.pop} gen={a.gen} jobs={a.jobs}")
    print(f"        {waves} wave(s) of {a.jobs}; estimated ~{est:.0f} min")
    if a.seeds % a.jobs:
        idle = a.jobs - (a.seeds % a.jobs)
        print(f"        ! last wave leaves {idle} core(s) idle — "
              f"use {waves * a.jobs} seeds for the same wall clock")
    print()
    print(f"loading panel  train={TRAIN}  valid={VALID}  test={TEST}")
    d_tr = load_gp_data(window=TRAIN)
    d_va = load_gp_data(window=VALID)
    d_te = load_gp_data(window=TEST)
    T_tr = build_terminals(d_tr)
    T_va = build_terminals(d_va)
    T_te = build_terminals(d_te)
    print(f"  train {d_tr['close'].shape[0]}d x {d_tr['close'].shape[1]}tk, "
          f"{len(T_tr)} terminals")

    evals = a.seeds * a.pop * (a.gen + 1)
    print(f"  {a.seeds} seeds x pop {a.pop} x gen {a.gen} "
          f"= ~{evals:,} evaluations\n")

    jobs = [(s, a.pop, a.gen, a.hof) for s in range(a.seeds)]
    raw: list[dict] = []

    if a.jobs > 1:
        # Parallelise across SEEDS, not individuals. Each worker runs a whole
        # seed single-threaded, so nothing large crosses the process boundary
        # every generation — and the worker function is module-level, hence
        # picklable (a nested closure is not).
        import concurrent.futures as cf
        _CACHE["d"], _CACHE["T"] = d_tr, T_tr        # parent already has them
        with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(_seed_worker, j): j[0] for j in jobs}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:                # noqa: BLE001
                    print(f"  seed {s:2d}: FAILED {type(e).__name__}: {e}")
                    continue
                for formula, fit, names in res:
                    raw.append({"formula": formula, "fitness": fit,
                                "terminals": ",".join(names), "seed": s})
                best = max((r[1] for r in res), default=float("nan"))
                print(f"  seed {s:2d}: {len(res):3d} candidates, "
                      f"best {best:.4f}")
    else:
        for j in jobs:
            t = time.perf_counter()
            res = run_seed(j[0], T_tr, d_tr, a.pop, a.gen, a.hof)
            for formula, fit, names, _ in res:
                raw.append({"formula": formula, "fitness": fit,
                            "terminals": ",".join(names), "seed": j[0]})
            print(f"  seed {j[0]:2d}: {len(res):3d} candidates, "
                  f"best {max([r[1] for r in res], default=float('nan')):.4f}, "
                  f"{time.perf_counter()-t:.0f}s")

    # de-duplicate identical formulas before the expensive PnL step
    seen, uniq = set(), []
    for c in raw:
        if c["formula"] not in seen:
            seen.add(c["formula"])
            uniq.append(c)
    print(f"\n{len(raw)} candidates, {len(uniq)} structurally unique")

    # Recompute alphas in the parent from the formula strings — cheaper than
    # pickling ~4 MB frames back from every worker.
    ok = []
    for c in uniq:
        try:
            arr = O.recompute(c["formula"], T_tr)
            if isinstance(arr, pd.DataFrame):
                c["alpha"] = arr
                c["pnl"] = pnl_series(arr, d_tr)
                ok.append(c)
        except Exception:
            continue
    uniq = ok
    print(f"{len(uniq)} recomputed successfully")
    kept = greedy_prune(uniq, a.corr_limit)
    print(f"{len(kept)} survive the <={a.corr_limit} correlation gate")

    rows = []
    for i, c in enumerate(kept[: a.topk]):
        m = report(c["alpha"], d_tr)
        try:
            f = O.to_trexsim(c["formula"], O.TERMINAL_TREXSIM)
        except Exception as e:
            f = f"<untranslatable: {e}>"
        # HELD-OUT CHECK on 2018-2020 — recompute from that window's own
        # history; reindexing the training alpha would be wrong, since every
        # rolling operator has to be rebuilt from validation-window data.
        v_ir = v_f = t_ir = np.nan
        try:
            va = O.recompute(c["formula"], T_va)
            if isinstance(va, pd.DataFrame):
                mv = report(va, d_va)
                v_ir, v_f = mv["ir"], mv["ir_over_sqrt_tvr"]
        except Exception:
            pass
        # TEST (2021-2022) is REPORTED but never selected on — `qualifies` uses
        # VALID only. It exists so you can see, after choosing, whether an alpha
        # also held up in a window nothing touched.
        try:
            te = O.recompute(c["formula"], T_te)
            if isinstance(te, pd.DataFrame):
                t_ir = report(te, d_te)["ir"]
        except Exception:
            pass
        rows.append({
            "rank": i, "formula": c["formula"], "trexsim": f,
            "fitness_ir_over_sqrt_tvr": round(c["fitness"], 4),
            "annualized_ir": round(m["ir"], 4),
            "ir_daily_vs_platform_0.07": round(m["ir_daily"], 4),
            "net_ir": round(m["net_ir"], 4),
            "annualized_return": round(m["ret"], 4),
            "tvr": round(m["tvr"], 4),
            "max_drawdown": round(m["max_dd"], 4),
            "numstk": round(m["numstk"], 1),
            "valid_ir": round(v_ir, 4) if np.isfinite(v_ir) else "",
            "valid_ir_over_sqrt_tvr": round(v_f, 4) if np.isfinite(v_f) else "",
            "test_ir": round(t_ir, 4) if np.isfinite(t_ir) else "",
            # positive in ALL THREE windows — the strongest evidence available
            # locally that an alpha is not a fit to one regime
            "robust_all_windows": bool(m["ir"] > 0 and np.isfinite(v_ir)
                                       and v_ir > 0 and np.isfinite(t_ir)
                                       and t_ir > 0),
            "survives_oos": bool(np.isfinite(v_ir) and v_ir > 0
                                 and m["ir"] > 0),
            # every platform gate in one column — this is what to filter on
            "qualifies": bool(m["ir_daily"] > 0.07 and m["tvr"] < 0.5
                              and m["numstk"] > 160
                              and np.isfinite(v_ir) and v_ir > 0),
            "max_corr": c.get("max_corr", 0.0),
            "seed": c["seed"], "terminals": c["terminals"],
            "days": m["days"],
        })

    df = pd.DataFrame(rows)
    p = OUT / "gp_results_topk.csv"
    df.to_csv(p, index=False)
    pd.DataFrame([{k: v for k, v in c.items()
                   if k not in ("alpha", "pnl")} for c in uniq]
                 ).to_csv(OUT / "gp_results_all.csv", index=False)

    if kept:
        pn = pd.DataFrame({f"a{i}": c["pnl"] for i, c in enumerate(kept[:a.topk])})
        pn.corr().to_csv(OUT / "gp_pnl_corr.csv")

    print(f"\nwrote {p}  ({len(df)} rows)")
    print(f"total {time.perf_counter()-t0:.0f}s")
    if len(df):
        print("\n" + df[["rank", "annualized_ir", "tvr",
                         "fitness_ir_over_sqrt_tvr", "max_corr",
                         "formula"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
