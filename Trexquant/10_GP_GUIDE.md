# Genetic search — how to run it, and what changed

## Run this

```bash
cd "~/Desktop/Quant Model/Trexquant/pipeline"
python3 -m pip install deap
python3 tq_gp_run.py --seeds 16 --pop 500 --gen 40 --jobs 8
```

**≈ 1.1 hours on 8 cores.** Output → `Trexquant/gp_results/gp_results_topk.csv`.

---

## Optimal parameters — measured, not guessed

Benchmarked on the real panel shape (1,760 days × 579 tickers):

| Expression | eval | fitness | total |
|---|---|---|---|
| `sub(ret1, ret5)` | 0.2 ms | 54.5 ms | 55 ms |
| `ts_zscore(div(co, vol20), 20)` | 31 ms | 57.5 ms | 89 ms |
| 6-op nested tree | 59 ms | 57.4 ms | 116 ms |
| `ts_corr(c_ma20, v_ma20, 60)` | 80 ms | 57.0 ms | 137 ms |

**≈ 99 ms per individual.** Note the shape: **fitness costs ~57 ms regardless of
expression complexity**, so it dominates and tree depth is nearly free. That
argues for a wider search rather than a shallower one.

| Config | Evaluations | 8 cores |
|---|---|---|
| 300 × 25 × 8 seeds | 62,400 | 0.2 h |
| **500 × 40 × 16 seeds** | **328,000** | **≈1.1 h** ⭐ |
| 800 × 50 × 12 seeds | 489,600 | 1.7 h |

### Why these numbers

- **Seeds ≫ population, for this competition.** Each seed draws a random 70%
  subset of terminals, so seeds buy *structural* diversity — different runs are
  forced to explore different data — while population only buys refinement
  *within* one basin. Since half your score is alpha count under a ≤50%
  correlation gate, diversity is the binding constraint. **16 seeds is the
  single most valuable parameter here.**
- **gen 40** — past ~40 generations `eaSimple` has usually converged; extra
  generations deepen the same basin rather than finding new ones.
- **Tree depth capped at 8**, not your QIP setting of 17. Depth 17 is what
  produced the unreadable 9-level expression at rows 10–19 of your old file,
  and deep trees overfit far more per unit of in-sample IR.
- **`--jobs 8`** — set to your physical core count. Run `sysctl -n hw.perflevel0.physicalcpu`.

If you have the evening, `--seeds 24 --pop 600 --gen 45` (≈2.2 h) is strictly
better. Nothing below `--seeds 12` is worth running.

---

## New functions added to the search

Your QIP GP had **10 primitives**, and every time-series one was hard-wired to a
5-day window: `add sub mul div neg rank diff5 ts_max5 ts_min5 ts_ma5`.

The new set (`tq_gp_ops.py`) has **21 operators × 4 windows**. Every one maps
1:1 to a Trexsim builtin — if the GP could build something the platform can't
evaluate, the search time would be wasted.

| Added | Why it matters |
|---|---|
| **Variable windows {5, 10, 20, 60}** via strongly-typed GP | Alpha101 uses windows from 2 to 250. A 5-day-only search **cannot express momentum or slow reversal at all** — arguably the biggest limitation of your old run. |
| **`ts_corr`** | The most-used operator in the reference libraries — **44 occurrences** across `alpha101.py` + `gtja191.py`. Price–volume correlation is Alpha101's core idiom and your GP had no way to express it. |
| `ts_delta` | 47 uses in the reference libs; your `diff5` was the fixed-window version. |
| `ts_std`, `ts_zscore` | Volatility and self-normalisation. |
| `ts_decay` → `ts_mean_exp` | **The turnover lever.** The tutorial's own case study used it to cut TVR from 1.00 to 0.21. Directly raises IR/√TVR. |
| `ts_sum`, `ts_median`, `ts_skew`, `ts_max`, `ts_min`, `ts_delay` | Standard vocabulary, now with real windows. |
| `cs_zscore` alongside `cs_rank` | Preserves relative distances where rank throws them away. |
| `at_signlog`, `at_signsqrt` | Tail compression that keeps sign. |

**Deliberately excluded:** anything Trexsim lacks. No daily VWAP, no `amount`
as a primitive field, no operators outside the platform's 32.

---

## The three changes that actually fix the clone problem

1. **Dimensionless terminals** (`tq_gp.build_terminals`). 13 terminals, all
   ratios, returns or bounded ranks — asserted in testing that none has median
   |x| > 50. Raw price *levels* are why your old run collapsed onto
   `neg(closes)`: levels differ across stocks by orders of magnitude and barely
   move, so the size factor is found in generation 1 and never left.
2. **Diversity by construction.** 16 independent seeds, each on a different 70%
   terminal subset. `HallOfFame` keeps the N best with no novelty pressure, so
   a single run *will* fill with mutations of one winner — the fix is to run
   many, differently-constrained searches.
3. **Greedy correlation prune at ≤0.5 on PnL series** — the platform's own
   qualification rule, applied locally and for free, before you spend
   submissions on clones.

---

## What the output contains

`gp_results/gp_results_topk.csv` — same columns as your old file, plus:

| Column | |
|---|---|
| `trexsim` | **Paste-ready platform expression.** Handles infix conversion, `cs_rank` [1,2]→[0,1] rescaling, and `div` → `/ at_zero2nan(...)`. |
| `fitness_ir_over_sqrt_tvr` | The competition's own override metric |
| `valid_ir`, `valid_ir_over_sqrt_tvr` | **Held-out 2018–2020**, recomputed from that window's history (not reindexed) |
| `survives_oos` | Positive IR in *both* train and validation |
| `max_corr` | Correlation against already-accepted alphas |
| `numstk`, `tvr`, `max_drawdown` | Against the platform thresholds: `numstk > 160`, `tvr < 0.5` |

Also written: `gp_results_all.csv` (every unique candidate) and
`gp_pnl_corr.csv` (the correlation matrix — sanity-check it before submitting).

---

## Reading the results

**Sort by `valid_ir`, not `annualized_ir`.** In-sample IR from a 328,000-
expression search is the maximum of 328,000 draws and is inflated by
construction. The held-out column is the one that means anything.

Then, in order:

1. Drop anything with `survives_oos = False`.
2. Drop anything with `tvr > 0.5` (platform threshold) or `numstk < 160`.
3. Check `gp_pnl_corr.csv` — anything above 0.5 against an alpha you've already
   submitted will not qualify.
4. Paste the `trexsim` column into the platform and confirm **fast and slow
   mode agree**. Give slow mode a lookback well above 60, since `w60` operators
   need it.

Expect roughly **5–15 survivors** from 328,000 candidates. That is the normal
yield, and it is a fine outcome — remember Track A hand-crafted alphas are
still half the score and need none of this.
