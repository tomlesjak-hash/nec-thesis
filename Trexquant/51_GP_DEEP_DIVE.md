# Genetic programming — full technical account

*Deep-dive companion to note 50 §3. Every figure traced to the source files:
`pipeline/tq_gp.py`, `pipeline/tq_gp_ops.py`, `pipeline/tq_gp_run.py`, the QIP
baseline `quant_lib/genetic_factor.py`, and `gp_results/*.csv`.*

---

## 1. The inherited implementation, and its four structural limits

The starting point was a DEAP genetic program written for a China A-share
course project (`QIP/data_and_models/quant_lib/genetic_factor.py`, 817 lines).
Its configuration:

```python
pset = gp.PrimitiveSet('Main', 5)                    # UNTYPED, 5 arguments
pset.renameArguments(ARG0='closes', ARG1='highs',
                     ARG2='lows',   ARG3='opens', ARG4='amounts')
pset.addPrimitive(add, 2);  pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2);  pset.addPrimitive(div, 2)
pset.addPrimitive(rank, 1); pset.addPrimitive(neg, 1)
pset.addPrimitive(diff5, 1)
pset.addPrimitive(ts_max5, 1); pset.addPrimitive(ts_min5, 1)
pset.addPrimitive(ts_ma5, 1)

params = {'lag_periods': 2, 'quantile': 0.2, 'n_trading_days': 243,
          'trading_cost': 0.0012, 'fitness': 'ir',
          'number_population': 100, 'number_generation': 10,
          'number_hall_of_fame': 10}
```

Four limits, in descending order of how much they mattered:

**(a) Raw price levels as terminals.** `closes`, `highs`, `lows`, `opens`,
`amounts` differ across stocks by orders of magnitude and are extremely stable
over time. The first structure any cross-sectional search finds in such data is
the **size factor** — `neg(closes)` ranks small companies first and barely
changes day to day. All ten hall-of-fame members from that run correlated
**0.99+** with one another. The search was not finding ten alphas; it was
finding one, ten times.

**(b) Hardcoded 5-day windows.** `diff5`, `ts_max5`, `ts_min5`, `ts_ma5` are
fixed at five days. The GP **cannot express a 20-day moving average**, so an
entire dimension of the space — the time horizon of every operator — is closed
off. Any effect operating at a monthly horizon is unreachable by construction.

**(c) Untyped primitive set.** With `PrimitiveSet` rather than
`PrimitiveSetTyped`, every function accepts every argument. That is workable
when all operators are unary over frames, but it makes parameterised windows
impossible: nothing prevents `ts_mean(x, closes)`.

**(d) A fitness function measuring a different strategy from the target.**
`calc_ir` scores a **quintile long/short bucket** at `lag_periods = 2`:

```python
factors = factor_data.shift(2)          # 2-day delay
ranking = factors.rank(axis=1, pct=True)
top_qt  = ranking < 0.2                 # bottom quintile by value
bot_qt  = ranking > 0.8
ls_returns = (bot_returns - top_returns) / 2.0
```

Trexsim scores a **fully-weighted, dollar-neutral, industry-neutral
cross-section at delay 1**. Equal-weighting the extreme 20% and discarding the
middle 60% is a materially different portfolio from weighting all 1,000 names —
different turnover, different concentration, different exposure. The GP was
optimising a strategy nobody was going to run.

Plus two calibration mismatches: `n_trading_days = 243` (Chinese calendar; US is
252) and `trading_cost = 0.0012` — 12bp two-way, appropriate for A-shares and
roughly double a realistic US large-cap figure.

---

## 2. Rebuild 1 — strongly-typed GP with searchable windows

```python
class Win(int):
    """Distinct type so windows can only appear in window slots."""

pset = gp.PrimitiveSetTyped("MAIN", [pd.DataFrame] * n, pd.DataFrame)

for f in (add, sub, mul, div):                    pset.addPrimitive(f, [D, D], D)
for f in (neg, signlog, signsqrt, cs_rank, cs_zscore):
                                                  pset.addPrimitive(f, [D], D)
for f in (ts_mean, ts_std, ts_sum, ts_median, ts_max, ts_min,
          ts_skew, ts_delay, ts_delta, ts_zscore, ts_decay):
                                                  pset.addPrimitive(f, [D, Win], D)
pset.addPrimitive(ts_corr, [D, D, Win], D)

for w in (5, 10, 20, 60):
    pset.addTerminal(Win(w), Win, name=f"w{w}")
pset.addPrimitive(w_id, [Win], Win)               # see below
```

Subclassing `int` as `Win` creates a type DEAP's tree generator treats as
distinct, so a window argument can only ever be filled by a window terminal.
That makes **the horizon of every operator a searchable parameter** — the single
largest expansion of the space relative to the baseline.

**One non-obvious detail.** DEAP's `genHalfAndHalf` fails with *"tried to add a
primitive of type Win, but there is none available"* if a type has terminals but
no primitives. The fix is `w_id`, an identity function `Win → Win` that exists
solely to satisfy the generator. It is unwrapped during translation
(`if fn == "w_id": return args[0]`) so it never reaches the platform.

**Every primitive maps 1:1 onto a Trexsim operator**, so a discovered expression
translates directly rather than being reimplemented by hand. Two translations
are non-trivial and would silently change the alpha if done naively:

```python
if fn == "div":     return f"({args[0]} / at_zero2nan({args[1]}))"
if fn == "ts_decay": return f"ts_mean_exp({args[0]}, {args[1]}, 0.5)"
```

and `cs_rank`, which returns **[1, 2]** on Trexsim rather than [0, 1], is
rescaled on the way out.

---

## 3. Rebuild 2 — dimensionless terminals

Seventeen terminals, every one a ratio, a return, or a bounded rank:

| terminal | definition | class |
|---|---|---|
| `c_ma20`, `c_ma60` | `close / ts_mean(close, N)` | price shape |
| `hl` | `high / low` | range |
| `co` | `(close − open) / open` | intraday return |
| `gap` | `(open − close₋₁) / close₋₁` | overnight return |
| `stoch` | position in the 20-day high/low range, bounded [0,1] | range |
| `ret1`, `ret5`, `ret20` | returns | momentum/reversal |
| `v_ma20` | `volume / ts_mean(volume, 20)` | participation |
| `dv_adv` | dollar volume / 60-day ADV | liquidity |
| `vol20` | `ts_std(ret1, 20)` | volatility |
| `size` | `cs_rank(adv)`, bounded | size |
| `beta60` | `ts_corr(ret1, ret1_spx, 60)` | market exposure |
| `amihud` | `|ret1| / dollar_volume` | price impact |
| `hi52` | `close / ts_max(close, 252)` | 52-week anchor |
| `idio` | `ret1` minus its industry mean | residual |

No terminal carries a price level, so there is no size attractor to collapse
onto. This also satisfies Trexsim's dimensionless rule, which protects against
corporate-action retro-adjustment — the failure mode documented in note 43.

**Every terminal is winsorised cross-sectionally at 1%/99% before the search
sees it.** Ratios like `v_ma20` spike 50× on a news day, and unwinsorised a GP
expression can earn most of its in-sample IR from a handful of observations.
Clipping removes that degree of freedom at almost no cost.

---

## 4. Rebuild 3 — the fitness function

`fitness()` mirrors the platform rather than the course project:

1. mask to the point-in-time universe
2. **industry-neutralise** (the sim runs `Neut=industry`)
3. winsorise the alpha at 1%/99%
4. subtract the cross-sectional median (as the sim does)
5. divide by gross exposure → unit-gross, dollar-neutral book
6. PnL at **delay 1**, turnover from position changes
7. score = **IR / √TVR**, the metric the correlation-override clause uses

### 4.1 One correction that mattered more than the rest

IR/√TVR alone is the wrong objective, because **the platform's primary gate is
IR, not IR/√TVR**. An alpha below 0.07 does not qualify at all, so its
tie-break metric is irrelevant. Without a correction the search prefers a
low-turnover alpha at IR 0.053 to a real one at IR 0.107, purely because the
smaller √TVR wins the division.

The fix is a soft multiplicative gate:

```python
ir_daily = ir / np.sqrt(252)
if ir_daily < 0.07:
    score *= (max(ir_daily, 0.0) / 0.07) ** 2
```

Squared so the pull is firm; multiplicative so ordering is preserved among
sub-threshold candidates and the search keeps a usable gradient early on.

---

## 5. Reward hacking, and the five degeneracy guards

Every guard below exists because the search found the exploit first.

| # | guard | threshold | what it caught |
|---|---|---|---|
| 1 | cross-sectional dispersion | `1e-4` | `div(ret20, ret20)` = 1 everywhere. **16 of 26 "winners"** contained it, at TVR ≈ 0.0018 |
| 2 | minimum turnover | `0.01` | static holdings scoring as signals |
| 3 | minimum positions | `100` | expressions NaN almost everywhere, concentrating the book in 1–2 names: **in-sample IR 3.28, held-out −2.40** |
| 4 | **effective N** = 1/Σw² | `50` | four alphas reporting `numstk ≈ 400` while holding **99% of the book in one name** — invisible to a position count |
| 5 | TVR floor in the denominator | `0.05` | IR/√TVR diverging as TVR → 0 |

**Guard 4 is the one worth describing in an interview.** `numstk` counts
non-NaN positions and cannot see concentration. Inverse Herfindahl measures what
is actually held. The platform has the same blindness, so an alpha can pass its
`numstk > 160` check while being a one-stock bet.

**Guard 1 is a textbook specification-gaming result.** The objective
IR/√TVR is unbounded as TVR → 0, and the search located that singularity
independently in 16 of 26 runs before any floor existed. It did not find a
better alpha; it found a better exploit of the scoring function.

---

## 6. Anti-overfitting, and the −0.70

Run 1 used depth 8. Its winners were ~40 nodes and had **in-sample IR 3.28
against held-out IR −2.40 — opposite signs**. Across the whole run the rank
correlation between in-sample fitness and held-out IR was **−0.70**.

That number is the single most important result of the GP phase. It says
sorting candidates by fitness was **worse than sorting them at random**. Every
selection decision made on local fitness was, on average, pointed the wrong way.

Two changes followed:

```python
MAX_DEPTH = 5      # was 8; node count is the best single predictor of overfitting
PARSIMONY = 0.004  # mild multiplicative size penalty, so a simpler tree wins ties
v *= max(0.5, 1.0 - PARSIMONY * len(ind))
```

and the training window widened from 2011–2017 (7 years, **no 2008 at all**) to
2006–2018, with 2019–2020 held out for validation and 2021–2022 untouched.

**Both runs still produced zero platform qualifiers unaided.** Run 2 was better
engineered in every respect and produced *fewer* local qualifiers (7 vs 21) and
the same number of platform qualifiers: zero.

---

## 7. The search space — the number that reframes everything

With 11 terminals per seed (a 70% random subset of 17), 5 unary and 4 binary
primitives, 11 window-parameterised time-series operators and 4 window values,
the count of syntactically valid trees of height ≤ 5 is approximately:

| height | distinct trees |
|---|---|
| ≤ 1 | 1.0 × 10³ |
| ≤ 2 | 4.3 × 10⁶ |
| ≤ 3 | 7.5 × 10¹³ |
| ≤ 4 | 2.2 × 10²⁸ |
| **≤ 5** | **2.0 × 10⁵⁷** |

Against 1.57 × 10⁶ expressions actually evaluated, that is a covered fraction of
**8 × 10⁻⁵²**.

**This is the correct way to think about what a GP does.** It is not searching
the space — no conceivable amount of compute searches 10⁵⁷. It is running a
stochastic local hill-climb from a few hundred random starting points, and
everything it finds is reachable from those starts by mutation and crossover
under a fitness gradient. When that gradient is anti-correlated with the true
objective, more compute makes matters worse, not better.

---

## 8. What the search actually spent its time on

Terminal usage across all 450 hall-of-fame members (`gp_results_all.csv`):

| terminal | appears in | share |
|---|---|---|
| **`size`** | **225 / 450** | **50.0%** |
| `c_ma20` | 117 | 26.0% |
| `dv_adv` | 117 | 26.0% |
| `c_ma60` | 97 | 21.6% |
| `gap` | 83 | 18.4% |
| `ret1` | 80 | 17.8% |
| `v_ma20` | 75 | 16.7% |
| `beta60` | 70 | 15.6% |
| `ret20` | 63 | 14.0% |
| `hl` | 49 | 10.9% |
| `co` | 36 | 8.0% |
| `ret5` | 29 | 6.4% |
| **`stoch`** | **23** | **5.1%** |
| **`vol20`** | **15** | **3.3%** |
| `amihud`, `hi52`, `idio` | **0** | **0.0%** |

Two findings here, and both are more interesting than any alpha the search
produced.

### 8.1 The size attractor survived the fix

`size` appears in **half** of all surviving expressions — more than double the
next terminal. Removing raw price levels stopped the 0.99-correlation collapse,
but `cs_rank(adv)` is still a slow-moving, highly stable cross-sectional
ordering, and the search still reaches for it more than anything else.

**The clone problem was mitigated, not solved.** A stronger version of the fix
would have been to orthogonalise every terminal against `size` before the
search, or to drop it entirely and let the sim's own neutralisation handle it.

### 8.2 The two winners came from the terminals the search neglected

Of ~50 GP cores tested on the platform, exactly two ever qualified:

| core | key terminal | share of HOF using it |
|---|---|---|
| A1 | `stoch` | **5.1%** |
| A3 | `vol20` | **3.3%** |
| — | `size` | 50.0%, **zero qualifiers** |

The search allocated half its surviving population to a terminal that produced
nothing, and the two cores that worked came from terminals it used in 3–5% of
expressions. **This is the −0.70 rank correlation made concrete**: local fitness
did not merely fail to rank candidates, it systematically misdirected the search
toward the wrong region of the terminal space.

### 8.3 Three terminals were built but never ran

`amihud` (price impact), `hi52` (52-week anchor) and `idio` (industry residual)
appear in **zero** of 450 expressions. File timestamps explain why:

```
2026-08-09 01:00   GP_Alphas_run1.txt      run 1 output
2026-08-09 06:23   gp_results_all.csv      run 2 output
2026-08-09 12:52   pipeline/tq_gp.py       terminals added HERE
```

They were added **after both searches completed** and never used. The code
comment describes them as *"three terminals carrying information the others do
not"* — Amihud illiquidity is a documented priced factor, `hi52` is the anchor
behind momentum and the disposition effect, `idio` is what actually gets traded
under industry neutralisation. All three are economically distinctive, and none
was ever searched.

**A second, compounding defect:** all three are also **missing from
`TERMINAL_TREXSIM`**, the translation map. `to_trexsim` resolves unknown names
with `terminals.get(node.id, node.id)` — it falls through to the bare token
silently. So even if a winner had used them, the emitted expression would have
contained the literal string `amihud`, which is not a Trexsim variable.

The search space and the deployable space had diverged, and nothing in the
pipeline would have reported it.

---

## 9. Diversity by construction, and the correlation prune

Two mechanisms addressed the competition's ≤50% correlation rule *during* the
search rather than after it.

**Random terminal subsets.** Each of the 30 seeds draws its own random 70%
subset of terminals, so different runs are forced to explore different data:

```python
k = max(5, int(len(terms) * 0.7))
names = sorted(rng.sample(sorted(terms), k))
```

**Greedy PnL prune** at the platform's own threshold — sort by fitness
descending, accept a candidate only if its correlation against everything
already accepted is ≤ 0.5. This is what stops twenty submissions qualifying two.

---

## 10. Two bugs in the scaffolding itself

Both were silent, both would have invalidated the output, and neither raised an
exception.

**Industry neutralisation left unclassified names untouched.** Under a constant
alpha every *classified* name demeans to ~0, so the handful of tickers with a
missing industry code became **the only non-zero positions in the book**. The
search could score well by producing a constant and letting the entire portfolio
fall on names whose sector classification happened to be absent. Fixed by
setting them to NaN — which also matches the platform, where there are no
unclassified names.

**The correlation prune compared un-neutralised PnL.** `fitness` and `report`
both neutralised; `pnl_series` initially did not. Two alphas could therefore
pass the correlation gate on raw PnL and be near-duplicates once neutralised —
which is the single failure that gate exists to prevent.

---

## 11. Engineering

**Thread pinning before numpy imports.** numpy on macOS uses Accelerate, which
multithreads internally. With 10 worker processes each spawning its own thread
pool, threads contend for the same 10 cores and throughput collapses. One thread
per process is correct when parallelism is already at the process level — and it
must be set in the parent *and* every spawned worker, so it sits at module top
level.

**Per-process panel cache.** macOS spawns rather than forks, so each worker
re-imports the module and must load the panel itself. A module-global cache
makes that once per worker rather than once per seed.

**Results are returned as formula strings, not DataFrames.** Each alpha frame is
~4 MB; shipping 450 of them back through IPC would cost more than recomputing
each from its formula in the parent (~50 ms).

**Measured cost and total compute:**

| | trees | cost/eval | evaluations | wall clock (10 jobs) |
|---|---|---|---|---|
| run 1 | ~40 nodes, depth 8 | 416 ms | 828,000 | **9.6 h** |
| run 2 | ~14 nodes, depth 5 | 337 ms | 738,000 | **6.9 h** |

Run 2 trained on a window **1.9× longer** yet cost *less per evaluation*,
because the depth cap made trees 3× smaller and node count dominates evaluation
time.

**~16.5 hours of compute across both runs.**

---

## 12. The honest verdict

**What the GP produced:** two cores that ever qualified on the platform, and
only after a hand-built conditioner was attached to each. Neither cleared the
0.07 gate unaided.

**What the GP taught, which was worth more:**

1. **Search quality is bounded by fitness quality.** Run 2 improved every aspect
   of the machinery and changed nothing, because the objective was the problem.
2. **A −0.70 rank correlation makes selection actively harmful**, and §8.2
   quantifies the damage: half the population spent on a terminal that produced
   nothing, while the two that worked came from terminals used 3–5% of the time.
3. **Specification gaming is the default outcome**, not an edge case. Five
   separate guards were needed, and each was written only after the search had
   already found the exploit.
4. **10⁵⁷ versus 1.57 × 10⁶.** A GP is a stochastic local search from a handful
   of random starts, not an exploration of a space.

**The correct division of labour**, and the account I would give: the machine
searched a structured space of price-and-volume expressions and found two usable
building blocks. It could not have found the conditioner that made them work,
because the local panel contains **no earnings dates, no FOMC calendar and no
expiration schedule**. Adding that conditioner roughly doubled the IR of both.
The human contribution was not a better expression — it was a variable the
search could not see.

---

## 13. What I would do differently

1. **Fix the fitness before scaling the search.** Cheap diagnostic: sample 500
   random expressions, evaluate locally, evaluate on the platform, measure the
   rank correlation. Had I done that first, the −0.70 would have surfaced in an
   afternoon rather than after 16 hours of compute and 50 platform submissions.

2. **Orthogonalise terminals against `size`**, or remove it. Half the search
   collapsed toward it and it produced nothing.

3. **Assert that every terminal is translatable** at pipeline start. A three-line
   check would have caught §8.3 — a set-difference between `build_terminals` and
   `TERMINAL_TREXSIM`, raising rather than falling through.

4. **Use the platform as the fitness function directly** via a small number of
   queries, with a surrogate model over expression features to choose which to
   spend them on. That is Bayesian optimisation with an expensive oracle, and it
   is the correct formulation of this problem — which I only recognised after
   the search had failed.

5. **Multi-objective (NSGA-II) rather than a scalarised IR/√TVR.** Scalarising
   is what created the singularity at TVR → 0 that guards 1 and 5 exist to
   patch. A Pareto front over (IR, TVR, correlation) has no such exploit.
