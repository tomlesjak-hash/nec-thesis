# Translating a GP alpha → Trexsim

Source: `QIP/Projects/my_project/gp_results_topk.csv`, row 10 — the best of the
20 by IR (2.537 gross, 1.407 net, dd 0.114).

## The raw expression

```
add(
  amounts,
  neg(ts_ma5(diff5(
    mul(
      mul(
        add(sub(amounts, ts_min5(closes)), highs),
        div(add(amounts, closes), amounts)
      ),
      add(mul(amounts, lows), div(ts_ma5(opens), ts_ma5(highs)))
    )
  )))
)
```

---

## 1. Faithful translation (paste-ready)

```python
# `amounts` is CN turnover in currency; Trexsim has no such field.
# The exact equivalent is price x shares traded.
amounts = close * volume

A = (amounts - ts_min(close, 5)) + high
B = (amounts + close) / at_zero2nan(amounts)
C = amounts * low + ts_mean(open, 5) / at_zero2nan(ts_mean(high, 5))

X = (A * B) * C
alpha = amounts - ts_mean(X - ts_delay(X, 5), 5)
```

### Operator mapping used

| QIP | Trexsim | Note |
|---|---|---|
| `amounts` | `close * volume` | no native field |
| `ts_ma5(x)` | `ts_mean(x, 5)` | |
| `ts_min5(x)` | `ts_min(x, 5)` | |
| `diff5(x)` | `x - ts_delay(x, 5)` | ⚠️ **not** `ts_diff(x, 5)` — see below |
| `div(x, y)` | `x / at_zero2nan(y)` | QIP's `div` patches `y==0` to `1e-10`; `at_zero2nan` is the Trexsim idiom and gives NaN instead of a 1e10 explosion |
| `neg(x)` | `-x` | |
| `rank(x)` | `cs_rank(x) - 1` | [1,2] → [0,1]; not used in this expression |

> ⚠️ **`ts_diff` is a trap here.** Your `diff5` is `df - df.shift(5)` — a
> *lag-5* difference. Trexsim documents `ts_diff(A, n)` as "the n-th discrete
> difference", which is `np.diff` semantics — differencing applied **n times
> iteratively**, a completely different operation. `x - ts_delay(x, 5)` is
> unambiguous. Use it.

---

## 2. Why this will almost certainly disappoint on Trexsim

**Simplify it by orders of magnitude.** In CN data `amounts` ≈ 1e8 while prices
≈ 1e1. So:

| Term | Approximates to |
|---|---|
| `(amounts - ts_min5(closes)) + highs` | `amounts` |
| `(amounts + closes) / amounts` | `1` |
| `amounts * lows + ts_ma5(opens)/ts_ma5(highs)` | `amounts * lows` |
| `X = (A*B)*C` | **`amounts² × lows`** |

So the whole thing collapses to:

```
alpha ≈ amounts - ts_mean(Δ₅(amounts² × lows), 5)
```

and since `amounts²` dwarfs `amounts`, the first term is decorative. The real
alpha is **`-ts_mean(Δ₅(amounts² × lows), 5)`** — a smoothed 5-day reversal on
dollar-volume-squared times price.

Three reasons that won't survive the trip:

1. **Dimensionally meaningless.** Units are currency² × price. The tutorial is
   explicit that expressions should be dimensionless, both for interpretability
   and to cancel corporate-action retro-adjustment. `amounts²` does the
   opposite — it *amplifies* split effects quadratically.
2. **Scale-dominated.** A handful of mega-liquid names will have `amounts²`
   several orders above everything else and absorb essentially the entire book.
   No `cs_rank` or `cs_zscore` anywhere to control it.
3. **Wrong market.** It was fitted on China A-shares, where the size/liquidity
   effect is far stronger than in US large caps. Row 0 of the same file is
   `neg(amounts)` at IR 1.208 — that *is* the CN small-cap effect, and it's the
   attractor the whole run collapsed onto.

**Submit the faithful version anyway** — it costs one simulation and tells you
empirically how much transfers. Just don't be surprised by a low IR.

---

## 3. The version actually worth submitting

Same economic idea — *reversal on a surge in trading activity* — made
dimensionless and scale-controlled:

```python
dv     = close * volume
rel_dv = dv / at_zero2nan(ts_mean(dv, 20))          # dimensionless
chg    = rel_dv - ts_delay(rel_dv, 5)               # 5-day change
alpha  = -cs_rank(ts_mean(chg, 5)) + 1.0            # ranked, [-1, 0]
```

What changed and why:

- `dv / ts_mean(dv, 20)` — each stock measured against **its own** history, so
  mega-caps no longer dominate. Also kills the split-adjustment problem.
- Dropped `× lows` — that was a price-level (size) artifact, not economics.
- Dropped `amounts²` — squaring had no economic justification, only scale.
- `cs_rank` at the end — bounded positions, controlled turnover.
- `+ 1.0` shifts `cs_rank`'s [1,2] output to [0,1] before negation.

This is close to the tutorial's own High-Volume Return Premium case study
(Gervais, Kaniel & Mingelgrin), so **check correlation against any volume alpha
already in your pool** before submitting — it may well breach the 0.5 gate.

---

## 4. ⚠️ What this file really shows: the clone problem, quantified

Rows 10–19 are ten hall-of-fame slots. Count the *distinct* expressions:

- Row 10: `div(add(amounts, **closes**), amounts)`
- Rows 11, 12, 13, 15, 16, 17, 19: `div(add(amounts, **highs**), amounts)` — **identical to each other**
- Rows 14, 18: same as row 11 with the two `add` arguments swapped — **commutatively identical**

**Ten slots, two distinct alphas, differing at a single leaf node.** All ten
report dd 0.114 and net IR 1.389–1.407, which is what near-perfect correlation
looks like.

Rows 0–9 (the DEAP run) are the same story with a different attractor:
`neg(amounts)`, `neg(highs)`, `neg(lows)`, `neg(closes)`, `neg(add(opens, closes))`
— every one a price/liquidity level, all correlating 0.99+ in
`factor_corr_near_duplicates.csv`.

Under Trexquant's rules you'd submit twenty and qualify **two**.

The fixes, in order of impact:

1. **Dimensionless terminals.** With raw levels available, GP finds the size
   factor in generation 1 and never leaves. Give it `close/ts_mean(close,20)`,
   `high/low`, `volume/ts_mean(volume,20)`, `ret1`, `(close-open)/open`.
2. **Diversity pressure.** `HallOfFame` keeps the N highest-fitness individuals
   with no novelty term — it fills with mutations of one winner by
   construction. Penalise correlation to already-accepted alphas, or run many
   seeds with different terminal subsets and prune afterwards.
3. **Fitness = IR/√TVR**, industry-neutral — matching the scorer rather than
   quintile long/short at `lag_periods=2`.
