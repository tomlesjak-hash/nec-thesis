# Trexquant Alpha Competition — Platform & Rules

**Source:** "Trexquant Competition Tutorial" (27pp) + PySim UI. Saved 2026-08-07.

---

## What data are we actually trading? — two different numbers

| | Number | What it is |
|---|---|---|
| **Database** | **~3,250 US names** | What's physically in the data files. Every daily price/volume var reports coverage 3237–3251; intraday 3246–3249; returns 3250. This is roughly the full liquid US listed cross-section. |
| **Simulation universe** | **top1000 US** | A *sim setting*, not a property of the data. The engine screens to the top 1000 each day (liquidity/size ranked) and you only take positions there. |

Confirmed from the tutorial's own PnL legend, which is the full settings string:

```
ir 0.025  ret 0.044  tvr 1.442  dd 48.4  ir/sqr(tvr) 0.021  cs 0.00
bk 1.00x1.00  stk 773-772  liq 287  liqn 122
d0  top1000  industry  fast  us   (20060331-20211230)
```

→ delay **0** · universe **top1000** · neutralization **industry** · mode **fast** · region **us** · dates **2006-03-31 → 2021-12-30**. Correlation panel in the same example ran **2020-01-08 → 2021-12-30**.

**Two caveats before trusting the date range:**
1. Those are the *tutorial's example* settings. Verify the competition's actual date range and universe in the UI settings panel — don't assume.
2. The FRED variables' stated bounds fingerprint **2022 → ~mid-2023** (CPI 288.66–301.81, WTI $73.28–114.84, U3 3.1–3.4%, GDP $24.4–26.1T), which is *outside* the tutorial's 2006–2021 window. Either the competition data extends further, or the bounds column is computed differently. Worth resolving — it changes which regimes you're fitting.

**Why the two numbers matter for alpha design:** every Coverage figure in the data dictionary is against the ~3250 database, *not* against the top1000. A fundamental with coverage 800 might be 800 large caps (dense inside your universe) or 800 microcaps (nearly empty inside it). Never infer usability from the coverage number alone — run it and read `stk`, the realised long×short stock count, against the `numstk > 160` threshold.

---

## Scoring — read this before writing a single alpha

### Qualification (per alpha)
- **IR > 0.07** (information ratio = mean(daily ret) / std(daily ret))
- **Correlation ≤ 50%** to *any* alpha already in your qualified pool
- **Override:** if corr > 50%, the new alpha still qualifies if its **IR/√TVR is ≥ 20% higher** than the alpha it correlates with

### Final rank = rank(Alpha Count) + rank(Combined Performance)
1. **Alpha Count** — number of qualified, uncorrelated alphas in the pool
2. **Combined Performance** — all qualified alphas aggregated into one combined alpha, evaluated **out-of-sample**

Lowest summed rank wins.

### Strategic implication
This is **not** a "find one great alpha" competition. It is a **diversification** competition scored twice over.

- The `≤ 50% correlation` gate means the 20th alpha is worth as much in sub-score (1) as the 1st.
- The combined-alpha sub-score (2) rewards **low pairwise correlation**, because combining N alphas with average pairwise ρ gives combined IR ≈ IR_avg × √(N / (1 + (N−1)ρ)). At ρ = 0.5, 10 alphas buy you only ~1.35× the IR of one. At ρ = 0.1, the same 10 buy ~2.4×.
- Therefore: **maximise the number of genuinely orthogonal ideas, not the IR of any single one.** An IR-0.08 alpha off an untouched data group is worth more than an IR-0.15 alpha that is 0.6-correlated with three you already have.
- Practical target: sweep *across data groups* (intraday / fundamental / seasonality / macro-beta / volume), because different input data is the cheapest source of decorrelation. Sweeping parameters on one idea produces correlated clones that fail the gate.
- Since the override clause is on **IR/√TVR**, a high-IR but high-turnover alpha is penalised. Keep turnover down where possible.

---

## Trexsim mechanics

**Alpha = a matrix. Rows = stocks, columns = dates.** The value is the *desired dollar position*.

```python
alpha = (open - close) / open
```

- Positions scale so long book and short book each sum to the booksize (e.g. 1.00 × 1.00 = $1M long, $1M short).
- **The median is subtracted from the alpha matrix** → a value of `0` becomes a real position. Use `np.nan` for "no position".
- Expressions should be **dimensionless** (ratios), both for interpretability and to cancel corporate-action adjustment bias.

### Operator naming convention
| Prefix | Scope | Examples |
|---|---|---|
| `at_` | element-wise | `at_nan2zero(A)`, `at_zero2nan(A)` |
| `ts_` | time series, per stock | `ts_mean(A, d)`, `ts_delay(A, d)`, `ts_rank(A, d)`, `ts_norm(A, d)`, `ts_std(A, d)`, `ts_mean_exp(A, d, decay)`, `ts_delta` |
| `cs_` | cross-sectional, per day | `cs_zscore(A)`, `cs_rank(A)`, `cs_winsor(A)`, `cs_remove_middle(A)` |

Custom operators are plain Python functions registered on the **Operator** tab:

```python
def n_days_return(close, days):
    out = np.full(close.shape, np.nan)
    past_close = ts_delay(close, days)
    out = (close - past_close) / past_close
    return out
```

Write one only if (a) no built-in does it and (b) you'll reuse it. One-off helpers can be defined inline in the alpha editor.

**Trexsim is not a general ML environment** — memory and runtime constraints rule out complicated models. Vectorize; the tutorial's own example sped up several orders of magnitude by replacing a day loop with `ts_rank`.

---

## Fast vs Slow mode

| Fast | Slow |
|---|---|
| Simulates whole history at once | Day-by-day, only past data, with a given lookback |
| Much faster — use for idea screening | Accurate — **required for submission** |
| Can differ from reality | The truth |

### Why they diverge
1. **Insufficient lookback** — slow mode only runs your code on days `i-lookback … i`. `ts_mean(-ret1, 30)` needs ≥ 31 days of lookback; give it 5 and you get N/A everywhere.
2. **Forward bias** — fast mode can see the future. Extreme case: `ts_delay(close, -1)`. Fast shows a beautiful line; slow shows flat zero.
3. **Corporate adjustments** — splits/dividends are applied retrospectively, and stocks that split are more likely to be ones that went up. `alpha = -close` differs between modes for this reason. Dimensionless expressions mitigate it.

---

## Metrics

| Abbr | Name | Definition |
|---|---|---|
| `ir` | Information Ratio | mean(daily return) / std(daily return) |
| `ret` | Annualized Return | cumulative_pnl / ((longbook + shortbook)/2) × (252 / simulation_days) |
| `tvr` | Turnover | mean(dollars_traded) / ((longbook + shortbook)/2) |
| `dd` | Drawdown | peak-to-trough of cumulative pnl / ((longbook + shortbook)/2) |
| `ir/sqr(tvr)` | IR/√TVR | the metric the correlation-override clause uses |
| `liq` | Liquidity | capital allocatable with minimal market impact (relative gauge) |
| `liqn` | Liquidity (New) | booksize at which the alpha's avg daily stock volume = 1% of market volume |
| `bk` | Booksize | long × short size, $m |
| `stk` | Number of Stocks | long-count × short-count, averaged over last 120 trading days |

Observed thresholds from the tutorial's OS test panel: `ir > 0.07`, `tvr < 0.5`, `numstk > 160`, `max corr (yours) < 0.50`, `max corr (others) < 0.90`.

> ⚠️ Note the `tvr < 0.5` threshold in that panel — tighter than the case-study alphas achieved (tvr 1.58). Verify the live threshold in the UI before optimising.

---

## Case study — High-Volume Return Premium (Gervais, Kaniel & Mingelgrin 2001, JoF)

**Claim:** stocks with unusually high volume outperform; unusually low volume underperform, over the following month. Attention/visibility mechanism, not autocorrelation or liquidity.

Paper's setup: 1-day formation vs 49-day reference, top/bottom 10%, 20-day hold. Reported ~11% annual excess return on small/mid NYSE names, 1963–96.

### Evolution of the implementation (the actual lesson)

```python
# v1 — literal transcription, day loop, ~40 lines
# ir 0.063, tvr 1.584

# v2 — vectorized, same idea
alpha = (ts_rank(volume,50) > 1.9).astype('float') - (ts_rank(volume,50) < 1.1).astype('float')

# v3 — continuous weights instead of decile buckets: bets more often, on more stocks
alpha = ts_norm(volume, 50)
# ir 0.013, tvr 1.003 — turnover too high

# v4 — exponential decay = "hold with exponentially decreasing size"
alpha = ts_mean_exp(ts_norm(volume,50), 20, 0.1)
# ir -0.000, tvr 0.214 — turnover fixed, signal died (deteriorates from 2016, loses in 2017)

# v5 — condition the signal on its own volatility
alpha = ts_mean_exp(ts_norm(volume,20) * (cs_rank(ts_std(volume,10)) < 1.0), 3, 0.5)
# ir 0.015, tvr 0.753, ir/sqrt(tvr) 0.018, numstk 618x650
```

**Takeaways:**
- Move from **buckets → continuous weights** (bet on more stocks, more often).
- Move from **fixed holding period → exponential decay** (controls turnover).
- **Condition** the signal on a second variable to strengthen it.
- Trexquant's own framing: *"medium-frequency statistical arbitrage to profit from a large number of favorable bets."* Design for breadth, not conviction.

### The paper's own suggested variation axes (each = a decorrelated alpha)
size · returns · momentum · bid-ask spread · short interest · industry classification · different weights · different holding times

And the tutorial's forward-looking prompts:
- Do other volume datasets have predictive power? (**intraday volume**, option volume, buy/sell volume)
- What other variables show the attention effect? (**returns, news, time of day**)
- Combine with other classification strategies (**industry**)
- Different trading rules (weights, holding times)

> Two of those four point directly at `GROUP_intra1`, which this competition provides in full.

---

## Common errors checklist

- [ ] `0` used where `np.nan` was meant → phantom positions
- [ ] `np.mean` / `np.sum` instead of `np.nanmean` / `np.nansum` → NaN contamination (built-in operators already handle this)
- [ ] Insufficient lookback in slow mode
- [ ] Forward bias from future data leakage (`ts_delay(x, -1)`, or Delay0=No vars used at delay 0)
- [ ] Mixing variables of different units/scales
- [ ] Ignoring splits/dividends — keep expressions dimensionless
- [ ] Slow Python loops instead of vectorized expressions
- [ ] Adding alphas without NaN handling:
      `alpha = at_zero2nan(at_nan2zero(alpha1) + at_nan2zero(alpha2))`

---

## Workflow notes

- Alpha list page has filters; **keep a personal sheet** of link + expression + hypothesis for every simulation worth remembering.
- Correlations: select alphas via checkbox on the alpha list → **Compute Correlation** in the bottom action bar. Do this *before* submitting, to protect the ≤ 50% gate.
- Research loop: Observations/Theory/Data/Articles/Experience → Idea → Code → Simulation → Alpha → back to Idea.
- Idea sources the tutorial names: Google Scholar, SSRN, Investopedia, Quantpedia, Bloomberg/MarketWatch.
