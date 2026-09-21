# Data Acquisition & Training Plan — Trexquant Competition

**Status:** DRAFT — awaiting Tom's confirmation before any download
**Date:** 2026-08-07 · **Deadline:** ~2026-08-29 (≈21 days)

---

## 0. The finding that shapes everything

Your QIP genetic program, as configured, **produces clones**. This is not a hypothetical — it's in your own output files:

`gp_results_topk.csv` hall of fame:
```
neg(amounts)                              ir 1.208
sub(rank(opens), add(opens, amounts))     ir 1.204
neg(highs)                                ir 0.984
neg(ts_min5(amounts))                     ir 0.932
neg(diff5(closes))                        ir 0.783
neg(lows)                                 ir 0.713
neg(closes)                               ir 0.685
neg(add(opens, closes))                   ir 0.671
```

`factor_corr_near_duplicates.csv`:
```
gp_deap_01 ↔ gp_deap_02    0.9991
gp_deap_01 ↔ gp_deap_03    0.9951
gp_deap_01 ↔ gp_deap_04    0.9982
gp_deap_01 ↔ gp_deap_05    0.9983
...all pairs 0.99+
```

**All ten hall-of-fame members are one factor.** `neg(closes)` = "short high-priced stocks" = a size proxy. Everything else is a cosmetic mutation of it.

Under Trexquant's rules that means: submit 10, qualify 1. Both halves of the score (alpha count, combined OS performance) punish exactly this failure.

### Root causes and fixes

| Cause | Why it breaks | Fix |
|---|---|---|
| **Raw price *levels* as terminals** (`closes`, `highs`, `lows`, `opens`, `amounts`) | Level differences across stocks are enormous and stable; they swamp every subtler signal. GP finds the size factor in generation 1 and never leaves. | Feed **only dimensionless terminals**: `close/vwap`, `high/low`, `volume/ts_mean(volume,20)`, `ret1`, `ret5`, `(close−open)/open`, `cs_rank(mktcap)`. This also matches Trexsim's "make expressions dimensionless" rule. |
| **Fitness = IR alone** | DEAP's `HallOfFame` keeps the N highest-fitness individuals with no diversity pressure. It fills with mutations of one winner by construction. | Correlation-penalised fitness: `IR − λ·max(|corr| to already-accepted)`. Or multi-objective NSGA-II on (IR, −max_corr). Simplest workable version: many independent runs with **different random seeds and different terminal subsets**, then prune by correlation afterwards. |
| **Search is tiny** (pop 100 × 10 gens) | ~1000 evaluations total. Converges to the first strong attractor. | pop 500–1000 × 30–50 gens, and run 10–20 independent seeds. |
| **Fitness doesn't match the scorer** | `calc_ir` uses quintile long/short at `lag_periods=2`, no industry neutralisation. Trexsim uses a rank-weighted, dollar-neutral, **industry-neutralised** book. | Rewrite `calc_fitness` to mirror Trexsim: industry-demean the alpha, rank-weight the full cross-section, compute IR of daily PnL. Otherwise local IR won't predict client IR. |
| **Optimising raw IR** | The competition's tie-break clause is on **IR/√TVR**, and there's a turnover threshold. | Optimise `IR/√TVR` directly. You already have the turnover machinery in `calc_net_ir`. |

---

## 1. Recommendation: what to download

### Fields — 5 minimum, not 2

You said open + close only, because "the rest would be too much." **The storage math says otherwise.** For 1000 stocks × 20 years (≈5,040 trading days):

| | Per field | Notes |
|---|---|---|
| Values | 5.04 M | 1000 × 5040 |
| float64 | **40 MB** | |
| float32 | **20 MB** | half, and plenty of precision for prices |
| Parquet + snappy | **~10–20 MB** | what you'd actually store |

So each additional field costs about **40 MB**. Your QIP folder is **17 GB**. Adding high, low and volume costs 120 MB — 0.7% of what you're already storing.

And they are not optional. From your own `alpha101.py`, field usage across the 88 implemented factors:

```
close   28×      volume  27×      open  11×
amount  11×      high    10×      low    6×
```

**`volume` is used almost as often as `close`.** Dropping it removes the entire liquidity/attention factor family — including the tutorial's own worked case study (the High-Volume Return Premium). Dropping high/low removes the intraday range, which is the closest daily-bar proxy to intraday information you have.

**Download list (Tier 1 — all essential):**

| Field | Size (20yr, float32) | Why |
|---|---|---|
| `open` | 20 MB | |
| `high` | 20 MB | intraday range, gap analysis |
| `low` | 20 MB | intraday range |
| `close` | 20 MB | adjusted |
| `volume` | 20 MB | liquidity/attention family — non-negotiable |
| `close_unadj` or `adj_factor` | 20 MB | lets you reconstruct raw prices; guards against the retro-adjustment bias the tutorial warns about |
| `shares_outstanding` | 20 MB | → market cap, buyback signal |
| `sector` / `industry` code | ~1 MB | **required** — Trexsim neutralises by industry, so your local fitness must too |
| universe membership mask | ~5 MB | boolean, which names are in-universe on each date |

**Total: ~150 MB.** Derive rather than download: returns (from close), dollar volume (close × volume), market cap (close × shares), VWAP approximation.

**Tier 2 — skip, per your decision:** intraday bars. Your instinct is right on effort, though not on storage. Aggregated first-hour/last-hour bars would only be ~480 MB; the problem is that *acquiring* them means pulling minute data (≈2 billion rows, 80 GB+) and aggregating. Not worth it in 21 days.

> **But you don't need to download intraday to use it.** Trexsim already has `GROUP_intra1` pre-computed — 30 variables, first-hour and last-hour OHLC, VWAP, volume, trade count, std, skew, information ratio. So: **GP-search the daily alphas locally, hand-craft the intraday alphas directly on the platform.** You get the least-crowded data group with zero download. This is probably the single highest-value split of your effort.

### Source — recommendation with an honest caveat

| Option | Verdict |
|---|---|
| **yfinance** (already wired in `nec_baseline`) | **Start here today.** Free, working, 592 tickers already cached for 2015–2024. Fatal flaw: **no delisted tickers** → survivorship bias. Fine for building and debugging the pipeline; not fine for final alpha selection. |
| **Sharadar SEP** | **The right answer if you'll spend money.** Covers active *and* delisted US equities back to the 1990s, explicitly "nearly completely free of survivorship bias", unadjusted + split-adjusted + split/div/spinoff-adjusted EOD OHLCV. Available via Nasdaq Data Link, and Sharadar launched a direct offering in July 2026. ⚠️ **Pricing is not published** — I could not verify it. You'd need to check sharadar.com/subscribe yourself. |
| **Polygon.io** (rebranded **Massive.com** as of July 2026) | Stocks plan ~$29/mo, 15-min delayed, unlimited API calls, aggregates endpoint supports daily bars. ⚠️ **Delisted-ticker coverage not confirmed** in my search — verify before buying, because that's the whole point. |
| **WRDS / CRSP** | Gold standard — true point-in-time, delisting returns, no survivorship bias. **Check whether NEC gives you access.** If yes, use it. If access takes more than ~3 days to arrange, it's too slow for this deadline. |

**Recommended path:** check WRDS access today (free if you have it, best quality). In parallel, run the yfinance pipeline immediately so you're never blocked. If WRDS is unavailable, price Sharadar SEP — it's the cheapest route to a survivorship-free sample.

---

## 2. Universe — what "top 1000 by ADV" actually means

**ADV = Average Daily dollar Volume.**

```
ADV_i(t) = mean over trailing 60 trading days of ( close_i(d) × volume_i(d) )
```
i.e. how many dollars of that stock change hands on a typical day.

**The construction:**
1. Each month-end, compute ADV for every listed US equity using only data up to that date.
2. Rank descending; take the top 1000.
3. That set is the universe for the *following* month.
4. Store as a boolean matrix, dates × tickers.
5. **Critically: include names that later delisted.** If a company was in the top 1000 in 2011 and went bankrupt in 2013, it belongs in your 2011–2013 sample.

**Why ADV rather than market cap:** market cap includes large but thinly-traded names you couldn't actually trade. ADV selects on tradability, which is what a stat-arb universe is for. Trexquant's own `liqn` metric is defined in volume terms ("booksize at which the alpha's avg daily stock volume = 1% of market volume"), which points the same way.

> ⚠️ **Assumption to verify:** I don't know for certain how Trexquant defines *their* top1000 — it could be market cap, or a proprietary liquidity screen. Check the UI or ask. If they use mktcap and you use ADV, your local universe drifts from theirs and alphas transfer worse.

### Why survivorship bias matters *specifically for you*

You'll be searching mainly **reversal** alphas — that's what the tutorial's baselines are (`-ret1`, `-ret5`, `-ret20`).

If your universe is built from names that exist *today*, then every stock in your sample survived. A stock that crashed 60% and recovered is in there; a stock that crashed 60% and delisted is not. So "buy the losers" looks systematically better than it was. **Survivorship bias inflates precisely the alpha family you're hunting**, and it inflates the loser leg — the one doing the work.

`nec_baseline/nec_moe/market_data.py` already says this in its own docstring. It has point-in-time S&P 500 membership from Wikipedia with the changes table parsed, which handles index-membership survivorship — but yfinance still won't serve you price history for fully delisted tickers.

**My recommendation:**

- **Now (day 1):** reuse `nec_baseline`'s point-in-time S&P 500 universe. It works, it's cached, membership changes are handled. Accept that it's ~500–900 large caps rather than 1000 mid-to-large.
- **Then (day 3–5), if you get a survivorship-free source:** rebuild as true top-1000-by-ADV including delisted names.

Do **not** use "top 1000 current listings." It's the worst of both — the effort of a big universe with bias baked into exactly the signals you care about.

---

## 3. Timeline — download wide, train narrow

**Download: 2004-01-01 → present. Subset later.**

Three reasons:
1. **The download is the irreversible step; the training window is a free parameter.** Changing your training window later costs one line of code. Re-downloading costs hours. Never download less than you might plausibly want.
2. **2004, not 2006** — you need ~2 years of burn-in so a 252-day lookback feature is fully populated on 2006-01-01, matching the tutorial's window with no NaN edge.
3. **The extra years are ~40 MB/field.** Irrelevant.

**Train on a three-way split:**

| Split | Window | Use |
|---|---|---|
| **Train** | 2006–2015 | GP searches here. Only here. |
| **Validation** | 2016–2020 | Screen GP output. Iterate GP *hyperparameters* here. Anything that dies here is discarded. |
| **Local test** | 2021–present | Touch **once**, at the very end, on your final shortlist. |
| **Client OS** | hidden | The real test. |

**Why 2016 is the right train/validation boundary:** the tutorial says its own volume-premium alpha *"deteriorates from 2016, loses money in 2017."* That's Trexquant telling you where they see a regime break. Putting your boundary there means validation specifically tests whether an alpha survives that break — which is what the OS window will test too.

### The trap: don't use the client as your search loop

Submissions appear to be unlimited, and the score rewards alpha count. So it's tempting to submit → check IR → tweak → resubmit.

**That destroys the out-of-sample-ness of the OS window.** Every time you look at an OS result and change something in response, you've fitted to it. Do that fifty times and your OS score is in-sample.

The discipline:
- **Local validation decides what gets submitted.** The client just records the outcome.
- Submit *many* alphas — that's correct, alpha count is half the score.
- Do **not** submit-check-tweak-resubmit the *same* alpha.
- Run the correlation pruning **locally, before submitting**, so you don't burn the pool on clones.

---

## 4. On the PCA step

You proposed PCA to check correlation between variables. Worth separating three things:

1. **PCA on the input variables** (open, close, high, low, volume) — **skip.** They're trivially collinear; open/high/low/close correlate above 0.99 by construction. PC1 will be "price level" and you'll learn nothing.
2. **PCA on the factor matrix** — useful for one specific question: *how many independent bets do I actually have?* Count eigenvalues above 1, or the number of PCs needed for 90% of variance. If you have 40 factors and 4 meaningful PCs, you have 4 ideas.
3. **Greedy correlation filter — this is the operational tool.** The platform's rule is literally "≤50% correlation to anything already in your pool." So mirror it: sort candidates by IR/√TVR descending, walk the list, accept a candidate only if its max |corr| to everything already accepted is ≤0.5. That *is* the qualification rule, run locally and for free.

You already have the machinery — `factor_corr_matrix.csv`, `factor_corr_cluster_order.csv`, `factor_corr_near_duplicates.csv`.

> **One detail that matters:** correlate the **alpha PnL series** (or daily position vectors), not the raw factor values. That's what Trexsim's correlation tool compares. Two factors with different values can produce near-identical positions after ranking, and vice versa.

---

## 5. Proposed sequence

| Phase | Days | What |
|---|---|---|
| **0. Decide source** | 0.5 | Check WRDS access. Price Sharadar. Start yfinance regardless. |
| **1. Download** | 1 | ~150 MB, 2004→present, OHLCV + shares + sector + universe mask. Wide DataFrames, dates × tickers, parquet — same shape as your QIP pickles so the GP code ports directly. |
| **2. Port the GP** | 2 | Swap CN pickles for US parquet. Rewrite `calc_fitness` to mirror Trexsim (industry-neutral, rank-weighted, dollar-neutral, IR/√TVR). |
| **3. Fix the terminals** | 1 | Replace raw levels with dimensionless ratios. This alone should stop the clone problem. |
| **4. GP search** | 3–4 | 10–20 seeds × varied terminal subsets, pop 500+, 30+ gens. Train window only. |
| **5. Validate + prune** | 2 | Screen on 2016–2020. Greedy ≤0.5 correlation filter. PCA to count real ideas. |
| **6. Translate to Trexsim** | 2 | Rewrite survivors in Trexsim operator syntax (`ts_`/`cs_`/`at_`). Verify fast vs slow mode agreement. |
| **7. Hand-craft intraday** | 2–3 | In parallel with 4–6. `GROUP_intra1` + `trading_days_until_next_earnings_announcement`. No download needed. |
| **8. Submit + final local test** | 2 | Shortlist → 2021+ local test once → submit. |

**Slack: ~4 days.** Phase 7 is the one to protect — it's the least-crowded data and needs no download, so it's the best expected-value-per-hour in the whole plan.

---

## 6. Open questions before download

1. **WRDS access through NEC?** — determines everything else.
2. **How does Trexquant define top1000** — ADV, market cap, or proprietary? Check the UI.
3. **What is the competition's actual date range?** The tutorial shows 2006-03-31 → 2021-12-30, but the FRED variables' bounds fingerprint 2022–mid-2023. If the OS window is 2022+, your validation split should shift later.
4. **Confirm the delay setting** — tutorial example shows `d0`. That changes whether you shift by 0 or 1 day in local fitness.
