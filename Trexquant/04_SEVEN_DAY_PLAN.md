# Revised Plan — 7 Days

**Supersedes the timeline in `02_DATA_PLAN.md`.** Deadline ≈ 2026-08-14.

---

## 1. Connector findings

| Source | Status |
|---|---|
| **bigdata.com addon** | ❌ **Unavailable.** Requires OAuth authorization that can't be completed in this session. Even if authorized, it's a *research* platform — news, filings, transcripts, tearsheets, events. It has no bulk historical OHLCV endpoint. Wrong tool for this job. |
| **FMP** (connected) | ⚠️ **Free tier.** Tested live: `chart / historical-price-eod-dividend-adjusted` ✅ **works, 20+ years, and returns `adjOpen/adjHigh/adjLow/adjClose/volume` — exactly Trexsim's `CAX_ADJ_DIV_D1` convention.** `company / delisted-companies` ✅ works. But `directory` (ticker lists) ❌ blocked, `indexes` (S&P constituents) ❌ blocked, `historical-market-cap` with date range ❌ blocked. **So: prices yes, universe no.** |
| **Alpha Vantage** (connected) | `TIME_SERIES_DAILY_ADJUSTED` exists, `outputsize=full` = 20+ yrs. Viable backup, but per-ticker and free tiers are heavily capped. |
| **My sandbox** | ❌ **Cannot reach Yahoo** (SSL cert verification fails) and `yfinance` isn't installed. **Bulk download must run on your machine.** |

### Verdict on source
**yfinance, on your machine, extending `nec_baseline`.** It's already wired, `yf.download()` batches hundreds of tickers per call, no API key, and you already have **592 tickers cached for 2015–2024**. Universe comes from `nec_baseline`'s Wikipedia point-in-time S&P 500 (already parsed, membership changes handled). Use the FMP connector for spot-checks and gap-filling, not bulk.

---

## 2. ⚠️ The strategy change 7 days forces

With 22 days, GP-first was reasonable. With 7, **it risks scoring zero** — five days of data plumbing and nothing submitted.

Run two tracks in parallel, and **flip the priority**:

### Track A — hand-crafted alphas in Trexsim ⭐ PRIORITY
- **Zero dependencies.** All 255 variables already there, point-in-time, correctly adjusted. Starts today.
- **Alpha count is half the score.** This alone can place well.
- ~15–30 min per alpha including simulation → **20 alphas ≈ 2 focused days**.
- Guarantees a non-zero score even if Track B never finishes.

### Track B — local GP
- Needs download → port → fix terminals → search → validate → translate. Realistically 4–5 days *if nothing breaks*.
- Your GP currently produces clones (see `02_DATA_PLAN.md` §0), so it needs real fixes before it produces anything submittable.
- **Treat as upside, not critical path.**

> If you only do one thing this week, do Track A.

---

## 3. Survivorship bias — downgraded, and here's why

I made this sound more dangerous than it is for *your* situation. **Trexsim is a free, unbiased validator.**

Local data is for **search**. The platform is for **truth**. A survivorship-biased local sample means some GP candidates die when you simulate them on the platform — mildly annoying, not fatal, because the platform check costs minutes.

So with 7 days: **don't spend a day building a clean universe.** Use what's cached, generate candidates fast, let Trexsim filter them. Just stay suspicious of GP-discovered *reversal* alphas specifically, since that's the family the bias inflates.

---

## 4. Track A — the alpha slate (start now)

Build across **distinct data groups** — different input data is the cheapest decorrelation, and the ≤50% gate is what you're beating.

| # | Family | Sketch | Why decorrelated |
|---|---|---|---|
| 1–4 | **Intraday reversal/momentum** | `-(close_price_first_hour - open)/open`; `(close - vwap_last_hour)/close`; overnight `= (open - ts_delay(close,1))/ts_delay(close,1)` | `GROUP_intra1` — untouched by daily-bar competitors |
| 5–6 | **Trade-size flow** | `volume_first_hour / number_of_trades_first_hour` vs its own 20d mean; same for last hour | Institutional-vs-retail proxy, not derivable from daily data |
| 7–8 | **Intra-hour info ratio** | `cs_rank(information_ratio_last_hour)`; first-vs-last hour spread | Bounded ±2.6, already normalized, almost certainly under-mined |
| 9–11 | **Earnings-window conditioning** | Any of the above × mask on `trading_days_until_next_earnings_announcement` ∈ [0,5] or [−5,0] | Multiplies your alpha count cheaply — a gated alpha is genuinely new |
| 12–13 | **Volume premium** (tutorial case study) | `ts_mean_exp(ts_norm(volume,20) * (cs_rank(ts_std(volume,10)) < 1.0), 3, 0.5)` and variants | Their own worked example — start from the v5 form, not v1 |
| 14–15 | **Calendar gates** | Reversal × OpEx window; × turn-of-month; × turn-of-quarter | Seasonality group, orthogonal by construction |
| 16–17 | **Index beta** | `ts_corr(ret1, ret1_spx, 60)`; `ts_corr(ret1, ret1_vix, 60)` | Macro group |
| 18–20 | **Skew / dispersion** | `cs_rank(skew_return_last_hour)`; `skew_volume_first_hour`; `high_PE − low_PE` | Higher-moment, distinct from level signals |

**Discipline:** run **Compute Correlation** on the platform before submitting each new one. Anything >0.5 against your existing pool, drop or re-gate it.

---

## 5. Track B — download spec (you run it)

**Universe:** `nec_baseline`'s point-in-time S&P 500 (Wikipedia, already parsed). ~600–900 names across the window.
**Window:** 2004-01-01 → present. You already have 2015–2024 for 592 tickers, so this is mostly a backfill.
**Fields (yfinance gives all in one call):** `Open, High, Low, Close, Adj Close, Volume`
**Plus 3 tickers:** `^GSPC`, `^VIX`, `GLD`
**Compute locally, no download:** `ret1/5/10/20`, 15 calendar vars, 2 FOMC vars, 11 FRED macro.
**Format:** wide DataFrames, dates × tickers, parquet, float32 — matches your QIP pickle shape so `genetic_factor.py` ports with minimal change.
**Size:** ~250 MB.

**Skip entirely:** intraday (your call, correct), all ~175 fundamentals (technical-only per your call — and right for GP anyway, since 50–100 day update frequencies make step functions the GP can't use well).

### The GP fixes that matter most (in priority order)
1. **Dimensionless terminals only.** Replace `closes/highs/lows/opens/amounts` with `close/ts_mean(close,20)`, `high/low`, `volume/ts_mean(volume,20)`, `ret1`, `ret5`, `(close−open)/open`, `cs_rank(mktcap)`. **This single change is what stops the clones.**
2. **Drop `vwap` from terminals** — Trexsim has no daily VWAP. Any alpha using it is unsubmittable.
3. **Fitness = IR/√TVR** on an industry-neutral, rank-weighted, dollar-neutral book — mirroring Trexsim, not quintile long/short at `lag_periods=2`.
4. **Diversity pressure:** 10–20 independent seeds × varied terminal subsets, then greedy ≤0.5 correlation prune. Cheaper to implement than NSGA-II and works.

---

## 6. Day-by-day

| Day | Track A (priority) | Track B |
|---|---|---|
| **1** (today) | Alphas 1–6. Verify universe/date-range/delay settings in the UI. | Kick off yfinance backfill in the background. |
| **2** | Alphas 7–13. Correlation-check the pool. | Build parquet panel; port GP data layer. |
| **3** | Alphas 14–20. | Fix terminals + fitness function. |
| **4** | Re-gate anything that failed the 0.5 check. | GP search, multi-seed. |
| **5** | — | Validate on 2016–2020, greedy correlation prune. |
| **6** | — | Translate survivors to Trexsim syntax; check fast-vs-slow agreement. |
| **7** | Final correlation pass across the *combined* pool. Submit. | Buffer. |

**If Track B slips past day 5, abandon it and put the hours into Track A.** Twenty hand-crafted alphas beats five GP alphas that arrived too late to correlation-check.

---

## 7. Verify in the UI today (5 minutes, affects everything)

1. Actual **date range** — tutorial says 2006-03-31→2021-12-30, but FRED bounds fingerprint 2022–23.
2. **Delay** setting — `d0` or `d1`? Changes your local lag.
3. How **top1000** is defined — ADV or market cap?
4. Current **turnover threshold** — the OS panel showed `tvr < 0.5`, tighter than the tutorial's own case-study alphas achieved.
