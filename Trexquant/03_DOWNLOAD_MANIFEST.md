# Download Manifest — mapped to Trexsim variable names

**Status:** DRAFT — awaiting confirmation
**Governing rule:** the GP may only search over variables that **exist in Trexsim**. An alpha built on a field the platform doesn't have is unsubmittable. Every terminal must be either a Trexsim variable or exactly derivable from one.

---

## 1. Your plan's list → actual Trexsim varnames

| My plan called it | Trexsim varname | Group | D0 | Cov | Adj | Verdict |
|---|---|---|---|---|---|---|
| open | **`open`** | backoffice | Yes | 3251 | DIV | **Download** |
| high | **`high`** | backoffice | Yes | 3251 | DIV | **Download** |
| low | **`low`** | backoffice | Yes | 3251 | DIV | **Download** |
| close | **`close`** | backoffice | Yes | 3251 | DIV | **Download** |
| volume | **`volume`** | backoffice | Yes | 3251 | MUL | **Download** |
| shares_outstanding | **`shsout`** | backoffice | **No** | 3237 | MUL | **Download** (millions) |
| sector / industry code | **`sector`**, **`industry`**, **`subindustry`** | backoffice | **No** | 3237 | — | **Download all three** |
| close_unadj / adj_factor | ⚠️ **no Trexsim equivalent** | — | — | — | — | Download for *diagnostics only* — never as an alpha terminal |
| universe membership mask | ⚠️ **not a variable** — it's the `top1000` sim setting | — | — | — | — | Build locally |

**Correction to my earlier note:** I said derive market cap from close × shares. Trexsim has it natively as **`mktcap`** (cov 3237, **D0 = No**). Deriving it from adjusted close × adjusted shsout works — the DIV and MUL adjustment factors cancel — but download the vendor's version too as a cross-check.

---

## 2. What to ADD — all cheap, all daily, all non-intraday

Since storage isn't the constraint, these are all worth having.

### 2a. Download (vendor)

| Trexsim varname | Group | D0 | Cov | Freq | Why add it |
|---|---|---|---|---|---|
| **`mktcap`** | backoffice | No | 3237 | 1.03 | Size factor + universal normaliser. Native field, don't just derive. |
| **`shsfloat`** | backoffice | No | 3011 | **15.6** | Float shares (millions). Updates **4× faster** than `shsout` (15.6d vs 59d). `shsout − shsfloat` = restricted/insider holdings — a genuinely distinct signal. ⚠️ *Historical float time series is hard to source; yfinance gives current-only. Get it if your vendor has it, skip if not.* |
| **`cashdvd`** | backoffice | No | 1619 | 342 | Cash dividend, 0 when none. Easy from any vendor. Enables dividend yield + ex-div effects. |
| **`close_spx`**, **`ret1_spx`** | daily_idx | Yes | 3251 | 1.0 | Ticker `^GSPC`. **One ticker.** |
| **`close_vix`**, **`ret1_vix`** | daily_idx | Yes | 3251 | 1.0 | Ticker `^VIX`. **One ticker.** |
| **`ret1_gld`** | daily_etf_returns | Yes | 3791 | 1.0 | Ticker `GLD`. **One ticker.** |

> The three index/ETF tickers cost essentially nothing and unlock rolling-beta terminals — `ts_corr(ret1, ret1_spx, 60)`, `ts_corr(ret1, ret1_vix, 60)`. Those **are** cross-sectional signals even though the index itself is 1D. High value per byte.

### 2b. Compute locally — zero download

| Trexsim varnames | How | Effort |
|---|---|---|
| `ret1`, `ret5`, `ret10`, `ret20` | from `close` | trivial |
| all **15** `GROUP_days_to_calendar_events` vars | `exchange_calendars` / `pandas_market_calendars` US trading calendar. Monthly OpEx = 3rd Friday; quarterly OpEx = 3rd Friday of Mar/Jun/Sep/Dec. | ~1 hour |
| `days_since_last_fomc_meeting`, `days_until_next_fomc_meeting` | Published FOMC calendar — hardcode the ~8 dates/year since 2004. | ~30 min |
| all **11** `GROUP_fred` macro vars | FRED API (free, `fredapi`). CPI, GDP, unemployment, WTI. | ~30 min |

That's **28 Trexsim variables for zero download cost.** Do all of them.

### 2c. The one hard download worth attempting

| Trexsim varname | Cov | Freq | Verdict |
|---|---|---|---|
| **`trading_days_until_next_earnings_announcement`** | 7687 | 2.0 | ⭐ **Highest-value conditioner in the dataset.** Needs historical earnings announcement dates. Sharadar has an actions/events table; Polygon has earnings dates; yfinance gives only recent ones. **Try to get it. If you can't within a day, skip and use it directly in Trexsim instead** — see §3. |

---

## 3. What NOT to download — and the principle behind it

**Principle: download locally only what the GP genuinely needs to search. Anything hard to source *correctly* is already correct on the platform — hand-craft those alphas there instead.**

| Skipped | Count | Why |
|---|---|---|
| `GROUP_intra1` | 30 | Your call, and correct. Acquiring it means pulling minute bars (~2bn rows). **Already pre-computed in Trexsim.** |
| `GROUP_edgar8kfs` + 3 ratio groups | ~175 | The point-in-time trap: vendors serve *restated* fundamentals, so a 2011 backtest sees numbers published in 2014 → look-ahead bias that silently inflates every quality/value factor. Getting true PIT means Compustat PIT or Sharadar SF1. **In Trexsim these are already point-in-time by construction** — they're 8-K filings with Delay0 = Yes, i.e. available on the filing date. Fighting to replicate that locally, in 21 days, is a bad trade. |

**This is the same logic as intraday, applied twice.** Fundamentals and intraday together are **205 of the ~255 variables** — and you get both, correctly, by hand-crafting on the platform rather than downloading. Your local GP handles the daily price/volume/calendar space, which is exactly where GP search works well anyway (dense, daily-updating, dimensionless).

---

## 4. ⚠️ What Trexsim does NOT have — keep these out of the GP

Your QIP China dataset has fields with **no Trexsim equivalent**. If a terminal isn't on the platform, any alpha using it is dead on arrival.

| In your QIP data | Trexsim? | Note |
|---|---|---|
| `vwap` (daily) | ❌ **Absent** | Only `vwap_first_hour` / `vwap_last_hour` exist, both intraday. **There is no daily VWAP.** Your CN pipeline has one — do not carry it over. |
| `amount` (dollar volume) | ⚠️ derive | Not a field, but `close × volume` is exact. Safe. |
| `turnover_amount`, `turnover_volume` | ⚠️ derive | `volume / shsout`. Safe. |
| `pct_change`, `return`, `daily_return` | ✅ | = `ret1`. |
| `float_market_cap` | ⚠️ derive | `close × shsfloat`. |
| `hs300_beta`, `cs500_alpha`, etc. | ⚠️ derive | Compute betas vs `ret1_spx` yourself. |
| `days_since_ipo` | ❌ **Absent** | No equivalent. Drop. |
| `tradable` (suspension flag) | ❌ **Absent** | US names rarely halt; ignore. |
| `bonus_ratio`, `rights_issue_ratio` | ❌ **Absent** | China-specific corporate actions. Drop. |
| `industry_pct_change` | ⚠️ derive | Group-mean of `ret1` by `industry`. |

**Also absent entirely from Trexsim** (don't plan alphas around them): bid-ask spread, short interest, options volume / put-call ratio, analyst estimates, news sentiment, institutional holdings.

---

## 5. ⚠️ Delay-0 discipline — this will silently corrupt your local fitness

Trexsim marks each variable as usable at delay 0 or not. Your local backtest must match, or your IR won't transfer.

| Usable at **delay 0** | Must be **lagged ≥1 day** |
|---|---|
| `open`, `high`, `low`, `close`, `volume` | `mktcap`, `shsout`, `shsfloat`, `cashdvd` |
| `ret1`, `ret5`, `ret10`, `ret20` | `industry`, `sector`, `subindustry` |
| `close_spx`, `close_vix`, `ret1_spx`, `ret1_vix`, `ret1_gld` | all 15 calendar vars, both FOMC vars |
| `trading_days_until_next_earnings_announcement` | all `GROUP_fred` macro |
| all fundamentals (`GROUP_edgar8kfs`) | all `*_ratio_derived_d1` valuation ratios |

**The trap:** if you derive `mktcap = close × shsout` and use it same-day, you've given yourself delay-0 access to a field Trexsim marks delay-0 = No. Local IR looks fine, client IR is worse, and you won't know why. **Lag the whole right-hand column by one day in your local fitness function.**

Your QIP `calc_ir` uses `lag_periods = 2` uniformly. That's not what Trexsim does — it's per-variable. Fix this when you port.

---

## 6. Revised storage math (correcting my earlier estimate)

I previously said ~150 MB assuming exactly 1000 columns. **That was wrong** — a point-in-time top-1000 universe over 20 years touches far more than 1000 unique tickers, because names enter and leave. Realistically **3,000–4,500 distinct tickers**.

```
5,040 trading days × ~4,000 tickers × 4 bytes (float32) ≈ 80 MB per field
```

| | Fields | Raw float32 | Parquet (sparse, compresses well) |
|---|---|---|---|
| Price/volume | 5 | 400 MB | ~120 MB |
| Size/shares | 3 | 240 MB | ~50 MB |
| Classification | 3 | 240 MB | ~10 MB (int, highly repetitive) |
| Dividends | 1 | 80 MB | ~5 MB |
| Index/ETF | 3 tickers | <1 MB | <1 MB |
| Computed (calendar, FOMC, FRED, returns) | 28 | — | ~50 MB |
| **Total** | | **~1 GB** | **~250 MB** |

Still trivial next to your 17 GB QIP folder — but it's ~1 GB, not 150 MB. Store as parquet with float32.

---

## 7. Final manifest

**Download from vendor (12 fields + 3 tickers):**
```
open  high  low  close  volume            ← D0 = Yes
mktcap  shsout  shsfloat*  cashdvd        ← D0 = No, lag 1
industry  sector  subindustry             ← D0 = No, lag 1
^GSPC  ^VIX  GLD                          ← 3 tickers
close_unadj (or adj_factor)               ← diagnostics only, never a terminal
                    * skip if vendor lacks history
```

**Attempt (1 field):**
```
trading_days_until_next_earnings_announcement   ← try; fall back to platform-only
```

**Compute locally (28 vars, no download):**
```
ret1 ret5 ret10 ret20
15 × calendar vars · 2 × FOMC vars · 11 × FRED macro
```

**Deliberately skipped (205 vars) — hand-craft directly in Trexsim:**
```
30 × GROUP_intra1          (pre-computed on platform)
175 × fundamentals          (already point-in-time on platform)
```

**Window:** 2004-01-01 → present (2-yr burn-in before a 2006 train start).
**Format:** wide DataFrames, dates × tickers, parquet, float32 — same shape as your QIP pickles so `genetic_factor.py` ports with minimal change.
