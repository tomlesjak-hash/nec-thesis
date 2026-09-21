# Universe, Timeline & Pipeline — recommendation

**Decide these four, then we commence.** Code is written and waiting in `pipeline/`.

---

## 1. Is the NEC download protocol good?

**Architecturally yes, operationally no.** Keep the design, fix four things — which the copies already do.

### What's genuinely good (kept)
- **Cache-first, source-agnostic.** The CSV *is* the interface, so the feature layer never knows where bars came from. Clean.
- **Skip-and-report failures.** Individual ticker failures are collected, not fatal.
- **Point-in-time universe from Wikipedia** with the constituent-change table parsed — this is the valuable part and most people don't bother.
- **Honest bias documentation** in the docstrings. `universe_coverage_report` quantifies residual bias instead of hiding it.

### What breaks for us (fixed in the copies)

| # | Problem in `nec_baseline` | Why it matters here | Fix in `tq_data.py` |
|---|---|---|---|
| 1 | **One `yf.download()` call per ticker** | ~900 sequential round-trips. Hours, not minutes. | `download_batch()` — chunks of 100, `threads=True`. |
| 2 | **Cache key = `{sym}.us.{start}.{end}.csv`** | Changing the window invalidates all 592 cached files. Your existing cache is keyed `2015-01-01.2024-12-31`, so a 2010→2026 run would re-download everything. | Cache key is the **symbol alone**; full history stored, sliced at load. `migrate_nec_cache()` seeds it from your 592 files — **not re-downloaded**. |
| 3 | **`auto_adjust=True`, keeps only OHLCV** | Loses the unadjusted close, so the split/dividend factor is unrecoverable — and the tutorial names retro-adjustment as a real fast-vs-slow divergence source. | `auto_adjust=False`; stores `close` (= Adj Close, matching Trexsim's `CAX_ADJ_DIV_D1`) **and** `close_raw`. |
| 4 | **No shares / market cap / sector** | Can't build a size factor. | `download_shares()` + `add_marketcap()`. |
| 5 | Stooq default is dead (JS anti-bot); returns float64 per-ticker frames | Slow, wrong default | yfinance default; wide float32 parquet panel. |
| 6 | `universe.py` imports `torch` | Unnecessary heavy dep | Copies are pure pandas. |

**Nothing under `nec_baseline/` is modified.** The copies live in `Trexquant/pipeline/` and only *read* the old cache directory.

---

## 2. Market cap — you asked to download it, so here's the honest situation

I tried to do this properly and hit a wall worth knowing about.

**Tested live:** FMP's `historical-market-cap` returns only ~63 rows regardless of `limit=5000` or `page=4` — roughly **3 months of history** on your free tier. `directory` and `indexes` are paywalled too. So FMP gives you prices (20+ years, correct adjustment) but **not** market cap history.

**What I built instead:** `download_shares()` pulls real historical share counts via yfinance's `get_shares_full()`, and `add_marketcap()` computes `mktcap = close_raw × shares`. That's a genuine download of the share count, not a fabrication — and it correctly uses the *unadjusted* close, since market cap is a real dollar quantity.

**The caveat you need to hear:** Yahoo's share history is thin — typically the **last ~4 years**, and nothing for some tickers. There is no free source for 15 years of point-in-time share counts. CRSP/Compustat are the paid answer and you don't have them.

**So, practically:**
- **Track A (platform alphas):** use Trexsim's native `mktcap` freely — full history, already there, cov 3237. No problem at all.
- **Track B (local GP):** `mktcap` will be sparse before ~2022. Use **`adv60` rank as the size proxy** instead. Cross-sectionally, ADV rank and market-cap rank correlate strongly, and for a GP terminal you need *a* size dimension, not a precise one.

Run with `--no-shares` to skip the slow per-ticker loop if you'd rather not wait.

---

## 3. Timeline — the data decides this, not preference

`nec_baseline/nec_moe/universe.py` documents a measured constraint:

> `EARLIEST_RELIABLE = 2011-01-01` — the Wikipedia change table carries ~400 events, ~22/yr through the 2010s–2020s (matching real index turnover), but **only ~40 events for the entire 2000s** and ~8 for the 1990s.

So before 2011, "point-in-time membership" degenerates into *today's index backfilled* — maximum survivorship bias, precisely on the reversal alphas you're hunting.

**That kills my earlier 2004/2006 advice.** Downloading 2004–2010 prices is cheap, but *training* there is unsound with this universe source.

### ⚠️ REVISED 2026-08-08 — competition window confirmed as **2006 → 2022**

Tom confirmed the actual competition range. That changes the split, and it
creates a genuine tension worth stating plainly:

**The competition wants 2006–2022. The universe is only trustworthy from 2011.**
Five of the seventeen years (2006–2010) sit in the zone where Wikipedia's
change table is too sparse to reconstruct membership, so those years degrade
toward "today's S&P 500 backfilled" — maximum survivorship bias, on exactly
the reversal alphas you're hunting.

You can't fix that without a paid survivorship-free source. You can manage it.

| | Window | Days | Why |
|---|---|---|---|
| **Download** | 2006-01-01 → 2022-12-31 | ~4,280 | Exactly the competition window — no burn-in buffer (Tom's call) |
| **Optional early train** | 2006–2010 | ~1,260 | ⚠️ survivorship-biased universe. Usable as *extra* training data, never as validation, and discount any alpha that only works here |

> **Consequence of no burn-in:** a feature with an N-day lookback is NaN for the
> first N trading days of 2006. `ts_mean(x, 252)` isn't usable until ~2007-01.
> Harmless for short-window alphas (5–60 days fill within a quarter), and the
> tutorial's own run started **2006-03-31**, which suggests they consume a
> burn-in internally too. If you later want a long-lookback feature clean from
> day one, re-run with `--start 2005-01-01`.
| **Train** | 2011-01-01 → 2017-12-31 | ~1,760 | Reliable membership. Primary search window. |
| **Validate** | 2018-01-01 → 2020-12-31 | ~755 | COVID crash included — a brutal filter. |
| **Local test** | 2021-01-01 → 2022-12-31 | ~505 | Touch **once**. |

**Why test on 2021–2022:** the tutorial's own correlation panel ran
**2020-01-08 → 2021-12-30**, which suggests their evaluation window sits at the
end of the range. Testing on the last two years mirrors that.

> This also **retires the earlier FRED-bounds puzzle.** I'd flagged that the
> FRED variables fingerprint 2022–mid-2023 while the tutorial said 2006–2021.
> With the real window confirmed as 2006–2022, the FRED block is simply the
> tail end of it. No mystery, and no reason to plan for a 2023+ regime.

All windows are one-line changes: `--start` / `--end` on `run_download.py`.

---

## 4. Should you narrow to ~500? — **Yes, but your reasoning is backwards**

You said: narrow to ~500 "just to not make a fully generalized GP model."

**On overfitting, narrowing does the opposite of what you expect.** Effective sample size for a cross-sectional alpha is `n_days × n_stocks`. Halving the cross-section halves your observations and makes each daily cross-sectional rank noisier. A GP with fewer observations overfits **more**, not less. If overfitting were the only concern, you'd want the *widest* universe you could get.

**But your instinct is right for a different reason: liquidity homogeneity.** A GP searching across a universe that mixes mega-caps with thinly-traded small-caps will find alphas driven by illiquidity — wide spreads, stale prices, microstructure noise. Those score beautifully in a frictionless backtest and die instantly in live trading. Trexsim's `top1000` is a *liquid* universe, and its `liqn` metric explicitly penalises alphas that need more than 1% of daily volume. Restricting to liquid names makes discovered alphas more transferable. **That's the real argument for narrowing, and it's a good one.**

**And the decision is nearly moot anyway.** S&P 500 point-in-time membership gives you ~500 names on any given date by construction, and ~800–900 unique tickers across 2011–2026 as names enter and leave. So `--top-n 500` is close to a no-op on top of the membership filter; it mainly earns its keep if you later widen the ticker pool.

### Recommendation: **500/day. Default is already set.**

Accept the one real mismatch: S&P 500 is large-cap only, while Trexsim's top1000 reaches into mid-cap, where reversal and volume-attention effects are documented as **stronger**. So expect your GP alphas to test *better* on the platform than locally, not worse — an error in the safe direction. You said yourself the alpha should still relate well to the client universe, and I agree that's the right call at 7 days.

---

## 5. What the pipeline produces

```
Trexquant/
  pipeline/
    tq_data.py        universe + batched download + shares + wide panel
    tq_calendar.py    17 Trexsim seasonality vars, zero download
    run_download.py   entry point
  cache/              per-symbol CSVs (seeded from nec_baseline, not re-fetched)
  panel/*.parquet     wide dates x tickers, float32
```

**Panel fields (~30, all Trexsim-mappable):**

| | |
|---|---|
| Price/volume | `open` `high` `low` `close` `volume` `close_raw` |
| Derived | `ret1` `ret5` `ret10` `ret20` `dollar_volume` `adv60` |
| Size | `shsout` `mktcap` *(sparse pre-2022 — see §2)* |
| Context | `close_spx` `close_vix` `close_gld` `ret1_spx` `ret1_vix` `ret1_gld` |
| Seasonality | 17 × `cal_*` (calendar + FOMC) |
| Universe | `universe` boolean mask |

**To run:**
```bash
cd "~/Desktop/Quant Model/Trexquant/pipeline"
pip install yfinance pandas pyarrow lxml
python run_download.py                 # defaults: 2010→now, top-500
python run_download.py --no-shares     # skip the slow shares loop
```

Expect ~10–20 min with the migrated cache, longer with `--shares`. Output ~250 MB.

---

## 6. Updated 7-day plan

Unchanged in structure from `04_SEVEN_DAY_PLAN.md` — **Track A (hand-crafted platform alphas) is still the priority**, because it has zero dependencies and guarantees a non-zero score. This pipeline serves Track B, the upside play.

| Day | Track A ⭐ | Track B |
|---|---|---|
| **1** | Verify UI settings. Alphas 1–6 (intraday group). | `python run_download.py` in background. |
| **2** | Alphas 7–13. Correlation-check. | Port `genetic_factor.py` data layer to the parquet panel. |
| **3** | Alphas 14–20. | **Fix terminals** (dimensionless only) + fitness → IR/√TVR, industry-neutral. |
| **4** | Re-gate anything failing the 0.5 check. | GP search, 10–20 seeds. |
| **5** | — | Validate 2019–2022; greedy ≤0.5 correlation prune. |
| **6** | — | Translate survivors to Trexsim syntax; fast-vs-slow check. |
| **7** | Final correlation pass on the combined pool. Submit. | Buffer. |

**If Track B slips past day 5, abandon it.** Twenty hand-crafted alphas beat five GP alphas that arrive too late to correlation-check.

---

## Decisions needed

1. **Timeline** — download 2010→now, train 2011–2018 / validate 2019–2022 / test 2023+? *(recommended)*
2. **Universe size** — 500/day? *(recommended — and roughly what S&P 500 PIT gives anyway)*
3. **Shares download** — run it (slow, sparse pre-2022) or `--no-shares` and use `adv60` as the size proxy?
4. **Priority** — confirm Track A first, or do you want Track B started immediately?
