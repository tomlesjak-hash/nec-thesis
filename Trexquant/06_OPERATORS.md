# Trexsim Builtin Operators — all 32

Saved 2026-08-07. All operate on `ndarray, shape (n, d)`.

## ⚠️ Two conventions that will bite you

**1. Arrays are STOCKS × DAYS — transposed from your QIP pandas panel.**
`cs_rank` documents `A[:,day] = rankdata(A[:,day])`, so a *column* is one day
across all stocks. Your QIP DataFrames are dates × tickers. Operators hide this,
but any hand-written `np.*` with an `axis=` argument needs the axis flipped.

**2. `cs_rank` and `cs_scale` return [1, 2], NOT [0, 1].**
This silently wrecks any formula that multiplies two ranks. `rank_a * rank_b`
in [0,1]² spans [0,1]; in [1,2]² it spans [1,4] — dominated by a constant
offset, so the signal is swamped. **Subtract 1 to restore [0,1] semantics**
whenever you port a formula written against pct-ranks.

---

## Element-wise — `at_`

| Operator | Does | Notes |
|---|---|---|
| `at_nan2zero(A)` | `np.nan_to_num(a)` — NaN and ±inf → 0 | Use when **adding alphas** so one NaN doesn't null the sum. Also for slowing down sparse variables. |
| `at_zero2nan(A)` | `a[a==0] = np.nan` | Guard denominators; lets `nanmean` skip zeros. The idiomatic divide-by-zero fix. |
| `at_signlog(A)` | `sign(a) * log1p(abs(a))` | Zero maps to zero. For logging data with both signs. **Not** plain `log`. |
| `at_signsqrt(A)` | `sign(a) * sqrt(abs(a)+1)` | Tames heavy tails while keeping direction. |

## Cross-sectional — `cs_` (per day, across stocks)

| Operator | Does | Notes |
|---|---|---|
| `cs_rank(A)` | Dense rank, **scaled to [1, 2]** | `cs_rank(ret1) > 1.9` selects the top 10%. ⚠️ not [0,1]. |
| `cs_zscore(A)` | `(a - nanmean) / nanstd` | Best default for combining variables of different units. |
| `cs_scale(A)` | Min-max to **[1, 2]** | Preserves relative distances, unlike `cs_rank`. |
| `cs_winsor(A, filter_percentile, remove_extreme)` | Clip extremes to the percentile value | ⚠️ **one** percentile, not a (lo, hi) pair. `remove_extreme=True` zeroes them instead. |
| `cs_remove_middle(A, filter_percentile)` | Keep only the tails | `0.80` removes the middle 80%, keeps 10% per tail. Higher return, higher turnover and vol. |
| `cs_indneut(input, grouping)` | Subtract the group mean each day | `cs_indneut(-ret1, industry)`. The sim's `Neut=industry` setting already does this — only needed for a *different* grouping, e.g. `subindustry`. |
| `cs_factorneut(input, [f1, f2], int_factors=[i1])` | Orthogonalize against style factors | Example cites `barra_size, barra_momentum, barra_beta, barra_value` — **these weren't in the pasted data description; confirm they exist before relying on them.** |
| `cs_booksize(input, booksize)` | Scale so long and short books each sum to `booksize` | If the input is all one sign, one book ends up empty. |
| `cs_shrink_balance(input)` | Shrink the larger book to match the smaller | Use instead of `cs_booksize` when you must not violate earlier constraints (volume limits, locates). |
| `cs_poslimit(input, pos_limit)` | Cap any single stock's share of the book | `cs_poslimit(-ret1, 0.01)` = max 1% per name. |
| `cs_harmonic_mean(A)` | Daily harmonic mean, broadcast to every stock | Constant per day → **neutralizes to zero alone**. Only useful differenced, e.g. `cs_harmonic_mean(x) - ts_harmonic_mean(x, 10)`. |

## Time-series — `ts_` (per stock, along days)

| Operator | Does |
|---|---|
| `ts_delay(A, days)` | Value from `days` ago. First `days` columns → NaN. |
| `ts_diff(A, n)` | n-th discrete difference along time. |
| `ts_fill(A)` | Forward-fill. |
| `ts_mean(A, days)` | Rolling mean. |
| `ts_mean_exp(A, days, exp_factor)` | Exponentially weighted mean. **The turnover-control tool** — "hold with exponentially decaying size". |
| `ts_median(A, days)` | Rolling nanmedian. |
| `ts_harmonic_mean(A, days)` | Rolling harmonic mean. |
| `ts_sum(A, days)` | Rolling cumulative sum. |
| `ts_max(A, days)` / `ts_min(A, days)` | Rolling max / nanmin. |
| `ts_std(A, days)` | Rolling standard deviation of finite values. |
| `ts_skew(A, days)` | Rolling skewness of finite values. |
| `ts_rank(A, days)` | Rank of today's value within the past `days`. |
| `ts_norm(A, days)` | Center on the trailing mean. |
| `ts_zscore(A, days)` | Center **and** standardize on the trailing window. |
| `ts_std_normalized(A, days)` | Standardize today's value. |
| `ts_corr_binary(A, B, days)` | Per-stock rolling correlation between two variables. Docs note it gives **beta to the market** via `close_SPX` — the data description spells it `close_spx`; check the casing the sim accepts. |

---

## Porting cheat-sheet — QIP → Trexsim

| Your QIP code | Trexsim | Watch out |
|---|---|---|
| `df.rank(axis=1, pct=True)` → [0,1] | `cs_rank(A) - 1` | **Subtract 1.** Raw `cs_rank` is [1,2]. |
| `winsorize(x, 0.01, 0.99)` | `cs_winsor(x, 0.01, remove_extreme=False)` | One percentile, not two. Verify the convention on a small case. |
| `df.where(cond, np.nan)` | `at_zero2nan(x)` or `np.where(...)` | These are **ndarrays**, not DataFrames — no `.where` method. |
| `df.shift(n)` | `ts_delay(A, n)` | |
| `df.diff(n)` | `ts_diff(A, n)` | |
| `df.rolling(n).mean()` | `ts_mean(A, n)` | |
| `df.rolling(n).apply(rank)` | `ts_rank(A, n)` | |
| `np.log(a)` | `np.log(a)` fine for positive data | `at_signlog` is `sign*log1p(abs)`, **not** plain log. |
| `self.factor_data = X` | `alpha = X` | |
| industry neutralization | the sim's `Neut=industry` setting | Don't double-neutralize unless using a different grouping. |

**Combining alphas** (from the tutorial):
```python
alpha = at_zero2nan(at_nan2zero(alpha1) + at_nan2zero(alpha2))
```
But inside a *single* alpha, prefer natural NaN propagation — it correctly
drops stocks with no data. Only `at_nan2zero` a secondary *tilt* term so it
degrades to zero tilt rather than nulling the whole signal.
