# alpha101_057 / alpha101_019 improved → Trexsim

---

# alpha101_019_improved

**The cleaner of the two — translate this one first.**

```python
# ---- part A: 7-day return, self-normalised by its own volatility ----------
lr   = np.log(1.0 + ret1)                    # log return; ret1 >= -0.93 so safe
r7   = ts_sum(lr, 7)
vol7 = ts_std(lr, 7)
part_a = -np.tanh(r7 / at_zero2nan(vol7))

# ---- part B: long-horizon momentum as a cross-sectional rank -------------
r250   = ts_sum(lr, 250)
part_b = cs_rank(cs_winsor(r250, 0.01, remove_extreme=False))

alpha = part_a * part_b
```

### Three things that would go wrong if translated literally

**`1 + cs_rank(...)` becomes just `cs_rank(...)`.** Your QIP `cs_rank` is
`df.rank(pct=True)` → **[0,1]**, so `1 + cs_rank` spans **[1,2]**. Trexsim's
`cs_rank` *already* returns **[1,2]**. Writing `1 + cs_rank(...)` on the
platform gives [2,3] — same width, wrong offset, and since it multiplies
`part_a` the whole alpha is distorted. **`1 + rank_pct` and Trexsim `cs_rank`
are the same object.** This is the one error that silently changes the alpha
rather than erroring.

**`np.tanh` has no built-in equivalent** — Trexsim's element-wise set is only
`at_nan2zero`, `at_zero2nan`, `at_signlog`, `at_signsqrt`. But alphas are plain
Python with numpy (the tutorial's own example is `(open - close) / open`), so
`np.tanh` works directly. If the editor ever rejects it, `at_signlog` is the
nearest built-in — sign-preserving and compressive, though log-shaped and
unbounded rather than squashed to ±1:

```python
part_a = -at_signlog(r7 / at_zero2nan(vol7))
```

**`vol7.where(vol7 > 0)` becomes `at_zero2nan(vol7)`.** These are ndarrays, not
DataFrames — no `.where` method. The `+ 1e-8` is then unnecessary and better
removed: `at_zero2nan` yields NaN (no position) where the fudge factor would
have produced an enormous ratio.

⚠️ **`ts_sum(lr, 250)` needs a full year of history.** It is NaN for the first
250 trading days, so nothing trades until ~2007 in a 2006-start simulation, and
slow mode needs a lookback well above 250. If IR looks weak, try `ts_sum(lr, 120)`
— roughly the same signal at half the warm-up cost.

---

# alpha101_057_improved

Harder: it needs **two** operators Trexsim doesn't have.

```python
# ---- VWAP: no daily VWAP exists, so build one from the two hourly bars ----
vnum = vwap_first_hour * volume_first_hour + vwap_last_hour * volume_last_hour
vden = volume_first_hour + volume_last_hour
vwap = vnum / at_zero2nan(vden)

rel_dev = (close - vwap) / at_zero2nan(vwap)

# ---- peak-recency proxy, replacing ts_argmax -----------------------------
peak = ts_rank(close, 30)
prank = cs_rank(peak) - 1.0                  # [1,2] -> [0,1], matching QIP

# ---- wma with weights [1, 2] over 2 days, written out --------------------
denom = (2.0 * prank + ts_delay(prank, 1)) / 3.0

rel_w   = cs_winsor(rel_dev, 0.01, remove_extreme=False)
denom_w = cs_winsor(denom,   0.01, remove_extreme=False)

alpha = -1.0 * (rel_w / at_zero2nan(denom_w))
```

### The two missing operators

**`amount / volume` — Trexsim has no daily `amount` and no daily VWAP.** Only
`vwap_first_hour` and `vwap_last_hour` exist. The version above volume-weights
those two into a real, if partial, VWAP — genuine traded-price data covering the
two most active hours of the session. That is much closer to your CN `amount /
volume` than any daily-bar reconstruction.

Simpler fallback if you would rather avoid intraday fields — the classic typical
price:

```python
vwap = (high + low + close) / 3.0
```

**`ts_argmax` does not exist.** `ts_argmax(close, 30)` measures *how many days
ago* the 30-day high occurred. Nothing in Trexsim's 32 operators returns an
index. `ts_rank(close, 30)` is the closest available in spirit: it says where
today sits within the last 30 days, so a value near the top implies the peak is
recent. Related, not identical — a real substitution, not a faithful port.

An alternative worth simulating side by side:

```python
peak = close / at_zero2nan(ts_max(close, 30))   # how far below the 30d high
```

### Two smaller notes

**The `-1` after `cs_rank` matters here more than usual**, because `prank`
lands in the **denominator**. Left as [1,2] the divisor never approaches zero
and the alpha is tame; brought to [0,1] as QIP intended it explodes for stocks
whose peak-recency rank is lowest — and that tail *is* much of the signal.
`at_zero2nan` handles the exact-zero case; your `eps = 1e-6` guard has no direct
equivalent, which is fine since a near-zero divisor now yields a large finite
value rather than an error.

**`winsorize(x, 0.01, 0.99)` → `cs_winsor(x, 0.01, remove_extreme=False)`** —
one percentile, not a pair. Worth knowing that winsorising immediately before a
`cs_rank` is nearly a no-op (rank is invariant to monotone transforms), but here
`rel_dev` feeds an arithmetic division, so it genuinely matters.

---

## Suggested order

1. **019 first** — it's a faithful translation with one clean substitution, so
   if it underperforms that's information about the alpha, not the port.
2. **057 second**, with the intraday VWAP. If it fails, retry with
   `(high + low + close) / 3` to isolate whether the VWAP proxy is the problem.
3. If 057 still disappoints, the `ts_argmax` substitution is the most likely
   culprit — it's the least faithful part of the translation.

Both are **volume/price-microstructure** alphas, so check correlation against
each other and against anything from `11_TRACK_A_ALPHAS.md` that uses volume
(#4, #9) before submitting.
