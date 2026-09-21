# Worked translation — `alpha101_038_improved` → Trexsim

## Paste this

```python
N = 10
volume_tilt_strength = 0.3

# --- base: recent-high close x strong intraday move, shorted ---------------
# cs_rank returns [1,2] in Trexsim, so subtract 1 to get pct-rank semantics.
ts_rank_close_cs = cs_rank(ts_rank(close, N)) - 1.0

intraday_ret     = np.log(close / open)
intraday_ret_w   = cs_winsor(intraday_ret, 0.01, remove_extreme=False)
intraday_rank    = cs_rank(intraday_ret_w) - 1.0

base_factor = -1.0 * ts_rank_close_cs * intraday_rank

# --- tilt: today's volume vs its own N-day average -------------------------
advN      = at_zero2nan(ts_mean(volume, N))   # guards divide-by-zero
rel_vol   = volume / advN
rel_vol_w = cs_winsor(rel_vol, 0.01, remove_extreme=False)
vol_rank  = cs_rank(rel_vol_w) - 1.0

# at_nan2zero on the TILT only: a missing volume degrades the tilt to zero
# instead of nulling the whole position for that stock.
alpha = base_factor + volume_tilt_strength * at_nan2zero(vol_rank)
```

---

## The three changes that matter

### 1. `cs_rank` returns [1, 2], not [0, 1] — **the one that would break it**

Your QIP `rank()` is `df.rank(axis=1, pct=True)` → **[0, 1]**.
Trexsim's `cs_rank` is dense rank **scaled to [1, 2]**.

Your formula multiplies two ranks. Watch what that does:

| | range of each rank | product | after `-1 *` |
|---|---|---|---|
| QIP (pct) | [0, 1] | [0, 1] | [-1, 0] |
| Trexsim raw | [1, 2] | **[1, 4]** | [-4, -1] |

In the Trexsim version the product is dominated by a constant offset — the
*variation* is a small ripple on top of a large constant. Trexsim subtracts the
median before sizing, so it wouldn't be a total loss, but the relative weighting
between the two ranks is destroyed and the volume tilt (0.3 × [1,2] = 0.3–0.6)
gets crushed against a base term spanning 3 units instead of 1.

**Subtracting 1.0 after every `cs_rank` restores your intended semantics.**
This applies to `cs_scale` too.

### 2. `.where()` doesn't exist — these are ndarrays

```python
advN_safe = advN.where(advN > 0, np.nan)     # pandas — fails in Trexsim
advN      = at_zero2nan(ts_mean(volume, N))  # Trexsim idiom
```

`at_zero2nan` is exactly the built-in for this, and volume can't be negative so
it's fully equivalent here.

### 3. `winsorize(x, 0.01, 0.99)` → `cs_winsor(x, 0.01, ...)`

Trexsim takes **one** percentile, not a (lo, hi) pair. From the docs,
`filter_percentile` is the level at which to clip both tails, so your
1st/99th becomes `0.01`.

⚠️ The docs are ambiguous — `cs_remove_middle(A, 0.80)` uses its parameter as
the *middle* fraction, so `cs_winsor` might too. **Verify on one simulation:**
run with `0.01` and `0.02` and check whether `ret` and `tvr` move materially.
If they're near-identical, see below for why it barely matters anyway.

---

## Something worth knowing: your winsorize steps are nearly no-ops

**Rank is invariant to any monotone transformation.** Winsorizing clips extreme
values *to* the threshold — which only changes the result by creating ties at
the boundary. Everything else keeps its ordering, so `cs_rank` returns almost
exactly the same numbers with or without it.

```
raw:        [-0.9, -0.02, 0.00, 0.01, 0.85]
winsorized: [-0.1, -0.02, 0.00, 0.01, 0.10]
cs_rank:     identical ordering -> near-identical ranks
```

The only real effect is compressing the ranks of tied extremes. So:

- If you keep `cs_winsor`, expect it to change very little.
- **Dropping both `cs_winsor` calls costs almost nothing and runs faster** —
  and the tutorial is explicit that runtime matters in Trexsim.
- Winsorizing genuinely matters when the value feeds an **arithmetic** step
  (a mean, a sum, a product of raw values). Before a rank, it's close to free.

Your `rel_vol = volume / advN` is the one place I'd keep it, since volume
spikes are extreme — but even there the following `cs_rank` absorbs most of it.

**Leaner variant, worth simulating side by side:**

```python
N = 10
ts_rank_close_cs = cs_rank(ts_rank(close, N)) - 1.0
intraday_rank    = cs_rank(np.log(close / open)) - 1.0
vol_rank         = cs_rank(volume / at_zero2nan(ts_mean(volume, N))) - 1.0
alpha = -1.0 * ts_rank_close_cs * intraday_rank + 0.3 * at_nan2zero(vol_rank)
```

If IR is within noise of the full version, keep this one — fewer moving parts,
faster, and easier to vary later.

---

## Free variants for your alpha count

The competition scores **alpha count** as half the rank, with a ≤50%
correlation gate. Each of these changes the *input data*, which is the cheapest
way to land under that gate:

| Variant | Change | Why it decorrelates |
|---|---|---|
| Different horizon | `N = 5` or `N = 20` | Partly correlated — check before submitting. |
| Intraday close | `np.log(close_price_last_hour / open_price_last_hour)` | Different data group (`intra1`) → genuinely new. |
| Overnight leg | `np.log(open / ts_delay(close, 1))` | Splits the daily move into a distinct component. |
| Trade-size tilt | replace `volume` with `volume_first_hour / number_of_trades_first_hour` | Institutional-vs-retail proxy; not derivable from daily bars. |
| Earnings gate | multiply by `(trading_days_until_next_earnings_announcement < 5)` | Seasonality group — near-orthogonal by construction. |
| Turnover control | wrap in `ts_mean_exp(alpha, 5, 0.3)` | Lowers `tvr`, raising IR/√TVR — the metric the override clause uses. |

---

## Checks before you submit

- [ ] `open`, `close`, `volume` are all **Delay 0 = Yes** → no lag needed. ✔
- [ ] Run **fast** mode to screen, then **slow** mode to confirm — they must
      agree. `ts_rank(close, 10)` and `ts_mean(volume, 10)` need ≥ 11 days of
      lookback; give slow mode more than that.
- [ ] Expression is dimensionless (`close/open`, `volume/advN`) ✔ — this is
      what protects you from corporate-action retro-adjustment bias.
- [ ] Don't add `cs_indneut(..., industry)` — the sim's `Neut=industry`
      setting already does it. Only add it for a *different* grouping such as
      `subindustry`.
- [ ] **Compute Correlation** against your existing pool before submitting.
