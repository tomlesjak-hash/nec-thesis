# Track A — hand-crafted alphas, paste-ready

**Why this list looks the way it does.** The tutorial's own alphas score
`-ret1` → IR 0.025, `-ret5` → 0.043, `-ret20` → 0.026, and its refined
volume-premium alpha → 0.015. **All below the 0.07 gate.** Trexquant deliberately
set the bar above what textbook signals reach, so nothing here is plain
reversal. Every alpha below does one of three things instead:

1. **Uses `GROUP_intra1`** — 30 first-hour/last-hour variables. The only
   sub-daily structure in the dataset, and the part a daily-bar competitor
   cannot reach at all.
2. **Conditions on `trading_days_until_next_earnings_announcement`** — gating a
   signal to a window where it is strong beats running it always.
3. **Smooths with `ts_mean_exp`** — cuts turnover, and by averaging out noise
   often raises IR as well. The tutorial used it to take TVR from 1.00 to 0.21.

Two mechanical rules throughout: `cs_rank` returns **[1,2]** so subtract 1, and
every denominator gets `at_zero2nan`.

> Run each, note the IR, keep what clears 0.07. Expect maybe a third to. Then
> **Compute Correlation** before submitting — the gate is ≤0.50.

---

## 1. Overnight vs intraday decomposition ⭐ start here

The daily return is two different animals glued together. Overnight moves
(close→open) carry information from news and order imbalance; intraday moves
(open→close) are dominated by liquidity provision and revert. Trading them with
opposite signs is strictly more informative than trading `ret1`.

```python
prev      = ts_delay(close, 1)
overnight = (open - prev) / at_zero2nan(prev)
intraday  = (close - open) / at_zero2nan(open)

alpha = (cs_rank(overnight) - 1) - (cs_rank(intraday) - 1)
```

## 2. Same idea, smoothed

Same signal, lower turnover. Compare IR against #1 — if it holds up, this is the
better submission because IR/√TVR is the override metric.

```python
prev      = ts_delay(close, 1)
overnight = (open - prev) / at_zero2nan(prev)
intraday  = (close - open) / at_zero2nan(open)

alpha = ts_mean_exp((cs_rank(overnight) - 1) - (cs_rank(intraday) - 1), 5, 0.4)
```

## 3. Close-auction pressure ⭐ intra1

If the close prints well above the last hour's VWAP, someone was pushing into
the auction. That pressure is mechanical rather than informational, and it
reverts.

```python
push  = (close - vwap_last_hour) / at_zero2nan(vwap_last_hour)
alpha = -(cs_rank(push) - 1)
```

## 4. Average trade size — institutional vs retail flow ⭐ intra1

`volume / number_of_trades` is mean trade size. Rising trade size with flat
volume means larger participants are active. **This is not derivable from daily
bars** — no competitor without intra1 can express it.

```python
ats   = volume_last_hour / at_zero2nan(number_of_trades_last_hour)
rel   = ats / at_zero2nan(ts_mean(ats, 20))
alpha = ts_mean_exp(cs_rank(rel) - 1, 3, 0.5)
```

## 5. Intra-hour information ratio ⭐ intra1, likely under-mined

`information_ratio_last_hour` is return-per-unit-risk inside the final hour,
already bounded to about ±2.6. Bounded and pre-normalised means it needs no
winsorising, and almost nobody will have looked at it.

```python
alpha = -(cs_rank(information_ratio_last_hour) - 1)
```

## 6. First-hour / last-hour divergence ⭐ intra1

Opening hour is retail and overnight-order driven; closing hour is
institutional and index driven. When they disagree, the close usually wins.

```python
alpha = (cs_rank(return_last_hour) - 1) - (cs_rank(return_first_hour) - 1)
```

## 7. Pre-earnings drift gate ⭐ uses the best conditioner in the dataset

Multiplying an existing signal by an earnings-window mask produces a
**genuinely new alpha** rather than a correlated variant — different data group,
so it should pass the 0.50 gate against everything above.

```python
days = trading_days_until_next_earnings_announcement
gate = (days < 5)

alpha = at_zero2nan(-(cs_rank(ret5) - 1) * gate)
```

## 8. Post-earnings drift (the other side)

PEAD is one of the most durable documented anomalies. Same variable, opposite
window — and `at_zero2nan` keeps you flat on every stock outside it.

```python
days = trading_days_until_next_earnings_announcement
gate = (days > 120)          # just after a report, counting to the next

alpha = at_zero2nan((cs_rank(ret5) - 1) * gate)
```

## 9. Volume premium, conditioned (the tutorial's own case study)

Their v5, corrected. Note their published version uses
`cs_rank(...) < 1.0`, which selects **nothing** — `cs_rank` bottoms out at
exactly 1.0. Use 1.5 for the low-dispersion half.

```python
vol_z  = ts_norm(volume, 20)
stable = (cs_rank(ts_std(volume, 10)) < 1.5)

alpha = ts_mean_exp(vol_z * stable, 3, 0.5)
```

## 10. Idiosyncratic reversal — market beta removed

Plain reversal is crowded. Reversal *after* stripping market sensitivity is a
different signal, and the residual is what actually mean-reverts.

```python
beta  = ts_corr_binary(ret1, ret1_spx, 60)
alpha = -(cs_rank(ret5) - 1) + (cs_rank(beta) - 1) * 0.3
```

## 11. Subindustry neutralisation instead of industry

The sim already neutralises by `industry`. Neutralising by **`subindustry`**
(~500 groups vs ~70) is a genuinely finer control and a different alpha.

```python
prev      = ts_delay(close, 1)
overnight = (open - prev) / at_zero2nan(prev)

alpha = cs_indneut(cs_rank(overnight) - 1, subindustry)
```

## 12. Intraday range vs realised vol

Wide intraday range relative to recent realised volatility signals a stressed
book, which typically reverts.

```python
rng    = (high - low) / at_zero2nan(close)
stress = rng / at_zero2nan(ts_mean(rng, 20))

alpha = ts_mean_exp(-(cs_rank(stress) - 1), 5, 0.4)
```

---

## Working through these

**Order:** 1, 3, 5, 4, 7 first — those are the most differentiated. Then 6, 2,
9, 10, 12, 8, 11.

**If one nearly clears 0.07**, try in this order before discarding it:
1. Wrap in `ts_mean_exp(alpha, 5, 0.4)` — smoothing often lifts IR *and* cuts TVR
2. `cs_winsor(x, 0.02, remove_extreme=False)` before the rank
3. `cs_remove_middle(alpha, 0.6)` — trade only the tails; raises IR, costs TVR
4. Add an earnings gate from #7

**Before submitting anything:** run Compute Correlation against your existing
pool. These are spread across five different data groups precisely so they
should clear 0.50 against each other, but verify rather than assume.

**Check fast vs slow mode agree** on anything using a 20- or 60-day window, and
give slow mode a lookback well above the longest window in the expression.
