# Five original alphas — designed to clear 0.07

## What the evidence says about clearing 0.07

Three facts to design against:

1. **Plain signals don't reach it.** The tutorial's own alphas: `-ret1` → 0.025,
   `-ret5` → 0.043, `-ret20` → 0.026, refined volume premium → 0.015. Trexquant
   set the bar deliberately above textbook.
2. **Your #19 hit 0.090, and its shape is the clue.**
   `-tanh(r7 / vol7) × cs_rank(r250)` is not one signal — it is a **fast signal
   × slow signal interaction**, with the fast leg volatility-normalised. The
   interaction is what got it over the bar; neither leg alone would.
3. **`GROUP_intra1` is original by construction.** Alpha101 was written in 2015
   for daily bars. No published factor uses `vwap_last_hour` or
   `number_of_trades_first_hour` because that data wasn't in the set. Nothing
   built on it can collide with another competitor's Alpha101 submission under
   the `max corr (others) < 0.90` check.

All five below follow the #19 skeleton — **volatility-normalised fast leg,
`tanh` squash, optionally scaled by a slow leg** — applied to information
Alpha101 never had. Each draws on a different data group, so they should clear
0.50 against each other.

---

## 1. Overnight vs intraday reversal asymmetry

**Idea.** The daily return welds together two different processes. Overnight
moves (close→open) come from news and accumulated order imbalance and tend to
*continue*. Intraday moves (open→close) are dominated by market-makers earning
the spread and tend to *revert*. `ret1` averages them into mush — which is
exactly why the tutorial's `-ret1` only manages 0.025. The **spread between
them** is the clean signal.

```python
prev   = ts_delay(close, 1)
on     = (open - prev) / at_zero2nan(prev)
intra  = (close - open) / at_zero2nan(open)
spread = intra - on

alpha = -np.tanh(spread / at_zero2nan(ts_std(spread, 20)))
```

*Group: backoffice. Should be near-uncorrelated with anything using volume.*

---

## 2. Closing-auction pressure, gated away from earnings ⭐

**Idea.** A close printing well above the last hour's VWAP means someone pushed
into the auction. That pressure is mechanical — index rebalancing, MOC
imbalance, a VWAP algo finishing — and it reverts. **But not near earnings**,
where late buying may be informed. Gating the signal to the quiet window should
sharpen it considerably.

```python
push = (close - vwap_last_hour) / at_zero2nan(vwap_last_hour)
z    = push / at_zero2nan(ts_std(push, 20))
far  = (trading_days_until_next_earnings_announcement > 10)

alpha = at_zero2nan(-np.tanh(z) * far)
```

*Groups: intra1 + trading_days_fast. Two sources no Alpha101 factor touches.*

---

## 3. Intraday participation shift

**Idea.** Where a stock's volume sits within the day says who is trading it.
Institutional flow and index activity concentrate near the close (VWAP and MOC
algos finish there); retail and news reaction cluster at the open. When a
stock's volume share **shifts toward the close relative to its own history**,
that is accumulation, and it continues. Scaled by long-horizon momentum, as #19
scales by `r250`.

```python
share = volume_last_hour / at_zero2nan(volume_first_hour + volume_last_hour)
shift = (share - ts_mean(share, 20)) / at_zero2nan(ts_std(share, 20))
slow  = cs_rank(ts_sum(np.log(1.0 + ret1), 120)) - 1.0

alpha = np.tanh(shift) * slow
```

*Group: intra1. Literally impossible to compute from daily bars.*

---

## 4. Intraday churn — a variance-ratio signal

**Idea.** Compare volatility *within* the hour against volatility *across*
days. If a stock is thrashing intraday but going nowhere day to day, it is being
churned without information arriving — noise, not signal, and it reverts. If
intraday vol is low while daily vol is high, moves are informed and persist.
This is a variance-ratio in microstructure clothing, and it needs the intraday
standard deviations to express.

```python
intra_v = (std_dev_return_first_hour + std_dev_return_last_hour) / 2.0
daily_v = ts_std(ret1, 20)
churn   = intra_v / at_zero2nan(daily_v)
z       = (churn - ts_mean(churn, 20)) / at_zero2nan(ts_std(churn, 20))

alpha = -np.tanh(z)
```

*Group: intra1. Economically distinct from #3 — that one is about* who *trades,
this is about* whether information is arriving.

---

## 5. Post-earnings drift, conviction-scaled ⭐

**Idea.** PEAD is among the most durable documented anomalies, and
`trading_days_until_next_earnings_announcement` makes it directly expressible.
On a quarterly cycle (~63 trading days), a stock that reported 1–20 days ago
shows `days_until_next` in roughly **[43, 62]** — that is the drift window.
Scale the reaction by its own volatility so a 5% move in a quiet name outranks
5% in a jumpy one, exactly as your #19 does.

```python
days = trading_days_until_next_earnings_announcement
win  = (days > 40) * (days < 65)                    # ~1-20 days post-report
jump = ret5 / at_zero2nan(ts_std(ret1, 60) * 2.236) # 5d move in vol units

alpha = at_zero2nan(np.tanh(jump) * win)
```

*Group: trading_days_fast. Trades only ~1/3 of the calendar — check `numstk`
stays above 160; if not, widen to `(days > 35) * (days < 70)`.*

---

## Running them

**Order: 2, 1, 5, 3, 4.** #2 and #5 use the earnings conditioner, which is the
single most under-used variable in the dataset.

**If one lands just short**, in this order:
1. `ts_mean_exp(alpha, 5, 0.4)` — smoothing lifts IR *and* cuts turnover
2. Add the slow leg: `alpha * (cs_rank(ts_sum(np.log(1.0+ret1), 250)) - 1.0)`
   — this is the specific thing that got #19 over the bar
3. `cs_remove_middle(alpha, 0.6)` — trade only the tails; raises IR, costs TVR
4. Swap `np.tanh` for `at_signsqrt` — a softer squash, sometimes better

**Watch `numstk`** on #2 and #5. Both gate on earnings proximity, so they hold
positions on only part of the universe. Below 160 they fail regardless of IR.

**Correlate before submitting.** These span four data groups on purpose, but #3
and #4 both use intra1 and could overlap — check that pair specifically.
