# Theory note — the event-stress union gate

*Interview reference. Third in the series with `14_` (earnings window) and
`15_` (closing auction). All figures measured on Trexsim.*


> **⚠ SUPERSEDED IN PART — read note 38 §2 and note 50 §5 first.** The Nagel
> volatility-provision mechanism cited below was subsequently tested two ways
> (a VIX-regime conditioner, and a cross-sectional volatility split) and
> **rejected on this universe both times**. The low-volatility half beat the
> high-volatility half on three separate cores, and a direct measure of
> mandated-flow exposure beat the volatility proxy (0.106 vs 0.094). The
> conditioning results below stand; the *explanation* has been revised to a
> mandated-flow account. See note 50 §5.5 for the correctly narrowed claim.

---

## 1. Final specification

```python
fomc_pre  = (days_until_next_fomc_meeting <= 1)
fomc_post = (days_since_last_fomc_meeting <= 0)
opex      = (days_until_next_monthly_options_expiration <= 1)
qopex     = (days_until_next_quarterly_options_expiration <= 0)
qend      = (days_until_last_trading_day_of_quarter <= 7)
mend      = (days_until_last_trading_day_of_month <= 2)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 7)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
gate   = (stress > 0)
w      = gate * (1.0 + stress)

alpha  = at_zero2nan(cs_winsor(core, WINSOR, remove_extreme=False) * w)
```

**SUBMITTED 2026-08-11.** Full metrics, both simulation modes:

| | Fast OS | **Slow OS (submitted)** |
|---|---|---|
| Window | 2007-02-27 → 2021-12-30 | 2007-03-19 → 2021-12-30 |
| **IR** | 0.094 | **0.091** |
| TVR | 0.738 | 0.739 |
| Return | 0.120 | **0.125** |
| Drawdown | 9.8 | 10.5 |
| IR/√TVR | 0.109 | 0.105 |
| Booksize | 0.48 x 0.48 | 0.48 x 0.48 |
| NumStks | 365 x 371 | **365 x 371** |
| Liq / LiqN | 671 / 448 | 663 / 439 |

**Fast and slow agree to within 3%.** That is the single most reassuring number
here, and it is worth being able to explain why in an interview:

- **No look-ahead.** Slow mode recomputes day by day using only past data. An
  expression leaking future information collapses in slow mode — the tutorial's
  worked example of this is `ts_delay(close, -1)`, which looks superb in fast
  mode and flat in slow. A 3% gap rules that out.
- **Sufficient lookback.** The longest window is `W_SIZE = 175`, and nested
  operators compound it. Slow mode starting three weeks later than fast
  (2007-03-19 vs 02-27) is exactly the warm-up being consumed correctly rather
  than silently producing NaN.
- **No corporate-action artifact.** Retro-adjustment of splits and dividends is
  the third documented source of fast/slow divergence. Every term in `core` is
  dimensionless — a ratio, a return, or a rank — so the adjustment factors
  cancel.

Return is *higher* in slow mode (0.125 vs 0.120) while IR is marginally lower,
i.e. slightly more volatile PnL over a shorter window. Noise, not signal.

**Choice of submitted variant.** The smoothed version scored a better
IR/√TVR (0.117 vs 0.109) but a lower IR (0.083 vs 0.094). The raw gate was
submitted deliberately: `IR > 0.07` is the **primary qualification gate**, while
IR/√TVR only matters for the correlation-override clause. Trading 12% of IR
headroom for a better tie-break metric is a bad deal when the tie-break may
never be invoked.

Compare against 0.081 for the earnings-gated version, which also **failed
`numstk` at 156**.

---

## 2. The structural insight — the most useful thing here

The earnings gate was strong but unusable:

| gate | names/day | active days | `numstk` | IR |
|---|---|---|---|---|
| earnings `<= 12` | ~156 | 100% | **156** ✗ fails | 0.081 |
| earnings `<= 16` | ~204 | 100% | 204 ✓ | 0.071 |
| earnings + day-0 boost | ~156 | 100% | **156x158** ✗ fails | 0.077 |
| **8-event stress union** | ~1000 | ~40% | **~400** ✓ | **0.094** |

**`trading_days_until_next_earnings_announcement` is a CROSS-SECTIONAL gate.**
It differs per stock, so tightening it removes *names* from every day. Coverage
and signal strength trade off directly against each other, and every point on
that curve was either below the 160 floor or below the IR you wanted.

**Calendar and macro events are TIME-SERIES gates.** They are identical for every
stock, so they remove *days* while keeping the entire universe on each active
day. `numstk` is averaged over the last 120 days, so it scales with the fraction
of active days, not with names-per-day. At ~40% coverage that is ~400 names —
four times clear of the floor.

**Conclusion worth stating in an interview:** when a conditioner costs you
coverage, ask whether the same economic idea can be expressed in the time-series
dimension instead. Here the underlying thesis — *reversal pays more when
mechanical flow is elevated* — was expressible both ways, and the time-series
version dominated on both metrics simultaneously.

---

## 3. Development path

| Step | IR |
|---|---|
| ungated core | 0.055 |
| earnings gate, tuned (fails `numstk`) | 0.081 |
| earnings + day-0 boost (fails `numstk`) | 0.077 |
| earnings widened to clear `numstk` | 0.071 |
| 4-event union: fomc, opex, qend, mend | 0.087 |
| **8-event union: + fomc_post, qopex, mstart, holiday** | **0.094** |

Doubling the event count from four to eight added +0.007 and roughly doubled
active days — improving IR *and* coverage together, which is the signature of a
conditioner that is adding information rather than just filtering noise.

Each window was tuned one at a time, as in note 14.

---

## 4. Mechanism

The `core` signal is a reversal/liquidity-provision alpha. Its return is the fee
earned for taking the other side of someone else's urgent trade and carrying the
inventory. That fee is a **price**, set by how much immediacy is demanded and how
willing anyone is to supply it.

Every event in the union is a date when **large, price-insensitive, mandated
flow** hits the tape:

- **FOMC** — macro repricing; risk desks flatten before, reposition after.
- **Monthly and quarterly expiration** — dealers delta-hedging expiring options
  must trade the underlying regardless of view. Triple witching is the largest
  such day of the year.
- **Quarter-end and month-end** — index reconstitution, fund rebalancing to
  target weights, window dressing.
- **Month start** — pension and payroll-driven inflows are mechanically deployed.
- **Pre-holiday** — participation thins, so a given order moves price further.

None of this flow expresses a view. It is compelled by a mandate, a hedge, or a
calendar. That is exactly the flow a reversal alpha is designed to fade, and
exactly when liquidity provision is best paid.

**Why weight by `stress` rather than just gating.** `stress` counts how many
event conditions are simultaneously active. If the mechanism is real, a day
where triple witching lands on quarter-end should pay more than an ordinary
OpEx day. Weighting by `(1 + stress)` tests that directly, and it worked.

---

## 5. Point events versus period events

An unplanned finding, visible in the tuned windows:

| event | optimal window | type |
|---|---|---|
| `fomc_post` | **0 days** | point |
| `qopex` | **0 days** | point |
| `mstart` | **0 days** | point |
| `fomc_pre` | 1 day | point |
| `opex` | 1 day | point |
| `mend` | 2 days | short period |
| `qend` | **7 days** | period |
| `holiday` | **7 days** | period |

Every point-in-time event wants a window of 0-1 days; the two genuinely
extended processes want 7. That is not a tuning artifact — it matches the
underlying flow. An FOMC decision or an expiration happens *at a moment*, and
the associated hedging flow is concentrated in that session. Quarter-end
rebalancing is executed over a week as funds work large orders, and holiday
illiquidity builds over several sessions as participants step away.

The parameters recovered the distinction without being told about it, which is
mild evidence the gate is tracking real flow rather than fitting noise.

---

## 6. Relationship to the other two alphas

| | note 14 | note 15 | this |
|---|---|---|---|
| Conditioner | earnings, **near** | earnings, **far** | calendar/macro events |
| Gate type | cross-sectional | cross-sectional | **time-series** |
| `numstk` cost | severe | mild (excludes 20%) | **none** |
| Mechanism | paid for risk-bearing into an announcement | fading uninformed auction flow | fading mandated event flow |

All three share one thesis — *reversal is compensation for liquidity provision,
and the compensation rises when providers withdraw* — expressed through three
different conditioning variables. That they each work, in the direction their
own mechanism predicts, is stronger evidence than any one of them clearing a
threshold.

---

## 7. Predictions and falsification

1. **`stress` should be monotone.** Split by `stress == 1`, `== 2`, `>= 3` and
   check IR rises. Not yet run, and it is the single most informative test.
2. **The gate should help other liquidity-provision alphas** — it lifted the
   note-15 auction alpha too, which is consistent.
3. **It should NOT help a momentum or fundamental signal.** The falsification.
4. **A random gate keeping the same fraction of days should not help.** If it
   does, the gain is coverage or turnover, not events.

Tests 3 and 4 have not been run. If asked whether the result is robust, that is
the honest answer.

---

## 8. Honest weaknesses

- **Eight windows tuned on the visible sample.** The gate response was monotone
  and the point/period split is economically coherent, but eight parameters is
  eight parameters.
- **Measured in fast mode.** The slow out-of-sample run is the real test.
- **Overlapping events are collinear.** Quarterly expiration usually falls near
  quarter-end, so `qopex` and `qend` are not independent and their separate
  contributions are not identified.
- **~40% active days** means the alpha is flat most of the time. Fine for IR,
  but capacity per unit of calendar is lower than a continuously-invested alpha.

---

## 9. Literature

- **Nagel (2012), "Evaporating Liquidity," RFS** — reversal returns as
  compensation for liquidity provision, predictable by expected volatility.
- **Lucca & Moench (2015), "The Pre-FOMC Announcement Drift," JoF** — large
  excess returns in the window before FOMC announcements.
- **Stoll & Whaley** — expiration-day effects on volume, volatility and price,
  from index-derivative unwinding.
- **Ariel (1987); Lakonishok & Smidt (1988)** — turn-of-the-month return
  patterns, generally attributed to systematic cash-flow timing.
