# Theory note — closing-hour return skew, event-weighted

*Interview reference. Fourth in the series with `14_` (earnings window), `15_`
(closing auction) and `16_` (event-stress gate). All figures measured on
Trexsim.*

---

## 1. Final specification

```python
fomc_pre  = (days_until_next_fomc_meeting <= 0)
fomc_post = (days_since_last_fomc_meeting <= 0)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

sk    = ts_mean(skew_return_last_hour, 1)
score = cs_rank(sk) - 1.0

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)
```

**SUBMITTED 2026-08-11.** Second qualified alpha.

| | Fast | **Slow (submitted)** |
|---|---|---|
| Window | 2006-03-31 → 2021-12-30 | 2007-03-15 → 2021-12-30 |
| **IR** | 0.097 | **0.102** |
| TVR | 1.375 | 1.380 |
| Return | 0.047 | **0.049** |
| Drawdown | 4.5 | **4.7** |
| IR/√TVR | 0.083 | 0.086 |
| Booksize | 0.91 x 0.91 | 0.91 x 0.91 |
| NumStks | 706 x 706 | **707 x 707** |
| Liq / LiqN | 363 / 157 | 361 / 156 |

**Slow mode scored *higher* than fast (0.102 vs 0.097).** Look-ahead
contamination shows up as slow-mode collapse, so an increase rules it out. This
is a stronger result than note 16's alpha, which lost 3% going to slow mode.

**Why the start date moves nearly a year.** Note 16's alpha shifted three weeks
between modes and that was warm-up being consumed — its longest window is
`W_SIZE = 175`. This expression's longest window is `ts_mean(..., 1)`, so
warm-up cannot explain an 11-month shift. The explanation is data coverage:
`GROUP_intra1` reports 3,246–3,249 days against ~4,279 for the daily groups.
The intraday history simply starts later. Worth knowing, because it means the
alpha is never evaluated on 2006 or most of the 2008 crisis run-up — the
drawdown of 4.7 is measured on a sample that excludes the worst of the crisis,
and the comparison against note 16's 10.5 is therefore **not like-for-like**.

---

## 2. Comparison against the first submission

| | note 16 alpha | **this alpha** |
|---|---|---|
| Core data | daily bars (backoffice) | **intra1** |
| Core signal | reversal / liquidity provision | closing-hour return shape |
| IR | 0.091 | **0.102** |
| Drawdown | 10.5 | **4.7** (shorter sample) |
| NumStks | 365 | **707** |
| Booksize | 0.48 | **0.91** |
| TVR | 0.739 | 1.380 |
| Gate role | on/off, ~40% of days | **graded weight, most days** |
| Parameters tuned | ~22 | **9** |

Higher IR from **9 tuned parameters instead of 22**. That ratio matters more
than the IR difference — note 16's alpha has 14 internal windows fitted on
gradients of +0.001 to +0.002 each, which is where overfitting hides. This one
has a single-term core with no internal windows at all.

---

## 3. Mechanism

**What `skew_return_last_hour` measures.** Third moment of returns sampled
within the final trading hour. Positive skew = the hour was mostly small moves
punctuated by a few sharp up-ticks. Negative skew = mostly small moves with a
few sharp down-ticks. It is a statement about the *shape* of the hour, not its
direction — a stock can close up with negative skew and down with positive.

**Direction: long positive skew.** Names whose closing hour contained sharp
upward jumps outperform over the following day.

**Why the closing hour specifically.** It is where mechanical flow concentrates
— index tracking, ETF create/redeem, MOC algorithms, rebalancing (the argument
developed in note 15). A few sharp up-ticks inside an otherwise quiet hour is
the signature of a large buy order working through the book against thin
resting liquidity. That order is executing on a mandate, not a view, and it is
not finished: institutional orders are worked across sessions. Positive
closing-hour skew is therefore a **fingerprint of incomplete institutional
demand**, and going long collects the remainder.

**Why this is not simply momentum.** Skew is scale-free and direction-agnostic
by construction. A stock up 3% smoothly through the hour has near-zero skew; a
stock flat on the hour with two sharp spikes has high positive skew. The signal
is deliberately orthogonal to the hour's return, which is what makes it
survivable alongside a reversal alpha in the same pool.

**Why the gate helps.** Same argument as note 16 — mandated flow is heaviest
around expirations, rebalancing dates and macro events, so the "this jump was
mechanical" reading is most reliable then. `stress` counts how many such
conditions are simultaneously live, and weighting by `(1 + stress)` says the
inference is more trustworthy when more of them coincide.

---

## 4. What I got wrong

**I specified the sign backwards on all three intraday alphas.** I wrote
`-np.tanh(...)` for the volatility-asymmetry and variance-ratio signals and
`-(cs_rank(...) - 1)` for this one. All three were run with the sign flipped and
all three produced positive IR. My mechanism reasoning was inverted in every
case.

For this alpha specifically I had argued from the **idiosyncratic-skewness
literature** (Boyer, Mitton & Vorkink 2010): investors overpay for lottery-like
positive-skew stocks, so high-skew names should *underperform*. That is a real
and well-replicated effect — but it concerns **expected skewness of the return
distribution over months**, priced as a preference. This variable is **realised
skewness within a single hour**, which is a microstructure observation about
order execution, not a distributional property investors form preferences over.
Applying a monthly-horizon pricing result to an hourly execution statistic was
a category error.

**Lesson, and it is the same one as note 14:** the mechanism story was doing the
reasoning, and the mechanism was wrong. Both times the fix was testing the
parameter across its range rather than trusting the story. Sign is a parameter.

---

## 5. Development path

| Step | IR | TVR |
|---|---|---|
| `-` sign, `ts_mean(sk, 5)`, ungated | negative | — |
| `+` sign, `ts_mean(sk, 5)`, ungated | 0.061 | 0.553 |
| composite with two other intra1 signals | 0.054 | 0.861 |
| `+` sign, `ts_mean(sk, 10)`, ungated | worse | — |
| **`+` sign, `ts_mean(sk, 1)`, ungated** | **0.088** | 1.315 |
| **+ event-stress weight (this)** | **0.097 / 0.102** | 1.380 |

Two findings worth stating:

**Smoothing destroys the signal.** Window 1 (no smoothing) beats window 5 by
+0.027 and window 10 by more. The information decays within a day — it is an
execution-flow observation with a one-session half-life, and averaging five days
of it averages five unrelated events. This is the opposite of note 16's core,
where every component wanted a multi-day window.

**Combining made it worse.** Averaging the ranks of three intraday signals
that each showed positive IR gave 0.054, below the 0.061 of the best component
alone. The other two (volatility asymmetry at 0.042, variance ratio at 0.034)
were weak enough that equal weighting was net dilutive. Equal-weighting a set of
signals is only sound when they are comparably strong; here it was not tested
first, it was assumed.

---

## 6. Honest weaknesses

- **TVR 1.38 is high** and no cost model has been applied. IR is measured gross.
  A signal with a one-day half-life necessarily turns the book over, so this is
  intrinsic to the idea rather than a tuning failure — but it means the alpha is
  the most cost-sensitive of the three and would likely be the first to die net
  of spread. The names it trades are large (LiqN 156), which helps.
- **IR/√TVR 0.086 is below note 16's 0.105.** If this alpha ever needs the
  correlation-override clause it will fail it. Its qualification depends on
  staying under 50% correlation on its own merits.
- **Shorter sample.** 2007-03 onward, so the drawdown figure excludes 2006 and
  the early crisis period. Not comparable to the other alphas' drawdowns.
- **Gate windows were widened without a mechanism argument.** `mend <= 15` and
  `opex <= 10` mean `stress` is nonzero on most days, so the gate stopped being
  an event filter and became a smooth weighting. It works, but the note-16
  point/period story does not explain these windows, and I do not have an
  account of why 15 days before month-end should matter. Fitted, not reasoned.
- **`skew_return_last_hour` is a vendor-computed field.** The sampling frequency
  inside the hour is not documented in the data description. If it is computed
  from 1-minute bars the interpretation above holds; from 5-minute bars the
  "sharp up-tick" reading is weaker.

---

## 7. Predictions and falsification

1. **`skew_return_first_hour` should be weaker.** The opening hour has price
   discovery mixed with mechanical flow; the closing hour is more purely
   mechanical. If first-hour skew works *equally* well, the closing-auction
   story is not what is driving this.
2. **The effect should be stronger in larger names.** More index and ETF
   membership means more mandated closing flow.
3. **It should NOT work on a randomly chosen hour mid-session** — untestable
   with this data, which is the honest limitation of prediction 1 as a test.
4. **A random gate keeping the same day fraction should not add +0.014.** Same
   falsification as note 16, still unrun.

---

## 8. Interview preparation

### 8.1 The 60-second walkthrough

> "It's a cross-sectional equity signal on roughly 700 US large caps. The core
> is one variable: the skew of returns *within the final trading hour*. I rank it
> cross-sectionally each day and go long the top, short the bottom, with no
> smoothing — the signal has a one-day half-life. Then I weight the whole book by
> a count of how many mandated-flow events are live that day: FOMC, monthly and
> quarterly expiration, quarter-end, month-end and turn, pre-holiday. Daily IR
> is 0.102, which is a Sharpe of about 1.6, on 707 names with a 4.7% max
> drawdown. The economic story is that a few sharp up-ticks inside an otherwise
> quiet closing hour is the fingerprint of a large institutional buy order
> working through thin resting liquidity — and those orders aren't finished, so
> the residual demand shows up the next day. The gate says that inference is
> most reliable when mandated flow is heaviest."

Then stop. Let them pick the thread.

### 8.2 The cost question — expect this, and lead with it

This is the hardest question about this alpha and the numbers are not flattering.
Have them ready rather than being walked into them.

Turnover 1.38 means **348× the book per year**. Against a 4.9% gross annual
return:

| round-trip cost | net annual return |
|---|---|
| **1.41 bp** | **0.0% — break-even** |
| 1 bp | +1.4% |
| 2 bp | −2.1% |
| 5 bp | −12.5% |
| 10 bp | −29.9% |

**The alpha needs sub-1.4bp all-in execution to make money standalone.** For
comparison, the note-16 alpha breaks even at 6.7bp — **4.8× more cost
tolerance** — because it earns 12.5% on less than half the turnover.

The honest answer, in three parts:

1. **Standalone, at any realistic retail or mid-tier institutional cost, this
   loses money.** Say so first. Claiming otherwise on a 348×-turnover signal
   destroys credibility immediately.
2. **The competition scores gross IR**, so it qualifies on the stated criteria.
   That is a statement about the scoring rule, not a defence of the alpha.
3. **The real defence is portfolio-level.** Standalone turnover is the wrong
   denominator at a multi-alpha firm. When this book is combined with other
   alphas, its trades net against theirs before anything reaches the market —
   *marginal* turnover is far below standalone turnover, and marginal cost is
   what determines whether adding the signal is accretive. A high-Sharpe,
   low-correlation, high-turnover signal can be strongly positive at the margin
   while being negative on its own. That is a large part of why a firm runs
   hundreds of alphas rather than the best five.

If pressed on what you would do about it: `ts_mean_exp` on the ranks to damp
position churn (the operator docs name it as the turnover-control tool),
restricting to the top liquidity decile, and passive execution — though the
one-day half-life caps how patient you can afford to be, which is the real bind.

### 8.3 Question bank

**"Why long positive skew? Shouldn't lottery-like stocks underperform?"**
That is Boyer-Mitton-Vorkink, and it is about **expected skewness of the return
distribution over months**, priced because investors have a preference for
lottery payoffs. This is **realised skewness inside one hour** — a
microstructure observation about how an order was executed, not a distributional
property anyone forms a preference over. Different object, different horizon.
Worth adding: I initially reasoned from BMV and specified the sign backwards.
The data corrected me.

**"Isn't this just momentum?"**
Skew is scale-free and direction-agnostic by construction — a stock up 3%
smoothly through the hour has near-zero skew, a stock flat with two sharp spikes
has high positive skew. So it should be close to orthogonal to the hour's
return. *Caveat to volunteer:* I have not regressed the signal on `ret1` or on
`return_last_hour` to confirm the orthogonality is empirical rather than
theoretical. That is the test I would run next.

**"Sharpe 1.6 on 9 parameters — how do you know it isn't overfit?"**
Four things, in descending strength. Slow mode scored *higher* than fast (0.102
vs 0.097), which rules out look-ahead. The core has **no internal windows at
all** — it is one variable, ranked. The sign flip was a discrete binary test,
not a fitted parameter. And the smoothing-window response was monotone across
1/5/10, which is a gradient rather than a spike. *Against:* the eight gate
windows were widened without a mechanism argument, and `mend <= 15` in
particular I cannot justify economically. Those are fitted.

**"Why is your drawdown only 4.7%?"**
Partly low volatility — implied annual vol is about 3%, so the Calmar is 1.04,
which is respectable but not extraordinary. Partly the sample: `GROUP_intra1`
covers ~3,247 days against ~4,279 for the daily groups, so this alpha is never
evaluated on 2006 or the early crisis period. The drawdown is **not
comparable** to my other alpha's 10.5%, which does include it.

**"What's the capacity?"**
707 names, LiqN 156, all large caps — so ADV is not the binding constraint.
Market impact at 348× annual turnover is. Capacity is set by how much can be
traded in the final hour without moving the price you are trying to measure,
which I have not sized. It is a real limit and I would expect it to be modest.

**"Why should this persist?"**
Institutional execution has to happen, and large orders have to be worked across
sessions — that structure is durable. But the signal is cheap to compute from
widely available intraday data, so competitive erosion is the obvious risk.
*I have not run a decay test across 2007-2021*, and that is the single most
informative thing I could add. If the effect is flat over fifteen years it is
structural; if it is concentrated pre-2015 it is being arbitraged away.

**"Does the gate really add anything, or are you just trading fewer days?"**
It adds +0.014 IR. The falsification is a random gate keeping the same fraction
of days — if that helps equally, the gain is coverage or turnover, not events.
**I have not run it.** Same admission applies to the note-16 alpha.

**"How do your two alphas relate?"**
Different data groups (daily bars vs intraday), different mechanisms — one is
reversal and gets paid for supplying liquidity, this one is continuation and
gets paid for detecting unfinished demand. They share only the event gate. That
they work in *opposite* directions on the same days is the interesting part: it
suggests the gate is identifying elevated mandated flow generally, and different
signals monetise different pieces of it.

### 8.4 The question I would ask if I were interviewing me

*"You had the sign backwards on all three of your intraday signals. If your
mechanism reasoning was wrong three times out of three, why should I believe the
mechanism story you're telling me now?"*

The honest answer is that the mechanism did not generate the result — it was
written *after* the sign was determined empirically. What the reasoning is
actually good for is generating hypotheses to test and constraining what to try
next, and its track record here suggests treating the direction of any
microstructure signal as a free parameter to measure rather than a conclusion to
derive. The same thing happened in note 14, where I argued the last three days
before earnings should be excluded and the data showed day 0 was the single most
valuable day in the sample.

That is a defensible position, but only if stated before they get there.

---

## 9. Literature

- **Baltussen, Da & Soebhag, "End-of-Day Reversal"** — closing-price deviations
  and their reversal. Note this predicts *reversal*, and this alpha is
  continuation; the two are reconcilable only if skew and level displacement
  carry different information, which is assumed here and not demonstrated.
- **Boyer, Mitton & Vorkink (2010), "Expected Idiosyncratic Skewness," RFS** —
  the result I misapplied. Included deliberately: an interviewer who knows this
  paper will raise it, and the distinction between expected distributional
  skewness and realised intra-hour skewness is the answer.
- **Bogousslavsky (2016), "Infrequent Rebalancing, Return Autocorrelation, and
  Seasonality," JoF** — intraday return predictability from institutional
  rebalancing patterns.
- **Nagel (2012), "Evaporating Liquidity," RFS** — the anchor for the gate, as
  in note 16.
