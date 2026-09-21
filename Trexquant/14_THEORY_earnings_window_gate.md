# Theory note — event-time conditioning of a reversal alpha

*Interview reference. Rewritten 2026-08-10 after the day-0 result overturned the
first version. All figures measured directly on Trexsim.*


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
stoch  = (close - ts_min(low, 20)) / at_zero2nan(ts_max(high, 20) - ts_min(low, 20))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
gate   = (trading_days_until_next_earnings_announcement >= 0) \
       * (trading_days_until_next_earnings_announcement <= 12)

volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core = (-(ts_zscore(stoch, 10))
        - (((ts_median(cs_zscore(relvol), 20) - ts_mean(ret20, 20))
            + ts_skew(relvol - ts_delay(relvol, 3), 33))
           - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * gate)
```

**IR 0.124, annualised return 0.196**, from a starting point of IR 0.063.

---

## 2. How this alpha was built

### Phase 1 — infrastructure (2 days)

Built a point-in-time panel: yfinance daily bars, S&P 500 membership
reconstructed from Wikipedia's constituent-change table, 2006-2022, 4,279 days
x 584 tickers, monthly-rebalanced top-500-by-ADV universe.

Validation caught six real bugs that would each have silently corrupted results:

| Bug | Consequence if unfixed |
|---|---|
| `auto_adjust=False` mixes conventions | adjusted close sat **below** unadjusted low on 77% of bars |
| Dollar volume from adjusted price | pre-split turnover understated by the split factor, distorting the universe |
| Universe mask accumulated forward | names removed from the index stayed members forever — full survivorship bias |
| Mask re-admitted names after halts | mid-month additions, indistinguishable from look-ahead |
| Cached files keyed on date range | a 2010-start run silently accepted 2015-2024 data |
| 5 tickers with reused symbols | CPWR, EP, MI, COL, SLE — different companies wearing dead tickers |

Documented and unfixable: **212 of 796 index members (26.5%) are missing**
because Yahoo purges delisted tickers. The gap is concentrated in companies that
went to zero — Lehman, Countrywide, Fannie, Freddie, Ambac, SVB, First
Republic — which is the worst possible shape of bias for a sample containing 2008.

### Phase 2 — genetic search (~15 hours compute, 0 usable alphas)

**Run 1:** 30 seeds x 600 population x 45 generations plus initial population
= 828,000 evaluations,
depth cap 8, trained 2011-2017. Produced 450 candidates, 21 passing every local
gate after a <=0.5 correlation prune.

**On Trexsim: none of the 21 cleared IR 0.07.**

**Run 2**, rebuilt to reduce overfitting: depth capped at 5, parsimony pressure,
training widened to 2006-2018 to include the financial crisis, plus guards added
for cross-sectional dispersion, minimum turnover, position count and effective
position count. 738,000 expressions, 7 local qualifiers.

**On Trexsim: none of the 7 cleared either.**

~1.57M expressions searched, 28 locally-qualifying alphas, zero platform
qualifiers. Diagnosis: the local panel is ~370 large caps against the platform's
top-1000, missing 212 delisted names, and contains **no intraday data at all**.
Local IR was measuring something systematically different.

Failure modes worth naming, all found by validation rather than by inspection:

- **The clone problem.** The original QIP GP used raw price *levels* as
  terminals and collapsed onto the size factor — all 10 hall-of-fame members
  correlated 0.99+. Fixed by making every terminal dimensionless.
- **Reward hacking.** With fitness = IR/sqrt(TVR), the GP drove turnover toward
  zero instead of finding signal. Sixteen of 26 "winners" contained
  `div(ret20, ret20)` — which is 1 everywhere. Fixed with dispersion and
  turnover floors.
- **Concentration.** Four alphas reported `numstk` ~400 while holding 99% of the
  book in one name. `numstk` counts non-NaN positions and cannot see this; fixed
  with effective N = 1/sum(w^2).
- **Anti-predictive ranking.** In run 1 the rank correlation between in-sample
  fitness and held-out IR was **-0.70**. Sorting by fitness was worse than
  random.

### Phase 3 — manual development (the part that worked)

Took the single GP expression that had come closest on the platform (run 1,
rank 28, seed 11 — IR 0.063) and improved it by hand, changing **one thing per
simulation** and recording IR and return each time.

What did nothing: `cs_remove_middle` at any level (0.063 → 0.063 — the signal is
diffuse across the cross-section, not concentrated in the tails); winsorising
`volret` harder or softer.

What actively hurt: normalising all components to a common scale (the relative
magnitudes turned out to be load-bearing); exponential smoothing of the output;
longer accumulation windows.

What helped: winsorising, removing accumulation entirely, and — by an order of
magnitude more than anything else — **conditioning on time-to-earnings**
(+0.055 from the gate against +0.006 from every internal window combined).

The gate was walked systematically rather than guessed: upper bound 90 → 60 →
18 → 12, then lower bound 3 → 2 → 1 → 0. The final step, adding the announcement
day itself, produced the largest single jump in the exercise.

---

## 3. Where the improvement actually came from

| Change | IR | Δ |
|---|---|---|
| Starting expression, gate `<= 90` | 0.063 | — |
| gate `<= 60` | 0.069 | +0.006 |
| gate `<= 15` | 0.078 | +0.009 |
| gate `>= 3, <= 18` | 0.084 | +0.006 |
| gate `>= 2, <= 18` | 0.086 | +0.002 |
| gate `>= 1, <= 18` | 0.088 | +0.002 |
| **gate `>= 0, <= 18`** | **0.113** | **+0.025** |
| internal window tuning | 0.117 | +0.004 |
| gate `>= 0, <= 12` | 0.122 | +0.005 |
| remaining window tuning | **0.124** | +0.002 |

**Nearly all of the gain is the gate, and the single largest step is one day.**
Including day 0 — the announcement day itself — added +0.025, roughly a quarter
of the total improvement, from one day out of nineteen. Tuning every internal
window in the expression contributed +0.006 combined.

That asymmetry is the most important fact in this note.

---

## 4. What I got wrong, and why it matters

The first version of this note argued that the last ~3 days before an
announcement should be **excluded**: implied vol peaks, options hedging flow
dominates the tape, and a reversal signal has nothing to say about mechanical
delta-hedging. The `>= 3` bound appeared to confirm it.

**That was wrong.** Systematically walking the lower bound down — 3, 2, 1, 0 —
improved IR monotonically at every step, and the last step was the largest jump
in the whole exercise. The days I argued were noise are the most valuable days
in the sample.

The earlier apparent confirmation was confounded: when `>= 3, <= 18` beat
`<= 15`, the upper bound moved at the same time. I attributed the gain to the
exclusion when it came from the extension.

**Lesson to state explicitly if asked:** the mechanism story was doing the
reasoning, and the mechanism was wrong. What saved it was testing the parameter
across its whole range rather than stopping once the theory looked confirmed.

---

## 5. Revised mechanism

Two effects are plausibly operating, with different signatures:

**(a) Liquidity-provision premium — days 1-12.** Short-horizon reversal returns
are compensation for taking the other side of urgent trades and holding
inventory. That compensation is a *price*, set by supply and demand for
immediacy. Approaching a scheduled announcement, other mean-reversion traders
reduce inventory — nobody carries a position into a binary event — while demand
for immediacy rises as institutions reposition. Supply falls, demand rises, the
fee widens. This is Nagel's "Evaporating Liquidity" (RFS 2012) result applied
cross-sectionally in event time rather than through the VIX in calendar time.

**(b) Earnings-announcement premium — day 0.** Stocks earn elevated returns
around scheduled announcements as compensation for bearing event risk (Savor &
Wilson). On day 0 the position is taken into the print and earns the reaction.

The +0.025 jump from one day says **(b) is large**. The smooth monotone gain
from 90 → 12 days says **(a) is real too**. Most likely both, with day 0
carrying disproportionate weight per day.

**Not yet isolated.** Running `gate = (days == 0)` alone would decompose this
directly, and that number should be measured before making claims about which
mechanism dominates.

**No look-ahead.** `trading_days_until_next_earnings_announcement` is Delay 0 =
Yes, and "this company reports today" is public in advance. The position is
formed from information available before the announcement.

---

## 6. Human/machine complementarity — the interesting part

The base expression came from a genetic program searching ~1.5M expressions over
daily price and volume data. It reached IR 0.063 and was **structurally unable
to do better**, because the local panel used for the search contained no
earnings data at all. The GP could optimise the signal; it could not discover
*when* to apply it.

Adding the event-time conditioner doubled the IR. Tuning the GP's own internal
windows — the thing the GP was actually good at — added almost nothing (+0.006),
confirming it had already found near-optimal parameters within its search space.

So the division of labour was clean: **the machine found the signal, the human
supplied the conditioning variable it could not see.** That is a more defensible
account of a GP-derived alpha than "the search found it," and it is honest about
what the search did and did not do.

---

## 7. Predictions and falsification tests

The mechanism constrains what should happen next. Cheap to run:

1. **`gate = (days == 0)` alone.** Decomposes the two effects. If IR is very
   high on that single day, this is primarily an announcement alpha.
2. **`gate = (days >= 1) * (days <= 12)`.** The complement. If it still clears
   0.07, effect (a) stands independently.
3. **FOMC window.** Macro event risk, entirely different data. If
   `(days_until_next_fomc_meeting <= 12)` also lifts IR, the mechanism
   generalises beyond earnings and the result is considerably more interesting:
   ```python
   gate = (days_until_next_fomc_meeting >= 0) * (days_until_next_fomc_meeting <= 12)
   ```
4. **High-VIX conditioning.** Nagel's direct result. Effect (a) should be
   stronger when liquidity is scarcest.
5. **Decay over 2006-2022.** If liquidity provision has become more competitive,
   the effect should weaken. Worth knowing before claiming durability.

**Falsification — the test an interviewer will reach for:**

- Gate a signal with **no** liquidity-provision character (pure momentum, or a
  fundamental factor) on the same window. The theory says it should not benefit.
- Gate on a **random** window keeping the same fraction of stock-days. If IR
  improves similarly, the gain came from reducing position count or turnover,
  not from earnings.

Neither has been run. If asked whether the result is robust, the honest answer
is that these two tests are the ones that would settle it.

---

## 8. Honest weaknesses

- **Measured in-sample** on the platform's visible 2006-2022 window. The
  competition's out-of-sample block is the real test.
- **~12 parameters were tuned** on that same window. The gate response is
  monotone over a wide range, which argues against pure curve-fitting, but the
  internal windows were fitted on far smaller gradients (+0.001 to +0.002 each)
  and several of those are probably noise.
- **Concentration risk.** One day out of nineteen supplies a quarter of the
  improvement. If day 0 is unusable in production — reporting-time uncertainty,
  execution constraints around the print — much of the edge goes with it.
- **Transaction costs.** Turnover is modest, but names within days of reporting
  carry the widest spreads. The cost of trading is highest exactly where the fee
  is highest, and that has not been modelled.

---

## 9. Literature

- **Nagel (2012), "Evaporating Liquidity," RFS** — reversal returns as
  compensation for liquidity provision, predictable by expected volatility. The
  anchor for effect (a).
- **Savor & Wilson** — earnings announcement premium; elevated returns around
  scheduled announcements as compensation for event risk. The anchor for (b).
- **Bernard & Thomas (1989)** — post-earnings announcement drift; the
  after-the-fact counterpart to this before-the-fact window.
- **Gervais, Kaniel & Mingelgrin (2001), JoF** — high-volume return premium;
  Trexquant's own tutorial uses it as its worked case study.
