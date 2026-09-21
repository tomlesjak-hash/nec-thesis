# Theory note — vol-regime coupling minus MA reversal, event-weighted

*Interview reference. Fifth in the series with `14_`, `15_`, `16_`, `17_`.
All figures measured on Trexsim.*


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

core = (ts_max(ts_mean(ts_max(ts_corr_binary(
            (ts_mean(ts_std(ret1, 20), 20) - ts_zscore(ts_std(ret1, 20), 10)),
            ret20, 60), 10), 20), 5)
        - (close / at_zero2nan(ts_mean(close, 20))))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)
```

**SUBMITTED 2026-08-11.** Third qualified alpha.

| Metric | Value |
|---|---|
| Window | 2007-03-15 → 2021-12-30 |
| **IR** | **0.073** |
| **TVR** | **0.251** |
| Return | 0.071 |
| Drawdown | 11.8 |
| **IR/√TVR** | **0.145** |
| Booksize | 0.91 x 0.91 |
| NumStks | 712 x 693 |
| Liq / LiqN | **608 / 356** |

---

## 2. Why this is the best of the three, despite the lowest IR

| | A1 reversal | A2 skew | **A3 this** |
|---|---|---|---|
| Daily IR | 0.091 | **0.102** | 0.073 |
| Annualised Sharpe | 1.44 | **1.62** | 1.16 |
| Annual return | **12.5%** | 4.9% | 7.1% |
| Implied annual vol | 8.65% | **3.03%** | 6.13% |
| Max drawdown | 10.5 | **4.7** | 11.8 |
| Calmar | **1.19** | 1.04 | 0.60 |
| Turnover / year | 186× | 348× | **63×** |
| **Break-even cost** | 6.7 bp | 1.4 bp | **11.2 bp** |
| **IR/√TVR** | 0.105 | 0.086 | **0.145** |

**Net of realistic costs it is the only one that is comfortably viable:**

| round-trip cost | A3 net annual return |
|---|---|
| 1 bp | +6.5% |
| 2 bp | +5.8% |
| 5 bp | +3.9% |
| 10 bp | +0.8% |
| 15 bp | −2.4% |

Still profitable at 10bp. A2 is underwater at 2bp. On 63× annual turnover in the
most liquid names in the book (LiqN 356, nearly double either other alpha), this
is the one that would actually survive contact with a market.

**It is also the only one that can invoke the correlation override.** At
IR/√TVR 0.145 against A1's 0.105 and A2's 0.086, if this alpha ever collides
above 50% correlation with either, it clears the "≥ 20% higher IR/√TVR"
exemption comfortably — 0.145 is 38% above A1 and 69% above A2. The other two
have no such protection.

**The honest counterweight:** lowest Sharpe, highest drawdown, worst Calmar
(0.60 — it loses more in a drawdown than it makes in a year). It is the most
*tradeable* alpha and the least *comfortable* one to hold.

---

## 3. The GP result that finally worked

Note 14 §6 made a claim that had not yet been demonstrated:

> *the machine found the signal, the human supplied the conditioning variable it
> could not see.*

At the time that was an account of one alpha whose GP core scored 0.063 and
needed hand-tuning plus a gate to reach 0.124. Across two GP runs — ~1.57M
expressions, 28 locally-qualifying candidates — **zero cleared 0.07 on the
platform unaided**, and roughly fifty were tested.

**This is the first GP core to qualify with nothing but a gate bolted on.** No
internal window was touched; the expression is exactly as the search emitted it.
The gate windows were tuned, the core was not.

That converts the note-14 claim from a story about one alpha into a repeatable
procedure: *search for structure on daily bars locally, then condition on
event-time variables the local panel never contained.* The local panel has no
FOMC dates, no expiration calendar, no earnings dates — the GP could not have
found the gate if it had run for a year.

---

## 4. Reading the core

Two terms, subtracted.

### Term B — `close / ts_mean(close, 20)`

Price relative to its own 20-day average, **subtracted**, so the alpha is short
stocks trading above their moving average and long those below. Textbook
short-horizon mean reversion, and dimensionless by construction. Nothing novel;
it is the workhorse the GP reached for.

### Term A — the vol-regime coupling

Build it inside out.

```python
v = ts_std(ret1, 20)                    # 20-day realised volatility
X = ts_mean(v, 20) - ts_zscore(v, 10)   # vol LEVEL minus vol SURPRISE
```

`ts_mean(v, 20)` is the slow-moving **level** of a stock's volatility — its
volatility regime. `ts_zscore(v, 10)` is how far today's volatility sits from
its own recent history in standard-deviation units — a volatility **innovation**.
Subtracting one from the other produces a variable that is high when a stock is
*persistently* volatile but *not currently spiking*, and low when volatility is
spiking relative to its own norm.

That decomposition is the interesting part. Realised volatility conflates two
economically distinct things: how risky a stock is, and whether something just
happened to it. Term A separates them and keeps the persistent component net of
the shock.

```python
c = ts_corr_binary(X, ret20, 60)
```

The rolling 60-day correlation, **per stock**, between that vol-regime variable
and trailing 20-day returns. This is a stock-level coefficient, not a signal
value: it measures *whether this particular stock's returns co-move with its own
volatility regime*. High positive correlation means the name rallies as its vol
regime builds and falls as it calms — a vol-feedback or inverse-leverage
signature. Near zero means returns and volatility regime are unrelated.

```python
ts_max(ts_mean(ts_max(c, 10), 20), 5)
```

Then three nested smoothing/extremum operators. `ts_max` over 10 days, mean over
20, max over 5.

**Honest assessment of this part:** the first `ts_max(c, 10)` has a defensible
reading — it asks whether the coupling has been strong *at any point* recently,
which makes the flag persist once tripped rather than flickering as the
correlation wobbles. The outer `ts_mean(20)` then `ts_max(5)` is much harder to
justify economically, and a human would not write max-of-mean-of-max. It is most
plausibly a search artifact: the GP found a smoothing stack that happened to fit,
and some of it is noise. **I would not claim otherwise in an interview.** The
testable version is to replace the three operators with a single `ts_mean(c, 20)`
and see how much IR is actually lost — that experiment has not been run and it
is the first thing I would do.

### What the combination does

Long stocks whose returns have recently coupled to their own volatility regime,
short stocks extended above their moving average. The plausible economic reading
is that the first term identifies names where volatility is being *paid for* —
where risk-bearing is currently compensated rather than merely present — and the
second term supplies the reversal timing. It is a conditional reversal: fade the
extension, but preferentially in names whose risk is being rewarded.

**Stated with the right confidence:** this reading was constructed after the fact
from an expression a search produced. It is a hypothesis about why the alpha
works, not the reasoning that generated it, and the distinction matters.

---

## 5. The tuning log — what the gate search actually showed

Every window walked one at a time, IR / return / TVR recorded at each step.

| knob | values tried | chosen | note |
|---|---|---|---|
| `opex` | 5, **10**, 12, 15 | 10 | 0.061 / **0.065** / 0.057 / 0.057 — clean interior peak |
| `qopex` | **1**, 5 | 1 | 5 gave no change |
| `qend` | 7, **10**, 12, 4 | 10 | 0.065 / 0.064 / 0.063 / 0.063 |
| `mend` | 2, 5, 10, **15** | 15 | 0.064 → 0.068 → 0.069 → **0.070**, monotone |
| `mstart` | **0**, 5 | 0 | 5 worse |
| `holiday` | 7, 3, **1** | 1 | 0.070 → 0.071 → **0.072**, monotone |

### Finding 1 — widening `mend` improved IR *and* cut turnover by a third

`mend` from 2 to 15 days raised IR 0.064 → 0.070 **while TVR fell 0.348 →
0.226**. Two metrics improving together, which note 16 argued is the signature
of a conditioner adding information rather than filtering noise.

But the turnover half has a purely mechanical explanation worth understanding,
because it generalises: **a narrow gate creates turnover by switching off.**
Every time `stress` drops to zero the book liquidates, and every time it comes
back the book re-enters — round trips that have nothing to do with the signal
changing. Widening the window reduces the number of on/off transitions, so
turnover falls. Gate width therefore trades selectivity against transition
turnover, and at 63× annual turnover this alpha is sitting on the good side of
that trade.

### Finding 2 — the "gate" is no longer gating

`mend <= 15` covers roughly 71% of trading days and `opex <= 10` about half.
`stress` is nonzero almost always: booksize 0.91 and `numstk` 712 confirm the
book is near-fully invested. The `(stress > 0)` term excludes very little.

So what survives is `(1 + stress)` — an **event-intensity weighting scheme**,
not a filter. It says *scale up on days when several mandated-flow events
coincide*, and the exclusion is incidental. The same drift happened to A2.

This matters for how the result is described. The note-16 alpha genuinely gates:
40% of days, `numstk` 365, booksize 0.48. Calling what A2 and A3 do a "gate" is
inaccurate, and an interviewer who looks at the booksize will notice.

### Finding 3 — `fomc_pre` was measured at the wrong point in parameter space

The `fomc_pre` results (1 → 0.057, 5 and 0 both worse) were recorded when total
IR was ~0.057, i.e. **before** the other seven windows were tuned. Every later
knob was measured at 0.065–0.072. The final expression uses `<= 0` — the value
those early measurements rejected.

This is exactly the confounding documented in note 14 §4, where the `>= 3`
earnings bound looked confirmed because the upper bound had moved at the same
time, and walking it down to 0 later produced the largest single gain in that
exercise. **Coordinate descent gives no guarantee that an early decision remains
optimal after later knobs move.** `fomc_pre` should be re-walked at the current
settings before this alpha is described as tuned, and I would expect a small gain.

---

## 6. Honest weaknesses

- **Sits on the qualification line.** IR 0.073 against a 0.07 threshold, with
  eight gate windows fitted on the visible sample. A2 at 0.102 has real headroom;
  this does not. Out-of-sample degradation of 5% disqualifies it.
- **Worst drawdown-adjusted profile of the three.** Calmar 0.60 means a bad
  drawdown costs more than a good year returns.
- **The core is not fully interpretable.** Three nested extremum operators, at
  least one of which is probably fitting noise. Not run: the ablation replacing
  them with a single `ts_mean`.
- **`ts_corr_binary` semantics assumed.** The operator doc describes a per-stock
  rolling correlation and notes it yields market beta when passed `close_spx`.
  The interpretation above depends on that being a plain Pearson correlation over
  the window; the name "binary" is not explained anywhere in the documentation
  and I have not verified what it does with NaNs or with constant inputs.
- **`fomc_pre` untuned at current settings** (§5, finding 3).
- **Same intra1-era sample** (2007-03 onward) as A2, so drawdown is again not
  comparable to A1's — except that here the sample restriction comes from the
  gate calendar, not from the core, since the core uses only daily bars. That
  asymmetry is unexplained and worth checking.

---

## 7. Interview preparation

### 7.1 The 60-second walkthrough

> "This one came out of a genetic program I ran over daily price and volume data
> — about 1.5 million expressions across two runs. The core has two pieces: a
> per-stock rolling correlation between the stock's volatility *regime*, net of
> volatility *shocks*, and its trailing return — and, subtracted from it, price
> relative to its 20-day average. So it's a conditional reversal: fade extended
> names, preferentially where the stock's risk appears to be getting paid for.
> Then I weight the book by how many mandated-flow events are live that day.
> IR is 0.073, but the number I'd point at is turnover — 0.25, about 63× the book
> a year, which means it breaks even at 11bp of cost and is still making money at
> 10. It's the lowest-IR and by a distance the most tradeable of my three."

### 7.2 Question bank

**"Your IR is 0.073 against a 0.07 bar. Isn't that just noise?"**
It is the weakest of my three on that metric and it has no headroom — I would not
defend it as a standalone discovery. What makes it worth submitting is the
turnover: at 63× a year against 348× for my highest-IR alpha, it is the only one
whose gross IR survives translation into net. A portfolio wants both, and they
are not substitutes.

**"Explain the core."** — §4. Lead with the volatility level/innovation
decomposition, which is the genuinely interesting part, and **volunteer that the
nested max-mean-max stack is probably partly search artifact** before they ask.
Naming your own expression's weakest component is much stronger than defending it.

**"How much of this is you and how much is the search?"**
The search found the core; I did not modify a single window in it. I supplied
the event conditioner, which the search could not have found because my local
panel contains no FOMC dates, no expiration calendar and no earnings dates.
Across two runs and ~1.5M expressions, no GP core cleared the platform threshold
unaided — roughly fifty were tested. This is the first that qualified with only
a conditioner added, which is the cleanest evidence I have for that division of
labour.

**"You ran 1.5M expressions and kept the ones that worked. Isn't this mining?"**
Yes, and the honest defence is not that it isn't. It is that the selection was
made on a *different* dataset from the one it is evaluated on — the search ran on
my own 370-name local panel, and every candidate was then tested on the
platform's 1000-name universe with a different sample and different data. In-
sample fitness on the local panel had a rank correlation of **−0.70** with
held-out IR in run 1, so local selection was worse than random. That the
platform-side survivors are so few (this is one of about fifty tested) is
consistent with genuine selection, not with a mined result transferring.

**"Why does the event weighting help a reversal signal?"**
Reversal returns are compensation for supplying liquidity — the fee for taking
the other side of someone's urgent trade. That fee is a price, and it rises when
mandated flow is heavy and other providers step back. Expirations, quarter- and
month-end rebalancing and FOMC repricing are all dates when flow arrives that
expresses no view. Nagel (2012) is the anchor. *Volunteer the caveat:* the
falsification — a random gate keeping the same fraction of days — has not been run
on any of my three alphas.

**"Your booksize is 0.91 and you have 712 names. What is the gate excluding?"**
Almost nothing, and calling it a gate is imprecise on my part. At these windows
`stress` is nonzero on nearly every day; what is doing the work is the
`(1 + stress)` weighting, which scales the book up when several events coincide.
It is an event-intensity weight, not a filter. My first alpha does genuinely
gate — 40% of days, booksize 0.48.

**"Talk me through your tuning."** — §5. The three findings, in order: the `mend`
result where IR and turnover improved together and why that is partly mechanical;
the fact that the gate stopped gating; and that `fomc_pre` was measured before
the other seven knobs moved and is therefore **not reliably tuned** — the same
confounding that cost me the day-0 finding on my first alpha.

**"Which of your three would you actually trade?"**
This one, on cost. A2 has the best Sharpe at 1.62 but breaks even at 1.4bp, so
standalone it is a paper alpha. A1 is the best risk-adjusted at Calmar 1.19 and
survives 6.7bp. This one survives 11.2bp. In a combined book I would want all
three — they span turnover from 63× to 348× a year and their trades net against
each other, so marginal turnover is well below the sum of standalone turnovers.

### 7.3 The question I would ask if I were interviewing me

*"You've submitted three alphas that share the same eight-event weighting scheme.
How much of your combined performance is three signals, and how much is one gate
applied three times?"*

The honest answer is that I do not know, and it is the right question. The three
cores are genuinely independent — daily-bar reversal, intraday return skew, and a
GP expression on volatility structure — drawn from three different data groups.
But they share a conditioner, and a shared conditioner induces shared
day-structure in the returns whether or not the cores are related. The
correlation check I ran is between finished alphas, which is the gate the
competition applies, but it does not decompose *where* the correlation comes
from. The decomposition I would run: measure each core's IR ungated, then measure
how much the gate adds to each. If the gate contributes most of the performance
in all three, I have one alpha and two decorations.

For the record, the ungated numbers I have are A1 0.055, A2 0.088, A3 unmeasured
— so on the two I can check, the gate adds 0.036 and 0.014 respectively against
cores that stand up on their own. **A3's ungated core has not been measured, and
it is the one where the answer matters most**, since it is the one sitting on the
qualification line.

---

## 8. Literature

- **Nagel (2012), "Evaporating Liquidity," RFS** — reversal as compensation for
  liquidity provision, predictable by expected volatility. The anchor for both
  the reversal term and the event weighting.
- **Ang, Hodrick, Xing & Zhang (2006), "The Cross-Section of Volatility and
  Expected Returns," JoF** — the volatility level/innovation distinction that
  term A's construction recovers, and the reason to expect them to price
  differently.
- **Black (1976); Christie (1982)** — the leverage effect; the return-volatility
  coupling term A measures per stock.
- **Stoll & Whaley** — expiration-day effects from index-derivative unwinding.
- **Ariel (1987); Lakonishok & Smidt (1988)** — turn-of-the-month effects.
