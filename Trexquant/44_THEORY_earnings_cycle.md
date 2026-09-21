# Theory note — earnings cycle position, low-volatility half

*Interview reference. Follows `14_`–`18_`, `30_`, `38_`, `43_`.*

---

## 1. Final specification

```python
fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 15)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)
earn_pre  = (trading_days_until_next_earnings_announcement <= 15)
earn_post = (trading_days_until_next_earnings_announcement >= 12)

stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0) \
       + (earn_pre * 1.0) + (earn_post * 1.0)
w      = (stress > 0) * (1.0 + stress)

vl    = cs_rank(ts_std(ret1, 60))
lovol = (vl <= 1.5)

du    = trading_days_until_next_earnings_announcement
score = -(cs_rank(du) - 1.0)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)
```

**SUBMITTED 2026-08-12.**

| Metric | Value |
|---|---|
| IR | *(fill from sim)* |
| TVR | |
| Return | |
| Drawdown | |
| IR/√TVR | |
| Booksize | |
| NumStks | |
| Liq / LiqN | |

*Rejected sibling for reference:* close-vs-typical-price on the same gate and
split reached IR 0.090 / TVR 0.632 / Ret 0.075 but **failed the 50% correlation
check** and was not submitted.

---

## 2. Mechanism

**Long stocks approaching an earnings announcement, short stocks far from one.**

This is the **earnings announcement premium** (Savor & Wilson): stocks earn
elevated returns in the window around scheduled announcements as compensation
for bearing event risk. The risk is undiversifiable at the stock level and
concentrated in time, so someone holding through the print must be paid for it.
Ranking the whole cross-section by position in the reporting cycle harvests that
premium continuously — some subset of names is always approaching a report.

**Why this is a characteristic, not a forecast.** The alpha holds no view on
whether any company will beat or miss. It expresses only that bearing scheduled
event risk is compensated. That is the same structural shape as the alphas that
have worked here (M4's unsigned efficiency, A2's skew, P8's depth) and unlike
the directional forecasts that failed uniformly.

**Relationship to note 14.** `trading_days_until_next_earnings_announcement` has
been the single most valuable variable in this project — walking its window as a
*gate* was worth +0.061 there, against +0.006 from every internal window
combined, and adding day 0 alone gave +0.025. **This is the first time it has
been used as a signal rather than a conditioner**, and the fact that it works in
both roles is evidence the underlying effect is real rather than a gating
artifact.

---

## 3. Low volatility wins again — third occurrence

| alpha | split | result |
|---|---|---|
| S8 | lovol | 0.094, beat hibet 0.076 |
| P2 | lovol | 0.106, beat small/large/hivol/lobet/hibet |
| **this** | **lovol** | **beat hivol** |

Three independent cores — a GP-derived reversal expression, a mandated-flow
weighted variant, and a pure calendar-position signal — and the low-volatility
half wins every time.

This is now the most replicated finding in the project, and it is **not** what
the original thesis predicted. Note 38 documented the first falsification;
this is the third instance. The accumulated evidence:

- volatility-regime gating in calendar time (gate B) added nothing to any core
- the high-volatility cross-sectional half loses to the low-volatility half in
  three separate alphas
- the calendar event gate has added IR to every core it has touched

**Reading:** the premium is attached to *mandated, price-insensitive flow*, not
to volatility-driven liquidity withdrawal. Low-volatility mega-caps are where
index, ETF and pension flow is the largest share of total volume, and that is
where fading forced flow pays best.

The Nagel attribution in notes 14, 16 and 18 remains partially wrong and should
be presented that way (see note 38 §2.3 for the replacement wording).

**Caveat worth stating:** the three splits are not independent tests of the
mechanism. All three cores are gated the same way and evaluated on the same
universe, so a common factor could produce the pattern. It is suggestive rather
than decisive.

---

## 4. A construction error in the gate

```python
earn_pre  = (du <= 15)
earn_post = (du >= 12)
```

**These overlap and together span every possible value of `du`.** The pair
contributes a constant **+1 to `stress` on every stock, every day**, plus **+2**
in the narrow 12–15 day band. It is not a pre/post decomposition.

The intended construction was `earn_post = (du >= 55)`. The counter resets to
~62 the day after a report, so a high value identifies names that have **just
reported** — the post-earnings-announcement-drift window (Bernard & Thomas),
a mechanism distinct from the pre-announcement premium.

**Consequences:**
- The gate's earnings contribution is nearly a constant, which after booksize
  normalisation does almost nothing.
- The 12–15 day bump is the only live earnings term, and it sits nowhere near
  day 0 — the day note 14 identified as carrying a quarter of the entire gain.
- **The PEAD window is not covered at all**, despite the note claiming it is.

**Untested upside.** Re-running with `earn_post = (du >= 55)` gives a genuine
two-window structure. Given how much the earnings variable has been worth
elsewhere, this is the highest-expected-value single change available to this
alpha.

Recorded rather than quietly fixed because the submitted version is the one with
the error in it, and the note must describe what was actually run.

---

## 5. Honest weaknesses

- **This is the least original alpha in the pool.** The core is
  `-cs_rank(days_to_earnings)` — one line, and the announcement premium is a
  well-known published result. What is mine is the gate and the volatility
  split, not the signal.
- **`max corr (others)` is a live risk.** It is simple enough that other
  competitors will plausibly have submitted something close. Worth checking if
  the platform exposes that figure.
- **The signal and the gate use the same variable.** `du` appears in both
  `score` and in `earn_pre`/`earn_post`. Because the gate's earnings terms are
  nearly constant (§4) the interaction is small, but it would not be if the
  gate were fixed — and the fix in §4 would introduce a real interaction that
  needs re-checking rather than assuming.
- **Metrics not yet recorded.** Table in §1 is incomplete.

---

## 6. Interview preparation

### 6.1 The 60-second walkthrough

> "It ranks the cross-section by how close each stock is to its next earnings
> report — long the ones about to announce, short the ones that just did —
> restricted to the low-volatility half of the universe and weighted by a
> ten-term calendar event gate. The mechanism is the earnings announcement
> premium: bearing scheduled, undiversifiable event risk is compensated, and
> since some subset of names is always approaching a report, you can harvest it
> continuously. It's the simplest alpha I submitted, and the same variable
> driving it was also the most valuable conditioner in my other work — which is
> some evidence the effect is real rather than a gating artifact."

### 6.2 Question bank

**"This is one line. Where's the work?"** In the conditioning, not the signal.
The raw ranking is a known published effect; the ten-term event weight and the
low-volatility restriction are mine, and the volatility split is what took it
from marginal to submittable. I would not claim the signal as a discovery.

**"Why the low-volatility half?"** §3 — and this is the more interesting answer.
It is the third alpha where low volatility beat high, which contradicts the
liquidity-provision thesis I started from and supports a mandated-flow account
instead. Lead with the fact that it falsified my own prior.

**"Isn't the announcement premium arbitraged away by now?"** A fair challenge,
and I have not run the decay test — splitting 2007-2021 into halves and
comparing would settle it, and it is the first thing I would check. The effect
persisting in a large-cap sample through 2021 is weak evidence against full
arbitrage, but weak is the right word.

**"Your two earnings windows overlap."** Yes — §4. They span every value of the
counter, so they contribute a near-constant rather than a pre/post
decomposition, and the PEAD window I intended to capture is absent. I found it
writing this up rather than before submitting.

### 6.3 The question I would ask if I were interviewing me

*"You used this variable as a gate in one alpha and as a signal in another. Are
those two alphas independent, or are you selling me the same effect twice?"*

They are not fully independent, and the correlation check only establishes that
the finished books differ by less than 50% — it does not decompose *why*. The
gate version conditions a reversal signal on event proximity; this version
trades event proximity itself. If the reversal premium near announcements is
simply the announcement premium showing up in a reversal book, then they are the
same effect in two costumes. The test I would run: strip the earnings terms from
the gated alpha entirely and see how much of its IR survives. That number is not
currently in any of my notes.

---

## 7. Literature

- **Savor & Wilson** — the earnings announcement premium; elevated returns
  around scheduled announcements as compensation for event risk. The anchor.
- **Bernard & Thomas (1989)** — post-earnings announcement drift; the
  after-the-fact counterpart, and the window §4 fails to capture.
- **Frazzini & Lamont, "The Earnings Announcement Premium and Trading Volume"**
  — announcement-return persistence at the firm level; the natural extension of
  this alpha and untested here.
- **Nagel (2012)** — the original thesis for the gate, partially rejected by the
  evidence in §3.
