# Method note — diagnosing a fast/slow divergence

*Interview reference. Not about one alpha — about a class of bug, how it was
found, and what it implies for every expression in the pool.*

---

## 1. What happened

The liquidity-resilience family (file 40) produced strong fast-mode results and
**failed badly in slow mode**. Same expression, same data, same window.

The framing matters. This is **not a market phenomenon.** It is a data artifact,
and calling it anything else would be the mistake. What makes it worth writing
up is not that the numbers moved — it is that the divergence is *diagnosable*,
that the cause has a specific and non-obvious mechanism, and that the same
mechanism silently threatens other alphas in the pool.

**In an interview this is a bug-hunting story, not a discovery story.** That is
the stronger of the two. Anyone can report a backtest; being able to say "my
alpha looked excellent, I did not believe it, here is exactly why it was wrong"
is the part that distinguishes a researcher from a backtest operator.

---

## 2. The three causes, and ruling two out

Trexsim's fast mode computes the whole panel at once; slow mode recomputes day
by day using only information available at that date. Three things cause them
to disagree:

| cause | applies here? |
|---|---|
| **1. Explicit look-ahead** — a negative `ts_delay`, or any forward reference | **No.** No negative delays anywhere in the expression. |
| **2. Insufficient warm-up** — long windows consuming sample | **No.** 250-day windows shift the start date, but slow mode consumes warm-up correctly and the shift was accounted for. |
| **3. Retrospective corporate-action adjustment** | **Yes.** |

---

## 3. The mechanism

Splits and dividends are applied **retrospectively**, and the adjustment
convention differs by field:

| field | convention |
|---|---|
| `close`, `open`, `high`, `low`, `vwap_*` | **DIV** — divided by the split factor |
| `volume`, `number_of_trades` | **MUL** — multiplied by it |
| returns, `skew_*`, `information_ratio_*` | already invariant |

In fast mode the whole history carries adjustments as of the *end* of the
sample. In slow mode each day carries only the adjustments known by that date.

So an expression is safe **only if its adjustment exposure cancels**:

| expression | exposure | safe? |
|---|---|---|
| `close * volume` | DIV × MUL | **invariant** |
| `(high - low) / close` | DIV / DIV | invariant |
| `volume / ts_mean(volume, N)` | MUL / MUL | invariant unless a split falls inside the window |
| `ts_sum(vol·s) / ts_sum(vol)` | MUL / MUL | invariant — **P2's `mflow` is fine** |
| `ts_sum(vol·s) / ts_sum(\|ret\|·s)` | MUL / none | **exposed — the resilience family, and P8** |

The resilience measure divides **share volume** (MUL-adjusted) by **absolute
return** (invariant). Nothing cancels. Its value in fast mode is scaled by every
split the stock will ever undergo.

---

## 4. Why the error is not neutral — the important part

A scaling artifact would be harmless if it were random. It is not.

**Companies split their stock after the price has risen.** A split factor is
therefore a function of *future* returns. Any expression whose value depends on
the adjustment level rather than being invariant to it is reading a variable
that encodes what the stock is going to do.

This is a genuine look-ahead. It arrives through the *data* rather than through
a forward reference in the expression, which is exactly why it survives visual
inspection — there is nothing wrong with the code.

**Concretely:** Apple split 7:1 in 2014 and 4:1 in 2020. In fast mode its
2007–2014 share volume is multiplied by **28**. A measure of "volume per unit of
price move" therefore ranks Apple enormously high across the whole early
sample — and Apple was one of the best-performing stocks in it. Repeat across
every large-cap splitter in a 2007–2021 universe and the ranking acquires a
systematic tilt toward future winners.

Fast mode does not just mis-scale the signal. It hands it the answer.

---

## 5. Pool audit

| alpha | exposure | status |
|---|---|---|
| A2 closing-hour skew | skew is invariant | safe |
| A3 vol-regime coupling | returns and price ratios | safe |
| A1 / S8 core | `relvol = volume / ts_mean(volume, 20)` — MUL/MUL | safe, small residual risk if a split lands inside 20 days |
| D2 adaptive blend | ranks and returns | safe |
| P2 mandated-flow weight | `mflow` is MUL/MUL | **safe** |
| **P8 cheap absorption** | `vol / \|ret\|` | **exposed — do not submit on fast-mode numbers** |
| L1–L11 resilience family | same | exposed; explains the failure |

P8's fast-mode figures (IR 0.096, TVR 0.042, break-even 73bp) should be treated
as unverified until re-run with the fix.

---

## 6. The fix

Use **dollar volume**, where DIV × MUL cancels exactly:

```python
dv    = close * volume
cheap = ts_sum(dv * stress, 250) / at_zero2nan(ts_sum(absr * stress, 250))
```

The repaired quantity is also economically better defined. Dollars traded per
unit of price move is **Kyle's lambda inverted** — a standard price-impact
measure. Shares per unit of return was never comparable across stocks at
different price levels in the first place, so the invariance fix and the
economic fix are the same change.

---

## 7. The general rule

> **Every ratio must have matching adjustment exposure in numerator and
> denominator. Mixing a MUL-adjusted field with an invariant one creates a
> look-ahead that no amount of reading the code will reveal.**

This generalises the note-16 observation that "dimensionless expressions
mitigate" fast/slow divergence. Dimensionless is not quite the right test —
`volume / |ret|` has consistent units in a loose sense but mismatched
adjustment exposure. **Adjustment-invariant** is the correct criterion.

---

## 8. Interview framing

**Lead with the process, not the artifact.**

> "One family of alphas looked excellent in fast mode and collapsed in slow
> mode. There are three documented causes of that divergence — look-ahead,
> warm-up, and retrospective corporate-action adjustment. The first two didn't
> apply, so I looked at the third. The expression divided share volume by
> absolute return. Volume is multiplied by the split factor retroactively and
> returns aren't, so nothing cancelled — and since companies split after their
> price rises, the split factor is a function of future returns. My alpha was
> reading tomorrow's news through the adjustment factor, with no forward
> reference anywhere in the code. Fixed it by switching to dollar volume, where
> the price and volume adjustments cancel exactly."

**Anticipated follow-ups**

*"How do you know that's the cause and not something else?"* — I don't, with
certainty. The other two causes are ruled out structurally, the mechanism
predicts the direction of the error correctly, and the fix is testable. If the
divergence survives the dollar-volume fix, the next suspect is the `quiet`
denominator: at the current loose gate windows, non-event days are sparse, so
`ts_sum(|ret| · quiet, 250)` may rest on few observations and be unstable.

*"Why did you run fast mode at all?"* — Throughput. Fast mode is the screen;
slow mode is the test. The discipline is that no result is believed until slow
mode confirms it, and this is the case that justifies the discipline.

*"What else in your book has this problem?"* — §5. One submitted alpha is
exposed and I have flagged it rather than leaving it in on unverified numbers.

**What not to say:** that this is an interesting market phenomenon. It is a
data-handling artifact, and describing it otherwise would suggest I had not
understood it.

---

## 9. What this says about fast mode generally

Fast/slow agreement has been treated in earlier notes as a look-ahead check —
note 16 cited a 3% gap as evidence of cleanliness for the event-gated reversal
alpha. That reading holds, but this case sharpens it: **fast mode is not merely
an approximation of slow mode, it is a different information set.** Retro-
adjusted data is data from the future, and any expression sensitive to the
adjustment level is trading on it.

The practical consequence is that fast mode is safe for expressions built from
returns, ranks and same-convention ratios, and unsafe for anything that mixes
conventions. That is a sharper rule than "check fast against slow", because it
says *in advance* which expressions need checking.

---

## 10. Literature

- **Kyle (1985), "Continuous Auctions and Insider Trading," Econometrica** —
  lambda as price impact per unit of order flow; the correct form of the
  repaired measure.
- **Amihud (2002), "Illiquidity and Stock Returns," JFM** — the |return| /
  dollar-volume illiquidity ratio, which uses DOLLAR volume precisely because
  share counts are not comparable across stocks.
- **Fama, Fisher, Jensen & Roll (1969)** — the original split event study, and
  the source of the fact that makes this bug directional: splits follow price
  appreciation.
