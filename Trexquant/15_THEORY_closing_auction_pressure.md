# Theory note — closing-auction pressure reversal

*Interview reference. Companion to `14_THEORY_earnings_window_gate.md`.*

---

## 1. The alpha

```python
push = (close - vwap_last_hour) / at_zero2nan(vwap_last_hour)
z    = push / at_zero2nan(ts_std(push, 20))
far  = (trading_days_until_next_earnings_announcement > 10)

alpha = at_zero2nan(-np.tanh(z) * far)
```

Three steps: measure how far the close printed from the final hour's
volume-weighted average price, normalise that by the stock's own recent
dispersion, and short the result — away from earnings.

---

## 2. Mechanism

**The closing auction is where mechanical flow concentrates.** Index funds
must print at the official close to track NAV. ETF creation/redemption settles
there. VWAP and MOC algorithms finish there. Rebalancing trades are timed there.
None of that flow is expressing a view on the stock — it is compelled by a
mandate or a benchmark.

When that flow is one-sided, it pushes the closing print away from where the
stock actually traded through the final hour. `close - vwap_last_hour` measures
exactly that displacement. Because the flow is uninformed, the displacement
carries no information about value, and price returns toward the hour's true
average. Shorting the displacement collects that reversion.

**Why divide by `ts_std(push, 20)`.** The raw displacement is not comparable
across stocks — a volatile name deviates from its VWAP more in normal
conditions than a quiet one. Dividing by each stock's own 20-day dispersion
converts it into "how unusual is today's push *for this stock*", which is the
quantity that should predict reversion. This is the same volatility
normalisation that lifted alpha101_019 over the gate.

**Why `tanh`.** Bounded. A stock with a 10-sigma push should not receive ten
times the position of a 1-sigma push — the reversion is not linear in the
displacement, and unbounded weights concentrate the book. `tanh` saturates.

**Why gate away from earnings (`> 10`).** This is the interesting part, and it
runs *opposite* to the gate in note 14. There the alpha wanted to be near
earnings; here it wants to be far away.

The reason is that the two alphas monetise different things. Note 14's alpha is
a liquidity-provision strategy — it is paid more when providing liquidity is
risky, which is near an announcement. This one is **betting that a price move is
uninformed**. Near an announcement, late-day buying may well be informed:
someone positioning ahead of a print, or reacting to a pre-announcement leak.
Fading informed flow loses money. So the gate excludes the window where the
"this flow is mechanical" assumption is least safe.

**The two gates being opposite is evidence, not inconsistency.** If both alphas
had wanted the same window, the gate would more likely be picking up a generic
volatility or liquidity filter. That they pull in opposite directions, each in
the direction its own mechanism predicts, is what you would expect if the gates
are capturing something real.

---

## 3. Provenance — what is mine and what is not

Stated precisely, because the distinction matters in an interview.

**The expression is original.** It was constructed from Trexsim's data
description — `vwap_last_hour` exists in `GROUP_intra1` — plus general
microstructure reasoning about what the closing auction is. It is not a
transcription of any published factor.

**The underlying phenomenon is well documented**, which I only verified after
writing it. That is a strength, not a problem: it means the idea has independent
support rather than being a data-mining artifact.

- **Baltussen, Da & Soebhag, "End-of-Day Reversal"** — closing-price deviations
  reverse almost fully overnight, net of the half-spread.
- **"Who Trades at the Close? Implications for Price Discovery and Liquidity"**
  — market-on-close imbalance impact is large and transitory. The decile with
  the largest buy imbalances beats the largest sell imbalances by ~32bp into
  the close, and roughly **83% reverses over the next 3-5 days**. A long/short
  strategy on the imbalance earns ~13.2bp per day.
- **Wu, "Closing Auction, Passive Investing, and Stock Prices"** — passive funds
  trading at the close to track NAV create transient mispricing.

**The important distinction from Alpha101.** Building a factor on a documented
*phenomenon* is ordinary quant work. Copying a published *formula* is not the
same activity. This alpha uses no published expression, and it uses a data field
(`vwap_last_hour`) that did not exist in the Alpha101 dataset, so it cannot
collide with another competitor's Alpha101 submission under the
`max corr (others) < 0.90` check.

**What cannot be claimed:** that nobody has ever built this. Closing-auction
reversal is actively traded by many desks. Originality here means "constructed
independently, not copied", not "never done before".

---

## 4. Relationship to the note-14 alpha

| | note 14 alpha | this alpha |
|---|---|---|
| Signal | range position + relative volume | close vs last-hour VWAP |
| Data group | backoffice | **intra1** |
| Mechanism | paid for providing liquidity into event risk | fading uninformed mechanical flow |
| Earnings gate | **near** (0-12 days) | **far** (>10 days) |
| Horizon | days | overnight |

Different signal, different data group, different mechanism, and near-disjoint
trading days. They should be strongly decorrelated — worth confirming with
Compute Correlation, but the prior is that both qualify.

---

## 5. Open questions

- **Does `vwap_last_hour` capture enough of the auction?** It is the final
  hour's VWAP, not the auction print itself. The literature measures the
  imbalance directly; this is a proxy. A cleaner version would use
  `close - close_price_last_hour`, isolating the last trade against the hour.
- **Is the 20-day normalisation window right?** Untested. 10 and 60 are worth
  a run each.
- **Does it work better on larger or smaller names?** The literature finds
  NYSE closing auctions less efficient than Nasdaq, which suggests venue and
  size effects worth conditioning on.
- **Would the reverse gate work as a second alpha?** `<= 10` trades a disjoint
  day set. If late-day flow near earnings is *informed*, the sign should flip
  and the same expression with `+np.tanh(z)` becomes a separate submission.
