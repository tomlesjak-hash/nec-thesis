# Theory note — A1 core on the low-volatility half

*Interview reference. Follows `14_`–`18_`, `30_`. All figures from Trexsim.*

**This note contains the most important empirical finding in the project. It
falsifies the mechanism claimed in notes 14, 16 and 18.**

---

## 1. Final specification

```python
vl    = cs_rank(ts_std(ret1, 20))
lovol = (vl <= 1.5)

fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 80))
         - (((ts_median(cs_zscore(relvol), 10) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lovol)
```

**SUBMITTED 2026-08-12.** Slow mode, 2007-03-15 → 2021-12-30.

| Metric | Value |
|---|---|
| **IR** | **0.094** |
| TVR | 0.666 |
| Return | 0.071 |
| Drawdown | 12.0 |
| **IR/√TVR** | **0.116** |
| Booksize | 0.96 x 0.96 |
| NumStks | 368 x 367 |
| Liq / LiqN | 392 / 162 |

Derived: annualised Sharpe **1.49**, implied annual vol 4.76%, Calmar 0.59,
break-even cost **4.2 bp**.

**Note the booksize.** At 368 names — half the universe — booksize still reads
0.96, so the platform renormalises the book rather than halving it. Universe
splitting therefore costs coverage but *not* gross exposure, which is a
materially better trade than assumed when the split matrix was designed.

---

## 2. THE FALSIFICATION — read this before any interview

Two splits of the same core were run:

| split | IR |
|---|---|
| high beta | 0.076 |
| **low volatility** | **0.089** → 0.094 tuned |

**The prediction was the opposite.** Every theory note in this project rests on
Nagel (2012): reversal returns are compensation for liquidity provision, and
that compensation is *predictable by expected volatility*. That mechanism makes
an unambiguous prediction — the high-volatility half should carry most of the
IR, because that is where providers withdraw and the fee for immediacy widens.

The split matrix (file 34) stated this in advance: *"reversal pays most where
volatility is highest, so this half should carry most of A1's IR… if the
low-vol half is just as good, the liquidity-provision reading is wrong."*

**The low-volatility half did not merely match. It won by 0.013, and after
tuning it beats the whole-universe parent on IR/√TVR (0.116 vs 0.106).**

### 2.1 The second, independent falsification

This is not one anomalous run. **Gate B — the macro volatility regime
conditioner — never added anything to any core it was applied to.** That was
attributed to correlation at the time. In hindsight the two results say the
same thing from different directions:

- conditioning on high **market** volatility (gate B, calendar time) → no gain
- conditioning on high **stock** volatility (hivol split, cross-section) → loses
  to the low-volatility half

Two independent tests, one in calendar time and one cross-sectionally, both
reject the volatility channel. That is much stronger evidence than either alone.

### 2.2 The revised mechanism

The event gate works. The volatility conditioning does not. Both live under the
heading "liquidity provision", but they are different claims:

- **Volatility channel (Nagel):** providers withdraw when risk rises, so the
  fee widens. Predicts high-vol names pay more. **Rejected twice.**
- **Mandated-flow channel:** the fee is paid by traders who *must* transact
  regardless of price — index funds, ETF create/redeem, MOC algorithms, pension
  rebalancing. Predicts the premium is largest where mandated flow is a large
  share of total volume. **Supported.**

Low-volatility stocks are large, stable, heavily index-weighted names —
utilities, staples, mega-caps. Their idiosyncratic trading is light relative to
the mechanical index and ETF flow they receive, so **mandated flow is a much
larger fraction of their volume**. That is exactly where a flow-fading strategy
should be paid best, and it is what the data shows.

**This also explains, retrospectively, why the eight-event calendar gate has
been the single most reliable lever in the project (+0.014 to +0.036 every
time) while every volatility-based conditioner failed.** The gate is a direct
measurement of mandated flow. Volatility is a proxy for a mechanism that
appears not to be operating.

### 2.3 What this means for the earlier notes

Notes 14, 16 and 18 all cite Nagel as the primary anchor. That attribution is
now **partially wrong** and should be stated as such rather than quietly
dropped. The correct account:

> The strategy monetises mechanical, price-insensitive order flow. The original
> reasoning attributed this to Nagel's volatility-driven liquidity-provision
> channel. Two independent tests — a macro volatility regime gate and a
> cross-sectional volatility split — both reject that channel. The evidence
> instead supports a mandated-flow account, closer to the index-rebalancing and
> expiration-effect literature (Stoll & Whaley; Ariel; Lakonishok & Smidt) than
> to Nagel.

**Being the person who ran the test that broke their own thesis is a far
stronger position than having a tidy story.** Lead with it.

---

## 3. The tuning that mattered

| change | effect |
|---|---|
| `ts_zscore(stoch, 10)` → **80** | ungated split IR to 0.115 |
| `ts_median(cs_zscore(relvol), 20)` → **10** | included in the above |
| `mend 15 → 13`, `holiday 1 → 5` | gate refinement |

**The `ts_zscore` window went from 10 to 80 — an eightfold increase, and by far
the largest single lever in this alpha.** It converts the reversal from a
two-week signal into a quarter-long one: `stoch` is now measured against an 80-
day distribution, so a position is taken only when the range position is
unusual over months rather than over a fortnight.

This is consistent with the pattern across the project — on A2, window tuning
was worth +0.027 against the gate's +0.014. **Internal windows have repeatedly
outperformed gate tuning, and note 14 recorded the opposite conclusion (+0.006
from windows) based on a single alpha.** That earlier generalisation was drawn
from too small a sample.

It is also plausibly *why* this split decorrelated from A1 enough to qualify: a
slower core on half the universe is two independent departures from the parent.

---

## 4. Position in the pool

| | A1 | A2 | A3 | D2 | **S8** |
|---|---|---|---|---|---|
| Sharpe | 1.44 | 1.62 | 1.16 | 1.19 | **1.49** |
| Drawdown | 10.5 | 4.7 | 11.8 | 8.0 | 12.0 |
| Calmar | 1.19 | 1.04 | 0.60 | 0.71 | 0.59 |
| IR/√TVR | 0.106 | 0.087 | **0.146** | 0.080 | **0.116** |
| Break-even | 6.7bp | 1.4bp | 11.2bp | 2.6bp | **4.2bp** |

Second-best Sharpe and second-best IR/√TVR in the pool. Weakest on Calmar —
drawdown 12.0 against 7.1% return.

---

## 5. Honest weaknesses

- **It is half of an alpha already submitted.** It scores as a separate alpha
  and count is half the competition score, so taking it is correct — but it is
  not an independent discovery and must not be presented as one.
- **Drawdown 12.0 on a 7.1% return.** Calmar 0.59, second-worst in the pool.
  Splitting to the low-volatility half reduced return volatility (4.76% vs
  A1's 8.65%) without proportionally reducing drawdown, which suggests the
  losses are concentrated in a sustained regime rather than spread evenly.
  Unexplained; worth locating.
- **The core now has ~14 tuned windows plus 8 gate windows plus a split
  threshold.** High parameter count against 0.024 of IR headroom.
- **The lovol-vs-hibet comparison is two runs, not a controlled experiment.**
  The clean version splits on volatility *and* beta separately with everything
  else held fixed. The mechanism conclusion in section 2 rests on those two
  runs plus the gate-B failures — suggestive, not decisive. Say so.

---

## 6. Interview preparation

### 6.1 The 60-second walkthrough

> "It's my event-gated reversal core restricted to the low-volatility half of
> the universe. IR 0.094, Sharpe about 1.5, 368 names. But the interesting part
> isn't the alpha — it's that I expected the *high*-volatility half to win. My
> whole thesis was Nagel's: reversal is compensation for liquidity provision and
> it's predictable by expected volatility. The low-vol half beat the high-beta
> half and beat the parent. Combined with a macro volatility gate that never
> added anything to any of my alphas, I think the volatility channel isn't what
> I'm capturing. What I'm actually being paid for is fading mandated flow —
> index funds, ETF creation, market-on-close algorithms — and low-volatility
> mega-caps are exactly where that flow is the largest share of volume."

### 6.2 Question bank

**"Why does low volatility win?"** — §2.2. Mandated flow as a *share of volume*
is highest in large, stable, heavily-indexed names. The strategy fades flow,
not risk.

**"Isn't this just the low-volatility anomaly?"** No — the low-vol anomaly is a
long-only tilt toward low-beta stocks. This is a *universe restriction* on a
long/short reversal book; within the low-vol half it is still dollar-neutral
and still trading reversal. If it were the low-vol anomaly, the whole-universe
version would carry a systematic long-low-vol tilt, and it does not.

**"How is this different from the alpha you already submitted?"** It is the
same core on half the universe with a much slower `ts_zscore` window — 80 days
versus 10. Two independent departures, which is why it cleared the 50%
correlation gate. **It is not an independent idea and I would not claim it is.**

**"Your thesis was wrong. Why should I trust the rest of your work?"** Because
the test that found it was one I designed and wrote down in advance, with the
falsifying outcome specified before the run. The mechanism was wrong; the
process caught it. That has now happened three times in this project — the
earnings-window argument in note 14, the sign on three intraday signals, and
this — and each time the correction came from testing across the full parameter
range rather than stopping when the theory looked confirmed.

### 6.3 The question I would ask if I were interviewing me

*"If mandated flow is the mechanism, why is your gate built on calendar events
rather than on a direct measure of index membership or passive ownership?"*

Because the dataset has no ownership or index-membership fields — the calendar
gate is the best available proxy for *when* mandated flow arrives, and the
volatility split is a proxy for *where*. The direct test would use passive
ownership share or index weight, and would be the first thing I would build
with better data. The honest summary is that I have two proxies pointing at one
mechanism and no direct measurement of it.

---

## 7. Literature

- **Nagel (2012), "Evaporating Liquidity," RFS** — the anchor for the ORIGINAL
  thesis, and the one this note's evidence rejects. Retained deliberately.
- **Stoll & Whaley** — expiration-day effects from index-derivative unwinding.
  Now the better anchor.
- **Ariel (1987); Lakonishok & Smidt (1988)** — turn-of-the-month effects from
  systematic cash-flow timing. Mandated flow, directly.
- **Wu, "Closing Auction, Passive Investing, and Stock Prices"** — passive funds
  trading at the close create transient mispricing; the cleanest statement of
  the mandated-flow channel.
- **Frazzini & Pedersen (2014), "Betting Against Beta," JFE** — the low-beta
  anomaly this alpha must be distinguished from, per §6.2.
