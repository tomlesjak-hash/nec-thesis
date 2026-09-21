# Theory note — adaptive IC-weighted blend

*Interview reference. Follows `14_`–`18_`. All figures measured on Trexsim.*

---

## 1. Final specification

```python
sk = ts_mean(skew_return_last_hour, 1)
a  = cs_rank(sk) - 1.0                                    # component A

beta = ts_corr_binary(ret1, ret1_spx, 60)
res  = ret5 - (beta * ts_sum(ret1_spx, 5))
b    = cs_rank(-res / at_zero2nan(ts_std(ret1, 20))) - 1.0    # component B

ic_a = cs_rank(ts_corr_binary(ts_delay(a, 1), ret1, 60)) - 1.0
ic_b = cs_rank(ts_corr_binary(ts_delay(b, 1), ret1, 60)) - 1.0

score = (a * ic_a) + (b * ic_b)

# 8-event stress weight, windows tuned for this alpha
fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)
```

**SUBMITTED 2026-08-12.** Measured 2007-03-15 → 2021-12-30 (slow mode).

| Metric | Value |
|---|---|
| **IR** | **0.075** |
| TVR | 0.870 |
| Return | 0.057 |
| Drawdown | 8.0 |
| IR/√TVR | 0.080 |
| Booksize | 0.96 x 0.96 |
| NumStks | 719 x 757 |
| Liq / LiqN | 527 / 221 |

Derived: annualised Sharpe **1.19**, implied annual vol 4.79%, Calmar 0.71,
turnover **219× book/year**, break-even cost **2.6 bp**.

---

## 2. The result that matters most here

**It survived slow mode.**

This alpha puts *realised returns inside the signal* — `ts_corr_binary(ts_delay(a,1), ret1, 60)` correlates a lagged signal against the return it was supposed to predict. That is the single most likely place in this entire project for look-ahead to hide, and fast mode cannot detect it: an expression that peeks at the future looks superb in fast mode and collapses in slow, which is exactly what the tutorial's `ts_delay(close,-1)` example demonstrates.

The construction is legitimate on inspection — the correlation window is strictly trailing, and `ret1` is Delay-0 = Yes so today's return is known when the position is formed. But "legitimate on inspection" is worth very little against a class of bug that is invisible by construction. The slow-mode run is the evidence, and it is the reason this alpha is submittable at all.

**If asked why you trust it:** not because the logic looks clean, but because the test that would have caught the failure was run and passed.

---

## 3. Provenance — stated precisely

**The technique is not mine.** It is the second stage of **AlphaForge** (Shi et al., AAAI 2025, arXiv:2406.18394), which mines formulaic alphas and then *dynamically combines* them, re-weighting components at each time slice by recent performance. The paper's own ablation shows the "Dynamic" variant consistently beating the "Static" one — the combination stage, not the generation stage, is where its edge lives.

**What is mine:** the translation into a single-expression, per-stock form. AlphaForge learns combination weights with a neural network over a *time series* of factor performance. This computes the weight directly inside the alpha as a **per-stock rolling IC** using one operator, requiring no training, no held-out data and no model to deploy.

That distinction is worth being precise about. Rolling-IC weighting is not a novel idea in the abstract — practitioners have weighted signals by recent efficacy for decades. What is specific here is doing it **cross-sectionally per stock** rather than as a single time-varying scalar per factor, which is a genuinely different object and is what makes it expressible in one line.

**What cannot be claimed:** originality of the underlying concept.

---

## 4. Mechanism

**The economic claim is that signal validity varies by stock and that variation persists.**

Component A (closing-hour return skew) infers unfinished institutional demand from the shape of the final hour. That inference should be far more reliable in names with heavy index and ETF membership, where mechanical closing flow is large, than in names without. Index membership is stable over months, so the reliability difference persists — long enough for a 60-day rolling window to measure it and act on it.

Component B (beta-residual reversal) is paid for absorbing idiosyncratic shocks. Its reliability should track how much genuinely idiosyncratic flow a name receives, which depends on its shareholder base and analyst coverage — also stable.

So the weighting is not chasing noise. It is estimating a **structural property of each stock's information environment**, and the 60-day window is long enough to average out luck while short enough to track changes in coverage or index membership.

**Why weight rather than switch.** Weighting keeps both components live everywhere and lets the book express relative confidence. The switching version (D3, untested) would halve exposure to each parent and probably decorrelate further, at the cost of throwing away information whenever the two disagree mildly.

---

## 5. Position in the pool

| | A1 | A2 | A3 | **D2** |
|---|---|---|---|---|
| Sharpe | 1.44 | 1.62 | 1.16 | **1.19** |
| Annual vol | 8.65% | 3.03% | 6.13% | **4.79%** |
| Drawdown | 10.5 | 4.7 | 11.8 | **8.0** |
| Calmar | 1.19 | 1.04 | 0.60 | **0.71** |
| Turnover/yr | 186× | 348× | 63× | **219×** |
| Break-even | 6.7 bp | 1.4 bp | 11.2 bp | **2.6 bp** |

**It sits between its two parents on essentially every axis** — turnover, drawdown and volatility all fall between A2's and the residual-reversal core's. That is what a blend should do, and it is simultaneously the honest weakness: nothing here is better than the best of its parents.

Net of cost it is viable at 1–2 bp and underwater by 3 bp.

---

## 6. Honest weaknesses

- **It is built from two alphas already in the pool.** It cleared the 50% correlation gate against both, which is the empirical answer, but it is not an independent discovery and should not be presented as one.
- **The ablation has not been run.** A static 50/50 blend of A and B — no IC weighting — is the control, and it is one simulation. *Without it there is no evidence the adaptive weighting contributes anything.* This is the first thing an interviewer will ask and the honest answer is currently "I don't know."
- **The IC window (60 days) is untuned.** 20 and 120 were never tried. On A2 the window sweep was worth +0.027 against the gate's +0.014, so this is likely leaving IR on the table.
- **Marginal IR.** 0.075 against a 0.07 threshold, on an expression with eight gate windows plus two component cores plus an IC window. The parameter count is high relative to the headroom.
- **Complexity for its own sake is a real risk.** This is the most complicated expression in the pool and delivers the third-lowest Sharpe. If a static blend matches it, the complexity is pure overfitting surface.
- **Sample starts 2007-03**, so drawdown is not comparable to A1's.

---

## 7. Interview preparation

### 7.1 The 60-second walkthrough

> "It blends two of my alphas — closing-hour return skew and beta-residual
> reversal — but the weights aren't fixed. For each stock I compute a rolling
> 60-day correlation between each signal's lagged value and the realised
> return, which is a per-stock information coefficient, and weight each
> component by that. So the book leans on whichever signal has actually been
> working for that specific name. The idea comes from AlphaForge's dynamic
> combination stage; what's mine is compressing it into a single expression
> using a per-stock rolling IC rather than a trained model. IR 0.075, Sharpe
> about 1.2, and — the part I'd emphasise — it holds up in slow mode, which
> matters because putting realised returns inside a signal is where look-ahead
> hides."

### 7.2 Question bank

**"This is a blend of two alphas you already own. Isn't it double-counting?"**
It cleared the 50% correlation gate against both parents, which is the
mechanical answer. The reason it does is that adaptive weighting means it
tracks each parent only part-time — where a parent's IC is low it carries
little of it, so its exposure profile differs from a static blend. But it is
derived from existing components and I would not present it as an independent
idea.

**"Does the adaptive weighting actually do anything?"**
*I have not run the static-blend control.* That is the ablation that would
settle it and it costs one simulation. Say this plainly rather than arguing
from the mechanism — the mechanism has been wrong before in this project
(three intraday signs, and the earnings-window argument in note 14).

**"How do you know there's no look-ahead?"**
Slow mode. The correlation window is trailing and `ret1` is Delay-0, so the
construction is clean on inspection — but that class of bug is invisible to
inspection, so the argument rests on the test rather than on the reasoning.

**"Why per-stock IC rather than a single time-varying weight?"**
Because the claim is that signal validity varies *cross-sectionally*, not just
over time. Closing-hour flow is more informative in heavily indexed names than
in thinly held ones, and that is a persistent property of a stock's information
environment rather than a market regime. A scalar weight cannot express it.

**"Isn't a 60-day IC mostly noise at the single-stock level?"**
Probably in part, yes — which is why the IC is `cs_rank`'d before use rather
than applied raw. Ranking discards the magnitude and keeps only the ordering,
which is far more robust to a noisy estimate. It does not fix the underlying
noise, and a longer window would help if the underlying property is as stable
as I claim. Untested.

### 7.3 The question I would ask if I were interviewing me

*"You added a component whose only job is to say which of your other signals is
working. If you can estimate that reliably, why not use it to size your whole
book instead of as one more alpha?"*

The honest answer is that this is what the competition's scoring rewards —
alpha count is half the score, so a meta-signal is worth more submitted as an
alpha than deployed as a portfolio-construction layer. In a real book the
correct use is the second one: a per-stock, per-signal reliability estimate is
a position-sizing input, not a return forecast. Submitting it as an alpha is
optimising the scoring rule rather than the portfolio, and it is worth saying
so before being asked.

---

## 8. Literature

- **Shi et al. (2025), "AlphaForge: A Framework to Mine and Dynamically Combine
  Formulaic Alpha Factors," AAAI** — arXiv:2406.18394. The source of the
  technique; its dynamic-vs-static ablation is the direct evidence that the
  combination stage carries the edge.
- **Nagel (2012), "Evaporating Liquidity," RFS** — anchor for component B and
  for the event weighting.
- **Blitz, Huij, Lansdorp & Verbeek (2013), "Short-term residual reversal,"
  JFM** — component B's construction; residual reversal roughly doubles the
  Sharpe of raw reversal by removing non-reverting factor exposure.
- **Grinold (1989), "The Fundamental Law of Active Management"** — IR ≈ IC ×
  √breadth. The formal statement of why an IC estimate is the right weighting
  quantity, and the reason ranking a noisy IC is defensible where using it raw
  would not be.
