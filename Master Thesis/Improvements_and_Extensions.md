# Improvements and extensions (future study)

Ideas for improving or extending the model **beyond the thesis as designed**. Nothing here is planned,
decided or in the code. Each entry records the question, why it is hard, and where to start reading,
so it can be picked up later without redoing the thinking. Design decisions for the thesis itself live
in `Advisor_Questions.md`; this file is for what comes after.

Format per entry: the question, the context that produced it, why it is hard, candidate approaches,
how it connects to the current design, and a short reading list (about two hours).

---

## E1. How could the model be improved for one-day (next-day) forecasting?

**Raised:** 2026-10-05, when Q21 was decided for a 5-day horizon (option B). The thesis does **not**
forecast next-day returns; this entry is for later study only.

**Why one day is hard (from the Q21 discussion).**
- **Least signal per period.** On the market-neutral target (CRSP, S&P 500 members 2015-2024), textbook
  signals have daily rank ICs of about 1%; each day carries very little information.
- **Microstructure noise.** Closing prices alternate between bid and ask, which creates spurious
  next-day reversal (bid-ask bounce, Roll 1984).
- **Decay.** Daily machine-learning strategies on the S&P 500 were strongly profitable until about 2009
  and roughly zero net of costs afterwards (Fischer & Krauss 2018; Krauss, Do & Huck 2017).
- **Turnover.** Daily rebalancing makes transaction costs the dominant term; gross results are
  meaningless on their own.
- **The regime approximation is exact at h = 1**, which is the one thing in its favour for this model.

**Candidate approaches to study.**
1. **Cleaner targets.** Mid-quote returns instead of closing-price returns (removes bid-ask bounce);
   skip-a-day targets; or splitting the return into its **overnight** (close to open) and **intraday**
   (open to close) parts, which behave very differently across stocks (Lou, Polk & Skouras 2019).
2. **Faster information.** Features that move at the daily frequency: order imbalance, intraday
   volatility and range, overnight gaps, abnormal volume, short interest, news and sentiment. The
   current inputs are daily price and volume aggregates only.
3. **Decomposing reversal.** Short-term reversal mixes liquidity-driven moves (which revert) with
   news-driven moves (which do not). Separating them, for example with industry-adjusted returns or news
   flags, sharpens the signal considerably (Da, Liu & Schaumburg 2014).
4. **The liquidity-provision view of the regime gate.** Next-day reversal profits are compensation for
   providing liquidity and rise with market stress (Nagel 2012). A gate that tracks funding or
   volatility conditions (VIX, intraday market volatility) is the natural conditioning variable at this
   horizon, and fits the thesis's regime framing directly.
5. **Cost-aware objectives.** Training against a loss that penalises turnover, or evaluating with
   explicit cost models, instead of predicting returns and trading on them naively.
6. **Sequence models.** Next-day prediction is where recurrent or attention models over recent daily
   history have been tried most (Fischer & Krauss 2018 use LSTMs); the code already has a sequence
   encoder (`input_mode="snapshot_plus_hidden"`) that is switched off in the base case.
7. **Multi-horizon learning.** Training one model on h = 1, 5 and 21 jointly, so the short horizon
   borrows strength from the longer ones (Blitz, Hanauer, Hoogteijling & Howard 2023 on how alpha
   changes with the prediction horizon).

**Connections to the current design.** Market-neutral target (Q25) carries over unchanged; frequency is
already daily (Q21), so only `horizon` would change; the overlap corrections become unnecessary at
h = 1; the gate timing item (predicted vs filtered) matters most at h = 1.

**Reading, about two hours.**
- Fischer & Krauss (2018), "Deep learning with long short-term memory networks for financial market
  predictions", *European Journal of Operational Research* 270(2). The daily S&P 500 case and its decay.
- Lou, Polk & Skouras (2019), "A tug of war: Overnight versus intraday expected returns", *Journal of
  Financial Economics* 134(1). Why the overnight and intraday parts of a day's return differ.
- Nagel (2012), "Evaporating liquidity", *Review of Financial Studies* 25(7). Short-term reversal as
  liquidity provision, and why it depends on market conditions.
- Da, Liu & Schaumburg (2014), "A closer look at the short-term return reversal", *Management Science*
  60(3). Separating liquidity-driven from news-driven reversal.
