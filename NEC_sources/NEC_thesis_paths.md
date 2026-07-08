# NEC Thesis: Understanding Your Options

## A guide to NEC, its variations, and how to choose what to research

**Prepared for:** Tom Koreslesjak
**Purpose:** A plain-language guide to NEC, the research choices you face, and how to decide what your thesis is actually about. No math is used; all explanations are in words.

---

## Why this document exists

You have committed to NEC for your master's thesis. But "doing a thesis on NEC" is not yet a thesis topic — it is a starting direction. To define a topic that satisfies the SAIF §2.1.6 Empirical Research requirements, you need to make three decisions:

1. **Which variation of NEC** are you researching?
2. **What is the central empirical question** the thesis answers?
3. **What is the headline pitch** that frames the work?

These decisions interact. You can't make decision 2 cleanly without first understanding the variations available in decision 1, and you can't pitch the thesis (decision 3) without knowing what it actually does (decisions 1 and 2). This document explains all three in plain language so you can make informed choices.

You will not need to understand the underlying math to read this document. The math comes when you work through the textbooks. Here, we focus on the *ideas* and the *research choices*.

---

## Part 1: What NEC actually is, in plain language

NEC is a specific instance of a broader class of models called **Mixture-of-Experts** (MoE). To understand NEC, you first need to understand the MoE intuition. Then we'll come back to what makes NEC specifically NEC.

### The intuition behind mixture-of-experts

Imagine you are trying to predict next month's return for every stock in a universe of 1,000 US equities. The traditional approach is to train one big model on all the historical data — one generalist that has seen everything and tries to find one pattern that works on average across all market conditions.

The problem: financial markets behave very differently in different conditions. The pattern that predicts returns well in a calm bull market often *reverses* in a crash. A "generalist" model averages over all these conditions and ends up mediocre everywhere — moderately good in normal markets, badly miscalibrated in extreme markets.

Mixture-of-experts asks a different question. Instead of training one generalist, what if we had several specialists, each one trained on a different "type" of market condition? Then at prediction time we ask: what market condition are we currently in? And we send the prediction to the right specialist.

This requires three components:

- **Several "experts"** — each expert is its own predictive model (typically a small neural network). One might specialize in calm markets, another in high-volatility markets, another in transition periods.
- **A "gate"** — a smaller model that looks at the current data and decides which expert (or weighted combination of experts) to use for each prediction.
- **A training procedure** that simultaneously teaches the experts to specialize and teaches the gate to route correctly.

This is mixture-of-experts in one paragraph. It's a simple idea — multiple specialists with a router that picks who handles each input. The complexity is in how to train the system end-to-end so the experts and the gate cooperate rather than fighting each other.

### What NEC specifically is

In the `OP model` folder of your repository, there is a specific implementation of a mixture-of-experts system called NEC. Its structure is:

- An LSTM sequence encoder that looks at the recent history of a stock's features
- A classifier head that outputs probabilities over three classes
- Two regression heads — one called "extreme" and one called "normal"
- A routing rule that picks the "extreme" expert if the classifier output favors class 1, and the "normal" expert otherwise

The defect that makes NEC interesting as a thesis subject: the classifier outputs three classes but the routing only uses one bit of information (class 1 versus not class 1). Classes 0 and 2 both route to the same "normal" expert. So two-thirds of the classifier's expressiveness is wasted. The "corrected NEC" fixes this by making the routing properly differentiable (so the experts and gate can be trained jointly) and by using all the classifier's information (typically by having three experts to match the three classes, with soft routing).

This defect is what makes NEC research-worthy. The original implementation is broken in a documented way, and your thesis can be about fixing it and rigorously studying the corrected version against baselines.

### Why this matters for finance

The intuition behind mixture-of-experts is essentially the same intuition that drives Hamilton's regime-switching model from 1989 — a classical and respected finance idea that has been cited tens of thousands of times. Hamilton's model says: instead of assuming asset returns follow one stable distribution, assume there are several "regimes" (high-volatility, low-volatility, expansion, recession) and returns follow different distributions in each. The hidden regime variable is inferred from the data using statistical machinery.

Mixture-of-experts is a *modernized* version of this idea. The key differences:

- **Observation model.** Hamilton uses simple Gaussian distributions per regime. MoE uses flexible neural networks per regime.
- **Gating mechanism.** Hamilton uses a Markov chain — today's regime depends on yesterday's regime. MoE uses a learned gating function that looks at current features to decide regime.
- **Tradeoff.** Hamilton has a clean statistical foundation but is restrictive in what each regime can model. MoE is more flexible but is harder to interpret.

A natural thesis pitch becomes: "Modernize Hamilton-style regime-switching for return prediction using mixture-of-experts neural networks, and test whether the modernized version actually wins on out-of-sample US equity return prediction."

This thesis sits in a strong place. It engages directly with a classical finance literature (Hamilton 1989 and the thirty-five years of follow-up work). It uses modern ML methodology. It produces a clear empirical comparison. It is exactly the shape SAIF §2.1.6 wants.

---

## Part 2: Variations on NEC — what you could research

When I refer to "standard NEC" or "corrected NEC", I mean the version where:

- The architecture is fixed: LSTM encoder, soft-gating router, K experts (where K is 2, 3, 4, or 8 depending on testing), end-to-end training.
- The features are traditional quant features: prices, returns over various horizons, volatility, momentum, volume, factor exposures, market-relative measures.
- The routing is soft — meaning we use a weighted combination of all experts based on the gate's output probabilities, rather than hard-picking one expert.
- The training data is broad US equities over 15 or more years.

This is the "vanilla" NEC thesis. The data is conventional. The architecture is fixed. The interesting question is purely empirical: does the corrected NEC outperform classical and ML baselines?

Beyond standard NEC, there are several research-worthy variations. Each preserves the core idea (mixture-of-experts for regime-conditional prediction) but changes one important thing about the experiment or the architecture. Below I describe the four most credible variations.

### Variation 1: Sentiment-augmented NEC

In this variation, the architecture is unchanged but the input features expand significantly. In addition to traditional quant features, you include:

- **News sentiment scores** at the stock-date level, derived by running a sentiment classifier over a news corpus.
- **Text features from SEC filings** (10-K, 10-Q) — what topics the company discusses, how positively, how that changes over time.
- **Analyst-revision summaries** — counts of EPS estimate upgrades versus downgrades, target-price changes.
- **Options-market signals** — implied volatility level, skew, term structure, put-call ratios.
- **Insider transactions** — net buying versus selling by company executives and directors.
- **Short-interest features** — short interest as a percentage of float and how it's changing.

These features capture aspects of asset pricing that traditional price-volume features miss entirely. They are particularly relevant for narrative-driven and bubble-driven episodes — like the current AI and semiconductor cycle, or the dotcom era. If your thesis cares about these kinds of episodes, sentiment-augmented NEC is the natural variation. If your thesis is about the architecture and methodology in general, sentiment features are optional.

The cost: building a sentiment-feature pipeline is roughly a one-to-two-month data engineering project on top of the modeling work. There are shortcuts. If you use precomputed academic sentiment datasets (Loughran-McDonald 10-K sentiment, IBES analyst revisions through WRDS, Baker-Wurgler market sentiment), the engineering load drops to two-to-three weeks. If you build a custom pipeline using FinBERT or similar on raw news, it's closer to two months.

### Variation 2: Routing-mechanism comparison

In this variation, the universe and features are standard but the headline contribution is methodological: which routing scheme wins on financial data?

- **Soft routing** uses a weighted combination of all experts at every prediction (always all experts active, weights from the gate).
- **Hard routing** picks one expert per prediction (the highest-probability one chosen by the gate).
- **Sparse top-k routing** picks the top few experts and ignores the rest (the modern Mixtral / Switch Transformer approach).
- **Gumbel-softmax routing** provides a differentiable approximation of hard routing (used when you want hard routing at inference but need gradients during training).

Each has trade-offs in terms of bias, variance, and computational cost. There is no consensus on which is best for the low-signal-to-noise regime of financial data. The thesis tests them empirically.

### Variation 3: HMM-gated NEC

This is a hybrid between classical regime-switching and modern MoE. The experts remain flexible neural networks (the "modern" part) but the gating mechanism is a Hidden Markov Model with Markov persistence (the "classical" part) — today's regime depends on yesterday's regime through a learned transition matrix.

This variation has a clean theoretical story: it preserves Hamilton's regime-persistence assumption while modernizing the observation model. It is the most mathematically interesting variation and connects deeply to classical finance theory. It is also the most math-heavy. If you love regime-switching theory and want a thesis with strong econometric foundations, this is the variation to pick.

### Variation 4: Graph-gated NEC

In this variation, the gating mechanism considers cross-sectional relationships between stocks. If NVIDIA, AMD, and TSMC are clearly in the same regime (as semiconductor stocks often are during AI-cycle episodes), the model recognizes this through a graph structure that encodes their relationships.

This requires building a stock-relationship graph (using sector membership, supply-chain proximity, news co-mention frequency, or similar). It is genuinely state-of-the-art ML and would be a strong thesis if executed well, but it is unusually ambitious for a master's timeline. I would not recommend it as a primary thesis topic unless you have substantial extra time.

### Decision shape

For most thesis purposes, the live options are **Standard NEC**, **Sentiment-augmented NEC**, and **HMM-gated NEC**. Routing-mechanism comparison can be a chapter inside any of these three rather than a separate thesis. Graph-gated NEC is probably too ambitious.

---

## Part 3: What is the "research focus" — what is the central question?

Within whichever variation you choose, you still have to decide what the *central question* of the thesis is. This is the empirical question your hypothesis tests will address. There are roughly four options for the focus.

### Focus A: Architectural benchmark

**The central question:** Does the corrected NEC outperform classical regime-switching baselines (Hamilton 1989) and standard machine-learning baselines (single MLP, LightGBM, vanilla GRU) on cross-sectional return prediction in US equities?

**What you would do:** Implement the corrected NEC. Implement the baselines — at minimum a Hamilton regime-switching model, a single MLP, a LightGBM model, and a vanilla GRU. Train all of them on the same data with the same evaluation protocol (walk-forward cross-validation with purging and embargoing, deflated Sharpe ratio for honest performance evaluation). Compare them using rank-IC, decile long-short returns, and risk-adjusted Sharpe.

**Headline finding might look like one of these:**

- "We find that corrected NEC outperforms Hamilton-style regime-switching by [X percentage points] on out-of-sample rank-IC, suggesting that modern MoE methodology offers a meaningful improvement over classical regime-conditional models."
- "We find that NEC does not significantly outperform the Hamilton baseline after proper financial-ML validation, suggesting that classical regime-switching remains competitive against modern MoE under realistic evaluation."

Either outcome is publishable. The thesis works either way. A defensible negative result is a contribution.

**Strengths:** Most defensible empirically. Cleanest hypothesis tests. Lowest risk of running out of time. Very clear what "success" and "failure" of the thesis look like.

**Weaknesses:** Least exciting subject matter at first glance. Reads as "we benchmarked some models," which understates what the work actually involves.

### Focus B: Regime interpretability

**The central question:** Do the learned expert assignments in the corrected NEC correspond to *recognizable* market regimes — high-volatility periods, recessions, momentum crashes, or other known financial regime classifications?

**What you would do:** Train the corrected NEC. Analyze which expert each stock-date is routed to. Compute correlation between expert-assignment time series and known regime indicators (VIX levels, NBER recession dates, market drawdowns, the momentum-crash dates from Daniel & Moskowitz 2016, or other established regime markers). Visualize. Interpret.

**Headline finding might look like:**

- "We find that the corrected NEC learns regime structure that correlates [X] with VIX, with experts specializing in high-volatility versus low-volatility periods."
- "We find that the corrected NEC's learned regimes do not correspond to known financial regimes, suggesting that the model detects different structure than the finance literature recognizes."

Either outcome is publishable.

**Strengths:** Most interpretable. Strongest "bridges-ML-and-finance" feel. Deepest connection to classical finance literature. Good fit for a SAIF committee that values both rigor and insight.

**Weaknesses:** Requires substantial interpretability analysis work. Findings can be ambiguous (what counts as "correspondence"?). Requires more careful definition of what regimes you are testing for.

### Focus C: Routing-mechanism comparison

**The central question:** Which routing mechanism (soft, hard, sparse top-k, Gumbel-softmax) wins on cross-sectional return prediction under the low-signal-to-noise conditions of financial data?

**What you would do:** Implement multiple routing schemes within the corrected NEC framework. Train each one on the same data with the same evaluation protocol. Compare predictive performance and interpretability.

**Strengths:** Most original ML methodology contribution. Specific, technical, defensible.

**Weaknesses:** Slightly less obviously a "finance" thesis. The committee might wonder whether the routing comparison is more of an ML paper than a finance paper. Probably better suited as one chapter inside a broader thesis rather than as the headline focus.

### Focus D: Combined comprehensive study

**The central question:** All of the above as three sub-studies of one comprehensive empirical study.

**Strengths:** Strongest overall contribution. Most comprehensive treatment.

**Weaknesses:** Largest workload. Real risk of running out of time. Master's theses generally have one focused contribution rather than three. Higher chance of producing thin work in each of the three areas instead of strong work in one.

### Recommendation

For most students, the recommended path is **Focus A (architectural benchmark) plus a chapter on Focus B (interpretability)**. This is the pattern most empirical-ML quant theses follow — a benchmark study with an interpretability analysis to give it depth. It produces a thesis that is both defensible (the benchmark gives clear answers) and insightful (the interpretability chapter gives the thesis a story).

---

## Part 4: The headline pitches — how do you frame the same work?

The pitches are different ways of *framing* the same underlying empirical work. The framing matters because it signals to your committee and to future readers what kind of researcher you are and where the contribution sits.

There are four credible pitches. Each works with multiple combinations of variation and focus, but each is associated with a particular committee audience.

### Pitch 1: "Modernize Hamilton with MoE" — finance-anchored

This frames the thesis as engaging with a thirty-five-year-old finance tradition and modernizing it with modern ML methodology. The thesis is positioned as a *finance* contribution that uses ML methods — not as an ML contribution that uses finance data.

The introduction opens with regime-switching theory, walks through the Hamilton framework, identifies its limitations (restrictive Gaussian observation models, hand-tuned Markov gating, limited flexibility per regime), and then introduces MoE as the modernization. The empirical question becomes "does the modernized version beat the classical version?"

**Best fits:** A committee composed of econometricians, finance PhDs, or anyone with a classical finance background. Strongest fit for a SAIF MF program where the committee leans finance-trained. Strongest §2.1.6 alignment.

### Pitch 2: "First rigorous MoE-for-quant study" — ML-anchored

This frames the thesis as taking modern MoE methodology — currently a major topic in ML thanks to Mixtral, Switch Transformer, and DeepSeek-MoE — and providing the first rigorous empirical study of its application to cross-sectional return prediction.

The introduction opens with modern MoE in ML, walks through its successes in language modeling, identifies the gap (almost no rigorous quant applications), and then proposes the empirical study.

**Best fits:** A committee with ML or computer-science background. Weaker fit for a pure finance program. Probably not the best choice for SAIF given your program's profile.

### Pitch 3: "Interpretable regime detection via ML" — interpretability-anchored

This frames the thesis around the interpretability question. Can modern black-box ML methods learn regime structures that correspond to recognized financial regimes? Bridges the gap between classical interpretable models (which have known regimes by construction) and modern flexible ML (which is more accurate but less interpretable).

The introduction opens with the interpretability crisis in ML applied to finance, identifies the gap, and proposes MoE-based regime detection as a way to make ML predictions more interpretable.

**Best fits:** A committee that values interpretability and academic rigor. Strong fit for either finance or ML committee composition. Particularly attractive if your supervisor cares about interpretability.

### Pitch 4: "Regime-aware portfolio construction" — practical-anchored

This frames the thesis around the practical problem of portfolio management. The MoE serves as a regime-detection mechanism, and the thesis tests whether incorporating learned regime detection into portfolio construction improves risk-adjusted returns under realistic transaction costs and turnover constraints.

The introduction opens with the practical problem of portfolio construction under regime change, identifies the gap in how regimes are currently detected, and proposes MoE as the regime detector.

**Best fits:** A committee with practical or industry background. Strongest connection to a quant trading career — this pitch reads as "here's a real trading methodology."

### Pitch combinations

You don't have to pick exactly one pitch. The pitches can be combined. A thesis can be framed primarily as "Modernize Hamilton with MoE" with a sub-pitch of "tested via realistic portfolio construction" — that combines pitches 1 and 4 and is a strong fit for a SAIF MF defense.

---

## Part 5: Decision matrix — putting it all together

To finalize your thesis topic, you need to make three decisions. Below is a decision matrix that shows which combinations of variation, focus, and pitch are coherent and which would be confusing.

### Decision 1: Which variation of NEC?

| Variation | Description | Recommended for |
|-----------|-------------|-----------------|
| Standard NEC | Vanilla soft-routed MoE on broad US equities with traditional features | Most students. Best balance of risk and reward. |
| Sentiment-augmented NEC | Same architecture, expanded features including sentiment / narrative / options data | Students who find narrative-driven episodes (AI bubble, dotcom) essential to their thesis identity. |
| HMM-gated NEC | Modern experts with classical HMM Markov-persistence gating | Students who love classical regime-switching theory and want strong econometric foundations. |

### Decision 2: What is the central focus?

| Focus | Description | Recommended for |
|-------|-------------|-----------------|
| Architectural benchmark | Does corrected NEC beat baselines? | Most students. Safest, most defensible. |
| Regime interpretability | Do learned experts correspond to known regimes? | Students who want strong bridges between ML and classical finance. |
| Routing mechanism | Which routing scheme wins in low-SNR financial data? | As a chapter inside another focus, not as primary. |
| Combined | All three as separate chapters | Only if you have unusually high productivity and a clear plan. |

### Decision 3: How do you pitch it?

| Pitch | Description | Best committee fit |
|-------|-------------|-------------------|
| Modernize Hamilton | Frame as finance contribution using ML methods | Finance-leaning committee. Strongest §2.1.6 alignment. |
| First rigorous MoE-for-quant | Frame as ML contribution applied to finance | ML / CS-leaning committee. |
| Interpretable regime detection | Frame as interpretability-bridges-ML-and-finance | Either, especially if supervisor values interpretability. |
| Regime-aware portfolio construction | Frame as practical portfolio management contribution | Industry-leaning committee. Strongest career signal. |

### Coherent combinations

Below are the combinations that make sense and tell a unified story. Confused combinations (such as "sentiment-augmented NEC framed as a methodology comparison") would be harder to defend.

| Combination | Pitch | Variation | Focus | Why it works |
|-------------|-------|-----------|-------|--------------|
| A (safe and recommended) | Modernize Hamilton | Standard NEC | Architectural + Interpretability | Lowest risk. Strongest §2.1.6 fit. Classical finance framing. |
| B (sentiment) | Modernize Hamilton with sentiment | Sentiment-augmented NEC | Architectural + Interpretability | More exciting subject matter. Two months extra engineering. |
| C (theoretical) | Modernize Hamilton (preserving Markov gating) | HMM-gated NEC | Architectural + Interpretability | Strongest theoretical contribution. Most math-heavy. |
| D (practical) | Regime-aware portfolio construction | Standard NEC | Architectural + practical backtesting | Strongest career signal. Most directly useful for industry. |

Combinations A and D are the strongest defaults. A is best for a pure academic / SAIF-academic committee. D is best if you want the thesis to clearly translate to a quant trading career.

---

## Part 6: My honest recommendation

You have several months of preparation invested in NEC. You understand or are learning the math (mixture models, EM algorithm, soft gating, load balancing). The `OP model` folder has starter code. The syllabus is built around it. The textbook project is being written for it.

Given all of that, my honest recommendation for the topic is:

> **Variation: Standard NEC.**
> **Focus: Architectural benchmark with a chapter on regime interpretability.**
> **Pitch: "Modernize Hamilton-style regime-switching for cross-sectional return prediction with mixture-of-experts neural networks."**

In one sentence: "This thesis modernizes the classical Hamilton-style Markov-switching framework using mixture-of-experts neural networks and tests, on a broad US equity universe under proper financial-ML validation, whether the modernized version improves cross-sectional return prediction relative to both classical regime-switching baselines and standard machine-learning baselines, while analyzing whether the learned expert assignments correspond to recognizable market regimes."

This is your thesis. It is defensible. It is finance-anchored. It engages with classical and modern literature. It satisfies §2.1.6 directly. It preserves all the runway you have built. It is one focused contribution that can be executed in your timeline.

If you find this insufficiently exciting — particularly if the AI / semiconductor / bubble theme feels essential to your sense of what you want to research — we can shift to **Combination B** (sentiment-augmented NEC). That version is more exciting, more career-relevant if you're targeting an AI-finance role specifically, but adds one to two months of data engineering on top of your existing plan.

If you find classical regime-switching theory more interesting than modern flexible ML, we can shift to **Combination C** (HMM-gated NEC). That version has the strongest theoretical contribution but is the most math-heavy and may exceed the time budget.

If you are primarily targeting a quant trading career and want the thesis to read as a clear practical methodology, we can shift to **Combination D** (regime-aware portfolio construction). That version is essentially the same work as Combination A with a different chapter structure — the headline pitch is portfolio construction instead of return prediction, but the underlying MoE machinery is identical.

The choice between A, B, C, and D is yours. There is no wrong answer; there are only different tradeoffs.

---

## Part 7: What we do next

Once you have read this document and have a sense of which combination you want, the next step is to formalize the thesis topic. That means writing:

1. **A one-paragraph topic statement** — what the thesis is about, in 100-150 words.
2. **A three-sentence empirical question** — the specific question your hypothesis tests will answer.
3. **A one-paragraph methodology sketch** — what models you'll compare, what data you'll use, what evaluation protocol.
4. **A list of three to five hypotheses** — formal claims you will test.

This becomes the basis of the proposal you submit to SAIF in September. We do not need to write the full proposal now (the proposal goes into much more detail). We just need to agree on the topic before we go further.

When you've read this document, come back and tell me which combination resonates (A, B, C, D, or a custom variant). We will then write the topic statement together.

---

## Quick reference: the key terms

| Term | Plain-language meaning |
|------|-----------------------|
| NEC | A specific implementation of mixture-of-experts in your `OP model` folder. Currently broken in a documented way. |
| Mixture-of-experts (MoE) | A class of models with multiple specialist predictors and a gate that routes inputs to the right specialist. |
| Expert | One of the specialist predictors in an MoE. Each expert is its own small neural network. |
| Gate | The component that decides which expert (or weighted combination) handles each input. |
| Soft routing | Use a weighted combination of all experts. The default for low-noise problems. |
| Hard routing | Pick one expert per input. Cheaper but non-differentiable. |
| Sparse top-k routing | Pick the top few experts and ignore the rest. The modern Mixtral / Switch Transformer approach. |
| Regime | A market condition (high-volatility, recession, momentum crash, etc.) under which the data-generating process is approximately stable. |
| Regime-switching | The classical finance approach of modeling returns with multiple regimes. Hamilton 1989 is the foundational paper. |
| Hamilton 1989 | The classical regime-switching paper. Cited tens of thousands of times. The "before" model in our thesis. |
| Cross-sectional return prediction | Predicting which stocks will outperform which other stocks within a single date. The standard quant prediction problem. |
| Rank-IC | The information coefficient (Spearman rank correlation) between predicted returns and realized returns within a cross-section. Standard quant evaluation metric. |
| Walk-forward CV | A cross-validation procedure that respects time ordering. Standard for financial ML. |
| Purging and embargoing | Techniques for removing data leakage in walk-forward CV. From López de Prado's AFML textbook. |
| Sentiment-augmented features | Features derived from text (news, filings) and non-price data (analyst revisions, options markets) rather than from prices and volumes alone. |
| §2.1.6 | The Empirical Research category in the SAIF MF thesis requirements. The category our thesis falls under. |
