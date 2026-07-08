# NEC Thesis Proposals: Three Directions and Their Literature

## A summary of three candidate thesis directions and the academic literature supporting their novelty

**Prepared for:** Tom Koreslesjak and supervisor pitch discussions
**Purpose:** A combined explanation of three candidate NEC thesis directions, with the academic literature review for each. The directions are explained in words, not math, so they can be understood without prior NEC study.

---

## Background: What NEC is, in one paragraph

NEC is a specific implementation of a class of machine learning models called Mixture-of-Experts (MoE). The intuition: instead of training one big general-purpose predictor on all market conditions, you train several smaller "expert" predictors and a "gate" that decides which expert to use for each prediction. Each expert can specialize on a particular type of market condition (calm markets, high-volatility markets, momentum-driven markets, etc.) and the gate routes new inputs to whichever expert is best suited. The original NEC code in the OP model folder has a documented defect — the gate's information is partly wasted — and the thesis builds on the corrected version. This connects to a classical finance idea, Hamilton's (1989) regime-switching model, which used a simpler statistical version of the same intuition. Modern MoE is the flexible deep-learning generalization of regime-switching.

---

## Direction 1: Routing-mechanism comparison in MoE for cross-sectional return prediction

### What the idea is, in plain language

When a Mixture-of-Experts model needs to decide which expert handles a given input, it uses a "routing mechanism." There are several different ways to do this routing, and they have different trade-offs.

The most common options are:

**Soft routing.** The gate produces a set of probabilities (one per expert) and the final prediction is a weighted average of every expert's prediction, with the weights being the gate probabilities. All experts contribute to every prediction. This is the smoothest, most stable choice.

**Hard top-1 routing.** The gate picks the single highest-probability expert and uses only that expert's prediction. Cheaper computationally, but the gate decision becomes non-differentiable, which makes training harder.

**Sparse top-k routing.** The gate picks the top *k* experts (typically 2 or 3) and uses a weighted combination of only those. A middle ground that gives you specialization without throwing away the rest of the experts entirely. This is what Mixtral and Switch Transformer use.

**Gumbel-Softmax routing.** A clever mathematical trick that lets you approximate hard routing during training (which gives the model gradient signal) while behaving exactly like hard routing at prediction time.

The question of *which routing mechanism is best* is genuinely open in machine learning, and there is no consensus. The trade-offs depend heavily on the signal-to-noise ratio of the underlying data. In language modeling, where signal is high, sparse top-k tends to dominate. In financial return prediction, where signal-to-noise is much lower, the answer is unclear and arguably different.

### What the thesis would do

Implement the corrected NEC with each of the four routing mechanisms (soft, hard top-1, sparse top-k, Gumbel-Softmax) and run a clean head-to-head benchmark on cross-sectional US equity return prediction. The same architecture, the same data, the same evaluation protocol — only the routing scheme changes between runs.

The contribution: provide the first systematic empirical answer to "which routing mechanism wins in the low-signal-to-noise regime of cross-sectional return prediction." This is methodological work that has direct implications for anyone building MoE-based quant models.

### Literature review

**What has been done elsewhere.** Routing-mechanism comparison exists in the broader machine-learning literature. Nguyen et al. (arXiv 2405.13997, 2024) compared sigmoid gating versus softmax gating for MoE regression and showed the choice matters. Shazeer et al. (2017) introduced noisy top-k gating in the foundational sparsely-gated MoE paper. Mixtral (2024) and Switch Transformer (Fedus, Zoph, Shazeer 2022) use top-k routing as their default. There is a substantial body of work comparing routing schemes for language modeling.

**What has been done in finance.** Several MoE-for-stock-prediction papers have appeared in 2024–2025, but each picks one specific routing scheme rather than comparing across schemes:

- MIGA (arXiv 2410.02241, October 2024) uses a trainable router with top-k selection on Chinese stock indices.
- LLMoE (arXiv 2501.09636, January 2025) uses an LLM as the router.
- Adaptive Market Intelligence (Vallarino, arXiv 2508.02686, July 2025) uses volatility-aware gating.
- MERA (ACM Web Conference 2025) uses retrieval-augmented routing.

None of these systematically compares the four standard routing mechanisms against each other on the same task.

**The novelty gap.** A systematic head-to-head comparison of soft, hard top-1, sparse top-k, and Gumbel-Softmax routing on cross-sectional equity return prediction does not appear in the literature. The defensible contribution claim is: "We provide the first systematic empirical comparison of routing mechanisms in Mixture-of-Experts models applied to cross-sectional equity return prediction, hypothesizing that the low signal-to-noise ratio of financial data favors different routing schemes than the high-SNR language-modeling settings where these mechanisms were developed."

This is a real and defensible contribution. Adjacent work exists; the specific test does not.

### Strengths and weaknesses

**Strengths.** Clear methodological contribution. Defensible novelty. Clean experimental design (only one thing changes between runs). Results are directly useful to practitioners building MoE-based models. Works whether the headline finding is positive ("soft wins") or negative ("the choice doesn't matter much for low-SNR data") — both outcomes are publishable.

**Weaknesses.** Reads as a methodology paper, which might land better with the ML/statistics faculty than with the finance faculty. The contribution is incremental rather than revolutionary. If the result is "soft wins by 2%," it is a small finding.

---

## Direction 2: HMM-gated NEC

### What the idea is, in plain language

Standard NEC uses a "feature-conditioned" gate — the gate looks at the current features (price, volume, etc.) and decides which expert to use right now, based only on the current state. It has no memory: the gate decision today does not directly depend on the gate decision yesterday.

Classical finance models, particularly Hamilton's (1989) regime-switching framework, do the opposite. They assume that the market is in one of several "regimes" (e.g., high-volatility and low-volatility), and that today's regime depends on yesterday's regime through a transition matrix — once you are in a high-volatility regime, you tend to stay there for a while. This is called Markov persistence. The model uses a Hidden Markov Model (HMM) to track the latent regime over time.

HMM-gated NEC combines the two:

- The *experts* remain modern flexible neural networks (the modern part).
- The *gate* uses an HMM with Markov persistence (the classical part).

The result is a model that has the best of both worlds. The experts can learn flexible, non-linear, complex relationships between features and returns. But the regime-detection mechanism preserves Hamilton's insight that regimes are persistent and slowly-evolving rather than reset every period from features alone.

### What the thesis would do

Implement standard NEC (feature-conditioned gate, neural network experts) as the baseline. Implement HMM-gated NEC (HMM gate with Markov persistence, neural network experts) as the proposed model. Implement classical Hamilton-style regression with Gaussian observations as the finance baseline. Compare all three on cross-sectional US equity return prediction.

The hypothesis being tested: in financial data, regimes are persistent (today's regime correlates with yesterday's), and a gate that imposes Markov persistence should outperform a gate that has no memory. This is a clean, finance-anchored hypothesis with direct ties to the regime-switching literature.

### Literature review

**What has been done elsewhere.** The closest precedent is Markovian RNN (Karatzas & Ozyildirim, arXiv 2006.10119, 2020, also published in IEEE Transactions on Neural Networks and Learning Systems). This paper uses an HMM for regime transitions inside an RNN — each regime controls the hidden-state transitions of one recurrent cell. The paper demonstrates significant performance gains over Markov-switching ARIMA and over standard RNN variants on financial time series.

Related but more distant work includes GenHMM (which combines HMMs with neural-network generative models per state), Hidden Markov Neural Networks (HMNNs, which combine Factorial HMMs with neural networks), and recent work on neural HMMs with adaptive granularity attention for high-frequency order flow.

In classical finance, Hamilton's (1989) regime-switching paper has been cited tens of thousands of times. Extensions like Markov-switching multifractal (Calvet & Fisher) and various time-varying transition matrices exist.

**The novelty gap.** Markovian RNN does HMM-gated *single recurrent network with regime-conditioned hidden state*. HMM-gated NEC would do HMM-gated *Mixture-of-Experts where the experts themselves are separately-parameterized neural networks*. The distinction matters substantively:

- In Markovian RNN, one network exists. Its dynamics change per regime.
- In HMM-gated NEC, multiple networks exist. Each is fully trained as a specialist; the HMM decides which network handles each input.

This second formulation — HMM persistence combined with full MoE architecture — does not appear in the literature in the form proposed. The closest precedent is one step away.

**Defensible contribution claim.** "We extend the Markovian-RNN paradigm to mixture-of-experts architectures by replacing the single regime-conditioned RNN with separately-parameterized expert networks routed by an HMM-derived posterior over regimes, and test whether this hybrid model — combining classical regime-persistence assumptions with modern flexible expert specialization — outperforms purely feature-conditioned MoE and purely classical Markov-switching regression for cross-sectional return prediction in US equities."

This is a strong novelty claim. Among the three directions, this one has the cleanest "first-of-its-kind" framing.

### Strengths and weaknesses

**Strengths.** Strongest novelty of the three directions. Strongest finance framing because it directly modernizes Hamilton. Clean experimental design with a clear hypothesis. Bridges classical and modern literatures, which the SAIF committee will appreciate. Strong fit for a supervisor with econometrics background.

**Weaknesses.** The most math-heavy of the three directions — implementing HMM-gated MoE requires understanding the forward-backward algorithm, the Baum-Welch updates, and how to integrate them with end-to-end neural-network training. This is a real engineering task. Risk of being one of those "elegant theory, ambiguous empirical results" theses if the HMM persistence assumption does not actually hold cleanly in the data.

---

## Direction 3: Sentiment-augmented NEC

### What the idea is, in plain language

Standard NEC uses traditional quantitative features: prices, returns, volatility, momentum, volume, factor exposures. These are all derived from price-and-volume data. They miss a lot of what actually moves markets in narrative-driven and sentiment-driven episodes — bubbles, manias, news-driven crashes, hype cycles like the current AI and semiconductor narrative.

Sentiment-augmented NEC keeps the same model architecture but expands the input features to include:

- News sentiment scores derived from running a text classifier (such as FinBERT) over financial news articles about each stock.
- Text features from SEC filings (10-K, 10-Q) — what topics the company discusses, how positively, how language changes over time.
- Analyst-revision summaries — counts and intensity of EPS estimate upgrades versus downgrades.
- Options-market signals — implied volatility level, skew, term structure, put-call ratio.
- Insider transactions — net buying versus selling by company executives.
- Short interest changes.

These features capture aspects of asset pricing that traditional price-volume features miss. They are particularly relevant for narrative-driven episodes.

### What the thesis would do

Build a sentiment-feature pipeline. Implement the corrected NEC. Train two versions: one with only traditional features and one with the expanded sentiment-augmented feature set. Compare performance on cross-sectional return prediction. Analyze whether the sentiment features specifically help during narrative-driven periods (like the dot-com era or the current AI cycle) and how they interact with the regime-detection mechanism of the MoE.

### Literature review

This is where the honest report becomes difficult. The space is meaningfully populated by recent (last 6–18 months) work.

**Direct precedents in 2024–2025.**

- **FTS-Text-MoE: Learning Explainable Stock Predictions with Tweets Using Mixture of Experts** (Xu et al., arXiv 2507.20535, July 2025). Combines numerical price data with summarized news and tweets in an MoE architecture. Reports improved Sharpe and returns over baselines. This is the closest direct precedent to sentiment-augmented NEC.

- **LLMoE: LLM-Based Routing in MoE for Trading** (Liu & Lo, arXiv 2501.09636, January 2025, AAAI workshop). LLM acts as the router and uses both price data and stock news. Multimodal real-world stock datasets.

- **H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts** (arXiv 2510.25091, October 2025). Three modalities: historical features, daily news encoded via Llama-3.2-1B, and timestamp embeddings. Style-structured MoE with shared market experts plus industry-specialized experts.

- **STONK / Towards Unified Multimodal Financial Forecasting** (arXiv 2508.13327, August 2025). Multimodal fusion of numerical features and sentiment-annotated textual features via cross-modal attention, with RoBERTa for domain-adaptive sentiment scoring.

- **DASF-Net** (MDPI 2025). Diffusion-aware sentiment fusion network. Uses FinBERT for sentiment extraction and diffusion over financial graphs. Not strictly MoE, but a sentiment-augmented multimodal architecture.

- **MERA: Mixture of Experts with Retrieval-Augmented Representation** (ACM Web Conference 2025). MoE for stock patterns with retrieval-augmented representations.

- **From news to trends: financial forecasting with LLM-driven news sentiment and selective state spaces** (Springer 2025). Sentiment integrated with state-space models for financial forecasting.

**The novelty problem.** The straightforward claim "we apply MoE with sentiment features to stock prediction" cannot be defended. Multiple direct precedents exist as recent as the past twelve months.

**Narrow novelty gaps that remain.** A few specific framings still have defensible novelty:

1. *Cross-sectional US equity ranking with rigorous financial-ML validation.* Most existing papers focus on single-stock movement or Chinese markets. They do not generally use walk-forward cross-validation with purging and embargoing, do not report deflated Sharpe ratios, and do not measure rank-IC rigorously. A contribution claim such as "we apply sentiment-augmented MoE to the cross-sectional ranking task on US equities with rigorous López de Prado–style validation" is defensible as a methodology-validation contribution.

2. *Sentiment-augmented MoE with regime-interpretability testing.* No paper I found tests whether learned experts in a sentiment-augmented MoE correspond to known financial regimes. The existing papers report performance metrics but do not analyze what their experts learn.

3. *Sentiment-augmented MoE combined with routing-mechanism comparison.* No paper systematically tests how routing-scheme choice interacts with the inclusion of sentiment features.

4. *Comparison against classical Hamilton-style regime-switching with sentiment proxies.* No paper compares sentiment-augmented MoE against a Hamilton-style baseline augmented with sentiment proxies.

**Strategic implication.** Sentiment-augmented NEC as a standalone primary contribution would be harder to defend than the other two directions. The supervisor or thesis committee can quickly point to FTS-Text-MoE, LLMoE, and H3M-SSMoEs as evidence that the basic idea has been done.

However, sentiment-augmented features combine naturally with either of the other two directions, where the headline contribution stays distinct. For example: "Routing-mechanism comparison in MoE for cross-sectional return prediction, with both traditional and sentiment-augmented features, to test whether routing-scheme choice interacts with feature modality." Or: "HMM-gated NEC with sentiment-augmented features, testing whether classical Markov-persistence gating handles sentiment-driven regimes differently from feature-conditioned gating." These hybrid framings preserve novelty while incorporating the sentiment angle.

### Strengths and weaknesses

**Strengths.** Most intellectually exciting subject matter, particularly given current interest in AI and semiconductor cycle dynamics. Sentiment features connect to a real research interest. Provides a feature-set extension that pairs well with either of the other two directions.

**Weaknesses.** Weakest standalone novelty due to crowded recent literature. Highest data-engineering cost (one to two months for a full FinBERT pipeline; two to three weeks if using precomputed academic sentiment datasets like Loughran-McDonald 10-K scores, IBES analyst revisions, and the Baker-Wurgler sentiment index). Less defensible as a primary contribution.

---

## Cross-cutting extension: Regime interpretability analysis

### What this extension is, in plain language

Once a Mixture-of-Experts model is trained, you can ask a question that the model itself does not directly answer: *what does each expert actually specialize in?* The routing decisions made by the gate produce a time series of expert assignments — at each date, for each stock, the model picked some expert. By analyzing this assignment time series, you can characterize what each expert has learned to handle.

The key question of the interpretability extension is: *do the learned expert assignments correspond to recognizable financial regimes that the finance literature already identifies?* Concretely, this means testing whether the model's experts align with things like:

- **High-volatility versus low-volatility periods** as defined by VIX levels.
- **Recession versus expansion periods** as defined by NBER business cycle dates.
- **Momentum crash periods** as defined by Daniel and Moskowitz (2016).
- **Cross-sectional dispersion regimes** as defined by the spread of factor returns.
- **Macroeconomic regimes** such as monetary easing versus tightening cycles.

The analysis is mechanically simple: you compute the time series of which expert was active when, and you correlate it with the time series of these known regime indicators. If a particular expert is mostly active during high-VIX periods and rarely active during low-VIX periods, you have evidence that the model has learned to detect volatility regimes from features. If the correlations are uniformly weak, you have evidence that the model is detecting structure that does not match the standard finance taxonomy — which is also an interesting finding.

### Why this adds value across all three directions

The interpretability analysis is not a separate thesis on its own — it is a complementary analytical layer that strengthens whichever primary direction is chosen. It converts black-box performance numbers into testable claims about *what the model has actually learned*. A thesis that reports "our model outperforms baselines by X" is weaker than a thesis that reports "our model outperforms baselines by X, and the gain comes from correctly identifying high-volatility regimes where the optimal predictor differs from the average-condition predictor."

For a SAIF committee with finance expertise, the interpretability chapter is also where the thesis most directly connects to classical finance theory. The empirical question shifts from "is this model better?" to "what does this model say about the structure of financial markets?" — which is a strictly stronger contribution.

The answer to your question is yes — interpretability analysis can be done for all three thesis directions. It is not exclusive to any one of them.

### How interpretability integrates with each direction

**Direction 1 (Routing-mechanism comparison).** Each routing scheme produces a different time series of expert assignments. Comparing the interpretability of experts across routing schemes is a natural extension of the main contribution. The empirical question becomes: not only which routing scheme produces the best out-of-sample returns, but also which routing scheme produces the most interpretable expert specialization. It is possible — and would be a publishable finding — that soft routing produces the highest accuracy while sparse top-k produces the most interpretable expert clustering, or vice versa. This adds a second axis to the comparison that goes beyond pure predictive performance. It also helps answer the question "which expert is which" that motivates the analysis in the first place.

**Direction 2 (HMM-gated NEC).** The interpretability fit is most natural here because the HMM states are explicit, named latent regimes by construction. The question becomes: do the learned HMM transition probabilities and emission patterns correspond to known financial regime classifications? Hamilton's original framework was designed to be interpretable — you could read off the regime probabilities at each date and compare them to economic conditions. HMM-gated NEC inherits this property. The interpretability chapter for Direction 2 essentially writes itself: report the HMM posterior regime probabilities, correlate with VIX and NBER, and compare against pure Hamilton-style models.

**Direction 3 (Sentiment-augmented NEC).** The interpretability analysis becomes specifically interesting here because you can test whether including sentiment features changes what regimes the model learns. Without sentiment features, the model might learn volatility regimes. With sentiment features, the model might learn additional narrative-driven regimes — bubble-mode, momentum-chase mode, post-shock recovery mode — that traditional features alone cannot detect. This is a substantive contribution to the question "what do narrative features actually add?"

### Implementation cost

The interpretability extension is comparatively cheap to add. It does not require training new models — it operates on the trained model's output (the expert-assignment time series) and on already-available regime indicators (VIX is free, NBER recession dates are free, momentum-crash dates are computable from CRSP data, Daniel-Moskowitz crash classifications can be reconstructed from published data).

The cost is roughly:

- Compute expert assignments for each stock-date in the test window: minutes of compute.
- Acquire regime indicator time series (VIX from FRED, NBER from NBER website, momentum crash dates from CRSP-derived computations): a few hours of data acquisition.
- Compute correlations, build visualizations, write the interpretability chapter: roughly one to two weeks.

Total budget: two to three additional weeks of work on top of the primary direction. Reasonable to include in any of the three directions without putting the timeline at risk.

### Contribution claims with interpretability added

Each direction's defensible contribution claim is strengthened by adding the interpretability extension:

- **Direction 1 with interpretability.** "We provide the first systematic empirical comparison of routing mechanisms in Mixture-of-Experts models applied to cross-sectional equity return prediction, *with analysis of how each routing scheme affects expert interpretability and correspondence to canonical financial regimes*."
- **Direction 2 with interpretability.** "We extend the Markovian-RNN paradigm to mixture-of-experts architectures by combining Hamilton-style Markov-persistence gating with separately-parameterized neural-network experts, *and demonstrate that the resulting model preserves the interpretability of classical regime-switching while gaining the flexibility of neural-network experts*."
- **Direction 3 with interpretability.** "We test whether sentiment-augmented Mixture-of-Experts models for cross-sectional equity return prediction learn meaningfully different regime structures than traditional-feature models, *and whether the additional regimes correspond to narrative-driven episodes such as bubble formation, momentum reversal, and post-shock recovery*."

### Verdict on the extension

The interpretability extension is a strict improvement for any of the three directions. It should be included in whichever direction is ultimately chosen — possibly as a dedicated chapter near the end of the thesis, or as an analytical section within the empirical results chapter. The supervisor pitch should mention it as a built-in component of the chosen direction rather than as a separate option.

---

## Comparison: novelty ranking and recommended pitch framing

### Updated novelty ranking

Based on the literature review, the three directions rank in clear order:

1. **HMM-gated NEC — strongest novelty.** Closest precedent (Markovian RNN, 2020) uses HMM gating with a single regime-conditioned RNN, not separately-parameterized MoE experts. The "HMM gate + MoE experts in finance" combination does not appear in the literature in the proposed form.

2. **Routing-mechanism comparison — strong novelty.** Adjacent work exists in language modeling (Nguyen et al. 2024 sigmoid vs softmax). MoE-for-stocks papers each pick one routing scheme. The systematic head-to-head comparison in cross-sectional return prediction has not been done.

3. **Sentiment-augmented NEC — weakest novelty as a standalone.** Six 2024–2025 papers cover overlapping ground (FTS-Text-MoE, LLMoE, H3M-SSMoEs, STONK, DASF-Net, MERA). Works as a feature-set extension paired with either of the other two, rather than as a standalone contribution.

### Recommended supervisor-pitch framing

The advice is to be transparent with the supervisor about the literature density of each direction, so the conversation focuses on which framing the supervisor finds most compelling rather than on litigating whether each idea is novel.

A sensible pitch script:

> "I'm considering three directions for the thesis, all built on the corrected NEC architecture for cross-sectional US equity return prediction. The first is a systematic comparison of routing mechanisms — soft, hard, sparse top-k, Gumbel-Softmax — to determine which wins in the low signal-to-noise regime of financial data. This head-to-head comparison has not been done in finance. The second is an HMM-gated extension that combines Hamilton-style Markov-persistence gating with separately-parameterized neural-network experts, extending the Markovian-RNN work of Karatzas and Ozyildirim from single-RNN to mixture-of-experts. This combination does not appear in the literature. The third is a sentiment-augmented extension that adds news-derived and options-market features to the standard NEC inputs. This direction has more recent precedents — FTS-Text-MoE, LLMoE, H3M-SSMoEs — so I would treat the sentiment angle as a feature-set extension that combines with one of the other two rather than as a standalone contribution. I would like your guidance on which of these you think is the most promising direction and which framing would land best with the SAIF committee."

This framing gives the supervisor clear information, demonstrates careful literature review, signals that the student understands what is and is not novel, and offers genuine flexibility on direction.

### Hybrid combinations that preserve novelty

Several hybrid framings combine the three directions while preserving the strong novelty of HMM-gated NEC or routing-mechanism comparison:

- *HMM-gated NEC with traditional + sentiment-augmented features.* Headline contribution remains HMM-gated MoE. Sentiment features become a feature-set ablation in the empirical chapter.
- *Routing-mechanism comparison with both feature sets.* Headline contribution remains the routing comparison. Sentiment features become a robustness ablation testing whether routing-scheme choice interacts with feature modality.
- *HMM-gated NEC with routing-mechanism ablation.* Two methodological contributions combined: introduce HMM-gated MoE, then within it test different routing schemes for the inner expert selection.

Any of these hybrid combinations gives a stronger thesis than picking only one direction in isolation.

---

## What we do next

This document is intended as the basis for the supervisor pitch conversation in July 2026 (when advisor matching happens at SAIF, June 29 – July 26, 2026). Tom is *not* yet writing the formal proposal — that step waits until after the supervisor is matched and gives direction.

When the supervisor has been assigned and has given input on which direction they support, the next steps are:

1. Formalize the chosen direction into a topic statement (a one-paragraph description of the thesis).
2. Draft the central empirical question with formal hypotheses.
3. Build the experimental design (universe, features, baselines, evaluation protocol).
4. Submit the Proposal Registration Form by September 13, 2026.
5. Pass the proposal defense October 12–18, 2026.

This timeline is consistent with the SAIF MF program agenda (Chapter 5 of the Basic Requirements PDF).

---

## Quick reference: papers cited

### Direction 1 (Routing-mechanism comparison)

| Paper | Year | Relevance |
|-------|------|-----------|
| Shazeer et al., Sparsely-Gated MoE | 2017 | Foundational MoE paper; noisy top-k routing |
| Fedus, Zoph & Shazeer, Switch Transformer | 2022 | Top-1 routing at scale |
| Nguyen et al., Sigmoid vs Softmax Gating (arXiv 2405.13997) | 2024 | Closest routing-comparison precedent (language modeling, not finance) |
| Mixtral of Experts (Jiang et al.) | 2024 | Top-2 routing reference |
| MIGA (arXiv 2410.02241) | 2024 | MoE for Chinese stocks; uses top-k routing only |
| LLMoE (arXiv 2501.09636) | 2025 | LLM as router; one routing scheme |
| Adaptive Market Intelligence (Vallarino, arXiv 2508.02686) | 2025 | Volatility-aware gating; one routing scheme |
| MERA (ACM Web Conference) | 2025 | MoE with retrieval-augmented routing |

### Direction 2 (HMM-gated NEC)

| Paper | Year | Relevance |
|-------|------|-----------|
| Hamilton, *A New Approach to the Economic Analysis of Nonstationary Time Series* | 1989 | Classical regime-switching foundation |
| Karatzas & Ozyildirim, Markovian RNN (arXiv 2006.10119) | 2020 | Closest precedent: HMM-gated single RNN |
| Hidden Markov Neural Networks (HMNNs) | various | Factorial HMM + NN; not MoE |
| GenHMM | 2019 | HMM with neural generative models per state |
| Neural HMM with Adaptive Granularity Attention | 2026 | High-frequency order flow modeling |
| Guidolin, *Markov Switching Models in Empirical Finance* | 2011 | Comprehensive survey |

### Direction 3 (Sentiment-augmented NEC)

| Paper | Year | Relevance |
|-------|------|-----------|
| FTS-Text-MoE (arXiv 2507.20535) | 2025 | Direct precedent: MoE with tweets/news for stocks |
| LLMoE (arXiv 2501.09636) | 2025 | LLM-based routing using price + news |
| H3M-SSMoEs (arXiv 2510.25091) | 2025 | Multimodal MoE with LLM-encoded news |
| STONK (arXiv 2508.13327) | 2025 | Multimodal fusion with RoBERTa sentiment |
| DASF-Net (MDPI) | 2025 | FinBERT-based sentiment fusion with diffusion graphs |
| MERA (ACM Web Conference) | 2025 | MoE with retrieval-augmented representation |
| From news to trends (Springer JIIS) | 2025 | LLM-driven sentiment with SSMs |
| Loughran-McDonald, dictionary-based 10-K sentiment | 2011, 2016 | Foundational financial-text sentiment |
| Baker-Wurgler, sentiment index | 2006, 2007 | Foundational market-level sentiment |
| Tetlock, *Giving Content to Investor Sentiment* | 2007 | WSJ sentiment predicts returns |
