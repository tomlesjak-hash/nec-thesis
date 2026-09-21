# Thesis Supervisor Meeting — Prep Notes

**Prepared:** 20 July 2026
**Purpose:** pitch three candidate directions for the NEC → gated-Mixture-of-Experts thesis and get the supervisor to steer the *scientific* choice. The engineering is largely done.

---

## 30-second summary

The thesis takes **NEC** — a crude regime-conditional predictor that already exists in my repo — and studies it rebuilt as a **proper gated Mixture-of-Experts (MoE)** for **cross-sectional US-equity return prediction**. The corrected baseline is already implemented and passing tests, and it's deliberately architected so that two of the three ideas below drop in without a rewrite. I'm bringing three directions and want the supervisor to pick the one that fits their taste and expertise. Framing is **ML-first, with finance as the empirical domain** — safe given the physics/statistics-leaning committee.

---

## Where things stand (lead with this — strongest card)

- **Corrected NEC baseline: implemented and passing** — a green multi-test suite, deterministic. K = 2 experts, a small GRU encoder, a soft gate trained **end-to-end** by the mixture negative-log-likelihood, snapshot-based experts.
- **The baseline was built to serve either Idea 1 or Idea 2 with no rewrite.** The gate is decomposed into a *shared expert-likelihood core* plus a *pluggable prior*: memoryless (Idea 1) or Markov-transition / forward-filter (Idea 2). The HMM gate is a strict generalization of the routing gate, not a separate codebase.
- **Comparison scaffolding already exists:** capacity-matched single-model baselines (Ridge, MLP), a classical **Hamilton** regime-switching baseline, a point-in-time S&P 500 universe (survivorship-aware), and López de Prado-style validation (walk-forward with purging, deflated Sharpe, a trial registry with selection-aware inference).
- **Translation for the meeting:** the plumbing is done and tested. What I need from you is the scientific direction — not a feasibility check.

---

## Background: what NEC is, and why it's the anchor

**NEC = "Normal-versus-Extreme Conditional."** The prototype (an LSTM plus a 3-class classifier) hard-routes each sample to either a "normal" or an "extreme" regressor. It's a sound idea, crudely executed.

Its documented defects *are* the research questions: hard `argmax` routing (the 3 classes collapse to a 2-way switch), a gate trained on separate hand-made labels instead of end-to-end, only two fixed experts, an over-parameterized 8-layer LSTM, and a silent undocumented feature slice, among others.

**Why anchor a thesis on a flawed model** — worth saying explicitly, because it pre-empts the obvious "why not start clean?" question:

- It gives me a **baseline and a before/after story** — a thesis needs a "before."
- It **hands me the research questions** for free; each defect is a chapter.
- It keeps a **finance-native idea** (regime-conditional prediction — the Hamilton lineage) while upgrading the machinery.
- It's the right **career signal**: "I diagnosed messy, production-style code and rebuilt it rigorously with ML theory" beats "I re-implemented a paper."

**The fix in one line:** shared encoder → learned soft/sparse gate → *N* emergent experts → weighted blend, all trained on a single loss. Every idea below is just *which box on that picture I touch*.

---

## The three ideas

| Idea | What changes | Core question | Novelty | Cost / risk |
|---|---|---|---|---|
| **1. Routing comparison** *(= Variation 2)* | the gate's **rule** | Which routing scheme wins on low-SNR returns? | Strong in finance | **Lowest** — swap one component on the working baseline |
| **2. HMM-gated NEC** *(= Variation 3)* | the gate becomes an **HMM** | Does persistence-aware gating beat a memoryless gate? | **Highest** | Medium — joint training of gate + experts, choose # regimes |
| **3. Sentiment-augmented** | the encoder's **inputs** | Do narrative signals add predictive power? | Weakest alone | Medium–high — text-data pipeline |

### Idea 1 — Routing-mechanism comparison (the gate's rule)

First systematic head-to-head of **soft / hard top-1 / sparse top-k / Gumbel-softmax** routing in an MoE for cross-sectional return prediction. The hypothesis is that the **low signal-to-noise ratio** of returns favors a different routing scheme than the high-SNR language-modeling settings where these mechanisms were designed — a confident "pick one" gate may latch onto noise, while soft blending hedges. Adjacent ML work exists (Nguyen et al. 2024, sigmoid vs. softmax routing) but nobody has run this for returns. Lowest engineering risk, cleanest experiment, and it **pays off even as a null result** ("the ChatGPT-style defaults don't transfer to finance" is a legitimate finding).

**Pitch line:** *"Do the routing tricks from large language models survive contact with noisy financial data?"*

### Idea 2 — HMM-gated NEC (swap the gate for a Markov filter)

Replace the learned feed-forward gate with a **Hidden Markov Model** gate: a persistence-aware belief over regimes (a learned K×K transition matrix applied through forward filtering) routes separately-parameterized neural experts. This extends the "Markovian RNN" (Karatzas–Özyıldırım, 2020), which did HMM-gated *single* RNN; **nobody has done HMM-gated multi-expert MoE for the cross-section.** Most novel of the three, strongest finance story ("Hamilton's regime-switching, but the per-regime models are modern neural experts"), and the most **interpretable** almost for free — each expert corresponds to a regime and the gate is the probability of being in it. The trade-off is more moving parts: gate and experts learned together, and the number of regimes to choose.

**Pitch line:** *"Hamilton's regime-switching model, with the linear per-regime pieces replaced by flexible neural experts."*

### Idea 3 — Sentiment-augmented NEC (widen the inputs)

Add **narrative features** to the feature set: news/text sentiment (FinBERT, or Loughran–McDonald word-lists), options-market signals (implied volatility, put/call ratios), and analyst revisions (IBES). This is the natural home for my **AI/semiconductor-bubble interest**, since bubbles are narrative-driven. But it's the **weakest standalone**: at least six 2024–25 papers already staple sentiment onto MoE stock models (FTS-Text-MoE, LLMoE, H3M-SSMoEs, STONK, DASF-Net, MERA), so I **cannot** claim "first sentiment MoE for stocks." It works best as a **second axis bolted onto Idea 1 or 2**, where the novelty is in the *interaction* (does the winning routing change with text features? does the HMM find different regimes once it can read the news?). Cost is the honest downside — roughly 1–2 months for a full text pipeline, or 2–3 weeks leaning on precomputed academic datasets (Loughran–McDonald, IBES, Baker–Wurgler).

**Pitch line:** *"A feature-set extension that combines with either of the above — pitched honestly, not as a standalone contribution."*

---

## Cross-cutting extension: regime interpretability

Whichever direction is chosen, a light add-on (~2–3 weeks, no new training) asks: **do the learned experts correspond to known financial regimes** — VIX spikes, NBER recessions, Daniel–Moskowitz momentum crashes? No paper does this systematically for the cross-section. It's most natural for Idea 2 (the regimes are explicit) but strengthens all three, and I'd treat it as built-in for whichever primary is picked, not a separate option.

---

## How I'll pitch it (novelty + honesty)

- **Novelty ranking: Idea 2 > Idea 1 > Idea 3.**
- Present **Ideas 1 and 2 as the two standalone candidates**, and **Idea 3 as a combinable feature extension** — and be transparent about the crowded sentiment literature. Knowing the landscape is a strength; overclaiming novelty is the one thing that could hurt me.
- **Two headline framings** to offer, so the supervisor can pick the one that fits their field:
  - *ML-first* — "the first rigorous study of MoE routing for cross-sectional return prediction."
  - *Interpretability-first* — "interpretable regime detection via machine learning."
  - Not leading with "modernize Hamilton" as the headline — that's the structural motivation, not the pitch.

---

## Backup if none land

Standard corrected-NEC architectural benchmark vs. the classical Hamilton regime-switching model and the single-model ML baselines (Ridge, MLP; LightGBM available behind an import guard). Clean, defensible, and already largely built — a safe fallback if the supervisor wants something lower-risk.

---

## Likely questions (and my answers)

- **"Isn't Idea 1 just benchmarking?"** → The low-SNR hypothesis makes it a scientific question, not a leaderboard: I'm predicting *why* the finance winner should differ from the language-model winner.
- **"How do you keep the comparison fair?"** → Everything held fixed but the routing rule; walk-forward validation with purging and the deflated-Sharpe / selection-aware machinery already in the repo.
- **"Idea 2 sounds complex — can you actually train it?"** → The forward-filter gate already ships with causality/correctness tests; joint training is the plan, with a two-stage fallback (fit the HMM, then the experts) if needed.
- **"Sentiment's been done."** → Agreed — that's exactly why it's an extension, not a standalone; the novelty is the interaction with routing/gating plus rigorous validation.
- **"How many regimes / experts?"** → Treated as a hyperparameter with an explicit selection criterion; I'll report sensitivity rather than hard-code it.
- **"What data?"** → Staged: synthetic → free daily US prices on a point-in-time universe → regime overlay → (extension) fundamentals/sentiment.

---

## What I want to walk out with

1. **Which standalone direction** — routing comparison (Idea 1) or HMM-gated (Idea 2) — fits your taste and expertise.
2. Whether to **commit sentiment (Idea 3) as an extension** or keep it in reserve.
3. **Preferred framing** — ML-first or interpretability-first.
4. Any **scope guardrails** — universe (currently US equities, to be narrowed), sample window, and timeline expectations.
