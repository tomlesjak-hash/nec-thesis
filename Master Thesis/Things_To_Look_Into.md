# Things To Look Into

Running list of open theoretical and methodological questions for the thesis. Separate from
`Advisor_Questions.md`, which holds design choices that need the advisor to decide. This file holds
topics I need to read up on or work out myself, whether or not the advisor is involved.

Last updated: 15 September 2026

**Index**

1. Distribution shift and whether regimes are distributions
2. Queue (not yet written up)

---

## 1. Distribution shift and whether regimes are distributions

**The question.** Are the training and test distributions different in financial data? And is entering
a new regime the same thing as entering a new distribution, such that assumptions have to change and
some error mitigation is needed?

### 1.1 Three kinds of shift, only one of which is dangerous

The standard taxonomy is in Quiñonero-Candela, Sugiyama, Schwaighofer and Lawrence, *Dataset Shift in
Machine Learning*, MIT Press 2009.

**Covariate shift.** P(X) changes while P(Y|X) stays fixed. The characteristics are distributed
differently across periods, but the mapping from characteristics to returns is stable.

**Prior or label shift.** P(Y) changes. Return volatility and mean vary over time, which is
uncontroversially true.

**Concept drift, also called conditional shift.** P(Y|X) itself changes. Momentum pays in one period
and reverses in another. This is the damaging one, because reweighting the inputs cannot fix it.

The thesis is about the third. Worth stating plainly, because the first two are largely handled by
things the literature already does.

### 1.2 The regime intuition, made precise

Write s_t for the regime at date t. A regime switching model says that conditional on s_t = k the data
follow a distribution F_k, and the marginal distribution actually observed is the mixture
Σ_k P(s_t = k) · F_k. Entering a new regime is entering a different conditional law. This is exactly
Hamilton's construction (Hamilton 1989, *Econometrica* 57(2), 357 to 384).

**The subtlety that matters.** If the chain governing s_t is ergodic, meaning irreducible and aperiodic
so that every regime keeps recurring, then the whole process is **strictly stationary**. The
non-stationarity is local, not global. There is one fixed joint law, and what looks like drift is the
chain moving through states of a distribution that never actually changes.

Consequence: **under ergodicity the test set is not out of distribution.** It is a draw from the same
joint law as the training set, with a different realised mix of regimes. The classical learning problem
is intact. What fails is not the theory but the estimator, because a single unconditional model must
average over regimes and so averages a relationship that is not constant.

### 1.3 What the binding requirement actually is

Not stationarity. **Regime coverage.**

If every regime present in the test period also appeared in the training period, the model is
interpolating within a known mixture. If a regime appears at test time that never appeared in training,
it is genuine extrapolation and nothing in the architecture or the theory protects against it.

This is the same constraint as the earlier count of roughly ten to fifteen usable stress episodes in
forty to sixty years. That number is not about sample size in the ordinary sense. It is about how many
draws are available from the rare mixture components, and it is the substantive argument for CRSP over
a short sample.

### 1.4 What genuinely breaks

**Regime switch versus structural break.** A regime switch recurs; a structural break does not.
Regulation NMS, decimalisation, the growth of passive ownership are one way doors and violate
ergodicity outright. Reference for detection: Bai and Perron 1998, *Econometrica* 66(1), 47 to 78. The
only defence is to avoid training across a break, or to downweight far history, which is what the
exponential decay window already does. **To decide: whether to test for breaks formally or handle it
purely through the decay window.**

**Independence.** Never held anyway. Returns are correlated within a date, and regimes make that
correlation cluster in time. This does not stop the estimator being consistent, but it destroys any
standard error computed as though observations were independent. Needs clustering by date and a block
bootstrap for inference, with blocks long enough to contain whole regime episodes.

**Validation set representativeness.** Validation is only a valid model selection device if its regime
mix resembles the test period's. With a 63 day test window in walk forward this is often violated. It
argues for judging a specification on the distribution of results across all thirty windows rather than
on any single one.

### 1.5 Something already engineered away

GKX and most of this literature rank normalise every characteristic cross sectionally to [-1,1] each
period. That removes most covariate shift by construction, since the marginal distribution of each
input is fixed at every date by design. If the thesis follows that convention, P(X) is roughly stable
and essentially all remaining shift is conditional. Convenient, since conditional shift is what the
thesis claims to model. **To verify: that this normalisation is in fact being adopted, and that it does
not destroy level information the gate needs.**

### 1.6 The diagnostic to run

**Domain classifier.** Label observations by period, train a classifier to tell the periods apart, look
at the AUC. AUC near 0.5 means the periods are indistinguishable in feature space. This is the
empirical version of the H-divergence in Ben-David, Blitzer, Crammer, Kulesza, Pereira and Vaughan
2010, *Machine Learning* 79(1-2), 151 to 175, whose bound states that test error is bounded by train
error plus this divergence plus an irreducible term.

It is the domain adaptation analogue of the VC question in Q17, and unlike VC it is computable and non
vacuous.

Run it twice: once on P(X), once on the residuals of a fitted model. If P(X) looks stable but residual
behaviour separates cleanly by period, conditional shift is isolated as the source. That is a one
paragraph result that justifies the whole architecture.

### 1.7 Why the architecture is the mitigation

The classical fix for covariate shift is importance weighting, reweighting training observations by the
density ratio of test to train (Shimodaira 2000, *Journal of Statistical Planning and Inference* 90(2),
227 to 244). It does **not** help under conditional shift. The fix for conditional shift is to condition
on the variable that indexes the shift.

Which is what a regime gated mixture of experts does. The gate is an estimate of which mixture component
the data are in; the experts are the component specific conditional models. So the model is not
something that suffers from distribution shift and then gets patched. It is an estimator built for it.

**Possible change to Section 2.** Reframing from "regimes exist and might matter" to "conditional shift
is the problem and conditioning on the regime is the estimator" is stronger and more standard. It also
supplies a clean null: if there is no conditional shift, the gate should collapse to uniform and the
corrections to zero. That is falsifiable rather than hopeful. **Not yet applied to the proposal.
Decide first.**

**Caveat on the claim.** The gate must estimate the regime from information available at date t. If
regime identification is only reliable in hindsight, the architecture works in sample and fails live.
This is a real risk with smoothed probabilities from a Hamilton filter, and argues for pre-committing
to **filtered** rather than smoothed estimates.

### 1.8 Reading, roughly two hours

- Quiñonero-Candela et al. 2009, chapter 1 only, for the taxonomy.
- Ben-David et al. 2010, sections 1 to 4, for the divergence bound and the classifier construction.
- Shimodaira 2000, skim, enough to see why importance weighting stops at covariate shift.
- Hamilton 1994, *Time Series Analysis*, chapter 22, for the stationarity and ergodicity conditions.

---

## 2. Queue (not yet written up)

- Verify references [21], [22], [23] in Section 1 (titles reconstructed, not confirmed).
- Verify [11], the ECoG / REW-MSLM paper, PMID 35234665.
- Verify author lists for [14] and [15].
- Verify RAVEN's 0.036 to 0.039 IC and ICIR figures against the source.
- Verify PRISM-VQ codebook size 512 and the persistence figures.
- Find the exact premise sentence in Ang and Timmermann rather than paraphrasing it.
- **No Chinese language or CNKI sources anywhere in the proposal.** Shanghai committee. Search
  机制转换 and 马尔可夫区制转换 together with 股票收益.
- Decide whether the persistence counting argument from Q17 becomes a short theoretical section.

## 3. From reading the existing code, 2026-09-21

Full factual record in `Code_State_2026-09-21.md`; work items in `Code_Change_Brief_2026-09-21.md`.
Only the things that need thinking about rather than coding are listed here.

- **SETTLED 2026-09-21: the gate is fitted separately and frozen, for all four gates.** The code
  trains it jointly, so the code implements a model the thesis is no longer making. Full record in
  Q19 of `Advisor_Questions.md`. The separate-fit interface is now the first thing to build.
- **The load-balancing term is inert under the decided design**, because a frozen gate has no
  trainable parameters for `lambda_LB` to act on. This is a real methodological note and can be
  written up, with the contrast against Ye & Borde and Shazeer et al., where the gate is trained
  and the term does work. It is *not* true of the code as it stands today, whose gate is trainable;
  keep the two statements apart.
- **Pin down the exact DeepSeek citation** for the point that jointly optimised routing does better
  on the fitted objective, which is the objection the frozen-gate choice has to pre-empt.
- **The existing HMM prior is not Hamilton's Markov switching model.** It learns a homogeneous,
  covariate-independent transition matrix by backpropagation through the forward filter, jointly
  with the emissions. Decide how to name it in the methodology, and whether a conventionally
  estimated reference model is needed to support the naming.
- **Encoder and gate are dead parameters under the HMM prior** with snapshot-only expert input,
  because that prior ignores the gate logits. Any table comparing prior kinds is comparing
  different live parameter counts. Decide whether to equalise or to report.
- **Expected regime duration `1/(1-A_kk)` is the quantity the persistence-counting argument is
  about**, and it is computable from a fitted transition matrix at no cost. Worth writing into the
  theoretical section alongside the counting argument.
- **Ye & Borde is arXiv:2608.12251** (from Q5); the residual architecture it specifies is not
  implemented in the code.
