# Questions for the Advisor

Running list. Add as they come up; mark resolved with the date and the answer.

---

## ADVISOR MEETING, 18 September 2026

The arguments are sound but the scope is not. His words were that everything as written will be
impossible, and that on several points I have to be subjective: pick something, argue for it, and move
on. The governing constraint is training cost, so the architecture choices get settled only once the
compute budget is actually calculated rather than guessed at.

What this changed:
- **Data is CRSP.** The universe is US equities. Wind remains a possibility for a China extension but
  is not the primary source. This closes Q3.
- **The mechanism-by-depth grid survives** and he told me to do it. The number of depths it carries is
  set by the compute budget, not chosen in advance. This closes Q11.
- **The width sweep is out.** Width follows the pyramid rule from a fixed first layer, and only the
  first hidden layer materially affects the parameter count anyway.
- **No sweep over the expert count.** K is fixed by argument and held across all four gates. K itself
  is still unchosen, which is Q18.
- **Start experimenting now** with the base model and the most generic loss function. Refine the loss
  later rather than designing it up front.
- **Defining generic market conditions comes first.** Whether the base has generalised to them is not
  answerable until that phrase has an operational definition.

Sequence he set: data, then compute budget, then parameters. Possible access to a university compute
node rather than my own machine.

---

## OPEN

### Q28. How is the industry-level gate implemented?

**Status:** open, raised 2026-10-08 when Q1 was decided (partly stock-specific, industry level). Tom wants
the implementation kept open for now.

**What is fixed by Q1.** Each stock's gate weight combines the shared market regime with a regime of its
industry; both parts are fitted separately and frozen (Q19). The networks' training cost does not change,
because frozen gate weights are only looked up per row; the extra cost is in fitting the gates.

**Candidate forms.**

- **(a) Factorial gate.** Two hidden chains side by side, market and sector, independent of each other
  (Ghahramani & Jordan 1997). One expert per pair of market state k and sector state l:

$$\pi_{(k,l)}(i,t)=\pi^{\text{mkt}}_k(t)\,\pi^{\,s(i)}_l(t)$$

  where s(i) is stock i's sector. 2 × 2 states gives 4 experts, inside the K ≤ 4 envelope. Most
  expressive; more experts, less data per expert.
- **(b) Coupled chains.** As (a), but the sector chain's transition probabilities depend on the market
  state, so sector stress is more likely when the market is stressed. More realistic, harder to fit and
  explain.
- **(c) Sector gate only, market regime as an input.** Each stock's gate weight is its sector's regime
  probabilities (ordered by variance so labels mean the same in every sector); the market regime enters
  elsewhere. Simplest to fit, but the market regime loses its central role.
- **(d) Pooled industry chains.** One model with shared regime definitions and transition matrix, each
  sector following its own chain. Labels comparable across sectors; one fit per fold; needs a native
  implementation.

**Sub-questions.**

1. What the industry chain is fitted on: sector-relative returns (sector minus market; the 2026-10-08
   suggestion, since raw sector returns mostly repeat the market) or raw sector returns.
2. Granularity: 11 GICS sectors, or finer industry groups (fewer stocks per group).
3. Number of states for each level (ties to Q18).
4. How each of the five gate types gets its industry counterpart.
5. Identification (Q10): the experts already see sector dummies, industry momentum and within-industry
   reversal; the pilot must show the industry regime adds information beyond them.
6. A descriptive check on 2000-2009 only: how often a sector is stressed while the market is calm.

**Ask the advisor:** which form, and is the factorial expert count acceptable given the data per expert?

---

### Q29. Should a gate variant define regimes by factor returns instead of the market return alone?

**Status:** open, raised 2026-10-08. Nothing decided; Q23 (the Hamilton gate on the CRSP S&P 500 index)
stands until this is answered.

**How the current gate reads the regime.** The Hamilton gate sees one series, the market's daily return.
Each regime has its own mean and volatility; every day the filter updates its belief by Bayes' rule from
yesterday's belief (moved forward by A) and how likely today's return is under each regime:

$$\xi_{t\mid t}(k)=\frac{f_k(r_t)\,\big[\xi_{t-1\mid t-1}^\top A\big]_k}{\sum_{j} f_j(r_t)\,\big[\xi_{t-1\mid t-1}^\top A\big]_j}$$

Daily means are tiny next to daily volatility, so in practice the regimes are volatility states, and the
persistence in A stops the gate flipping on single noisy days.

**Why stock characteristics do not belong in the gate.** After ranking, a characteristic is uniform on
[-1, 1] every day, so it says which stocks, not what state the market is in. It belongs in the experts.
Market-level versions (factor returns, market volatility and trend, cross-sectional dispersion, VIX,
spreads) are legitimate gate inputs, and the planned gates already use some: the jump model a feature
vector of the market series, TVTP covariates for the switching probabilities, the Wasserstein gate the
return distribution over a window.

**The idea.** The experts predict cross-sectional returns, so the regimes that matter most may be
**factor regimes** rather than market-volatility regimes. Example: momentum crashes happen in market
rebounds, when volatility is falling (Daniel & Moskowitz 2016), which a market-volatility gate can miss.
A gate variant whose observation is the vector of daily factor returns would define regimes by how the
cross-section behaves:

$$f_t=(r^{\text{mkt}}_t,\;r^{\text{size}}_t,\;r^{\text{value}}_t,\;r^{\text{mom}}_t)^\top,\qquad f_t\mid z_t=k\sim\mathcal N(\mu_k,\Sigma_k)$$

Precedent for multivariate regime models: Liu, Maheu & Song (2024), JAE 39(5) (cited in Q1).

**Costs and risks.**

- More parameters: a 4 × 4 covariance matrix per regime (10 numbers instead of 1), so less stable
  estimation and regime labels.
- Overlap with the experts' inputs (Q10): the experts see size, value and momentum characteristics; a
  gate built on their factor returns is a different object (time series, not cross-section), but the
  pilot must check the gate adds information.
- Data source: factors built from the CRSP/Compustat panel (consistent with Q23, but small- vs large-cap
  spreads inside the S&P 500 are narrow) or Kenneth French's daily factors (standard, external file).

**Sub-questions.**

1. A sixth gate type, or the input to an existing one (Hamilton made multivariate, or the jump model's
   feature vector)?
2. Which factors: market, size, value, momentum; also short-term reversal or cross-sectional dispersion?
3. Source: built from the panel, or French's factors?
4. Interaction with Q28: should the industry level use the same kind of input?
5. Compute and the K ≤ 4 envelope (Q18): full or diagonal covariance per regime.

**Ask the advisor:** is a factor-regime gate a worthwhile addition to the comparison, or scope creep?

---

### Q5. How do I position against Ye & Borde (arXiv:2608.12251) without overclaiming?
**Status:** open — raised 2026-09-11 after reading the paper in full

**What the paper actually does.** Despite the name, RG-ResMoE contains **no regime model**. Its
"regime state variable" is two numbers: 20-day rolling volatility of the equal-weighted market
return, and 20-day rolling volatility of the residual return per stock. The gate is a learned
softmax on those features plus stock features. No latent state, no transition matrix, no
persistence, no filter. Its axis is *where* regime information enters the network (input vs gate);
mine is *what generates* the state. Its target is 5-day realised volatility, not returns.

**Why this is good for me.** Their own Table 8 shows the IC advantage is 4.3x the full-sample gain
in the top volatility decile and 6.7x during COVID, but only **0.6x in regime-transition windows** —
below the full-sample gain. They write that the benefit is "greatest during sustained periods of
elevated market volatility rather than during regime transitions themselves." A gate built from a
regime *process*, with transition dynamics and persistence, is aimed squarely at that weakness.
Their limitations section also names "alternative forms of market state" as future work.

**The honest problem.** If I adopt their architecture — frozen base, zero-initialised residual
experts, soft gate, shrinkage plus load-balancing penalties — my architectural contribution is zero.
I think that is fine and I should state it openly: the architecture is theirs, adopted so the
comparison is clean, and my contribution is the gate.

**Ask the advisor:** is adopting a very recent paper's architecture wholesale acceptable for a
master's thesis, provided it is clearly attributed and the contribution is isolated to the gate? Or
does the thesis need architectural distance from it as well?

---

### Q6. Residual MoE, or a standard MoE?
**Status:** empirical half **SETTLED by the paper 2026-09-11**; design half still open

**RESOLVED — the comparison exists, and I should not claim it as a contribution.** Ye & Borde do run
a standard MoE with the same gate. Their §3.2 defines it explicitly: each expert produces a full
forecast, `ŷ = Σ_k π_k e_k(x)`, and **the gate receives the same `u = (x, z)`** as RG-ResMoE's gate.
Everything is trained jointly. It appears in Table 3, Table 6 and the Japanese replication:

| Variant (US panel) | IC | ΔIC vs RG-ResMoE | Collapsed seeds |
|---|---|---|---|
| RG-ResMoE (frozen base, zero-init experts) | 0.5469 ± 0.0012 | ref. | **0 / 30** |
| RG-ResMoE, random init instead of zero init | 0.5430 ± 0.0021 | −0.0039 | 13 / 30 |
| Standard MoE (no base, joint training) | 0.5413 ± 0.0027 | −0.0056 | **24 / 30** |
| MLP-L (capacity-matched, no MoE) | 0.5421 ± 0.0033 | −0.0048 | 3 / 30 |

Japan reproduces the ordering: MoE 0.4805 with 22/30 collapses against RG-ResMoE 0.4858 with 1/30.

So the sentence "the principal contribution comes from the residual architecture" is backed by
Table 6, not merely inferred from the expert-count sweep. **Arm A versus arm C is replication, not
novelty. Do not put it in the proposal as an original result.**

**But read the decomposition carefully — it changes the design.** Of the 0.0056 IC gap between the
standard MoE and RG-ResMoE, **0.0039 (about 70%) disappears simply by initialising the residual
experts randomly instead of at zero**, and the collapse count goes 0 → 13 on that change alone. Most
of the benefit is therefore not "having a base network" in the abstract; it is *starting at the base
and being penalised for moving far from it* — zero initialisation plus the shrinkage term. That is a
training-stability mechanism, and it should be described as one rather than as an architectural
insight.

**THE CAVEAT THAT MATTERS MOST FOR THIS THESIS.** The headline evidence for the residual design is
a stability result — 0/30 against 24/30 — and stability there is defined as **mean test QLIKE > 2**.
QLIKE is a volatility-specific loss (`log σ̂² + σ²/σ̂²`), undefined for a signed return forecast, and
it explodes precisely when a volatility forecast approaches zero. **That failure mode does not exist
for cross-sectional returns**, where the target is signed and centred near zero. So the single
strongest piece of evidence for the residual architecture may not transfer to this thesis's target
at all. Before relying on it, a stability criterion appropriate to returns has to be defined — seed
dispersion of rank IC, and the count of seeds whose out-of-sample IC falls below some threshold —
and it should be stated in the proposal that the transfer of the stability result is an open
empirical question, not an assumption.

**What remains open is the design argument, not the horse race.** Even granting that residual wins
on accuracy and stability for volatility, the reason to adopt it here is different: it isolates the
gate's contribution in the single term `Σ_k π_k r_k(x)`, which can be measured, plotted through
time, and tested for concentration in transition windows. The thesis is an experiment *about the
gate*, so the architecture that makes the gate's effect a measurable quantity is the right one
regardless of which wins on IC.

**Ask the advisor:** is "we chose the architecture that isolates the variable under study" a
sufficient justification to state in the proposal, given that the accuracy comparison is already
settled in the literature for a different target?

---

### Q8. Expert and base architecture — what is settled by literature and what is not

**Update 2026-10-05 (compute envelope, Tom's decision).** Measured on the laptop (Apple M4 Pro; Progress_Tracker
section 1c), the whole design runs in about 2-6 days of machine time if it stays within a depth scan of 1-3
(pyramid widths), a first width of at most 128, and K at most 4. Depths 4-5 and a first width of 256 are ruled
out by compute (1-1.5 months). The widths and K inside the envelope are still open (this question, Q17, Q18).

**Partly settled 2026-09-19.** Width is no longer a free parameter: it follows the pyramid rule
(halve at each layer) from a fixed first hidden layer, and no width sweep is run. The first hidden
layer carries roughly 83% of the parameters at d=100, so the rest of the widths barely matter. Depth
remains open and is swept, see Q11. Base/expert symmetry is still open.

**Status:** open — raised 2026-09-12. Partly resolved by literature, partly requiring an experiment.

**SETTLED — why a multilayer perceptron at all.** This does not need the advisor and does not need an
experiment; the argument is available from the literature and should simply be written down.
1. **Data representation determines architecture.** The finance MoE literature splits cleanly. Where
   one model observation is a point-in-time vector of firm characteristics with no time axis, the
   experts are multilayer perceptrons (Ye & Borde; the empirical asset pricing literature). Where one
   observation is a T x D window of a stock's history, the experts are temporal models — TCN, LSTM,
   Transformer (MIGA, RAVEN, PRISM-VQ). Nobody uses a Transformer on a tabular characteristic panel
   and nobody uses an MLP on a raw price sequence. Choosing the characteristic panel therefore
   chooses the MLP.
2. **Temporal experts would compete with the gate.** This is the strongest reason and it is specific
   to this thesis. If each expert can itself model temporal structure, it can absorb the
   regime-conditional variation that the gate is supposed to supply, and the contribution being
   measured leaks into the experts. The object of study is the gate; the experts must therefore be
   the simplest thing that can represent a cross-sectional mapping.
3. **Feasibility.** Roughly six gates, several pilot configurations, thirty walk-forward windows and
   thirty seeds is tractable with compact perceptrons and is not tractable with Transformer experts.
4. **Comparability.** The closest precedent uses compact MLP blocks, so matching it keeps the
   comparison with that paper meaningful.
5. **Benchmark class.** Gu, Kelly and Xiu establish feed-forward networks as the benchmark model
   class for exactly this problem, and find shallow beats deep because of the weak signal.
**The objection to pre-empt:** Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022, arXiv:2207.08815) show
gradient-boosted trees still beat deep learning on typical tabular data, and characteristic panels are
tabular. The answer has two parts — include a gradient-boosting baseline, which is nearly free; and
note that trees cannot serve as experts, because no gradient flows through a tree ensemble, which
would force every arm to be two-stage and remove the differentiable-gate comparison entirely.

**NOT SETTLED — how many hidden layers the experts should have.** I had been treating two hidden
layers as established. It is not.
- **Ye & Borde fix two hidden layers and never test depth.** Their §3.1 states every network is built
  from the same two-hidden-layer block; their ablations cover expert *count* (K=2/4/6) and residual
  design (zero-init / random-init / no base). There is no depth ablation in the paper. It is
  precedent, not evidence.
- **Gu, Kelly & Xiu** test depth 1-5 but for a *standalone* predictor with no mixture.
  **RESOLVED 2026-09-12 from their tables, and the resolution matters.** Monthly, all stocks:
  NN1 0.33, NN2 0.39, NN3 0.40, NN4 0.39, NN5 0.36 per cent OOS R². Annual, all stocks: NN1 2.64,
  NN2 2.70, NN3 3.40, NN4 3.60, NN5 2.79 — the four-layer network leads at the annual horizon. Across
  the top-1000 and bottom-1000 subsamples the winner alternates between three and four layers.
  **Verified against the full published paper 2026-09-12, and the rest of it splits along a clear
  line.** NN3 wins every *forecast-accuracy* criterion: monthly stock-level R² 0.40%, large-cap 0.70%,
  aggregate index monthly R² 1.80%, the largest market-timing Sharpe gain (0.77 on the S&P 500, +26
  points over buy-and-hold), and the annual market portfolio at 15.7%. NN4 wins every *realised
  portfolio* criterion: the best decile-spread strategy at 2.3% per month (27.1% annualised, Sharpe
  1.35 value-weighted, 2.45 equal-weighted), the smallest maximum drawdown (51.8% value-weighted and
  14.7% equal-weighted, against 69.6% and 84.7% for OLS-3), the mildest worst month at −9.01%, and
  Figure 9, of which the authors write that "NN4 dominates the other models by a large margin in both
  directions." **The paper's headline claim — "shallow learning outperforms deeper learning," with the
  peak at three layers — is drawn from the forecast-accuracy tables, and its own portfolio results
  point one layer deeper. The paper does not reconcile this.**
  **Their Diebold-Mariano tests find no significant difference between ANY pair of the five depths**
  (statistics from −1.04 to 0.59, all far below conventional thresholds), and the same tests fail to
  separate the neural networks from random forests or gradient-boosted trees, while clearly separating
  all of them from OLS and the generalised linear model. The authors state this themselves: neural
  networks "improve over tree models, but the difference is not statistically significant," and under
  their Bonferroni correction the neural advantage over penalised linear models "becomes only
  marginally significant." So the flat claim "NN3 is best" is not supported by their own testing;
  Lai's "3-4 hidden layers" characterisation is the fair reading, and the result that survives is the
  *shape* — shallow suffices, deeper does not help — not a depth.

  **The implication for the decision rule below is now stronger than a non-result.** Depth is not
  merely unresolved; the preferred depth *depends on which metric family is optimised*, and the two
  families disagree by one layer in the only study large enough to look. The metric this thesis will
  decide on must therefore be fixed in advance and justified, because choosing it after seeing results
  would be choosing the answer. Rank information coefficient and its information ratio sit in the
  forecast-accuracy family, where three layers lead; any decile-spread or Sharpe-based reporting sits
  in the portfolio family, where four lead. If both are reported, the possibility that they disagree
  should be anticipated in the write-up rather than discovered in it.
- **Lai (2025), arXiv:2505.01921**, tests depth 1-5 on characteristic-sorted portfolio factors for 420
  large-cap US stocks and finds **two hidden layers best** (avg OOS R² 3.66%) ahead of three (2.86%).
  Caveat: portfolio-level R², not stock-level, so use the ordering and never the level.
- **No study anywhere ablates expert depth inside a mixture of experts on cross-sectional returns.**
So the honest position is that the literature supports "shallow", is split between two and three, and
says nothing at all about experts specifically.

**NOT SETTLED — width.** Width is not a free choice once capacity matching is imposed. The comparison
requires the whole mixture to match the parameter count of the large single-network baseline, so
fixing the number of experts and the depth determines the width. What has to be chosen is the
*capacity budget*, not the width directly. Ye & Borde use expert width 16 against a capacity-matched
baseline of width 44; Gu, Kelly and Xiu use a fixed 32-16-8 pyramid; Lai uses a dynamic pyramid rule.
None of these is transferable without re-deriving it for this configuration.

**NOT SETTLED — whether the base and the experts should have the same depth.** Ye & Borde use the same
block for both. Nobody has tested an asymmetric specification. There is a substantive argument for
asymmetry: the base is a standalone predictor doing the whole job, which is precisely Gu, Kelly and
Xiu's object, while the experts only correct a residual, which is a smaller function to represent.
There is also a measurement argument that the base should be *strong*: if the base is weak, ordinary
predictability survives into the residual, the experts absorb general signal rather than
regime-conditional correction, and the claim that the correction term measures the gate's
contribution stops being true. If the base is too strong, the correction shrinks toward zero and the
gates become indistinguishable.


**IS THERE A GATE × DEPTH INTERACTION? (Tom, 2026-09-12 — yes, and there is a mechanism.)**
The optimal expert depth is not obviously gate-independent, because **the effective sample size each
expert is trained on is determined by the gate.** A sharp gate — a jump model with a large switching
penalty, or a near-deterministic filter — partitions dates into regimes with little mixing, so each
expert effectively trains on a subsample and must learn a full regime-specific mapping from fewer
observations, which argues for less capacity. A diffuse gate — Ye and Borde report a mean maximum
routing weight of 0.36 against 0.25 for uniform, so barely above averaging — gives every expert nearly
the whole panel at moderate weight, so more capacity is affordable, though the differences each expert
must encode are subtler. The two effects point in opposite directions and there is no reason to assume
they cancel at the same depth for every gate.

**This is measurable before any mixture is trained.** Once a gate is fitted, the effective sample size
it gives expert k is the Kish quantity (Σ_t π_k(t))² / Σ_t π_k(t)², computable directly from the
fitted weight sequence. Doing this for every candidate gate, before the expert sweep, costs nothing
beyond fitting the gates — which has to happen anyway — and tells us immediately whether the gates
differ enough on this axis for the interaction to be a live concern.

**The design conflict this creates.** If expert depth is allowed to vary by gate, each gate runs at its
own best configuration but the comparison is no longer controlled — differences between gates would
confound the mechanism with the architecture, which is exactly the criticism Ye and Borde level at the
field. If depth is fixed across gates, the comparison stays controlled but some gates may be
handicapped. **Proposed resolution: fix neither and report the full mechanism-by-depth grid, which contains the
fixed-depth comparison as one column and exhibits the interaction rather than concealing it. See Q11.** If the ranking of gates is the same under
both, the conclusion is robust and the matter is closed in a sentence. If the ranking changes, that is
a finding in its own right — it would mean the relative merit of regime mechanisms depends on expert
capacity, which nobody has shown — and it is reported rather than resolved.

**How the two sweeps differ, and why the base is the easy one.**
- **Base depth can be settled independently**, because the base is trained *standalone*, before any
  expert or gate exists. Sweeping it is literally Gu, Kelly and Xiu's experiment repeated on this
  panel and metric: train at each depth, evaluate out-of-sample, pick. No gate is involved and no
  interaction is possible. The one caveat is that the best *standalone* base is not automatically the
  best base *for the mixture* — a marginally weaker base leaves a larger and more measurable residual
  — so one variant a layer shallower should be carried into the mixture stage and compared on
  correction magnitude, not on standalone accuracy.
- **Expert depth cannot be settled without a gate**, since experts only exist inside a mixture. Tom's
  instinct to use the control gate is right in principle, but a single gate risks tuning the
  architecture to that gate's particular weight concentration. **Better: run the expert sweep under
  two gates chosen to span the range — the sharpest and the most diffuse candidate, identified by the
  effective-sample calculation above.** If depth ranks the same under both, fix it with justification
  and the interaction question is answered empirically rather than assumed away. If it ranks
  differently, the interaction is real and the secondary per-gate arm becomes necessary rather than
  optional. The extra cost is one additional pass of the depth sweep.

**Order of work that follows from this:** fit all candidate gates first (cheap, no mixture training);
compute effective sample size per expert for each; select the sharpest and most diffuse; sweep base
depth standalone in parallel; then sweep expert depth under both selected gates at matched total
capacity; then proceed to the construction arms.


**PROPOSED EXPERIMENT — Stage 1 of the construction pilot.**
Run before the construction arms, with one gate held fixed and everything else constant:
1. **Expert depth** in {1, 2, 3} hidden layers, **at matched total mixture capacity**, so that width
   adjusts as depth changes. This is what isolates depth from capacity — note that Gu, Kelly and Xiu's
   NN1-NN5 ladder does *not* do this, since capacity grows with each layer added, so their result
   confounds the two.
2. **Base depth** in {2, 3} hidden layers, run at the winning expert depth rather than as a full
   factorial, to keep the cost proportionate.
3. **Capacity budget** swept coarsely at the winning depths.
**How the experiment must be judged — this follows directly from the GKX resolution above.** If a
study with sixty years of data and thirty thousand stocks cannot statistically separate one depth
from another, a smaller study will not separate them on accuracy either. Expecting the depth sweep to
produce a significant accuracy winner is therefore unrealistic, and designing the decision rule around
accuracy would leave the choice undetermined. The criteria, fixed in advance, are instead:
1. **Rank information coefficient and its information ratio** — reported, but used only to exclude
   configurations that are clearly worse, not to pick a winner among those that are close.
2. **Across-seed stability** — dispersion of rank IC and the proportion of seeds below a stated
   threshold. This is where differences between configurations have actually shown up in the closest
   precedent, where accuracy gaps were small and training-reliability gaps were large.
3. **The magnitude and variability of the aggregate correction** — the decisive criterion here,
   because a configuration whose correction term is indistinguishable from zero cannot support the
   analysis the thesis depends on, however well it forecasts.
4. **The base's own standalone out-of-sample performance**, reported separately, so that a reader can
   see the residual is not an artefact of a weak base.
Stating in advance that the depth sweep is expected to be inconclusive on accuracy, and that the
choice will be made on stability and measurability, is more defensible than reporting an
insignificant accuracy ranking as though it settled something.

**See Q9** for the related and arguably larger question this raised: which metric family the thesis
should commit to, given that Gu, Kelly & Xiu's accuracy and portfolio criteria select different depths.

**Ask the advisor:** (a) is it acceptable to settle the expert architecture by experiment rather than
by citation, given that no published study addresses it; (b) does an asymmetric specification — a
deeper base than experts — weaken the capacity-matching argument, or is matching on total parameter
count sufficient; and (c) is the measurement argument for a strong base, that a weak base contaminates
the residual, one that should be stated explicitly in the thesis or treated as an implementation
detail?

---

### Q9. Which metric family should the thesis commit to, and what if the families disagree?
**Status:** open — raised 2026-09-12 after verifying Gu, Kelly & Xiu against the full published paper.

**The observation.** Gu, Kelly and Xiu's own results select *different network depths depending on
which family of criterion is applied*, and the paper reports this without reconciling it.

*Forecast-accuracy criteria all select the three-layer network:*
- Monthly stock-level out-of-sample R² of 0.40%, the peak across NN1–NN5
- Large capitalisations (top 1,000) 0.70%, the highest of any model
- Aggregate S&P 500 monthly out-of-sample R² of 1.80%, the highest of any model
- Table 5: "NN3–NN5 produce a positive R² for every portfolio analyzed, with NN3 dominating"
- Market-timing Sharpe of 0.77 on the S&P 500, 26 points above buy-and-hold
- Annual market portfolio R² of 15.7% — "NN3 continues to dominate"

*Realised-portfolio criteria all select the four-layer network:*
- Best decile-spread strategy: 2.3% per month, 27.1% annualised, Sharpe 1.35 value-weighted and
  2.45 equal-weighted
- Smallest maximum drawdown: 51.8% value-weighted and 14.7% equal-weighted, against 69.6% and 84.7%
  for OLS-3
- Mildest worst single month at −9.01%, the least extreme of any model under either weighting
- Figure 9, of which the authors write: "NN4 dominates the other models by a large margin in both
  directions"
- Annual stock-level R² of 3.60 against 3.40 for the three-layer network

Their headline claim — "Shallow learning outperforms deeper learning… performance peaks at three
hidden layers then declines" — is drawn from the forecast-accuracy tables. Their own portfolio
results point one layer deeper.

**Why this is not simply a curiosity about their paper.** Three consequences follow for this thesis.

1. **The choice of decision metric is not neutral, and it is not separable from the architecture
   choice.** In the only study of depth large enough to be authoritative, the two metric families
   disagree by a full layer. Selecting the metric after seeing results would therefore amount to
   selecting the architecture — and, in a comparison of gates, potentially to selecting which gate
   wins. The metric has to be fixed and justified before any run.

2. **This thesis's planned primary metrics sit in the accuracy family.** Rank information coefficient
   and its information ratio measure the quality of a cross-sectional ranking, which is a
   forecast-accuracy criterion, and that family points to three layers. But a finance committee will
   reasonably expect decile-spread returns, Sharpe ratios and drawdowns to be reported as well, and
   that family points to four. The two are not interchangeable and the difference is not a detail.

3. **Neither family can be settled statistically at the depth level anyway.** Their Diebold–Mariano
   tests find no significant difference between any pair of the five depths — statistics ranging from
   −1.04 to 0.59 — and the authors state that neural networks "improve over tree models, but the
   difference is not statistically significant." So the ranking that each family produces is a
   point-estimate ordering rather than a tested result, in both cases.

**What I would propose.** Designate rank information coefficient and its information ratio as the
primary criteria, on the grounds that the object of study is the conditional expected-return ranking
rather than a trading strategy, and that portfolio construction introduces weighting, turnover and
cost assumptions that are not part of the research question. Report decile-spread performance,
drawdown and turnover as secondary, descriptive evidence. State in advance that the two families may
disagree, and if they do, report the disagreement as a result rather than resolving it silently —
since Gu, Kelly and Xiu's experience suggests disagreement is the expected outcome rather than an
anomaly.

**Ask the advisor:**
(a) Is it acceptable to designate ranking metrics as primary and treat portfolio performance as
    secondary, given that a finance committee may regard realised portfolio performance as the more
    meaningful evidence?
(b) If the two families select different configurations, is reporting the disagreement openly an
    acceptable outcome for a master's thesis, or does the committee expect a single resolved
    recommendation?
(c) Should transaction costs enter the secondary portfolio evidence at all, given that neither the
    closest precedent nor the comparison itself depends on them, and that Gu, Kelly and Xiu report
    neural-network turnover of 110–130% per month, which is high enough that cost assumptions would
    materially affect any strategy conclusion?

See also Q8, which covers what is and is not settled about the architecture itself.

---

### Q10. How is the frozen base actually isolated, and how would we know if it failed?
**Status:** open — raised 2026-09-12. Partly answerable now by diagnostics; one structural option needs a decision.

**The problem, stated honestly.** Freezing the base does not enforce specialisation. The expert
correction r_k(x) is an unrestricted function of the same characteristic vector the base sees, and the
training objective rewards reducing squared error by any means available. If the base leaves
systematic predictability on the table, the experts will learn *that*, because doing so lowers the
loss. Nothing in the architecture distinguishes "regime-conditional correction" from "general
predictability the base missed." The frozen base buys identifiability of *the correction term*, not
identifiability of *what the correction contains*.

**Two distinct failure modes, needing different tests.**
- **Mode A — the base underfits.** General signal survives into the residual and the experts absorb
  it. The correction term is then large but is not the gate's contribution; it is ordinary
  predictability relearned one level up. This is the failure a stronger base mitigates.
- **Mode B — the experts converge.** Even with a good base, all K experts may learn approximately the
  same correction function. The aggregate correction is then nonzero but almost independent of the
  gate weights, so the gate is decorative. **The load-balancing penalty does not prevent this**: it
  balances the routing *weights*, not the expert *functions*, and a perfectly balanced gate over K
  identical experts still does nothing.

**Measures that detect these — all computable, none requiring new machinery.**

1. **Decomposition of the correction into gate-independent and gate-conditional parts.** Write the
   aggregate correction as the mean expert output plus the part attributable to the gate departing
   from uniform weights: the first term is a *general* correction to the base and is direct evidence
   of Mode A; the second is the genuinely regime-conditional part. Reporting the share of explained
   variation carried by each is the single most informative diagnostic, and it is the number that
   should appear in the thesis whenever the gate's contribution is claimed.

2. **Gate-permutation test.** At inference on the trained model, replace the fitted gate weights with
   (a) uniform weights and (b) weights drawn from a randomly chosen other date, and re-evaluate. If
   performance is unchanged, the gate contributes nothing regardless of what the correction magnitude
   suggests. If it degrades, the degradation *is* the gate's contribution, measured directly. Costs
   nothing — no retraining — and should be run for every gate.

3. **Cross-expert dispersion.** The weighted variance of the expert outputs at each observation. Near
   zero means Mode B. Note this is precisely the quantity the shrinkage penalty does *not* touch,
   since that penalty acts on the aggregate correction rather than on individual experts.

4. **Explaining the correction from the base's own inputs.** If the aggregate correction is well
   predicted by the base forecast or by the characteristic vector alone, without reference to the
   regime variables, it is general structure rather than conditional structure.

5. **Timing.** A genuinely regime-conditional correction should covary with the regime state and
   change around transitions. One that is flat through time is a general patch on the base.

6. **A null-gate training arm.** Retrain the identical architecture with the gate replaced by constant
   uniform weights. Any advantage of the real gate over this arm is the honest measure of what the
   gate adds. More expensive than the permutation test but stronger, since it also removes whatever
   the gate contributes through its effect on optimisation.

**On using a stronger base — helpful, but mitigation rather than solution.** A stronger base leaves
less general signal available to absorb, so it does attack Mode A directly. Three caveats. It cannot
eliminate the problem, since even the best available model explains well under one per cent of monthly
stock-level variation and some of the remainder is structure rather than noise. It trades against
measurability: a stronger base leaves a *smaller* correction, and if the correction shrinks far enough
the gates become indistinguishable, which is the failure the whole comparison is designed to avoid.
And per Q9, "stronger" is itself metric-dependent — the accuracy and portfolio criteria select
different depths. So the base should be strong and its own standalone performance reported, but the
isolation claim has to rest on the diagnostics above, not on the base specification.

**One structural option worth a decision.** The current shrinkage penalty acts on the *aggregate*
correction. An alternative is to penalise the *mean* expert output instead — pushing the experts to
cancel on average at every observation, so that a nonzero correction can only arise from the gate
weights departing from uniform. That would make the correction gate-conditional by construction rather
than by hope, and it is a one-line change to the objective. It also constrains the model's ability to
make a general correction at all, so it may cost accuracy, and it is untested as far as I can find.
It belongs as an arm rather than a default.

**THE COUNTER-ARGUMENT, and it is a good one (Tom, 2026-09-12).** If the base were strong enough to
leave nothing worth correcting, the residual mixture would collapse to the base and there would be no
model. The fact that the construction demonstrably beats a capacity-matched single network is itself
evidence that useful residual structure exists. The concern above should not be allowed to become
generalised scepticism about an architecture the literature shows to work.

**Where the concern does survive, sharpened by the numbers.** Ye and Borde's advantage over a
capacity-matched single network is +0.0048 in information coefficient. Their own ablation removing the
regime variables from the gate — keeping the gate, the residual construction and everything else, and
feeding it only stock features — costs **+0.0004**, against an across-seed standard deviation of
0.0012. So of the total advantage, roughly one part in twelve is attributable to the *regime
information*, and the remainder to the *residual construction*. Their result is strong evidence that
residual routing works and thin evidence that regime information in the gate does much for accuracy;
what it does robustly is avoid the catastrophic instability of input concatenation.

**This reframes the diagnostics rather than justifying scepticism.** The architecture is not in doubt.
The quantity this thesis is about — what the *gate signal* contributes — is small in the only
measurement available, smaller than seed noise in that study. Diagnostics are therefore instrumentation
for detecting a small effect, not a defence against being fooled by a broken model. It also sets a low
bar in the thesis's favour: if a rolling-volatility gate is worth +0.0004, a gate with transition
dynamics has room to do better, and their own transition-window result suggests where. The practical
requirement is statistical power — enough seeds and paired testing to resolve differences of this
order.

**Ask the advisor:**
(a) Is the decomposition in (1), reported alongside the gate-permutation test in (2), a sufficient
    basis for claiming the correction term measures the gate's contribution — or would the committee
    expect a stronger identification argument?
(b) Is the null-gate training arm in (6) worth its cost, given that the permutation test gives a
    cheaper approximation to the same quantity?
(c) Is penalising the mean expert output, so that corrections are gate-conditional by construction, a
    legitimate design choice or an over-constraint that would invite the criticism that the model was
    handicapped to produce the desired interpretation?

---

### Q12. Which baseline models should the mixture be compared against?
**Status:** open, raised 2026-09-12. Left deliberately unspecified in Section 3 pending this discussion.

Every claim the study makes is relative to a baseline, so the choice determines what any result means.
Four candidates, with the argument for each.

**A capacity-matched single network.** The most important of the four and effectively compulsory. If a
single network with the same total parameter count performs as well as the mixture, the architecture
has added nothing and the comparison across gates is measuring differences within a construction that
does not earn its place. This is the baseline Ye and Borde treat as their reference, and their
headline advantage of +0.0048 in information coefficient is stated against it. Without it the study
cannot distinguish "the gate mechanism matters" from "more parameters help".

**A small single network matched to one expert.** Cheap, and it separates two things the capacity-
matched baseline conflates: whether the benefit comes from the total budget or from how the budget is
divided.

**A penalised linear model.** The floor. Establishes that the nonlinear machinery is doing something
at all on this panel, and in the reference study the linear and generalised-linear models are the only
comparisons that machine learning methods beat by a statistically significant margin.

**A gradient-boosted tree ensemble.** The contested one. The argument for including it is that tree
ensembles remain strong on tabular data of the kind a characteristic panel represents, so its absence
is a predictable objection; and that in Gu, Kelly and Xiu's own Diebold-Mariano tests the neural
networks do not significantly outperform random forests or gradient-boosted trees, which makes the
objection substantive rather than procedural. The argument against is that it is a baseline for the
*predictor*, not for the *gate*, and the thesis is about gates; trees also cannot serve as experts,
since no gradient flows through a tree ensemble, so including one invites a comparison the
architecture cannot act on. It is close to free to run, since the pipeline already exists.

**A no-gate mixture.** The same architecture with the gate replaced by constant uniform weights. Not a
conventional baseline but arguably the most informative one here, since the difference between it and
any real gate is the honest measure of what gating contributes. See Q10.

**Ask the advisor:** which of these the committee would expect to see, whether the gradient-boosting
comparison helps the thesis or invites a distraction from its actual question, and whether the no-gate
mixture should be presented as a baseline in its own right or as a diagnostic.

---

### Q13. Should a recurrent gate be added as a fifth mechanism?
**Status:** open, raised 2026-09-13 after a full read of Dynamic TMoE (arXiv:2605.20678).

**Where the idea comes from.** That paper diagnoses memoryless routing as a limitation, which Section 1
already cites, but its remedy is more specific than the citation currently suggests. Its Temporal
Memory Router is a **GRU that carries a hidden state across time steps**, so the routing decision
depends on the trajectory and not only on the current input. Their ablation replacing it with a linear
or MLP router costs 4.2% in MSE and 1.8% in MAE, so the memory is doing measurable work.

**Why this matters for the comparison as currently designed.** A recurrent router has **memory without
a regime process**: no transition matrix, no persistence parameter, no dwell time, no interpretable
states. The present gate set does not contain anything of that kind. The two reference gates, a
clustering assignment and a smooth-transition function, have no memory at all, and the four candidate
mechanisms all carry both memory and explicit regime structure. Nothing in the set separates the two.

**The consequence is a hole in the claim.** If the regime mechanisms beat the memoryless references,
the result shows that memory in the gate helps. It does not show that *structured* memory helps, which
is the actual thesis. A referee can ask whether a plain recurrent router would have done just as well,
and on the present design there is no answer. Adding one supplies the control that closes the question.

**Cost.** Low. A small recurrent network reading the same conditioning series and emitting weights on
the simplex fits the existing gate interface without modification, and it adds one row to the grid.

**What else the paper offers, and what should be left alone.**
- Its Post-Addition Alignment freezes existing experts and the router backbone and trains only the new
  component; fine-tuning the base experts instead produced a 13.0% MSE increase through catastrophic
  forgetting. That is independent support for the frozen base in Q7, from a different architecture.
- It finds top-k of three optimal and larger expert pools degrading, which with Ye and Borde's finding
  that two and four experts perform alike while six is worse gives two independent studies agreeing
  that expert count matters little and excess hurts. Supports fixing K.
- Its MMD drift detector emits a binary signal and cannot serve as a gate, but MMD divergence between
  a reference and a current window is a natural **covariate for the time-varying transition
  probability gate**, which would connect the change-point literature to that mechanism at no
  structural cost.
- Its dynamic instantiation and pruning of experts should **not** be adopted: the parameter count then
  changes over time, which destroys capacity matching and with it the controlled comparison.
- Its heterogeneous experts, specialised for trend, seasonality and fluctuation, depend on the input
  carrying a time axis. A cross-sectional characteristic vector has no seasonality to extract, so the
  design does not transfer.

**Ask the advisor:** is the recurrent gate worth adding as a fifth mechanism to separate memory from
regime structure, or does it broaden the study past what the schedule supports? And if it is added,
should it count as one of the compared mechanisms or as a third reference gate alongside the
clustering assignment and the smooth-transition function?

---

### Q14. Should the experts produce full magnitude contributions, or only shrunk corrections?
**Status:** open, raised by Tom 2026-09-13. Partly answered by the literature, partly untested.

**First, a distinction that is easy to lose.** Three binary choices get bundled together in discussion
and are actually separate:
1. Is there a base at all? Residual mixture versus standard mixture. That is Q6, and the literature
   answers it: the residual form wins.
2. Is the base frozen or trained jointly? That is Q7, and the literature does not answer it.
3. **Are the expert outputs constrained to be small perturbations of the base, or free to contribute
   at full scale?** That is this question, and it is orthogonal to the first two.

**A further split inside the third choice.** "Constrained to be corrections" is doing two jobs at once
and they should be separated, because the evidence differs:
- **Zero initialisation** of the expert output layer, so the model begins exactly at the base. This is
  about the starting point. **It is tested.** Ye and Borde replace it with ordinary random
  initialisation and lose 0.0039 in information coefficient, with collapsed runs rising from 0 to 13
  out of 30. Zero initialisation earns its place.
- **The shrinkage penalty on the aggregate correction.** This is about the objective throughout
  training, not the starting point. **It is not tested anywhere.** Ye and Borde never ablate the
  coefficient, and it is carried in the pilot as arm E precisely because of that.

**A correction about the language model comparison.** In DeepSeekMoE the shared expert is not a base
being corrected. All experts, shared and routed, are trained together and their outputs summed, so the
correction framing does not exist there at all. It is therefore not quite right to say that version
uses "full magnitude corrections"; it uses no corrections. The genuinely comparable configuration is a
frozen base whose experts are free to contribute at full scale, which is close to Ye and Borde's
random initialisation arm.

**The argument for keeping corrections small.** Beyond the zero initialisation evidence, the
statistical case is the ridge and James-Stein one: the correction is estimated from very little signal,
so its estimate has high variance, and shrinking a high variance estimate toward zero trades a little
bias for a large reduction in variance. The downside is also bounded, since a useless gate then leaves
the model at the base.

**The argument for allowing full scale, which is specific to this thesis and is the stronger one.**
Regime effects are not supposed to be small perturbations. Henkel, Martin and Nardari find that
short-horizon predictability is essentially absent in expansions and substantial in recessions, which
is a different mapping between predictors and returns, not a mild adjustment to a common one. If a
regime genuinely reverses the sign of a predictor, the correction has to be large enough to cancel the
frozen base's output and then overshoot it. A penalty calibrated to keep corrections small fights
exactly the effect this study is looking for, and could suppress the signal it is trying to measure.
The same logic applies to the comparison: if the penalty compresses every gate's correction toward
zero, the gates become harder to tell apart.

**Where this leaves the design.** The two sub-components can be decided separately, and the evidence
points different ways for each. Keep zero initialisation, which is supported. Treat the shrinkage
coefficient as an experimental variable, not a fixed part of the architecture, and report results at
the validation-selected value and at zero. That is arm E of the pilot, which now carries more weight
than it did when it was added as a curiosity. A third option, from Q10, is to penalise the mean expert
output instead of the aggregate correction, which forces the correction to be gate-conditional without
capping its magnitude.

**Ask the advisor:**
(a) Is it acceptable to keep zero initialisation on the strength of published evidence while treating
    the shrinkage coefficient as an experimental variable, or should both be fixed by citation?
(b) Does the concern that a shrinkage penalty may suppress genuine regime effects, given that
    predictability is known to differ sharply between expansions and recessions, warrant reporting the
    unpenalised configuration as a headline result rather than as a robustness check?
(c) If the unpenalised configuration proves unstable across seeds, is falling back to a penalised one a
    legitimate response, or does that amount to choosing the configuration that produces the tidier
    answer?

See Q6 and Q7 for the two related but distinct choices, and Q10 for the alternative penalty.

---

### Q15. Adaptive overfitting: how many times may the same test set be used?
**Status:** open, raised by Tom 2026-09-15. This is the most serious methodological risk in the design.

**The literal question first.** Yes, the same test set may be used for different models. Comparing
models on a common test set is what a benchmark is, and every study cited in Section 1 does it. The
problem is not reuse. The problem is **using test performance to choose**, because a number that
informed a decision is no longer an unbiased estimate of generalisation for the thing it chose. The
bias grows with the number of decisions the test set informed, which is the adaptive data analysis
result of Dwork and co-authors, and in finance it is the same concern as Harvey, Liu and Zhu on
multiple testing and Bailey and Lopez de Prado on backtest overfitting.

**Why this design is exposed.** The mechanism-by-depth grid means roughly six gates by three depths,
so on the order of eighteen configurations will be looked at, plus five construction arms before them.
Selecting the best of eighteen noisy estimates and reporting it as though it had been chosen in
advance is precisely the error, and the effects in question are small enough that selection noise
could exceed them.

**One distinction that removes part of the worry.** Running thirty seeds and reporting a distribution
is **not** multiple testing. It estimates the sampling distribution of one configuration. Multiplicity
comes from the number of **configurations** examined, not from the number of seeds within one.

**What the design already does right.** The walk-forward protocol puts a validation block inside every
window. Every model selection decision, expert depth, capacity, the shrinkage coefficient, the
construction arm, belongs there, and none of it needs the test set. **The rule is that the test set is
for reporting and the validation block is for choosing.** If that rule holds, the exposure is limited
to the act of looking at the grid.

**Four further defences, in order of strength.**

1. **A sealed set of final windows.** Hold back the last five or six walk-forward windows and do not
   compute anything on them until every decision is frozen, then report them once as confirmation.
   This is the only defence that produces a genuinely untouched estimate, and it costs only the
   ability to iterate on the most recent period. The rolling design makes it natural: develop on
   windows one to twenty-four, confirm on twenty-five to thirty.
2. **One pre-registered primary hypothesis.** A single directional prediction, declared before any run,
   requires no multiplicity correction because it is one test. The transition-window hypothesis in
   Section 2 is already framed this way, and this is the reason it is worth more than rhetoric: it is
   the strongest statistical position available in a study that will otherwise examine dozens of
   configurations. Everything else in the grid is then descriptive.
3. **Multiplicity correction where a claim is made.** Six gates give fifteen pairwise comparisons. Gu,
   Kelly and Xiu apply a Bonferroni correction for twelve comparisons and report a critical value of
   2.64, so the precedent comes from the benchmark paper itself. **The uncomfortable implication is
   that a correction of that size may erase significance entirely**, given that the differences at
   stake in the closest precedent run from 0.0004 to 0.0056 in information coefficient against an
   across-seed standard deviation of 0.0012. Reporting that honestly is better than not applying the
   correction.
4. **Report the number of configurations examined**, which Section 3 already commits to, so that a
   reader can discount accordingly. If Sharpe ratios are reported at all, the deflated Sharpe ratio of
   Bailey and Lopez de Prado adjusts explicitly for the number of trials.

**The honest summary.** With this many configurations, a claim of the form "mechanism X is best" will
be difficult to support at conventional significance after correction. A claim of the form "the
pre-registered prediction about transition windows held, and the ordering of mechanisms was stable
across the grid" is defensible and is what the design is actually built to deliver. Framing the
results that way from the start is a design decision, not a retreat.


**Addendum, 2026-09-15: developing the base without spending the test set.**

A related worry, raised separately: if the base network turns out not to have generalised, it has to be
fixed, and fixing it after seeing test performance would burn that test set. Three clarifications
resolve most of this.

*Validation is the generalisation check, not the test set.* The validation block inside each window is
data the model never trained on, so a base that fits training data and fails validation is visibly
failing to generalise, and can be diagnosed and repaired without the test set being touched at all.
The test set is not there to tell you whether the model generalised; it is there to give an unbiased
estimate once decisions have stopped, because validation performance becomes optimistic as soon as it
has been selected on. **The whole development of the base therefore belongs inside the training and
validation portions of each window.**

*Not reporting a component's numbers does not make test-set use free.* It is tempting to think that
since the base's own performance is never reported, tuning it against test data costs nothing. It does
not work that way: contamination travels through the pipeline, not through the write-up. Every expert,
every gate and every arm sits on top of the frozen base, so a base that was fitted to the test period
passes that information into the results that *are* reported. In this design the base is maximally
load-bearing, which makes the exposure larger here than in an ordinary model, not smaller.

*Architecture and weights consume integrity differently.* Refitting the base's weights inside every
walk-forward window is routine and costs nothing, since each window trains only on its own development
data. What consumes integrity is choosing the base's **architecture**, because that choice is made once
and applies everywhere. So the architecture is settled on validation, in the pilot, and the weights are
then refit per window without further decisions.

*One case that should not be "fixed".* A base that fits training data and fails validation is a
modelling failure, and repairing it is free. A base that performs well on validation and worse on test
is a different thing: in a non-stationary market, degradation from one period to the next is expected
and is arguably the phenomenon the thesis is about. Iterating until that gap closes means fitting to a
particular test period's regime, which is the opposite of the study's purpose. The honest response
there is to report the degradation, not to chase it.


**Ask the advisor:**
(a) Should a block of final walk-forward windows be sealed and reported once, accepting that it cannot
    then be used for development?
(b) Is a single pre-registered directional hypothesis, with the grid reported descriptively around it,
    an acceptable primary result for a master's thesis, or does the committee expect a ranking of
    mechanisms with significance attached?
(c) If Bonferroni correction across the pairwise gate comparisons removes significance, is reporting
    that outcome acceptable, or does it read as a failed study?

---

### Q16. How much data is needed, and how should it be split across regimes? [PARTLY RESOLVED 2026-10-05]
**Status:** partly resolved 2026-10-05 (see the resolution block below); raised by Tom 2026-09-15. Bears directly on Q3, since it turns data access from a
convenience into a requirement.
**Updated 2026-09-25:** time-decay weighting worked out in detail under point 2, with reading and a
new question (d). Not implemented; no code until this is decided.
**Updated 2026-09-26:** ask (e) added, on keeping crash periods out of the decay and on testing which
historical periods matter. Not implemented.

**Resolution 2026-10-05 (partial; Tom's decisions).** Asks (a), (b), (d) and (e) are decided; ask (c)
stays open. The theory is written up for study in `Training_Window_and_Weighting_Theory.md`.

- **(a) Sample: 2000-2024** (was 2015-2024). It roughly doubles the stress episodes (2000-02, 2008-09,
  2010, 2011, 2015-16, 2018, 2020, 2022). 2000 is the earliest start with sectors, because Compustat's
  GICS history begins in mid-1999 (Q26 part 4). Feature history reaches further back (about 1995 for
  the 5-year windows); the sample is what is trained and tested on.
- **(b) Expanding window, never rolling.** A rolling window can contain no stress episode, which
  leaves the stress expert with nothing to learn from. Older data is down-weighted rather than dropped
  (d).
- **(d) Yes, decay, with one memory per component.** Bias against variance: old data adds bias
  because markets change (McLean & Pontiff 2016; Green, Hand & Zhang 2017; Chordia, Subrahmanyam &
  Tong 2014), but dropping it adds variance in a very low signal-to-noise setting, and the best
  weights after a break are smaller but not zero (Pesaran & Timmermann 2007; Pesaran, Pick &
  Pranovich 2013).
  - **Base: calendar-time exponential decay with a long half-life.** The base is where structural
    drift shows most, and every day informs it.
  - **Experts: regime-clock decay (ask (e), Tom's idea of 2026-09-26).** A row's age is the amount of
    later experience of the same regime, counted with the gate's filtered probabilities (no
    look-ahead), with one half-life in regime-days:
    $$w_s(t)=\sum_k\xi_{s\mid s}(k)\,\rho^{\,n_k(s,t)},\qquad n_k(s,t)=\sum_{u=s+1}^{t}\xi_{u\mid u}(k),\qquad \rho=2^{-1/H}$$
    Calm data ages quickly and crisis data slowly. With about 30% stress days and H = 500
    regime-days, the calm half-life is about 2.8 years and the stress half-life about 6.6 years. In
    the likelihood, the weight multiplies each row's term (a weighted composite likelihood; Q24).
  - **Gate: regime-clock forgetting too** (this reverses the "never the gate" option in (d)). The
    evidence for it: Nystrup, Madsen & Lindström (2017) find a 2-state HMM's parameters on daily
    S&P 500 returns drift well beyond estimation noise, and adaptive estimation forecasts better.
    The evidence against plain forgetting: that gain holds only after excluding the 20 worst days,
    and a short-memory gate can lose its stress state in a long calm spell.
    - Implementation: a weighted Baum-Welch in two passes. First an unweighted fit, then
      regime-clock weights, then weighted M-steps. Each regime's emission parameters use its own
      clock; transitions out of regime j use j's clock.
  - **Choosing the memories.** Each half-life comes from a small preset grid that always includes
    equal weights, as the reference arm for Gu, Kelly & Xiu's undecayed protocol. The choice is made
    once, in the pilot, on the early folds' validation blocks only, then frozen and pre-registered
    (Q15).
    - The gate's memory is chosen by its own job: the one-step predictive log-likelihood of the
      market series on validation. This keeps the gate fitted separately and frozen (Q19) and gives
      every gate the same tuning budget. The mixture's validation loss is reported as a diagnostic
      only.
- **(e)** Regime-clock decay is adopted, for the experts and the gate. Fixed exempt windows are not
  used. "Testing which historical periods matter" remains a possible extension, diagnostic only.
- **Test period, refits and validation (decided 2026-10-05, second step).**
  - **Test period: 2010-2024, refitted annually, 15 folds.** Each fold re-fits the gate, base and
    experts on the expanding block from 2000 to the year before, then tests on the next year.
  - **Why 2010 and not 2008.** Every setting must be chosen on data before the first test year.
    - A 2008 start leaves only the almost calm years 2005-2007 for tuning a regime model.
    - A 2010 start lets the pilot see the 2008 crisis, and every training block contains two crises.
    - Out of sample there are still six stress episodes (2010, 2011, 2015-16, 2018, 2020, 2022),
      including COVID, the sharpest transition test.
    - The 2008 crisis can still be tested out of sample through leave-one-episode-out (c).
  - **Why annual refits.** The gate's filter updates regime probabilities daily, so the model still
    reacts daily; refits only update slow parameters. Quarterly refits would cost 4 times as much for
    little gain. Gu, Kelly & Xiu also refit annually with an expanding window.
  - **Pilot.** Three folds with validation years 2007, 2008 and 2009 (training blocks 2000-2006,
    2000-2007, 2000-2008); it never touches 2010 or later. It chooses, once:
    - the half-lives and the gate memory;
    - the learning rate and weight decay;
    - the training length, as a fixed number of gradient steps.
    Its validation loss is **regime-balanced**: rows are reweighted with the gate's weights so every
    regime counts equally. The settings are then frozen and pre-registered.
  - **Main study: no held-out validation tail and no per-fold early stopping.** Each fold trains on
    its full block with the frozen settings, so the most recent and most heavily weighted data is
    never withheld. This resolves audit B-2.
    - A fixed step budget is expected to transfer across folds, because decay weighting keeps the
      effective sample roughly stable as the window expands.
    - The pilot checks this by comparing the best training length across its three folds. If it
      drifts strongly, the fallback is early stopping on a regime-balanced validation tail.
  - The 5-day purge stays wherever blocks meet.
- **Compute levers (decided 2026-10-05, third step).** With a fixed step budget, a run costs about
  steps × batch size × (6P - 2·d·w1) FLOPs, independent of how many training rows there are.
  - **Train on every day.** Thinning to every fifth day saves nothing under a step budget.
    Overlapping rows still add some information, because the price features change daily, and the
    overlap only affects standard errors, which are Hansen-Hodrick corrected.
  - **Fresh start every fold, no warm starts.** Warm-started networks generalise worse (Ash & Adams
    2020). Warm starts would also break the zero-initialised residual design (corrections would be
    relative to last year's base) and would tie the folds together. Fallback only if compute binds:
    "shrink and perturb".
  - **Seeds paired across arms** (common random numbers: every arm runs on the same seed list, and
    comparisons are seed by seed). The plan is 10 seeds per grid cell and 30 for the pre-nominated
    primary cell (Q11). The final count is the smallest for which the standard error of a paired
    difference, measured in the pilot, is below about a third of the smallest effect worth detecting.
  - **Engineering (no design change):**
    - one process per performance core, one thread each;
    - large batches (1,024-4,096), with the size fixed in the pilot together with the learning rate;
    - features and ranks built once for the whole panel;
    - base predictions cached;
    - optional: several seeds trained as one batched model, only if the CPU route is too slow.
- **Still open: (c)** leave-one-episode-out.
- **Minor open point:** the experts' decay as a mixture over regimes (the formula above) or per
  expert, where expert k's gradient uses only k's clock (closer to the wording of (e)).
- **Not implemented.** It needs a brief: row weights in the expert and base losses, and a weighted
  Baum-Welch for the Hamilton gate, which currently uses statsmodels without weights.


**The tension, stated plainly.** The regimes where the gate should matter most are the rare ones. They
are rare by construction, they must appear in training for an expert to specialise in them and in test
for the benefit to be demonstrated, and they differ from one another, so what is learned from one
crisis may not transfer to the next.

**The first thing to get right is the unit of observation, because it makes the problem look worse
than it is and then better than it is.** One month of a cross-section of three thousand stocks is three
thousand training rows, so the 2008 episode supplies tens of thousands of stock-month observations and
the experts are not starved of rows. But those rows are heavily cross-sectionally correlated, and
correlations rise in downturns, which is one of the settled findings in Section 1. For estimating
anything regime-conditional, the **effective** sample size is therefore much closer to the number of
months in that regime than to the number of stock-months. Plenty of rows, few independent observations.

**The binding number is the count of distinguishable stress episodes, and it is small.** Between 1970
and 2020 the United States had seven NBER recessions. Counting volatility episodes more loosely adds
the 1987 crash, 1994, the 1997 and 1998 emerging-market and hedge-fund events, the 2000 to 2002
decline, 2011, 2015 to 2016, late 2018, and 2022. That is on the order of **ten to fifteen usable
episodes in forty to sixty years**, and no amount of cross-sectional width changes it.

**What this implies for sample length, and it is the main practical conclusion.** Ye and Borde use
December 2015 to November 2025, which contains **one** major crisis. Their COVID result, the 6.7 times
relative gain that is among their most cited numbers, rests on 104 trading days. That is the standard
in this literature, and it is low. A study whose central question is whether a gate behaves better at
regime transitions cannot rest on one transition. **A longer sample is not a preference here, it is a
requirement that follows from the research question**, which makes the database access in Q3 a
methodological necessity rather than a convenience. A sample beginning in 1990 contains 2001, 2008,
2020 and 2022; the conventional asset-pricing sample beginning in the 1950s or 1960s contains roughly
twice as many episodes.

**The split question largely dissolves, and this is worth being clear about.** The confusion comes from
imagining a single fixed division of history into training and test, in which 2008 has to be assigned
to one or the other. The walk-forward design does not work that way. Each episode is **test** data for
the window that contains it and **training** data for every window after it. There is no choice to
make, and every crisis is used for both purposes at different points in the evaluation.

**Three problems that do not dissolve.**

1. **Early windows have no crisis in their training data at all.** A window testing 1992 trains on
   1990 to 1992, so whichever expert ought to handle crises has never seen one. This is unavoidable
   and should be reported rather than hidden: performance in the earliest windows will be worse, and
   showing it separately is more informative than averaging it away.

2. **Rolling versus expanding training window, which is a real design decision and currently unmade.**
   A rolling window of the kind Ye and Borde use, 504 days, means 2008 has left the training data by
   2012 and the model cannot recognise that 2020 resembles it. An expanding window retains everything.
   **Section 1 criticises rolling re-estimation precisely for having no recurrence**, for being unable
   to notice that current conditions resemble a period two decades ago because that period has fallen
   out of the window. Adopting a rolling scheme would put the design in tension with its own literature
   review. The expanding window is the more consistent choice, at the cost of departing from the
   convention in the papers being compared against. Note also that the two principal references differ
   here: Ye and Borde roll a fixed 504-day development period forward, while the empirical asset
   pricing convention is closer to a recursive scheme in which the training sample grows.

   **An argument from the architecture itself, which may settle it.** In a regime-gated model the
   division of labour is explicit: the base is meant to capture the stable relationship between
   characteristics and returns, and the gate is meant to handle adaptation to market state. Adaptation
   is therefore the gate's job, not the base's. That argues for giving the base the **longest possible
   training sample**, since its purpose is a low-variance estimate of the average relationship, and
   letting the gate do the work a rolling window would otherwise be doing crudely. On this reading the
   expanding window is not merely more consistent with the literature review, it follows from what the
   architecture is for. It also has a testable consequence: if a rolling base outperforms an expanding
   one, that is evidence the gate is not in fact absorbing the adaptation it is supposed to absorb.

   **A third option sits between the two and is standard industry practice: exponential decay.** Rather
   than a hard cut-off, which is what a rolling window is, or equal weighting of all history, which is
   what an expanding window is, each observation is weighted by a decay factor so that older data
   counts but counts less, with the rate set by a stated half-life. Commercial risk models are built
   this way. The half-life plays a persistence role for the training sample loosely analogous to the
   switching penalty in the jump model gate. An infinite half-life is exactly the expanding window. A
   rolling window is **not** a half-life of zero, which would put all the weight on the last date; it is
   a different shape, a hard cut-off, of which exponential decay is the smooth counterpart.

   **Time decay, worked out in detail (added 2026-09-25). Nothing is implemented; this is a design
   question, not a queued change.**

   *The scheme Tom wants.* Retrain as new data arrives, as the walk-forward already does, but weight
   each training date `t` by its age relative to the last training date `T`, with half-life `h`:

       w_t = 2^(-(T - t) / h)

   and train on the weighted average loss, `sum_{i,t} w_t * loss_{i,t} / sum_{i,t} w_t`. Weights are
   per **date**, so every stock on the same date gets the same weight. Dividing by the sum keeps the
   loss on the same scale for every `h`, so the learning rate and the correction penalty weight do not
   silently change meaning. `h` infinite recovers the current expanding window exactly.

   *The tension with the thesis, which is the reason this is a question rather than a feature.* Decay
   rests on the view that **markets drift**, so old data describes a world that no longer exists; that
   is López de Prado's stated rationale, citing Lo's adaptive markets hypothesis. The regime model rests
   on the opposite view for part of the variation, that **regimes recur**, so 2008 remains informative
   about the next crisis. The two are competing explanations of the same apparent non-stationarity.
   The collision is concrete: with a two-year half-life, at a fold ending in 2020 the 2008 data carries
   weight `2^-6`, about 1.6 per cent. The crisis regime is the one with the least data, and nearly all
   of it is old, so calendar-time decay starves it hardest. This is the same argument as the one above
   against rolling windows, in a softer form.

   *It caps the effective sample, whatever the length of history.* The effective sample size of
   weighted data is `n_eff = (sum w)^2 / sum w^2`. For exponential decay over a long history this is
   `2h / ln 2`, about **2.9 half-lives**, independent of how much data exists. A two-year half-life on
   monthly data gives roughly 69 effective months against 720 in sixty years, perhaps three or four
   stress episodes. Since episodes, not dates, are what limit the number of regimes (see the
   consequence for `K` below), decay feeds directly into Q18.

   *Where it could apply, component by component.*
   - **Base.** The most defensible place, and only as an option: the base captures the
     regime-independent relationship, which is precisely what slow drift would affect. Note that this
     cuts against the architectural argument above, which wants the base on the longest possible
     sample. A decayed base against an undecayed base is itself a test of whether the gate absorbs
     the adaptation.
   - **Experts.** Optional and risky. A crisis expert learns mostly from old, rare episodes.
   - **Gate.** No, by default. The regime model exists to learn from all past regimes. statsmodels also
     offers no weighted Markov switching likelihood, so decay here would mean writing a custom
     estimator.

   *If adopted.* `h` becomes a hyperparameter, set per component, selected on the training block only
   and logged as a selection event like every other. `n_eff` should be reported in every run. And an
   `h`-infinite arm must be kept regardless, because Gu, Kelly and Xiu, the benchmark, use an expanding
   window refitted once a year **with no decay**; dropping that arm would break comparability with them.

   *Reading, about an hour in total.*
   - López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley, **section 4.7 "Time
     Decay"**, with 4.4 and 4.5 on sample uniqueness for context (in the Books folder). Checked against
     the text: his scheme is **piecewise linear, not exponential**, with a parameter `c` running from no
     decay (`c = 1`) to erasing the oldest data (`c < 0`), and it decays over **cumulative uniqueness**
     rather than calendar time, because "a chronological decay would reduce weights too fast in the
     presence of redundant observations", meaning overlapping labels. The exponential version is left
     as his exercise 4.4.
   - Lo, A. W. (2017), *Adaptive Markets: Financial Evolution at the Speed of Thought*, Princeton
     University Press. The argument that market relationships evolve, which is the case for decay.
     Skim the framing chapters only.
   - J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*, 4th edition. The canonical
     exponentially weighted estimator in finance, a daily decay factor of 0.94 for volatility. Used
     for risk estimation rather than model training, but the same mechanism and the source of the
     "industry practice" claim above.
   - Gu, S., Kelly, B. and Xiu, D. (2020), *Review of Financial Studies* 33(5), the sample-splitting
     section, for the expanding, annually refitted, undecayed protocol of the benchmark.

3. **Crises differ from one another, and no split fixes that.** The honest response is to test it
   rather than assume it. A **leave-one-episode-out** analysis, training with 2008 excluded and testing
   on it, then the same for 2020, measures directly how far what is learned in one crisis transfers to
   another. As far as can be established nobody does this, it costs only re-runs of an existing
   pipeline, and it is informative whichever way it comes out.

**One consequence for the number of regimes.** If the stress regime occupies roughly fifteen per cent
of months, a thirty-five year sample gives about sixty-three months of it and a sixty year sample about
one hundred and eight. That is the effective sample available to the corresponding expert. At three
regimes this is thin but workable; at five it becomes very thin. This is an identification argument for
keeping the number of regimes small, independent of the testability argument in Section 2.

**Ask the advisor:**
(a) Given that the research question is about behaviour at regime transitions, is a sample containing
    only one or two major crises defensible, or does the study require the longer database sample?
(b) Rolling or expanding training window? The expanding one is consistent with the criticism the
    literature review makes of rolling re-estimation, but departs from the protocol of the papers being
    compared against.
(c) Is the leave-one-episode-out analysis worth the additional runs, and should it be a headline result
    or a robustness check?
(d) Time decay: should older data be down-weighted at all, given that the thesis argues regimes recur?
    If yes, on which components (base only, base and experts, never the gate), and is it acceptable
    that it departs from the undecayed expanding protocol of Gu, Kelly and Xiu, provided an undecayed
    arm is kept?
(e) Crash periods and decay. Tom's idea (2026-09-26): old data from crashes is exactly what the model
    needs to keep, so decay should not wash it out. Three versions, from simplest to most ambitious:
    - **Exempt fixed windows** (for example 2008 to 2009 and March 2020) from decay, weight 1
      throughout. Legitimate only if each window is labelled using data before that fold's training
      cutoff; choosing the windows after seeing test results is data snooping.
    - **Regime-clock decay** (proposed, untested). Each expert's decay clock ticks only while its own
      regime is active, using the frozen gate's probabilities, so for the stress expert 2008 is only
      "a few stress-months ago". It fits the architecture because the gate already supplies the
      clock, and it addresses the tension in point 2 between decay and recurring regimes directly.
    - **Test which historical periods matter** (an extension, not part of the main study). After the
      main analysis shows which periods each expert works best in, measure how much each historical
      period contributes to out-of-sample performance, and propose that as the basis for choosing
      what not to down-weight. It generalises the leave-one-episode-out analysis in point 3. Any
      period chosen this way must be reported as a diagnostic or future work, never fed back into
      the reported model, or it is selection on the test set (Q15).

    *Reading, about two hours.*
    - Ghorbani, A. and Zou, J. (2019), "Data Shapley: Equitable valuation of data for machine
      learning", *ICML*, PMLR 97. How much each training point, or block of points, is worth to a
      model's test performance.
    - Koh, P. W. and Liang, P. (2017), "Understanding black-box predictions via influence functions",
      *ICML*, PMLR 70. A cheaper approximation of the same quantity without retraining.
    - Mulliner, A., Harvey, C. R., Xia, C., Fang, E. and Van Hemert, O. (2025), "Regimes", *Journal
      of Portfolio Management* 52(4), 6-25 (SSRN 5164863). Weights historical dates by how similar
      their economic state variables are to today's, on six equity factors over 1985 to 2024, and
      finds information in the most dissimilar periods too ("anti-regimes"). The closest finance
      precedent to weighting history by regime rather than by calendar time.

    Ask: are crash-preserving weights worth including, and if so as a main-study arm or as the
    proposed extension?

---

### Q17. Can statistical learning theory tell us how large the base network should be?
**Status:** open, raised by Tom 2026-09-15. Short answer: not directly, but one strand of it suggests a
theoretical contribution the thesis does not currently have.

**Why VC dimension will not answer the sizing question.** Four reasons, in order of severity.

*It is the wrong quantity for this problem.* The Vapnik-Chervonenkis dimension is defined for binary
classification: it is the size of the largest set of points the hypothesis class can label in all
possible ways. Predicting a continuous excess return is regression, for which the analogues are the
pseudo-dimension, the fat-shattering dimension, and Rademacher complexity. Boucheron, Bousquet and
Lugosi (2005), the survey in question, is explicitly a survey of classification theory.

*The bounds are numerically vacuous at this scale.* For a rectified-linear network the VC dimension
grows on the order of W L log W in the parameter count W and depth L. A 32-16-8 base on about one
hundred characteristics has roughly 3,900 parameters, giving a VC dimension of order 130,000; on the
full 920-feature set it is roughly 30,000 parameters and of order 1.2 million. With about 2.2 million
stock-month observations the resulting bound on the generalisation gap is about **0.46** in the first
case and **0.85** in the second. The quantity being bounded is a predictive R-squared whose best
attainable value in this literature is **0.0040**. The bound is two orders of magnitude larger than the
entire signal and therefore says nothing.

*The effective sample size is far smaller than the nominal one.* Those figures already flatter the
bound, because they count every stock-month as an independent observation. Returns are strongly
correlated in the cross-section, and more so in exactly the stressed periods this thesis is about, so
the effective number of independent observations is much closer to the number of **months**. At that
sample size the bound is not merely loose but infinite.

*The independence assumption does not hold.* Classical bounds assume independent, identically
distributed samples. This is a panel with temporal dependence and a distribution that shifts by
assumption, since non-stationarity is the premise of the study. Versions exist for dependent data under
mixing conditions, and they are weaker still.

**What does transfer, conceptually.** Two ideas from that literature bear directly on the design.
- **Complexity should be measured on the function class actually reachable, not on the parameter
  count.** Regularisation, early stopping and zero initialisation all shrink the reachable class
  without changing the number of parameters. This matters here: the correction experts are
  zero-initialised and penalised toward zero, so they occupy a small ball around the origin rather
  than the whole parameter space. Norm-based complexity measures capture that; VC dimension does not.
  It is a further argument that the shrinkage coefficient in Q14 is a capacity control, not a
  cosmetic one.
- **Rademacher complexity is data-dependent and can be estimated.** It is the one measure in that
  survey that is computable on a real dataset.

**One diagnostic worth actually running.** The practical descendant of Rademacher complexity is the
randomisation test: shuffle the labels and retrain. Shuffling returns **within each date**, so that the
cross-sectional structure is preserved and only the pairing between characteristics and outcomes is
destroyed, gives a direct measure of how much of the fitted relationship is memorisation. If the base
achieves materially positive in-sample fit on shuffled returns, it has enough capacity to memorise and
is too large. This costs a few training runs, requires nothing new, and answers the sizing question
empirically where theory cannot. Effective degrees of freedom, estimable by simulation, is a second
such measure.

**The right theoretical anchor already exists and is more modern.** He and co-authors (2026) derive a
finite-sample oracle risk bound specifically for mixtures of experts, with a router-estimation cost of
order (M d / n) log(M d n). That is a capacity analysis for precisely this architecture and this
sample-size regime, and it is already cited in Section 1. VC theory is its distant and much looser
ancestor; using it instead would be a step backwards.

**A possible theoretical contribution, and the reason this question is worth the time.** The thesis
already argues informally that a regime process is a **restriction on the class of admissible gates**.
That argument can be made exact by counting, at a level of mathematics well within reach.

An unconstrained gate may choose any of K states on each of T dates, so the class has K^T members and
log-cardinality T log K. A gate constrained to switch at most k times, which is what the switching
penalty in a jump model enforces, may choose k change-point locations and the values between them, so
the class has at most C(T-1, k) K^(k+1) members, with log-cardinality of order k log(T/k) + k log K.
The reduction is large. Over 2,000 daily observations with three states, an unconstrained gate has
log-cardinality of about 2,197 nats; constrained to at most ten switches it has about 73, a factor of
thirty. Over 500 observations the factor is about nine.

This turns the thesis's central claim from an intuition into a statement with a proof: **persistence is
a capacity constraint, and its strength is controlled by a parameter the design already sweeps.** It
connects directly to the empirical question, since if the complexity reduction is real then the
constrained gate should generalise better precisely when the sample is small or the signal weak, which
is testable. It would also answer Q2, which asks whether the comparison needs a constructive element,
with a theoretical component instead of an additional experiment.

**Ask the advisor:**
(a) Is the counting argument above worth developing into a short theoretical section, given that it
    formalises the thesis's own premise and needs no new mathematics beyond combinatorics?
(b) Should the randomisation test be adopted as the capacity diagnostic for the base, in place of any
    attempt to derive a bound?
(c) Is it acceptable to state plainly in the thesis that distribution-free bounds are vacuous at this
    signal-to-noise ratio, with the numbers shown, or does the committee expect the classical theory to
    be presented without that qualification?

---

### Q18. How many economically distinct states should the model posit?
**Status:** OPEN. Raised 2026-09-16, narrowed 2026-09-19 after the advisor meeting. The per-gate
two-panel design is withdrawn, since it is a sweep. What remains is the choice of K itself, which is
still unmade and which nothing else can make for me.

**The correction that matters most.** K is an **input** to every one of the four gates, not an output.
A Markov switching likelihood is not defined until the number of states is fixed; the jump model and
the Wasserstein clustering both take K the way k-means takes k. The gate chooses where the boundaries
fall, how persistent each state is, which dates belong to which state, and what each state looks like.
It does not choose how many there are. Models that genuinely infer the component count exist, the
Dirichlet process mixture and the hierarchical Dirichlet process HMM among them, but none of the four
gates is one, and adopting one for the likelihood-based gates only would reintroduce exactly the
confound this design exists to remove.

**A quiet expert is not model selection.** With soft gating, setting K too high can leave one expert
with near-zero average weight, which looks like the model settling on a smaller number. It is not. The
parameters are still estimated, the model is still over-specified, and the convergence penalty below
still applies. Do not read a collapsed weight as evidence about K.

**Why the data caps K.** Three results, increasing in severity.
1. Heinrich and Kahn (2018), *Annals of Statistics* 46(6A), 2844-2870, correcting Chen (1995), same
   journal 23(1), 221-233: fitting m components when the truth has m0 gives an optimal local minimax
   rate of n^(-1/(4(m-m0)+2)) against n^(-1/2) when correctly specified. In sample-size terms, reaching
   accuracy eps needs n ~ eps^(-2) when correct, eps^(-6) when over by one, eps^(-10) when over by two.
2. Hardt and Price (2015), arXiv:1404.4997: for two Gaussians in one dimension, Theta(sigma^12) samples
   are necessary and sufficient; for k components, Omega(sigma^(6k-2)) are necessary. The requirement
   grows exponentially in k, and collapses to O(sigma^2) only when the components are well separated,
   which calm and mildly stressed equity states are not.
3. In a time series the usable count is **episodes, not dates**, roughly T divided by the average spell
   length. Sixty years of monthly CRSP with spells of eighteen to twenty-four months is about thirty to
   forty episodes in total, from which K(K-1) transition parameters and K expert networks must be
   estimated, and the rare state may contribute fewer than ten.

The same singularity behind point 1 is why the likelihood ratio test for the number of regimes is
non-standard: the information matrix is singular under over-specification. Hansen (1992), Garcia (1998),
Cho and White (2007) with the Carter and Steigerwald (2012) comment, and Kasahara and Shimotsu.

**Does the number differ by gate?** The number of economically distinct states is a property of the
world, so in principle no. What differs is how many each mechanism can **resolve**: Hamilton separates
states by conditional mean and variance, the jump model by proximity in feature space under a switching
penalty, Wasserstein clustering by distributional distance. Same sky, different instruments. Their
natural K will not agree and even at a common K their boundaries will not coincide, which is an argument
for fixing K rather than letting each gate pick, since a gate with more experts has more capacity and a
win could not be attributed to its notion of a regime.

**How I propose to choose it, without a sweep.** A distinction worth drawing. Sweeping K means
retraining the whole mixture at several values and comparing out of sample: that is out. Selecting K on
the gate alone is different and nearly free. Fit only the regime model, no networks, at K in {2,3,4}, on
the training portion of the first development window, and choose by BIC. Three fits of a tiny model, no
test data touched, reported once. Then cap the result at three regardless, on the episode-count
argument, because BIC does not know that the rare state has eight observations.

Economic naming has to happen in advance too. If the states cannot be named before fitting, calm,
stressed, and something between, they cannot be interpreted afterwards, and the gate-permutation
diagnostic in Q10 has nothing to test against. Label switching means the model returns K states without
names; assigning them is done by inspecting fitted parameters on training data, never by checking which
labelling improves test performance.

**Ask the advisor:**
(a) Is BIC on the gate alone, computed once on the first development window and capped at three,
    an acceptable basis for fixing K, or does he want the number fixed purely by economic argument?
(b) Should the states be named and interpreted in advance so the interpretability diagnostics have a
    target, or is post hoc labelling from the fitted parameters sufficient?
(c) Is it acceptable to state in the thesis that K is held equal across gates for comparability, while
    acknowledging that this may handicap whichever gate's own criterion would have chosen differently?

---

### PROPOSED PILOT — construction experiment
**Belongs in the proposal form under *Research schedule and expected outcomes*.**

Q6, Q7 and Q8 should not be argued in prose and then assumed. They are settled by a pilot run **before** the main gate comparison, so that the construction is locked and every candidate gate is evaluated on identical scaffolding. The pilot runs in **two sequential stages rather than as a full factorial**, which keeps the cost proportionate: first fix the building block, then fix the way the blocks are combined.

**Stage 1 — the block.** Fit all candidate gates first and compute the effective sample size each gives an expert, which is available from the fitted weight sequence alone; select the sharpest and the most diffuse. Sweep base depth standalone, since the base is trained before any gate exists and its optimum cannot depend on one. Sweep expert depth in {1, 2, 3} at matched total mixture capacity across **every** gate, reporting the complete grid, so that a gate-by-depth interaction is exhibited rather than assumed away (Q11); the effective-sample calculation then serves to explain any interaction the grid reveals rather than to choose which gates to test. See Q8 for why none of this can be settled by citation and for the mechanism behind the interaction.

**Stage 2 — the construction.** With the block fixed from Stage 1, the arms below:

| Arm | Construction | Precedent | What it isolates |
|-----|--------------|-----------|------------------|
| A | Standard MoE, `ŷ = Σ_k π_k f_k(x)`, joint | MIGA, PRISM-VQ; Ye & Borde §3.2 | replication of their result on a returns target |
| B | Residual MoE, base **pre-trained then unfrozen** and trained jointly with gate and experts | DeepSeekMoE shared-expert isolation, adapted | **the untested cell — isolates freezing alone** |
| C | Residual MoE, base pre-trained and **frozen** | RG-ResMoE (Ye & Borde) | the proposed construction |
| D | Arm C with **randomly initialised** experts | Ye & Borde, Table 6 | how much of the effect is zero-init, not the base |
| E | Arm C with the shrinkage coefficient **α = 0** | untested — Ye & Borde never ablate α | whether the penalty earns its place, and the upper bound on how visible the gate's contribution can be |

**Note on arm B's design.** A naive DeepSeek-style arm — base, gate and zero-initialised experts all trained from scratch together — would not isolate freezing, because the value of zero initialisation comes from the base already being a good forecaster. With a randomly initialised base, "start at the base and stay near it" means starting at noise, so such an arm would be handicapped for a reason unrelated to the question. Arm B should therefore **pre-train the base exactly as arm C does, then unfreeze it** for the second stage. Arms B and C then differ in precisely one bit: whether the base's weights are updated while the gate and experts are trained.

**Held fixed across all arms:** one gate (the simplest on the shortlist, so the pilot does not
pre-judge the main comparison), K, expert width and depth, features, windows, optimiser, early
stopping, ~30 seeds per arm.

**Decision rule, fixed in advance.** Rank IC and ICIR for accuracy. For stability, a
**returns-appropriate** criterion defined in advance — across-seed standard deviation of rank IC,
plus the count of seeds with out-of-sample IC below a stated threshold — because the paper's
collapse metric (QLIKE > 2) is volatility-specific and does not carry over. And the criterion that
decides ties in favour of C: whether `Σ_k π_k r_k(x)` is large enough and variable enough through
time to support the analysis the thesis depends on.

**Expected outcomes, stated honestly.**
- A vs C is **replication on a new target**, not a new result. Its value is establishing that the
  residual advantage — demonstrated for five-day realised volatility — survives the move to signed
  cross-sectional returns, where the QLIKE failure mode does not exist. That is a real and citable
  thing to check, and it may well fail.
- **B vs C is the genuinely open question** and the reason the pilot is worth scheduling. No
  published arm separates "having a shared base" from "freezing it."
- D quantifies how much of the residual benefit is really the zero initialisation.

**Cost:** four training loops over the same codebase. Roughly two to three weeks including the seed
sweep, early in the schedule.

**Ask the advisor:** is a construction pilot an appropriate item to write into the research schedule,
or does the committee expect the architecture to be settled from the literature before the proposal
is submitted?

---

### Q20. Which error function should the model be trained against?

**Status:** open — raised 2026-09-21. Deliberately excluded from
`Code_Change_Brief_02_Residual_Frozen_Base.md`, which leaves the existing objective untouched and
adds only a registry seam so that answering this later is a registration rather than a rewrite.

**The candidates.**

1. **Mixture negative log likelihood**, which is what `nec_baseline` implements today: Gaussian
   emissions, a learned per-expert noise scale, and responsibilities obtained by Bayes' rule.
2. **Squared error, or Huber, on the point prediction.** What Ye & Borde use, and what Gu, Kelly and
   Xiu use. Their preference for Huber over plain squared error is not cosmetic: returns are
   fat-tailed and the robust loss made a material difference in their results.
3. **A cross-sectional rank or information-coefficient objective**, matching the metric the thesis
   is actually scored on.

**The tension that is specific to this thesis, and the reason this cannot be waved through.** The
two leading candidates weight the experts by different things.

- The residual point prediction is `y_hat = f0(x) + sum_k pi_k r_k(x)`, weighted by the **prior**
  `pi` that comes out of the frozen gate. Under squared error, expert k's gradient is proportional
  to `pi_k` and to nothing else. The frozen gate fully determines which expert learns from which
  observation.
- The mixture likelihood weights expert k's gradient by the **posterior** responsibility
  `gamma_k = pi_k N_k / sum_j pi_j N_j`, which depends on the realised target through the
  likelihood term. An expert that happens to sit close to `y` receives more gradient than its gate
  weight alone would give it.

Under a **trainable** gate the posterior weighting is the whole point, because it is what supplies
the gate's training signal. **Under the frozen gate decided in Q19 that justification disappears**,
since the gate receives no gradient at all. What remains is that the realised target gets to
re-weight the experts *after* the regime assignment has already been made, which partially undoes
the constraint the frozen gate was introduced to impose. Given that the stated purpose of freezing
both the gate and the base is to leave the regime assignment as the only thing varying across arms,
this is an argument in the direction of the point-prediction loss. It is an argument, not a
conclusion, and it should be put to the advisor rather than settled unilaterally.

**Three secondary considerations.**

- **Objective versus metric.** The thesis is scored on rank-IC and long-short portfolio performance
  (Q9). Squared error optimises level accuracy; the metric rewards cross-sectional ordering. Whether
  it is acceptable for the training objective to differ from the evaluation metric is part of this
  question, and the rank objective is the candidate that closes the gap at the cost of leaving the
  benchmark literature's practice.
- **Raw target or residual.** Under squared error the two are equivalent: minimising
  `(y - f0 - sum_k pi_k r_k)^2` is exactly minimising the squared error of the mixture against the
  residual `y - f0`, because the frozen base contributes a fixed per-observation offset. Under the
  likelihood they are **not** equivalent, because the noise scale then describes a different
  quantity. So the choice of objective silently decides what `sigma_k` means.
- **The choice is a selection event.** Whichever objective is picked must be fixed before any
  out-of-sample result is looked at, and if more than one is tried, the number tried enters the
  multiple-testing accounting like any other candidate.

**What to ask.** Which objective, and is it acceptable for it to differ from the evaluation metric?
And, if the likelihood is retained, is the posterior re-weighting of experts under a frozen gate a
feature or a leak?

---

## RESOLVED

### Q2. Is the comparison framing enough, or does the thesis need a constructive contribution? [RESOLVED 2026-10-08]

**Answer: the comparison framing is enough; no constructive element is added.** Tom's decision,
2026-10-08: "Nothing is wrong with my framing." The contribution is the controlled comparison of
structurally distinct regime processes as the gate of a residual mixture of experts (now with an
industry-level component, Q1), including the statistical jump model and Wasserstein clustering as gates.
The differentiable continuous jump-model gate is not pursued.

*Original question, for the record:*

### Q2. Is the comparison framing enough, or does the thesis need a constructive contribution?
**Status:** open — raised 2026-09-10

A novelty audit found that the *mechanism* is not new: gated experts for regime discovery goes back
to Weigend, Mangeas & Srivastava (1995); an HMM has been used as an MoE gating model (REW-MSLM,
2022); and filtering-based gating on financial data is published (MoE-F, ICLR 2025). What appears
genuinely unclaimed is the **controlled comparison** of structurally distinct regime processes as
the gate, plus specifically the statistical jump model and Wasserstein clustering as MoE gates.

Given the routing comparison was rejected for having been done already, does a regime-mechanism
comparison stand on its own? Or should it be paired with a constructive element — the most natural
being to make the *continuous* jump model into a differentiable gate, which nobody has done?

---

### Q4. Which framing for the proposal — comparison, or design-led? [RESOLVED 2026-10-08]

**Answer: comparison-led.** Tom's decision, 2026-10-08, together with Q2: the proposal is framed as
"which regime mechanism makes the best gate", not as a new gate design. Q5 (positioning against Ye &
Borde) stays open.

*Original question, for the record:*

### Q4. Which framing for the proposal — comparison, or design-led?
**Status:** open

Two ways to write the same work:
- **Comparison-led:** "which regime mechanism makes the best gate?" Honest, but structurally similar
  to the rejected topic.
- **Design-led:** "here is a differentiable persistence-penalised gate, and here is the evidence it
  beats the alternatives." Same experiments, different emphasis.

Which reads better to the review committee?

---

### Q1. Should the regime gate be market-wide, or partly stock-specific? [RESOLVED 2026-10-08]

**Answer: partly stock-specific, at the industry level.** Tom's decision, 2026-10-08, stated as final.
The gate is no longer one weight vector per date shared by every stock. Each stock's gate weight combines
the **market regime** (shared by all stocks) with the **regime of its own GICS sector**, so a stock's
weight on date t depends on the market and on its industry. Stocks in the same sector on the same date
share a weight; stocks in different sectors can differ.

**Why.** Industries go through their own stress while the rest of the market is calmer: energy in
2014-2016, banks in 2008 and 2023, technology in 2000-2002 and 2022. A market-wide gate cannot express
this. The industry level keeps the market regime at the core of the thesis, which is the part best
supported by the literature and by the descriptive checks, and stays well short of per-stock routing.

**How it is implemented: open, see Q28.** Only the direction is decided here (market regime plus an
industry-level component, both fitted separately and frozen, Q19). The form of the industry component and
how it combines with the market regime are deliberately left open.

**Consequences.**

- **Q18 (K):** may become the choice of two numbers (market and industry states), depending on Q28.
- **Q24:** the per-row likelihood stays; with a sector-level latent it is closer to exact than before.
- **Q27 and Q16:** the window-average gate weight and the regime-clock decay will be computed per row from
  the combined weights; the formulas carry over whichever form Q28 picks.
- **Q10 (identification):** the experts already see sector dummies, industry momentum and
  within-industry reversal. The pilot must check that the sector regime adds information beyond them.
- **The five gate types:** each needs an industry-level counterpart. If that is one fit per sector per fold,
  it is 11 extra fits per fold and gate type, about 29 s each with statsmodels; small next to training.
- **Documents to update:** the PDF and Model_Derivation_BaseCase.md assume a shared weight vector per
  date; both need a section on the industry-level gate.
- **The advisor** should hear this explicitly, since it changes the object of study from market regime
  processes alone to market and industry regimes.

**Still open:** everything about the implementation, in Q28.

**Correction to the original proposal below.** Its variance-decomposition test (across dates against
across stocks within a date) cannot work with a frozen shared gate, whose weights are identical across
stocks by construction. The descriptive check in item 5 replaces it.

*Original question, for the record:*

### Q1. Should the regime gate be market-wide, or partly stock-specific?
**Status:** open — raised 2026-09-10

**The question.** In the model, the gate produces a weight vector over experts. Two designs:

- **Shared:** one weight vector per date, `π_t`, used by every stock in the cross-section.
- **Partly stock-specific:** `π_{i,t}`, a function of both a market-wide state (e.g. market
  volatility) and a stock-specific state (e.g. that stock's idiosyncratic volatility).

**Why it matters.** It changes what the thesis is. A shared gate makes the object of study a
*market regime process* and keeps the work clearly distinct from per-item routing. A stock-specific
gate makes it closer to routing — the axis I was steered away from — but is arguably more realistic,
since a semiconductor stock can be in turmoil on a day when utilities are calm.

**What the literature says (checked 2026-09-10):**
- The closest precedent, the regime-gated residual MoE for cross-sectional volatility
  (arXiv:2608.12251), uses **both**: its gate receives market volatility (shared across all stocks)
  *and* idiosyncratic volatility (per stock). So its gate is already partly stock-specific.
- "Dynamic Asset Allocation with Asset-Specific Regime Forecasts" (arXiv:2406.09578) generates
  **asset-specific** regime labels using the statistical jump model, explicitly contrasting this
  with "broad economic regimes affecting the entire asset universe."
- "Adaptive Market Intelligence" (arXiv:2508.02686) and arXiv:2410.07234 both gate between a
  high-volatility expert and a stable-equity expert **based on asset classification** — i.e. exactly
  per-stock assignment by volatility.
- Liu, Maheu & Song (2024), *Journal of Applied Econometrics* 39(5), 723-745, identify bull and
  bear markets using **multivariate** returns rather than the index alone.

**Arguments for keeping it shared.** Correlations rise and cross-sectional dispersion falls in
downturns, so industry differences matter least exactly when the regime matters most. Only ~20-50
distinguishable regime episodes exist in the whole record — splitting by sector leaves nothing to
estimate on. And the experts already handle stock heterogeneity through the characteristic vector.

**Arguments for making it partly stock-specific.** More realistic; the closest precedent does it;
and it costs almost nothing to add one stock-level state variable.

**What I would propose.** Keep the shared gate as the core (it is what makes the comparison a
controlled experiment), and decide the question empirically: decompose the variance in fitted gate
weights into an across-dates component and an across-stocks-within-date component. If most variation
is across dates, the shared gate is justified by evidence rather than assumption.

**Ask the advisor:** does adding a stock-specific state variable to the gate move the thesis back
toward the routing topic that was rejected, or is it a legitimate design refinement?

---

### Q22. Should the gate's filter run through the purge gap? [RESOLVED 2026-10-08]

**Answer: (b). The filter runs over every trading day, including the purge gap.** Tom's decision,
2026-10-08. The gate's parameters are still estimated on the training block only and then frozen; only
its regime belief is updated with the gap's market returns.

**Why.**

1. **The purge is for labels, not information.** The gap exists because the gap days' 5-day targets
   reach into the test block. Their market returns are public by the first test date, so feeding them to
   the filter is not look-ahead: a live trader's filter would have processed them.
2. **It removes a mis-specification.** The transition matrix A describes one trading day. Skipping the
   gap makes one step of A bridge h + 1 = 6 days at every fold boundary (15 times under annual refits).
3. **Consistency with the rest of the gate.** The window-average gate weight (Q27) and the regime-clock
   decay (Q16) both start from the filtered probability, so the state at the start of each test year
   should be exact.
4. **Simpler to state in the methodology:** "the purge removes labels, never information; the filter
   runs over every trading day."
5. **Small in size.** On the smoke panel the largest change in a regime probability was 0.008, on the
   first test date of a fold. This is a correctness point, not a results point.

**Implementation.** `evaluation.py` applies the gate to `train_dates + test_dates`; it will apply it to
the contiguous range from the first training date to the last test date. Test: the test-date
probabilities equal those of a single filter pass over the full series with the frozen parameters.
The same applies to the pilot (`pilot.py`). Goes into the next code brief.

*Original question, for the record:*

### Q22. Should the gate's filter run through the purge gap?

**Status:** open, raised 2026-09-26 from audit finding G-1. Nothing decided; the code keeps its
current behaviour until this is answered.

**What the purge gap is.** In walk-forward testing the last `h` days before each test block are
dropped from training, because their targets (forward returns) reach into the test period and would
leak test information into training. The gap exists for the **labels**.

**What the code does.** The frozen Hamilton filter is run over the training dates and then directly
over the test dates, so it also skips the market returns of the gap days. On the first test date it
has not seen the last `h` days of market data, and one transition step bridges `h + 1` trading days.

**Why it is a question at all.** The gap's market returns are public by the first test date, so
feeding them to the filter is not look-ahead: it is what a trader would know on that date. Skipping
them uses less information, not more, so the current behaviour is safe but slightly stale.

**Measured size (audit 2026-09-25, smoke panel).** The largest change in P(high-volatility regime) on
any test date is 0.0079, always on the first test date, and above 0.001 on at most two test dates per
fold. Small, but not zero.

**Options.**
- (a) Keep the skip. Simplest to describe; wastes `h` days of information at each fold boundary.
- (b) Run the filter over the contiguous span up to the last test date. Causal, and adds only the
  gap's market returns.

**Interaction with Q21.** The gap is `h` days long, so with `h = 1` the question almost disappears,
and with a monthly horizon it becomes about a month of unseen market data at every fold boundary.

**Ask the advisor:** (a) or (b), and is it worth a sentence in the methodology either way?

---

### Q23. Which market series should the Hamilton gate be fitted on? [RESOLVED 2026-10-08]

**Answer: (b). The CRSP value-weighted index of the S&P 500 universe (INDNO 1000500), daily total
return including dividends, as a log return.** Tom's decision, 2026-10-08. French's Mkt-RF stays
registered as a robustness check.

**Why.**

1. **One market definition.** The universe, the market-model features (beta, idiosyncratic volatility,
   coskewness) and the gate now all use the same index. The regime should describe the market the
   experts trade in.
2. **Provenance.** Same licensed CRSP release as everything else; no external file, no publication lag.
3. **Precedent.** Nystrup, Madsen & Lindström (2017), whose method the gate memory follows (Q16), fit
   their HMMs on S&P 500 daily log returns.
4. **The risk-free rate is immaterial.** About 0.005% a day against a daily volatility of about 1%; the
   gate separates regimes mainly by variance.

**What the gate sees (clarified 2026-10-08).**

- **One series only.** The Hamilton gate is univariate: its only input is the market's daily return. It
  does not see the 57 stock inputs. The gate answers "what state is the market in"; the experts answer
  "which stocks beat which in that state" (Q1: one shared weight vector per date; Q19: fitted
  separately and frozen). Later gates may use a few market-level inputs (the jump model's features of
  the same series, TVTP's covariates), never stock characteristics.
- **Not market-neutral.** Market-neutralising applies to the target only. The gate's input is the market
  itself, and the market minus the market is zero. The market's own level and volatility are exactly the
  regime signal.
- **Two different "markets" in the pipeline, on purpose.**
  - The **gate and the market-model features** use the value-weighted index (1000500), the standard
    measure of the market's state, dominated by the large caps that drive aggregate volatility.
  - The **target** is demeaned by the **equal-weighted** mean of the universe's 5-day returns on each
    date (Q25), so it sums to zero across stocks and matches an equal-weighted long-short book.
  - The networks never see the market return as an input: it is identical across stocks on a date, so
    ranking would remove it. It enters the networks only through the features built from it.

**Implementation.** A new `SERIES_REGISTRY` entry reading the CRSP market series already stored with
the panel (`market.parquet`; it is not one of the panel's sequence features), made the default. Goes
into the next code brief.

*Original question, for the record:*

### Q23. Which market series should the Hamilton gate be fitted on?

**Status:** open, raised 2026-09-26 from the CRSP integration run (brief 06 F). Nothing decided; the
code keeps its current input until this is answered. Something to settle later, not now.

**What the code does.** The gate is fitted on Kenneth French's daily market excess return (Mkt-RF),
series key `market_excess_return` in `markov_gate.py`'s `SERIES_REGISTRY`, as brief 03 specified.
Everything else in the pipeline now comes from CRSP: the features, the residual target and the
S&P 500 universe use the CRSP value-weighted S&P 500 index (INDNO 1000500).

**What French's series is.** It is not low-quality free data. French builds Mkt from CRSP: the
value-weighted return of all US-incorporated stocks on NYSE, AMEX and Nasdaq, minus the one-month
Treasury bill rate. The issues are provenance (an external file, published with a lag) and
consistency (the whole market, while the rest of the pipeline uses the S&P 500).

**Options.**
- **(a) Keep French Mkt-RF.** Standard in the literature, and an excess return.
- **(b) CRSP S&P 500 index total return (INDNO 1000500).** One market definition across the gate,
  the features and the target. Raw return rather than excess, because the CRSP download has no
  daily risk-free rate.
- **(c) CRSP whole-market index (INDNO 1000200, NYSE/NYSE American/Nasdaq/Arca value-weighted).**
  The closest CRSP copy of French's series, again without the risk-free rate.

**Why the risk-free rate barely matters here.** At daily frequency the T-bill return is of the
order of 0.005% a day, against a daily market standard deviation of about 1%, and the Hamilton gate
separates regimes mainly by variance. Dropping it shifts every regime mean by the same tiny constant.

**Cost of changing.** Small: the gate's input is a registry key, so (b) or (c) is one new entry.
Whichever is not chosen can stay registered as a robustness check.

**Ask the advisor:** (a), (b) or (c)? Is consistency with the pipeline's market definition worth
departing from the literature's standard excess-return series?

---

### Q27. Which gate weight: filtered, one-step predicted, or averaged over the 5-day window? [RESOLVED 2026-10-05]

**Answer: the average over the target window of the h-step-ahead regime probabilities.** The same
weight is used in training and in forecasting. Tom decided this on 2026-10-05. Before this, the item
was listed as "not yet a question" in the Progress Tracker and in the PDF (Section 8.9).

**The three candidates.** At the close of day t, the gate has today's filtered probability, the
probability of today's regime given the market series up to t:

$$\xi_{t\mid t}(k)=\mathbb P(z_t=k\mid\mathcal F_t)$$

From it, with the transition matrix A, it can form:

1. **Filtered:** today's regime, the filtered probability itself. Model_Derivation_BaseCase.md and
   the Hamilton gate in the code (`markov_gate.py`, `filtered_marginal_probabilities`) use this.
2. **One-step predicted:** tomorrow's regime.

   $$\pi_k(t)=\mathbb P(z_{t+1}=k\mid\mathcal F_t)=\big[\xi_{t\mid t}^\top A\big]_k$$

   The PDF's one-step derivation and the backprop-HMM baseline use this.
3. **Window average (chosen):** the expected share of the target window (t, t+h] spent in each
   regime, with h = 5 (Q21).

   $$\bar w_k(t)=\frac1h\sum_{j=1}^{h}\big[\xi_{t\mid t}^\top A^{\,j}\big]_k$$

All three use only the market up to day t (they are F_t-measurable), so none has look-ahead.

**Why the window average, in depth.**

1. **It is the one that is exact for the target.** The target is the sum of h daily market-neutral
   returns. Assume each day's expected return is set by that day's regime, with regime k contributing
   one h-th of expert k's window prediction per day. Then
   $$\mathbb E[y_{i,t}\mid\mathcal F_t]=\sum_{j=1}^h\sum_k\mathbb P(z_{t+j}=k\mid\mathcal F_t)\,\frac{\mu_k(x_{i,t})}{h}=\sum_k\bar w_k(t)\,\mu_k(x_{i,t})$$
   So the gate-weighted forecast with these weights is exactly the conditional expected 5-day return.
   The expected within-regime noise also adds up over days, so it is matched as well. The filtered
   weight assumes the regime never changes during the window. The one-step weight assumes it changes
   at most once, tomorrow. Both are approximations to the window average.
2. **It is consistent with the generative model.** Returns on each future day come from that day's
   regime (PDF Section 6). For h = 1 the window average reduces to the one-step predicted
   probability, so the one-step derivations in the PDF are the h = 1 case of this rule.
3. **It is the textbook multi-step forecast.** In Hamilton's framework, a forecast h days ahead uses
   the filtered belief moved h steps forward with the transition matrix (Hamilton 1994, ch. 22).
   Regime-switching asset allocation does the same over a holding period (Guidolin & Timmermann 2007).
   Using today's filtered probability for a future window is the practitioners' shortcut.
4. **It costs no information and no look-ahead.** It is built from the filtered belief and the frozen
   A.
5. **Training and forecasting use the same weight,** so the experts never face a different kind of
   gate weight at test time (the same principle as using filtered rather than smoothed probabilities
   on training dates).

**Descriptive check.** A 2-state Gaussian Hamilton model was fitted to the CRSP S&P 500 index (INDNO
1000500), daily, 2015-2024. This is a full-sample fit of the gate alone, not model results.

| Regime | Daily volatility | Expected duration |
|---|---|---|
| Calm | 0.6% | about 54 days |
| Stressed | 1.8% | about 27 days |

The weight on the stressed regime differs from the filtered weight as follows:

| Comparison | Mean gap | Largest gap |
|---|---|---|
| Window average | 0.055 | 0.10 |
| One-step predicted | 0.020 | — |

- The window-average gap exceeds 0.10 on about 10% of days, mostly when the filter is sure of stress.
  The average then pulls the weight back towards the long-run mix, because stress tends to end.
- About 18% of 5-day windows contain a switch of the filtered regime. This is a noisy proxy (crossings
  of 0.5), but it means the "one regime per window" approximation fails often enough to matter.
- The correction is modest and systematic, and largest in stress.

**Costs accepted.**

- The weights are slightly less sharp (about 0.86 instead of 1.00 when stress is certain), so the
  experts separate a little less strongly.
- The weight relies on A being well estimated, because errors in A compound over h steps.
- The mixture likelihood with these weights is still the "one regime per window" approximation. The
  exact distribution of a 5-day return is a mixture over regime paths. The window weights get its
  mean and its expected within-regime variance right; the likelihood is used for estimation, not
  inference.

**What it changes.**

- **Code.**
  - The Hamilton gate must output the window average instead of the filtered probability. Per the
    rule that every hyperparameter is a config field, this is a switch (`gate_weight`: `filtered` |
    `predicted` | `window`, default `window`), with h read from the target's horizon.
  - The backprop-HMM baseline (`HMMRegimePrior`) is **not** changed. Its latent is attached to each
    date's whole target window, one latent per 5-day target rather than a daily market regime. Its
    h-step predict from the last fully realised target is already the consistent weight for that
    model. Correction 2026-10-05: an earlier line here said it would move to window weights.
  - Gates without a transition matrix (the statistical jump model, the Wasserstein gate) need one
    estimated from their decoded state sequence on the training block. That is a detail for the
    implementation brief.
- **Docs.**
  - PDF Section 8.6 and 8.9 and the checklist are updated (revised 2026-10-05).
  - Model_Derivation_BaseCase.md section 1.7 now points here.

*Sources:* Hamilton (1994), *Time Series Analysis*, ch. 22; Guidolin & Timmermann (2007), JEDC 31(11);
Hamilton (1989), Econometrica 57(2).

### Q24. Is the regime latent per date or per observation in the likelihood? [RESOLVED 2026-10-05]

**Answer: per row.** The experts are trained on the per-row objective, which is what the code already
does (`losses.mixture_nll`, one log-sum-exp per row). Tom decided this on 2026-10-05. The model itself
is unchanged: one regime per date, shared by all stocks (Section 6 of `MoE_HMM_Gate_Formulation.pdf`).
Only the estimation objective is per row. No code change follows. The full theory is in Appendix D of
the PDF, revised 2026-10-05.

**The correction that settles it.** The original question (below) treated the per-row likelihood as
the likelihood of a *different* model, one where each stock-day draws its own regime. That is true,
but it is only half the picture. Under the per-date model, integrating out every other stock gives
each stock's own distribution, and that is exactly the per-row factor:

$$p\big(r_{i,t+1}\mid x_{i,t},\mathcal F_t\big)=\sum_{k=1}^K\pi_k(t)\,\mathcal N\big(r_{i,t+1};\mu_k(x_{i,t}),s_k^2\big)$$

So the per-row objective is the sum of the *correct* one-stock log-likelihoods of our model. It drops
only the dependence between stocks on the same date. That kind of objective is a **composite
likelihood**, here the independence likelihood (Lindsay 1988; Varin, Reid & Firth 2011). The choice is
therefore between the full likelihood and a composite likelihood **of the same model**, not between
the right model and a wrong one.

**Why per row, in depth.**

1. **It estimates the right parameters.** Each term is a true log-density, so its score has expectation
   zero at the true parameters, and a sum of such terms is an unbiased estimating equation whether or
   not the terms are dependent. Under the usual regularity conditions the estimator is therefore
   consistent. The frozen gate ties expert k to HMM state k, which removes the label-switching that
   would otherwise make the mixture unidentified.
2. **It is robust to the known failure of A3.** The per-date likelihood uses its extra information only
   through A3: given the regime, stocks are independent. That is false. Industry peers share shocks
   after the market is removed, and the market-neutral target leaves residual beta exposure (Q25).
   - The per-date likelihood then counts correlated stocks as independent evidence. With clusters of
     m stocks whose evidence terms correlate at rho, the cross-section is worth only
     N / (1 + (m - 1) rho) independent stocks (Kish's design effect). In an illustration with assumed
     values, m = 45 and rho = 0.1, the per-date posterior is about 5 times overconfident.
   - Its parameters become pseudo-true values: the best fit to the wrong joint distribution. A
     stylised simulation in Appendix D.8 shows this bites when regimes are close. With 600 dates,
     200 stocks in clusters of 20 and within-cluster correlation 0.5, the per-row estimates stay near
     the truth. The per-date estimates put the two experts' means nearly eight times too far apart
     (0.38 against 0.05) and squeeze their noise levels together. With no correlation, both objectives
     recover the truth.
   - The per-row objective needs only each stock's own distribution to be right. Appendix D.8 gives a
     model where A3 fails but every one-stock distribution is exactly right; the per-row objective
     stays correctly specified and the per-date one does not. This is the logic of working-independence
     estimating equations (Liang & Zeger 1986).
3. **It keeps the frozen gate in charge.**
   - Under the per-date likelihood, the responsibility (the regime posterior after seeing the returns)
     is almost always 0 or 1. With about 0.22 nats of evidence per stock, 500 stocks give about 110
     nats. The day's returns set it, whatever the gate said. Each expert learns from the days whose
     cross-section fits it, so the stocks redefine the regimes every day.
   - Under the per-row objective, one stock's evidence is small (a likelihood ratio of about 1.25), so
     the responsibility stays near the gate's weight. Each expert learns mainly from the days the gate
     assigns to its regime.
   - That is the purpose of freezing the gate (Q19): regimes stay market regimes.
4. **The forecast formula is identical.** Both use the gate-weighted average of the experts; only
   training differs.
5. **Practicalities.** The per-row objective is already implemented, mini-batches can be any rows, and
   training is smoother.

**Costs accepted.**

- **Efficiency.** The information in the co-movement of stocks on a date is not used. The relevant
  information measure is the Godambe (sandwich) information, which is never larger than the Fisher
  information.
- **No standard errors from the Hessian** (the matrix of second derivatives of the objective). The
  inverse Hessian is not a valid variance under a composite likelihood. This does not matter here,
  because inference comes from walk-forward rank ICs with Hansen-Hodrick corrections.
- **Softer specialisation.** The expert with the larger noise level can drift into absorbing outlying
  stock-days. Diagnostic after training: compare each expert's training weight with the gate's weight
  and with the size of the residuals.

**What this does not settle.** Q20, the error function. If the mixture likelihood is replaced by a
point-prediction loss, the per-date/per-row distinction disappears.

*Sources:* Lindsay (1988), Contemporary Mathematics 80; Varin, Reid & Firth (2011), Statistica Sinica 21;
Liang & Zeger (1986), Biometrika 73(1); Pakel, Shephard, Sheppard & Engle (2021), JBES 39(3); Kish (1965),
*Survey Sampling*; Godambe (1960), Annals of Mathematical Statistics 31.

*Original question, for the record:*

### Q24. Is the regime latent per date or per observation in the likelihood?

**Status:** open, raised 2026-10-01 while deriving the model from Bishop 14.5. Nothing decided; the
code keeps its current behaviour. Closely tied to Q20.

**The two likelihoods.** The thesis story is that a single regime holds for the whole market on a
date. Taken literally, all stocks on date t share one latent regime, so the product over stocks sits
**inside** the sum over regimes:

$$p(y_{1,t}, \dots, y_{N,t}) = \sum_{k=1}^{K} \pi_k(t) \prod_{i=1}^{N} \mathcal{N}\big(y_{i,t};\, \mu_k(x_{i,t}),\, \sigma_k^2\big)$$

The code (`losses.mixture_nll`, a log-sum-exp per row) uses the per-observation version instead,
where each stock-date draws its own regime independently and the sum sits inside the product:

$$\prod_{i=1}^{N} \sum_{k=1}^{K} \pi_k(t)\, \mathcal{N}\big(y_{i,t};\, \mu_k(x_{i,t}),\, \sigma_k^2\big)$$

**What differs in practice.**

- **Responsibilities.** Date-level: the posterior over regimes pools the evidence of all ~500 stocks
  on the date, so it becomes nearly 0 or 1 and each date is effectively assigned to one expert.
  Per-observation: each stock gets its own posterior, so two stocks on the same date can be
  attributed to different regimes.
- **Fidelity to the story.** The date-level form matches "one market regime per day" literally. The
  per-observation form is a softer approximation that typically trains more smoothly.
- **Interaction with Q20.** Under a point-prediction loss (squared error, Huber), the experts are
  weighted by the prior pi_k(t) alone and the distinction disappears. It only matters if the mixture
  likelihood is kept.
- **Interaction with the frozen gate (Q19).** With a date-level likelihood, the realised returns of
  the whole cross-section re-weight the experts on each date, which strengthens the posterior-versus-
  prior concern already raised in Q20.

**Ask the advisor:** if the mixture likelihood is kept, should the regime be latent per date (as the
regime story implies) or per observation (as implemented)? Is the per-observation form defensible as
an approximation, and should the other be reported as a robustness check?

### Q26. Which input features should the model use, and how are they prepared? [RESOLVED 2026-10-05]

**Status.** Decided by Tom on 2026-10-05, for the final model and not only the base case: the
selection rule, the scaling, the missing-value treatment, the industry treatment and the final list of
40 characteristics (57 inputs; part 6). The point-in-time fundamentals pipeline (part 5) is a proposal
for the implementation brief.

**The question.** Which characteristics go into the expert and base inputs, how many, and how are they
scaled, filled when missing, and combined with industry membership? The code currently has 14
price/volume snapshot features (`SNAPSHOT_FEATURES` in `features.py`), ranks them to [-0.5, 0.5],
and drops any row with a missing value (`_valid_rows`). Two of the 14 are exact duplicates after
ranking: `rel_ret_20d` (ret_20d minus the market's) and `rel_vol_20d` (vol_20d over the market's).
The market term is the same for every stock on a date, so their ranks equal those of `ret_20d` and
`vol_20d`.

#### 1. Feature families and the selection rule (decided)

**Decision.** There are three families:

- price/volume, from CRSP daily;
- fundamentals, from Compustat via the CRSP/Compustat link, point-in-time;
- industry membership, from GICS sectors (Compustat; ICB was the original choice, see part 4).

In the first two families, characteristics are chosen by one rule: **exactly two representatives for
each of the 13 themes of Jensen, Kelly & Pedersen (2023)**. The themes are Accruals, Debt Issuance,
Investment, Low Leverage, Low Risk, Momentum, Profit Growth, Profitability, Quality, Seasonality,
Short-Term Reversal, Size and Value. Theme membership follows JKP's own cluster file (`Cluster
Labels.csv` in bkelly-lab/ReplicationCrisis). That gives 26 characteristics. The industry block
(part 4) comes on top.

The two representatives of a theme are chosen by these criteria, in order:

1. robust in large caps, because the universe is the S&P 500;
2. relevant at the 5-day horizon (Q21);
3. a payoff known to depend on the market state, which gives the experts something to disagree about;
4. computable point-in-time from CRSP and Compustat;
5. different from each other, so the second adds information rather than repeating the first.

**Why a rule rather than a number.**

1. **The independent information in characteristics is low-dimensional.**
   - Green, Hand & Zhang (2017): 12 of 94 characteristics were independent determinants of returns
     in non-microcaps over 1980-2014, and only 2 since 2003.
   - Freyberger, Neuhierl & Weber (2020): about a dozen of 62 add information, and fewer among large
     stocks.
   - Hou, Xue & Zhang (2020): 65% of 452 anomalies fail once microcaps are controlled for.
   - JKP: 153 factors cluster into 13 themes.

   Adding features beyond the themes mostly adds near-copies: more parameters and noise, little
   information.
2. **Theme coverage** means no known source of cross-sectional return variation is left out.
3. **Exactly two, rather than one or two** (Tom's choice). One representative makes a theme hostage
   to a single definition. Two give a second measurement of the same economic idea, built a different
   way, which averages out the quirks of any one definition. The count is also fixed and pre-specified
   rather than tuned.
4. **No search.** The set is fixed before any result is seen. No feature selection is done on
   validation or test data, which protects the test set (Q15).

**Not decided here; to run later.** Two checks:

- A redundancy diagnostic on training data only: the average daily rank-correlation matrix,
  hierarchical clustering, and PCA eigenvalues.
- A pre-registered group ablation on validation: price/volume, then plus fundamentals, then plus
  industry.

PCA serves as a redundancy diagnostic, not as the selection rule, for three reasons:

- it never sees the target;
- it returns blends of features, not features;
- a direction that explains little variance can be the most predictive one. Short-term reversal is
  nearly uncorrelated with everything else, for example.

#### 2. Scaling: rank to [-1, 1] (decided)

On each date, each characteristic is ranked across the universe (ties get the average rank) and
mapped linearly:

$$
x_{i,t} = \frac{2\,(\mathrm{rank}_{i,t} - 1)}{N_t - 1} - 1 \in [-1, 1]
$$

**Why rank at all.**

- It is robust to outliers and to the heavy tails of accounting ratios. One firm with near-zero book
  equity cannot dominate.
- It removes the time variation in each characteristic's scale. The network sees the same uniform
  distribution every day, so what it learns in 2017 applies in 2022.
- The cost is that the size of differences between stocks is discarded and only their order kept.
  This is accepted, as in Gu, Kelly & Xiu (2020) and Freyberger, Neuhierl & Weber (2020).

**Why [-1, 1] rather than [-0.5, 0.5].**

- The information is identical: one range is twice the other, and the first-layer weights absorb the
  factor of 2.
- The input variance is 1/3 instead of 1/12. That is closer to the unit-variance inputs that
  standard weight initialisations (Xavier, He) assume.
- Under weight decay, the narrower range needs weights twice as large, penalised four times as
  heavily. That is an extra regularisation nobody chose.
- It is the Gu, Kelly & Xiu convention.

Implementation: `_cross_sectional_rank` currently maps to [-0.5, 0.5]. The 0/1 sector dummies are not
ranked.

#### 3. Missing values (decided)

**Decision.**

- (a) **Price/volume features** use rolling windows with a minimum count (80% of the window)
  instead of a full window.
- (b) **Fundamentals** carry the stock's own last point-in-time value forward until the next report,
  for at most 12 months after it became public.
- (c) **Anything still missing** after (a) and (b) is set to 0 after ranking, which is the
  cross-sectional median. One missing-value flag per family (price/volume, fundamental) is set to 1
  when any characteristic in that family was filled.
- (d) **Rows are no longer dropped** for a missing characteristic. They are still dropped when the
  target is missing.

**The theory behind it.**

1. **Why missingness matters.** Rubin (1976) separates three cases:
   - *missing completely at random*: unrelated to anything;
   - *missing at random*: related only to observed variables;
   - *missing not at random*: related to the missing value itself or to the outcome.

   Firm characteristics are not missing completely at random. Bryzgalova, Lerner, Lettau & Pelger
   (2025) show that over 70% of firms have gaps, that missingness is systematic, and that stocks with
   missing characteristics earn different returns.
2. **Why not drop the rows (complete-case analysis).** There are three reasons:
   - The model would be trained on an unrepresentative cross-section that systematically excludes
     young firms, recent index additions and financials. Financials are excluded because items such
     as cost of goods sold do not exist for them.
   - At forecast time every stock in the universe needs a forecast. A training rule that drops rows
     has no counterpart in deployment.
   - With 26 characteristics, the chance that at least one is missing compounds.
3. **Why the fill must be causal.** The forecast must be adapted to the information filtration
   (`MoE_HMM_Gate_Formulation.pdf`, Section 1.4): every input on day t must be computable from
   information available at the close of day t.
   - Interpolating a gap from values on both sides uses data from after t, even when weighted towards
     the trend. That is look-ahead, the same problem as smoothed regime probabilities.
   - Tom's first idea was a weekly per-stock median with a trend weight towards the later value. That
     is this kind of interpolation.
   - The causal alternative, extrapolating the recent trend forward, is a forecast of a number that
     has not been reported. It adds noise, and the market itself does not have that number.
4. **Why carrying forward is right for fundamentals.** It is hardly an imputation at all. Between two
   reports, the latest filing is exactly what the market knows about a firm's book equity or earnings.
   A step function that changes on report dates is the correct point-in-time representation of
   accounting information. It also keeps Tom's intuition that a fill should come from the stock's own
   history, not from the market.

   The 12-month cap exists because, after one missed annual cycle, the number no longer describes the
   firm (late filers, restructurings). Such firms are better identified by the flag than by a stale
   value.
5. **Why minimum-count windows for price features.** An average over 16 of 20 days estimates the same
   quantity with slightly more noise and negligible bias. The gain in coverage is large: new listings,
   trading halts, and stocks just added to the index.
6. **Why 0 plus a flag.**
   - In rank space, 0 is the median. It is the neutral value: no information, so assume a typical
     stock.
   - A feature enters the first network layer as weight times value, so at 0 a filled feature
     contributes nothing in either direction.
   - The flag lets the model learn a separate shift for stocks with missing data. This is how
     missingness that is itself informative enters the model: if those stocks earn different returns,
     the flag captures it instead of the fill distorting the feature.
   - Simple is enough for machine-learning portfolios. Chen & McCoy (2024) find that cross-sectional
     mean imputation performs about as well as EM. The reason is that missingness comes in large
     blocks (by time and by data source) and cross-sectional correlations are small, so the observed
     characteristics say little about the missing ones.
7. **Upgrade path, not adopted now.** Bryzgalova et al.'s causal B-XS method, which uses the stock's
   own past plus the cross-section through latent factors, can serve as a robustness check.
8. **Zero-if-missing items (decided 2026-10-05, after the first coverage report).** In total accruals
   (`taccruals_at`) and net operating assets (`noa_at`), IVAOQ, IVSTQ, MIBQ and PSTKQ count as 0 when
   blank. Compustat leaves them blank when a firm has none; book equity already treats TXDITCQ and
   PSTKQ this way, and it is the usual treatment in Richardson et al. (2005) and JKP. Core items (ATQ,
   ACTQ, LCTQ, LTQ, DLTTQ, ...) stay strictly missing, so a definition that does not apply (a bank's
   current assets) is still missing and flagged. Under the strict rule `taccruals_at` was present on
   5% of rows and `flag_fund_missing` was set on 95%; with this rule they are 79% and 30% (CRSP S&P 500
   panel 2015-2024, aggregate coverage only). Code: `FeatureSpec.zero_if_missing_items`.

#### 4. Industry (decided)

**Source: GICS from Compustat (changed from ICB on 2026-10-05, Tom's decision).** CRSP's ICB field
(`ICBIndustry` in `StkSecurityInfoHist` / `StkIssuerInfoHist`) is filled for S&P 500 members only up
to September 2023:

- from October 2023 every member is "NOAVAIL", and CRSP switches to a different scheme
  (`UESIndustry`, with different sector names);
- Compustat's dated GICS history (`gicshistory.dat` in the CRSP/Compustat Merged release, fields
  `INDFROM`, `INDTHRU`, `GSECTORH`) covers the whole sample point-in-time;
- GICS is the classification the S&P 500 itself uses.

The original ICB rationale below applies unchanged to GICS's 11 sectors.

*Original text:* ICB industry from the CRSP issuer history (`StkIssuerInfoHist`). It is dated, so it is
point-in-time. There are 11 sectors: Basic Materials, Consumer Discretionary, Consumer Staples, Energy,
Financials, Health Care, Industrials, Real Estate, Technology, Telecommunications and Utilities. Stocks
coded "not available" get all dummies equal to zero.

**Three uses, all adopted.**

1. **Sector dummies** (11 inputs, 0/1).
   - The base learns each sector's average market-neutral return.
   - The experts learn regime-dependent sector rotation, for example defensives beating the market in
     stress and technology leading in calm markets. This is the kind of switching the mixture is
     designed for.
   - The 11 sectors average about 45 stocks each. Gu, Kelly & Xiu's 74 two-digit SIC groups would
     leave about 7 stocks per group in the S&P 500, which is too noisy.
2. **Industry-relative characteristics** for a subset of the fundamentals: the characteristic is
   ranked within its sector instead of across the whole universe.
   - Accounting ratios differ structurally across industries (banks' book-to-market, software's thin
     book), so a universe-wide rank partly just sorts industries. A within-sector rank compares a firm
     with its peers (Asness, Porter & Stevens 2000).
   - Doubling every feature is not adopted. The subset is decided with the list; the proposal is the
     two Value and the two Profitability representatives.
3. **Industry return signals.**
   - Industry momentum: the sector's past return (Moskowitz & Grinblatt 1999).
   - Within-industry reversal: the stock's past return minus its sector's (Da, Liu & Schaumburg 2014;
     Hameed & Mian 2015).

**Resulting size.** See part 6: 57 inputs once the short-horizon market block is included.

#### 5. Point-in-time fundamentals (proposed; for the implementation brief)

**Data check (aggregate only).** The CRSP/Compustat Merged files are in `Data/`:

- the link history (gvkey to PERMNO with validity dates, link type and primary flag);
- quarterly income statement, balance sheet and year-to-date cash flow;
- the quarterly report date (RDQ), present for 99.9% of the 30,225 firm-quarters of S&P 500 members
  in fiscal 2013-2024;
- the 10-Q/10-K filing dates.

**Proposed rules.**

- A quarter's numbers become usable one trading day after the later of its report date and its
  filing date. The earnings and revenue surprises are the exception: they use the report date, since
  the announcement is the event.
- Flows are trailing four-quarter sums, with year-to-date cash flow differenced into quarters.
- Market values come from CRSP on day t.

**History needed.**

- Fundamentals from about 2013, since growth rates need a year of history.
- Some price features need up to five years, so CRSP history from about 2010 is needed for the
  features only. The training sample is still 2015-2024 unless the sample extension changes it.

**Known limitation.** Standard Compustat values can be restated after first release. The file contains
keysets for data "prior to company amendment", which may partly fix this; to be checked.

#### 6. The final list (decided 2026-10-05)

**Why a short-horizon market block was added.** Tom's point: 8 of the 13 JKP themes are accounting-based,
because they come from the monthly anomaly literature. At a 5-day horizon that balance is backwards.
Fundamentals barely change from week to week, while the signals known to predict weekly returns are
market-based, and price trends, liquidity and volatility dominate Gu, Kelly & Xiu's feature importance.
So the same rule, two representatives per theme, is applied to six further short-horizon market
themes, each backed by evidence at a daily or weekly horizon. The block was cross-checked against
Tom's Trexquant research (`Trexquant/`); see the end of this part.

**A. JKP block: 26 characteristics (13 themes × 2).**

| Theme | Representative 1 | Representative 2 | Data |
|---|---|---|---|
| Short-Term Reversal | 5-day return | 20-day return | CRSP |
| Momentum | 12-month return skipping the last month | price / 52-week high (George & Hwang 2004) | CRSP |
| Low Risk | 21-day volatility | beta, Frazzini-Pedersen construction (1-year volatility, 5-year correlation) | CRSP |
| Size | log market cap | Amihud illiquidity, 126 days | CRSP |
| Seasonality | same-calendar-month return, years 2-5 back (Heston & Sadka 2008) | coskewness with the market, 21 days (in this cluster in JKP's file) | CRSP |
| Value | book-to-market | earnings-to-price | Compustat + CRSP |
| Profitability | quarterly ROE | operating cash flow / assets | Compustat |
| Quality | gross profitability / assets (Novy-Marx 2013) | earnings consistency (consecutive quarterly earnings increases, up to 8) | Compustat |
| Investment | asset growth (Cooper, Gulen & Schill 2008) | sales growth | Compustat |
| Accruals | operating accruals (Sloan 1996) | total accruals (Richardson et al. 2005) | Compustat |
| Debt Issuance | 3-year debt growth | net operating assets / assets (Hirshleifer et al. 2004) | Compustat |
| Low Leverage | net debt / market cap | cash / assets | Compustat + CRSP |
| Profit Growth | standardised unexpected earnings (SUE) | revenue surprise (Jegadeesh & Livnat 2006) | Compustat, dated by report date |

**B. Short-horizon market block: 12 characteristics (6 themes × 2).**

| Theme | Representative 1 | Representative 2 |
|---|---|---|
| Volatility dynamics | idiosyncratic volatility, 21 days (market-model residual; Ang, Hodrick, Xing & Zhang 2006) | volatility shock: 5-day volatility / 60-day volatility |
| Tails and asymmetry | realized skewness, 21 days (Amaya et al. 2015) | maximum daily return, 21 days (Bali, Cakici & Whitelaw 2011) |
| Volume and liquidity | abnormal volume: today's volume against its trailing average (Gervais, Kaniel & Mingelgrin 2001) | quoted spread, (ask - bid) / midpoint at the close, 21-day average |
| Price path | overnight (close-to-open) return, 20 days (Lou, Polk & Skouras 2019) | price / 50-day moving average (Han, Zhou & Zhu 2016) |
| Earnings timing | announcement expected within the next 5 trading days (0/1, estimated causally from the report date four quarters earlier) | trading days since the last announcement (Barber et al. 2013) |
| Return dynamics | variance ratio: 5-day against 1-day return variance, 60 days (Lo & MacKinlay 1988) | correlation of daily return and volume, 60 days (Llorente, Michaely, Saar & Wang 2002) |

**C. Industry block: 17 inputs.**

- 11 GICS sector dummies (Compustat GICS history; ICB dropped, see part 4).
- 2 industry return signals:
  - industry momentum, the sector's 12-month return skipping the last month (Moskowitz & Grinblatt 1999);
  - within-industry reversal, the stock's 20-day return minus its sector's.
- 4 within-sector ranks, for book-to-market, earnings-to-price, quarterly ROE, and operating cash flow / assets.

**D. Missing flags: 2** (price/volume, fundamental).

**Total: 40 characteristics (22 market-based, 18 fundamental-based) and 57 inputs.**

**Data consequences.**

- The current CRSP extract keeps only price, return, market cap and volume. The open, high, low, bid
  and ask are in the raw daily file (`StkDlySecurityData`), with 100% coverage on S&P 500
  stock-days 2015-2024, so a re-extract is needed. The number of trades covers only 28% of those
  stock-days (Nasdaq only) and is not used.
- Shares outstanding for turnover-type measures are in `StkShares`.
- History needed: about 2010 onward for the 5-year windows, and fundamentals from about 2013.

**Cross-check against the Trexquant work.**

- *Not transferable:*
  - first-hour and last-hour bar features, because there are no intraday data;
  - calendar, FOMC, VIX and macro series, which are identical across stocks and so carry nothing after
    ranking; they could only enter through the gate;
  - trade size, because of the trade-count coverage.
- *Confirmed by Tom's own results:* volatility term structure, the overnight/intraday split, days to
  earnings, and volume relative to its own average.
- *Added from it:* the return-dynamics theme (proposed in `36_STOCK_CHARACTERISTICS.py`, not tested
  there).
- *Skipped as near-copies:*
  - position of the close in its 20-day range;
  - daily high/low ratio;
  - upside/downside volatility ratio;
  - announcement-return persistence (about 4 observations a year).
- *Lesson adopted:* express levels as surprises relative to the stock's own history where the level
  would otherwise act as a static tilt (volume, volatility).

*Sources:* Jensen, Kelly & Pedersen (2023), JF 78(5); Green, Hand & Zhang (2017), RFS 30(12);
Freyberger, Neuhierl & Weber (2020), RFS 33(5); Hou, Xue & Zhang (2020), RFS 33(5); Gu, Kelly & Xiu
(2020), RFS 33(5); Rubin (1976), Biometrika 63(3); Bryzgalova, Lerner, Lettau & Pelger (2025), RFS
38(3); Chen & McCoy (2024), JFE 155; Asness, Porter & Stevens (2000), working paper; Moskowitz &
Grinblatt (1999), JF 54(4); Da, Liu & Schaumburg (2014), MS 60(3); Hameed & Mian (2015), JFQA 50(1-2); George & Hwang (2004), JF 59(5); Heston & Sadka (2008), JFE 87(2);
Frazzini & Pedersen (2014), JFE 111(1); Novy-Marx (2013), JFE 108(1); Cooper, Gulen & Schill (2008), JF
63(4); Sloan (1996), TAR 71(3); Richardson, Sloan, Soliman & Tuna (2005), JAE 39(3); Hirshleifer, Hou,
Teoh & Zhang (2004), JAE 38; Jegadeesh & Livnat (2006), JAE 41(1-2); Ang, Hodrick, Xing & Zhang (2006),
JF 61(1); Amaya, Christoffersen, Jacobs & Vasquez (2015), JFE 118(1); Bali, Cakici & Whitelaw (2011), JFE
99(2); Gervais, Kaniel & Mingelgrin (2001), JF 56(3); Lou, Polk & Skouras (2019), JFE 134(1); Han, Zhou &
Zhu (2016), JFE 122(2); Barber, De George, Lehavy & Trueman (2013), JFE 108(1); Lo & MacKinlay (1988),
RFS 1(1); Llorente, Michaely, Saar & Wang (2002), RFS 15(4).

### Q21. At what frequency should the gate and the cross-section run, and what is the target horizon? [RESOLVED 2026-10-05]

**Answer: option B. A daily gate, a daily cross-section, and a 5-day target horizon ($h=5$).** This is
Tom's decision, taken 2026-10-05, after a review of the literature and a descriptive check on the
CRSP panel. It confirms what the code already does (`StageBSpec.horizon = 5`, daily panel), so no code
change follows from it.

**What it means.** Every trading day $t$ the Hamilton gate is updated with that day's market return,
and every stock in the S&P 500 universe is ranked on its characteristics $x_{i,t}$. The target is the
market-neutral return (Q25) over the next five trading days, $(t, t+5]$. Consecutive targets overlap by
four days.

**The options compared.** (A) daily gate, daily cross-section, $h=1$; (B) daily gate, daily
cross-section, $h=5$ (chosen); (C) daily gate, monthly cross-section, $h\approx21$; (D) monthly gate and
monthly cross-section.

**Why option B, in depth.**

1. **Statistical power on the sample we have.** The panel covers 2015-2024, 2,516 trading days. A
   monthly horizon leaves about 118 non-overlapping periods, and only about 59 of them in high-volatility
   conditions. That is far too few to train regime-specific experts or to detect regime differences:
   in the descriptive check below, a 5% rank IC at $h=21$ was not significant. At $h=5$ there are about
   500 non-overlapping weeks (about 250 in high volatility); at $h=1$ about 2,500 days.
2. **The regime contrast is clearest at $h=5$ while the statistics stay usable.** A descriptive check on
   the market-neutral target (CRSP, S&P 500 members 2015-2024, daily cross-sectional rank IC of three
   textbook signals, non-overlapping periods, days split at the median of trailing 20-day market
   volatility as a crude stand-in for regimes; full-sample signal diagnostics, not model results):
   one-week reversal had IC 2.27% (t = 2.73) at $h=5$, rising to 3.26% in high-volatility periods
   against 1.29% in low; one-month reversal 2.58% against -0.11%; at $h=21$ momentum changed sign
   (-2.22% against +1.78%), consistent with momentum crashes. The cross-section changes with the
   market's state, which is what the gate is meant to exploit, and at $h=5$ that change is both large
   and measurable.
3. **Less noise than next-day returns.** A one-day target is the noisiest choice: each period carries the
   least signal, it is contaminated by bid-ask bounce (Roll 1984), and daily machine-learning strategies
   on the S&P 500 have largely decayed since 2010 (Fischer & Krauss 2018; Krauss, Do & Huck 2017).
   Five days averages much of the microstructure noise out while keeping the short-horizon effects.
4. **Short-horizon effects are where regime dependence is strongest.** Short-term reversal returns are
   compensation for liquidity provision and rise sharply in turbulent markets (Nagel 2012). A weekly
   horizon keeps the study in the range where the cross-section is known to depend on the market state.
5. **It respects the regime timescale.** Under a daily Hamilton gate, regimes last weeks to months
   (expected durations of roughly 20-50 days for typical transition matrices). A 5-day window is short
   against that, so the base-case approximation "one regime per target window" is reasonable. A monthly
   window is comparable to the length of a stressed regime and would break it.
6. **The gate is best identified on daily data.** The Hamilton filter estimates its transition matrix and
   regime volatilities from about 2,500 daily observations, against about 120 monthly ones, and reacts
   to volatility clustering within days rather than weeks.
7. **It is already built and audited.** Overlapping targets are handled by Hansen-Hodrick t-statistics at
   lag $h-1$ (audit E-2), the purge between training and test equals $h$ (O-3), and the long-short book
   is horizon-consistent (E-3).
8. **It fits the market-neutral target.** At $h=5$ market demeaning removes about 28% of the target's
   variance (Q25), so the target is substantially cleaner without being reduced to next-day noise.

**Costs accepted deliberately.**

- **No direct comparability with Gu, Kelly & Xiu**, whose benchmark is monthly. Stated as a design choice
  in the methodology.
- **Overlapping targets.** Daily sampling of a 5-day target means consecutive observations share four
  days. Statistics are overlap-corrected, but the training likelihood counts overlapping dates as if
  independent, so it is used for estimation, not for inference.
- **Turnover.** A weekly holding period trades far more than a monthly one, so portfolio results are only
  meaningful net of transaction costs (Q9).
- **Compute.** About 1.27 million stock-days, roughly 21 times a monthly panel over the same years.
- **Slow characteristics** (fundamentals updated quarterly) barely change from one day to the next; most
  daily variation in the inputs comes from price-based features.

**What this does not settle.** Whether to extend the sample before 2015 (the loader's `start` is a config
field; the CRSP files appear to hold the full history), which would also make a monthly comparability arm
feasible; the long-short book's default (`portfolio_scheme`); whether the gate weight for a 5-day window
should be the one-step predicted probability, the filtered probability, or an average over the window
(the open timing item in `MoE_HMM_Gate_Formulation.pdf`, Section 8.9); and whether to add a non-overlapping
(every fifth day) robustness check.

*Sources:* Fischer & Krauss (2018), EJOR 270(2); Krauss, Do & Huck (2017), EJOR; Blitz, Hanauer,
Hoogteijling & Howard (2023), JFDS 5(4); Nagel (2012), RFS 25(7); Daniel & Moskowitz (2016), JFE 122(2);
Roll (1984), JF 39(4); Gu, Kelly & Xiu (2020), RFS 33(5).

**Follow-up (2026-10-08): horizon robustness.** h = 5 stays primary. Two additions, neither a change of
design: (1) an **IC decay curve**, the daily cross-sectional rank IC of the main signals against the
market-neutral forward return at h = 1, 2, 3, 5, 10, on 2000-2006 only (before the pilot's validation
years and the test period), to show how fast the signals' predictive power fades; (2) a **horizon profile**
(h = 1, 3, 5, 10) for the final best model only, reported as robustness (also Improvements E1). A 3-day
horizon was discussed: fresher signals and less overlap, but more noise, about 5/3 the turnover, and
windows that sometimes contain a weekend and sometimes not; no evidence it beats 5.

**Result of the IC decay curve (run by Tom 2026-10-09; commit 00b9604; `results/diagnostics/ic_decay_2000_2006.*`).**
Daily cross-sectional rank IC against the market-neutral forward return at h = 1, 2, 3, 5, 10, on
2000-2006 only (the last scored date moves back by h, so no 2007 return was read), Hansen-Hodrick
t-statistics at lag h - 1. **It supports h = 5; nothing points to 3 beating 5.**

- Yardstick: a signal whose information lasts the whole window has an IC growing like the square root
  of h with a roughly flat t-statistic; next-day-only information has an IC shrinking like one over the
  square root of h and a falling t-statistic.
- **One-week reversal** (`ret_5d`, the strongest signal): IC -1.67% at h = 1, -2.46% at h = 5 (t = -4.21),
  -2.08% at h = 10. It builds to h = 5 and adds nothing after.
- **Abnormal volume** (`volume_z`, the most robust): +0.86%, +1.07% (t = 4.05), +0.75%. Peaks at h = 5.
- **Fast signals** lose IC and t by h = 5 (most of their information sits in the first 2-3 days):
  idiosyncratic volatility, maximum daily return, momentum (peaks at h = 3, t = 1.43 at h = 5), the
  52-week high (gone by h = 5).
- **Slow signals** keep building to h = 10 and a one-day target would mostly miss them: distance from
  the 50-day average, bid-ask spread, the earnings-in-window dummy (t = 3.25 at h = 5), one-month
  reversal, return-volume correlation.
- No signal at any horizon (|t| under 1.3): volatility shock, realised skewness, overnight return,
  variance ratio.
- **Caveats.** 80 tests (16 signals x 5 horizons): under Bonferroni (|t| > 3.4) only `ret_5d` and
  `volume_z` (h <= 5) and `earn_next5` (h = 10) clear the bar. Regimes are pooled: 2000-2006 mixes the
  2000-02 bear market with the calm of 2003-06, so a signal with opposite signs in calm and stress
  (momentum around crashes) averages to near zero.
- **Decision (Tom, 2026-10-09): h = 5 confirmed, the horizon question is closed. The Q26 inputs are left
  unchanged**: the feature list was fixed by a rule (two per theme) so that it is not chosen on outcomes,
  and pooled-regime ICs near zero are not evidence against a signal whose sign depends on the regime.
- **Follow-up:** the same curve split by the gate's regime on 2000-2006 (brief 11).

*Original question, for the record:*

### Q21. At what frequency should the gate and the cross-section run, and what is the target horizon?

**Status:** open, raised 2026-09-26. Tom's current preference is a **daily gate and a daily
cross-section**, but it is kept open until it has been discussed. Nothing in the code commits to an
answer: the CRSP loader (brief 06) builds the daily panel the pipeline already uses, and the horizon
stays a config field.

**Three questions that have to be answered together.**

1. **Gate frequency.** How often the regime probabilities are updated. A daily Hamilton filter sees
   about 21 times as many observations as a monthly one, so its transition matrix is better
   identified and it reacts to volatility clustering within days rather than weeks.
2. **Cross-section frequency.** How often the stocks are ranked against each other, which is also how
   often the portfolio is rebalanced. The options are (a) daily gate with a daily cross-section,
   (b) daily gate with a monthly cross-section, where the gate probability on the last trading day of
   each month is the input to that month's cross-section, and (c) monthly for both.
3. **Target horizon `h`.** The forward return being predicted: next day (`h = 1`), next week
   (`h = 5`, what the code uses now as `fwd_ret_5d`), or next month (`h` of about 21). Any `h > 1` at
   daily frequency creates overlapping targets, which the code now handles with a Hansen-Hodrick
   long-run variance at lag `h - 1` (audit finding E-2), and sets the purge gap to `h` days (see Q22).

**Considerations.**

- **Predictability is weaker and noisier at daily horizons.** Next-day cross-sectional returns are
  dominated by short-term reversal and by bid-ask bounce: a trade at the bid followed by one at the
  ask looks like a return even when the price has not moved. Gu, Kelly and Xiu, the benchmark, use
  monthly data partly for this reason. A daily cross-section is a legitimate choice but needs one
  sentence of justification and a comparison that nets out transaction costs.
- **Turnover.** Daily rebalancing multiplies turnover, so the portfolio metrics in Q9 are only
  meaningful after costs.
- **Comparability.** A monthly cross-section is directly comparable with Gu, Kelly and Xiu and with
  most of the asset-pricing literature; a daily one is not.
- **Compute.** A daily cross-section has about 21 times the rows of a monthly one over the same
  window, which feeds straight into the compute budget the advisor asked for.
- **Fundamentals** (Compustat) update quarterly, so they are equally stale at either frequency and do
  not bear on the choice.

**Reading, about an hour.**
- Roll, R. (1984), "A simple implicit measure of the effective bid-ask spread in an efficient
  market", *Journal of Finance* 39(4). Why bid-ask bounce creates spurious negative autocorrelation in
  short-horizon returns.
- Jegadeesh, N. (1990), "Evidence of predictable behavior of security returns", *Journal of Finance*
  45(3), and Lehmann, B. (1990), "Fads, martingales, and market efficiency", *Quarterly Journal of
  Economics* 105(1). Short-term reversal at monthly and weekly horizons.
- Gu, S., Kelly, B. and Xiu, D. (2020), *Review of Financial Studies* 33(5), the data section, for
  why the benchmark is monthly.

**Ask the advisor:**
(a) Which pairing: daily/daily, daily gate with a monthly cross-section, or monthly/monthly?
(b) Which target horizon `h`?
(c) If the cross-section is daily, is comparability with Gu, Kelly and Xiu still required, for
    example through a monthly robustness arm?


### Q25. What target variable should the model be trained on? [RESOLVED 2026-10-01]

**Answer: the market-neutral return, defined as the cross-sectionally demeaned forward return.**
This is Tom's decision, taken 2026-10-01, after a literature review of the alternatives.

**Definition.** For stock $i$ in the universe $\mathcal S_t$ on formation date $t$, with forward return
$r_{i,t\to t+h}$ over $(t,t+h]$ (the same return the code already builds, delisting returns and the
post-delisting fill included):

$$y_{i,t} \;=\; r_{i,t\to t+h} \;-\; \frac{1}{N_t}\sum_{j\in\mathcal S_t} r_{j,t\to t+h}$$

where the average runs over the stocks of date $t$ that have a valid target (equal weights). Three
properties follow from the definition. The risk-free rate cancels, so raw and excess returns give the
same target. The target sums to zero across stocks on every date. The demeaning uses the realised
cross-section at $t+h$, which is legitimate because it transforms only the label, never a feature, and
the model's forecast at $t$ is a forecast of this relative return.

**Not to be confused with the code's existing `target_kind="residual"`.** That option subtracts
$\beta_{i,t}$ times the market's forward return, with a trailing 250-day beta (`features.py`), and
its docstring calls it "market-neutralized". It is a beta-residual (CAPM-style) target, which is
option 5 below and is *not* the decision. The decided target is a third kind that does not yet exist
in the code (implementation pending; the default stays `"raw"` until a brief changes it).

**The alternatives considered.** (1) Raw or excess return; (2) market-neutral, cross-sectionally
demeaned return (chosen); (3) cross-sectionally standardised or percentile-ranked return; (4)
volatility-scaled return (return divided by the stock's ex-ante volatility, "risk-adjusted" in the
Sharpe sense); (5) factor-residual or abnormal return (CAPM or Fama-French, "risk-adjusted" in the
alpha sense).

**Why the market-neutral return, in depth.**

1. **It is the quantity the thesis question is about.** The question is whether the market regime
   changes *which stocks outperform which*. The day's cross-sectional mean is the same number for
   every stock: it cannot be ranked, it carries no cross-sectional information, and predicting it is
   a market-timing problem, not a cross-sectional one. Removing it aligns what the experts are
   trained on with what the thesis claims to measure.
2. **It keeps the gate's information and the target's information apart, which is what makes the
   regime effect attributable.** The gate is fitted on the market series. A raw target contains,
   as its largest daily component, essentially that same market move. A regime that predicts the
   market's level would then appear to "help the experts" through the common move alone, and the
   comparison across gates would partly measure market timing. With the common move removed, any
   gain the gate brings has to come from regime-dependent *cross-sectional* structure. This is the
   same attribution logic that motivated freezing the gate (Q19).
3. **It makes the model's own assumptions defensible.** The derivation (`MoE_HMM_Gate_Formulation.pdf`,
   Section 6, assumption A3) assumes stock returns are independent given the regime and their
   characteristics, and independent of the market return. For raw daily returns this is badly
   false: on a $-4\%$ day almost every stock falls by more than any regime-conditional mean explains.
   The per-date likelihood (Q24) would then treat ~500 strongly correlated returns as independent
   evidence, and its responsibilities would mostly classify days by tomorrow's market move. Demeaning
   removes the dominant common factor, so A3 becomes a reasonable approximation; only residual
   dependence such as industry co-movement remains.
4. **The headline metrics are unchanged by it, so training and evaluation measure the same thing.**
   Rank IC, decile sorts and equal-weighted long-short returns are all invariant to subtracting a
   number common to every stock on a date. Training on the demeaned return therefore optimises
   exactly the variation those metrics score, instead of spending capacity on variation they ignore.
5. **It keeps the regime's dispersion information, unlike standardising or ranking.** Cross-sectional
   dispersion rises in stressed regimes. Dividing each date by its dispersion (option 3) would erase
   that, collapse the experts' noise levels $s_k$ towards a common value, and empty the variance-gain
   part of the NLL decomposition (`NLL_single` vs `NLL_full`). Demeaning removes the level only.
6. **It is compatible with the Gaussian mixture likelihood, unlike ranks.** Percentile ranks are
   bounded and uniform; a Gaussian expert is the wrong model for them, and a ranking target belongs
   with a ranking loss (Q20), not with the likelihood derived for the base case.
7. **It needs no estimated inputs, unlike volatility scaling or factor residuals.** Option 4 requires an
   ex-ante volatility estimate (estimation error, and a look-ahead risk if built carelessly) and
   down-weights high-volatility stocks and periods, which is where machine-learning signal is known
   to concentrate (Avramov, Cheng & Metzker 2023). Option 5 requires rolling betas, which are noisy
   and lagged.
8. **It does not strip out regime-dependent factor premia, unlike factor residuals.** Momentum
   crashes in rebounds after bear markets (Daniel & Moskowitz 2016) and value's cyclical swings are
   exactly the regime-dependent cross-sectional effects the experts should be able to learn. A
   Fama-French residual target would remove them by construction. Wang (2024) also finds no
   predictive gain from abnormal-return targets when only firm characteristics are used.
9. **It is simple, deterministic and reproducible**: one subtraction per date, no parameters, no
   warm-up period that drops early rows (unlike the 250-day beta window of the residual option).
10. **It has clear precedent.** A cross-sectional regression with an intercept (Fama-MacBeth) is
    equivalent to regressing on demeaned returns; production pipelines (Qlib, MASTER) normalise labels
    cross-sectionally by default; and Azevedo, Kaiser & Mueller (2023) find that removing the level
    and noise of raw returns from the training target improves long-short performance substantially.

**Costs accepted deliberately.**

- **Gu, Kelly & Xiu's 0.40% out-of-sample $R^2$ is not directly comparable**, since theirs is on excess
  returns. Our $R^2$ measures the explained share of *relative* returns and must be reported as such.
- **Heavy tails remain.** Demeaning removes the common shock, not the stock-specific outliers. How to
  handle them belongs to the loss function (Q20), which stays open.
- **Equal weights.** The demeaning gives every stock in the S&P 500 universe the same weight, which
  matches the equal-weighted long-short evaluation. A value-weighted version would differ slightly.
- **A technical violation of independence that is negligible here:** demeaned returns sum to zero on
  every date, which induces a correlation of about $-1/(N_t-1)\approx-0.002$ between any two stocks.
- **Residual cross-sectional correlation (industries) remains**, so the per-date posterior can still
  be overconfident (Q24).
- **Market timing is out of scope**: the model no longer forecasts the market's level at all.

**Clarifications added 2026-10-05.**

- **Market-neutral throughout, including reporting.** Tom's decision: results are reported in
  market-neutral returns, not raw returns. Demeaning preserves every difference between stocks and
  between industries exactly (the subtracted mean cancels in any difference), so "which stocks and
  which industries performed best" remains fully visible; only the level common to all stocks on a
  day is gone.
- **Measured noise reduction** (CRSP, S&P 500 members 2015-2024, ~504 stocks per day, daily log
  returns, aggregate only): demeaning removes **32%** of the variance of daily returns (28% at a
  5-day horizon), from 10-16% in calm years (2017, 2024) to 38% in 2022 and **50% in 2020**. The
  removed part is one shock shared by every stock, so it does not average out across the cross-section
  and is concentrated in the stressed periods the regimes are about.
- **Implicit assumption: unit beta.** Subtracting the equal-weighted mean removes the common move with
  the same loading for every stock. Where betas differ, $(\beta_i-\bar\beta)\,m$ remains in the target,
  so A3(b) holds only approximately and a regime-dependent beta tilt (e.g. low-beta stocks favoured in
  stressed regimes) can still be learned. That is a genuine cross-sectional effect; it should be
  measured (correlate each expert's predictions with stock betas, per regime), not removed.

**What this does not settle.** The horizon $h$ and frequency (Q21); log versus simple returns (the code
currently sums daily log returns); whether to run a volatility-scaled version as a robustness check; and
the loss function (Q20).

**Implementation (pending, not yet briefed).** A new target kind in `StageBSpec` that subtracts the
date's equal-weighted cross-sectional mean of the forward return over the stocks with a valid target;
the base $f_0$ and the experts are retrained on it; all metrics continue to be computed as now.

*Sources:* Gu, Kelly & Xiu (2020), RFS 33(5); Azevedo, Kaiser & Mueller (2023), J. Asset Management
24(5); Poh, Lim, Zohren & Roberts (2021), JFDS 3(2); Lim, Zohren & Roberts (2019); Wang (2024),
Financial Innovation 10; Blitz, Huij & Martens (2011), J. Empirical Finance 18(3); Avramov, Cheng &
Metzker (2023), Management Science 69(5); Daniel & Moskowitz (2016), JFE 122(2); Qlib documentation;
MASTER repository (SJTU).


### Q3. Data access, CRSP and Compustat through the university? [RESOLVED 2026-09-19]
**Answer: CRSP.** The thesis runs on US equities. Wind remains available and could support a China
extension, but is not the primary source. The feasibility section of the proposal can now be written
against CRSP, and the benchmark literature stays Gu, Kelly and Xiu rather than the Chinese
cross-sectional ML work. Access paperwork has the longest lead time of anything on the plan, so it
should be started immediately, in parallel with the theory reading rather than after it.

*Original question, for the record:*

### Q3. Data access — CRSP and Compustat through the university?
**Status:** RESOLVED 2026-09-19, see the answer above.

Section 3 of the proposal form requires a feasibility analysis, and that rests on the data plan.
Need to confirm whether SJTU's library subscription covers CRSP and Compustat, and what the access
process is. Fallback is free daily prices plus the Ken French data library, which changes the
characteristic set and therefore what the results are comparable with.

---

---

### Q11. Should the study report a full mechanism-by-depth grid? [RESOLVED 2026-09-19]
**Answer: yes, run the grid.** The advisor said it is likely feasible and told me to do it, so the
mechanism-by-depth table stands as the central reporting design. Two qualifications from the meeting.
The **number of depths** the grid carries is set by the compute budget once that is actually calculated,
not chosen in advance, which answers part (c) of the original question by deferring it to arithmetic.
And the grid covers depth only: **no width sweep** (see Q8) and **no sweep over the expert count**
(see Q18).

**Cost arithmetic supporting feasibility.** The models are small, roughly 19,000 parameters for base
plus three experts plus gate at 100 characteristics, and depth barely moves that figure because the
first hidden layer carries about 83 per cent of it. Four gates by four depths by thirty windows by ten
seeds is about 4,800 expert-stage runs. Wall clock is dominated by per-run overhead rather than
arithmetic, so budget 20 to 30 seconds a run and expect roughly 30 hours, or about four days at thirty
seeds. Two structural savings apply: the base never sees regime information, so it is trained once per
depth, window and seed and reused across all four gates; and each gate is fitted once per window,
independent of expert depth. Load the data once and loop inside a single process, since launching
thousands of separate processes would cost more than every matrix multiply in the thesis. Time the
Hamilton filter and the jump-model dynamic program early, because those are optimisations rather than
matrix multiplies and are the most likely source of an unpleasant surprise.

*Original question, for the record:*

### Q11. Should the study report a full mechanism-by-depth grid rather than fixing a depth?
**Status:** RESOLVED 2026-09-19, see the answer above.

**Tom's proposal (clarified 2026-09-12).** Run every gate at every candidate expert depth and
**report the complete grid** — mechanisms down one axis, hidden-layer counts across the other, in the
manner of Gu, Kelly and Xiu's NN1–NN5 ladder but with gates as a second dimension. This is a
*reporting* design, not a *selection* design: nothing is tuned away, and the table itself is the
result. His reasoning is that if the optimal depth is gate-dependent — and Q8 gives a mechanism for why
it would be — then fixing one depth handicaps whichever gates it does not suit, and a comparison that
handicaps some of its arms is not a fair test of the mechanisms. He notes the compute cost explicitly
as the objection to his own proposal.

**Assessment: this is the better design, and it is cleaner than either alternative I had proposed.**
Three reasons.

*It removes the selection problem entirely rather than managing it.* Reporting a full grid selects
nothing, so there is no cherry-picking to defend against and no need for a nested validation
procedure. The table is descriptive, exactly as the NN1–NN5 row is descriptive in the paper it
imitates, and no one accuses that paper of tuning.

*It makes the robustness claim checkable by the reader.* "The ordering of gates is unchanged across
every depth" is a far stronger statement than "each gate was tuned and mechanism X won," and a reader
can verify it from the table rather than trusting a selection procedure they cannot see.

*It turns a nuisance parameter into a result.* The interaction between regime mechanism and expert
capacity is itself unexamined in the literature. A design that optimises depth away answers the
question by discarding it; a grid answers it by displaying it.

**It also subsumes both designs previously discussed.** A fixed-depth comparison is one column of the
grid; a per-gate-best comparison is the row maxima. Both are readable from the same runs at no extra
cost, which means the grid is not a third option but the superset of the other two.

**The one discipline it still requires.** Reporting every cell is honest; taking the maximum over
cells and calling it the headline result is not, because the maximum of several noisy estimates is
biased upward. So **one cell must be nominated in advance as the primary configuration for
significance testing**, with the rest of the grid reported as descriptive evidence around it. That
costs nothing and closes the only opening a referee has.

**On feasibility, which is the real objection.** The grid at full seed count is roughly six gates by
three depths by thirty windows by thirty seeds, on the order of sixteen thousand training runs,
against fifty-four hundred for a fixed-depth design. That is a genuine threefold increase and Tom is
right to raise it. Three things reduce it without compromising the design.
- **Restrict the depth axis to one, two and three hidden layers** rather than the five of the ladder
  being imitated. Gu, Kelly and Xiu and Lai agree that four and five do not help, so the omission is
  justified by citation rather than by convenience, and it cuts the grid by forty per cent against a
  five-deep version.
- **Report the grid at a reduced seed count and the nominated primary cell at the full count.** Ten
  seeds is ample for a descriptive table whose purpose is to show whether an ordering is stable;
  thirty is needed only where paired significance tests are run. This roughly halves the total.
- **Screen the gate axis first.** Any mechanism that fails a basic sanity check on a single window
  need not enter the grid at all.
Applied together these bring the total to the order of ten thousand short runs. At the model sizes
involved — compact perceptrons on a characteristic panel, not sequence models — this is days of
compute rather than a change of feasibility class, and it is precisely the reason the expert
architecture must stay small.

**One qualification to Tom's wording.** "The only way to create a fair experiment" is slightly too
strong — a fixed-depth design is also fair, it simply answers a narrower question. But since the grid
contains the fixed-depth comparison as one of its columns, the distinction has no practical
consequence here: the grid is fair in both senses at once, which is what makes it the right choice
rather than merely the more generous one.

**A by-product worth claiming.** No published study reports a grid of regime mechanisms against expert
capacity, for any target. If the grid is run, that table is an original result regardless of what it
shows, and it is obtained from work the study has to do anyway.

**Ask the advisor:**
(a) Is reporting the full mechanism-by-depth grid the right design, given that it triples the compute
    relative to fixing a depth — or would the committee regard a single fixed architecture as
    sufficient and the grid as scope the thesis does not need?
(b) Is nominating one cell in advance for significance testing, with the remainder of the grid reported
    descriptively, the correct discipline — or is there a preferred convention for reporting a grid of
    this kind?
(c) How many depths should the grid carry? Restricting to one, two and three hidden layers is
    defensible by citation, but omitting four and five means the thesis cannot itself confirm the
    shallow-is-sufficient finding on its own data.

See Q8 for the mechanism behind the gate-by-depth interaction and for how it is measured before
training, and Q9 for the closely related question of which metric the selection should optimise —
these two interact, since a depth selected on ranking accuracy need not be the depth selected on
portfolio performance.

---

### Q19. Should the gate be fitted separately and frozen, or trained jointly with the experts? [RESOLVED 2026-09-21]

**Answer: fitted separately and frozen. All four gates.** This is Tom's decision, taken 2026-09-21,
and it is a design commitment rather than a finding from the literature.

**The motivation.** The object being measured is what the experts add on top of a fixed regime
assignment. Freezing the gate is what makes that attribution possible. If the gate were trained
jointly it would co-adapt with the experts, the partition would be shaped by whatever most improves
the prediction, and any improvement could no longer be credited to the regime structure rather than
to the extra capacity and the freedom to place the boundaries. With the gate frozen, every arm
receives the same regime assignment as an input it cannot influence, and the only thing that varies
across arms is the regime process that produced it. That is the comparison the thesis is making.

**The cost, accepted deliberately.** This is not the arrangement that optimises the fitted
objective. Jointly optimised routing does better on the training criterion, and the concern is
acknowledged in the DeepSeek work on mixture-of-experts routing. The trade is accepted: a cleanly
attributable comparison is worth more here than the best achievable loss, because the thesis is a
measurement rather than a system. **Write this into the methodology as an explicit choice with its
justification, not as an oversight**, and pre-empt the objection rather than waiting for it at the
defence. (The exact DeepSeek citation still needs pinning down.)

**Consequences that follow immediately.**

1. **The separate-fit interface becomes the primary path, not an accommodation.** The
   `fit(train_panel)` hook and the precomputed-prior family (section 3 of
   `Code_Change_Brief_2026-09-21.md`) were proposed because the jump model and the Wasserstein gate
   cannot be trained by backpropagation. They are now how *all four* gates work. Their priority
   rises accordingly: nothing else in the gate programme can be built first.
2. **The trainable priors become baseline arms.** `soft`, `hard`, `topk` and `gumbel` are no longer
   candidate mechanisms in the comparison. They remain valuable as a named baseline, since a
   jointly trained gate is the natural upper reference for what a co-adapted partition can achieve,
   and the gap between it and the frozen arms is itself a number worth reporting.
3. **The load-balancing term is inert.** With a frozen gate the mean gate weight comes from a model
   with no trainable parameters, so `lambda_LB` is constant with respect to everything being
   optimised and its gradient is exactly zero. This restores the observation that was retracted on
   21 September: it is false of the code as it stands today and true of the design as now decided.
   The term must be removed from the frozen arms or explicitly documented as inert, and the
   contrast with Ye & Borde and with Shazeer et al., where the gate is trained and the term does
   work, is a legitimate methodological note.
4. **`GateHead` becomes vestigial on the gate path**, and the dead-parameter audit matters more
   rather than less. Live parameter counts must be reported per arm.
5. **Fit once per window is already the harness's discipline**, so the walk-forward machinery needs
   no change to accommodate this: each fold fits its gate on the training block and freezes it for
   that fold's out-of-sample dates.

**What this does not settle.** The parameters remain open: the expert count K (Q18), expert depth
(Q11), the correction magnitude weight alpha, the jump penalty, and each gate's own
hyperparameters. This decision fixes the *architecture of the comparison*, not any quantity in it.

*Original question, for the record:*

### Q19. Should the gate be fitted separately and frozen, or trained jointly with the experts?

**Status:** RESOLVED 2026-09-21, see the answer above. Raised 2026-09-21 from reading the
existing code rather than from the literature. This is the one place where the proposal and the implementation currently describe
different models.

**The discrepancy.** The proposal describes four structurally distinct regime-switching processes,
each fitted as a regime model in its own right, whose output is then used as the gate of a mixture
of experts. That is a two-stage design: fit the regime process, freeze it, train the experts
against it. The code in `nec_baseline` does something else. Its gate is a trainable
`LayerNorm -> Linear` head fed by a GRU encoder, and every registered prior — including the Markov
one — is fitted jointly with the experts by backpropagation through the mixture likelihood. There
is no frozen-gate mode anywhere in the package.

**Why this is not a detail.** The two designs answer different questions.

- **Jointly trained.** The gate is optimised to partition the data in whatever way most improves
  the prediction. What is being compared across arms is then the *inductive bias* each prior
  imposes on a learned partition: memorylessness, Markov persistence, sparsity. This is the
  classical mixture-of-experts object (Jacobs et al. 1991; Jordan & Jacobs 1994) and it is what
  Bishop §14.5.3 describes.
- **Fitted separately and frozen.** The gate is a regime model estimated on its own terms, by its
  own criterion, with no knowledge of the downstream task. What is being compared is then whether
  *economically identified regimes* carry information the cross-section can use. This is the
  stronger and more interesting claim, and it is the one the proposal's framing implies.

**Two facts that bear on the choice.**

1. **Two of the four gates cannot be trained jointly at all.** The statistical jump model is fitted
   by an alternating assign-and-fit procedure with a jump penalty, and the Wasserstein gate by
   clustering. Neither is differentiable end to end. So a purely joint design cannot accommodate
   half the comparison, and the separate-fit interface has to be built regardless.
2. **A mixed design would not be a clean comparison.** Two gates trained jointly against two gates
   fitted separately confounds the mechanism with the estimation route, and the confound runs in a
   predictable direction, since a jointly trained gate is optimising the very metric the comparison
   is scored on.

Together these point toward fitting **all four** gates separately and freezing them, with the
jointly trained soft gate retained as a named baseline arm rather than as one of the four. But that
is an argument, not a decision, and it changes what the thesis claims.

**The secondary consequence.** Under a frozen gate the load-balancing term `lambda_LB` has zero
gradient, because the mean gate weight comes from a model with no trainable parameters. Under the
current joint design it works normally. So whether that term is inert is downstream of this
question, not independent of it.

**What to ask.** Which claim is the thesis making: that the choice of prior structure changes what
a learned gate discovers, or that independently identified regimes are useful to the cross-section?
And is it acceptable for the jointly trained gate to appear as a baseline rather than as one of the
compared mechanisms?

### Q7. Should the shared base be pre-trained and frozen, or trained jointly with the experts? [RESOLVED 2026-09-21]

**Answer: pre-trained and frozen.** Tom's decision, 2026-09-21, taken together with Q19. A design
commitment, not a finding.

**The motivation is the same one that froze the gate.** The quantity being measured is what a
regime-conditional correction adds to a fixed baseline. If the base moved during expert training it
would co-adapt with the experts, and any improvement could no longer be attributed to the regime
structure rather than to the base quietly reorganising itself around the mixture. Freezing both ends
leaves exactly one thing varying across arms: the regime process that produced the gate.

**What is being given up, stated plainly.** The original question identified the jointly trained
base as the cell Ye and Borde never test, and as DeepSeekMoE's shared expert isolation
(arXiv:2401.06066). That remains the configuration most likely to produce the best fitted objective,
and it is now deliberately out of scope. The reasoning: the novelty budget of this thesis is spent
on the gate mechanisms, and freezing the base is what keeps the comparison with the base paper
clean. Testing the untested cell would be a second thesis.

**Consequences.**

1. **Only the experts train.** Base frozen, gate frozen, so the trainable model is K small MLPs plus
   their noise scales. Report the live parameter count per arm so this is visible rather than
   inferred.
2. **The base is shared across all four gate arms** within a window and seed, since it never sees
   regime information. This is the amortisation Q11's cost arithmetic already assumed, and it is now
   load bearing for the compute budget.
3. **The base's own out-of-sample performance must be reported beside every result.** The mixture's
   number is uninterpretable without the floor it is measured from, and the headline quantity of the
   thesis becomes the *improvement over the base*, not the level.
4. **The base protocol must be fixed in advance and never tuned after seeing expert results.** An
   undertrained base makes the experts look good and an overtrained one makes them look useless.
   Because the base is shared, this does not bias the comparison *between* gates, but it does set
   the headline "does the residual help" number, so it belongs in the pre-registration.
5. **The residual may be close to noise, and a null is a real possible outcome.** With the base
   capturing most of a signal that is around 0.4 per cent monthly out-of-sample R-squared to begin
   with, there may be little left for regime-conditional corrections to find. Decide now how a null
   is reported, because deciding after seeing the results is not a decision.

**Also settled by this.** The base and the experts are both multilayer perceptrons on the
characteristic snapshot, per Q8. Implementation brief: `Code_Change_Brief_02_Residual_Frozen_Base.md`.

*Original question, for the record:*

### Q7. Should the shared base be pre-trained and frozen, or trained jointly with the experts?
**Status:** RESOLVED 2026-09-21, see the answer above. Raised as open; the jointly trained cell
that Ye & Borde do not test is now deliberately out of scope.

Their "standard MoE" arm changes **two things at once**: it removes the base *and* switches to joint
training. There is no arm in the paper with a base network that is trained jointly with the experts.
That configuration is exactly DeepSeekMoE's **shared expert isolation** (arXiv:2401.06066, used in
DeepSeek-V2/V3 and Qwen's MoE models) — an always-on shared expert plus routed experts, optimised
end to end — and it is now the mainstream choice in large-model practice. So the question "is the
benefit the base, or the freezing?" is genuinely unanswered by the closest precedent.

**Arguments for freezing.**
1. **It is what makes the residual interpretable.** If the base moves during training, the residual
   is defined against a moving target: the base can quietly absorb regime-conditional structure and
   shrink `Σ_k π_k r_k(x)` without the gate being any worse. Freezing turns the residual series into
   an analysable object — magnitude, timing, concentration in transition windows.
2. **It makes the cross-gate comparison exactly controlled** — every candidate gate faces the
   identical base, so differences between gates cannot come from differences in what the base
   happened to learn.
3. It gives the guaranteed floor described in Q6.
4. The closest precedent freezes, so the comparison with it stays clean.

**Arguments for joint training.** A strictly higher ceiling — the base specialises into whatever the
experts do not cover, and the system is optimised for the actual objective rather than in two
disconnected stages. It is also the mainstream choice. The cost is identifiability: base and experts
trade capacity back and forth and nothing pins down how much of the forecast the gate is responsible
for.

**My position:** freeze — measurement beats maximum performance when the point of the thesis is to
measure something. But the cost of freezing should be quantified rather than assumed.

**Ask the advisor:** does freezing the base invite the criticism that the model was handicapped, and
is reporting a jointly trained arm alongside it a sufficient answer?

---
