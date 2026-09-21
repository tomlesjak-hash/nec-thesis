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

### Q4. Which framing for the proposal — comparison, or design-led?
**Status:** open

Two ways to write the same work:
- **Comparison-led:** "which regime mechanism makes the best gate?" Honest, but structurally similar
  to the rejected topic.
- **Design-led:** "here is a differentiable persistence-penalised gate, and here is the evidence it
  beats the alternatives." Same experiments, different emphasis.

Which reads better to the review committee?

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

### Q7. Should the shared base be pre-trained and frozen, or trained jointly with the experts?
**Status:** open — and **this is the cell Ye & Borde do NOT test**

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

### Q8. Expert and base architecture — what is settled by literature and what is not
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

### Q16. How much data is needed, and how should it be split across regimes?
**Status:** open, raised by Tom 2026-09-15. Bears directly on Q3, since it turns data access from a
convenience into a requirement.

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
   this way. It is worth noting for this thesis in particular that **the half-life is a persistence
   parameter**, playing the same role for the training sample that the switching penalty plays for the
   jump model gate, so the three schemes are points on a continuum rather than separate choices: a
   half-life of zero is the rolling window and an infinite one is the expanding window.

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

## RESOLVED

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
