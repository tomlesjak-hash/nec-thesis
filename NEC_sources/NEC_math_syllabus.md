# NEC Math Companion Syllabus

## Mathematical Foundations for the NEC / MoE Thesis

**Purpose:** companion plan to `NEC_thesis_syllabus.md`  
**Role:** deepen only the math that actually supports the NEC thesis  
**Orientation:** connect new ML math to your existing physics/math background  
**Created:** June 2026  

---

## 1. Verdict and Scope

You should go deeper in math, but not equally everywhere. The NEC thesis needs a clear mathematical spine:

1. Mixture-model probability.
2. Optimization and training dynamics.
3. Load balancing and expert collapse.
4. HMM / Markov-switching math.
5. Financial validation statistics.

Information theory and ensemble bias-variance are useful supporting material. They should appear in the thesis if they help explain gates, entropy, regularization, and why multiple experts might help. They should not become the main thesis.

The key rule:

```text
Before proposal: understand and implement the math.
After proposal: deepen derivations for the thesis and defense.
```

---

## 2. How This Connects to the Current Syllabus

| Current syllabus module | Math companion topic | Thesis use |
|---|---|---|
| Module 2: Linear algebra | Matrix calculus, projections, SVD | regression, PCA, gradients, softmax gate |
| Module 3: Probability/statistics | conditional probability, likelihood, entropy | MoE probability, losses, gate diagnostics |
| Module 4: Linear/logistic models | softmax, convex losses, regularization | NEC classifier/gate math |
| Module 5: Validation | sampling error, purging, multiple testing | defend empirical claims |
| Module 6: Neural networks | backprop, optimization | train corrected NEC |
| Module 7: Sequence models | recurrences, hidden states | LSTM gate encoder |
| Module 8: Mixture models | latent variables, EM | mathematical heart of MoE |
| Module 9: HMMs/regimes | dynamic programming, Markov chains | Hamilton baseline |
| Module 10: MoE theory | routing, load balancing | corrected NEC design |
| Module 11: Empirical asset pricing | factor math, residual returns | finance interpretation |

---

## 3. Bridge From Your Previous Math

You are not starting from zero. Several pieces of your physics/math background transfer directly:

- **Linear algebra / tensors** -> neural network layers, PCA, covariance matrices, embeddings, matrix calculus.
- **Calculus and variational principles** -> optimization, regularized objectives, Lagrangian penalties, gradient descent.
- **ODEs / dynamical systems** -> state evolution intuition, but HMMs are stochastic state models rather than deterministic systems.
- **Sturm-Liouville / spectral thinking** -> useful for covariance/PCA intuition, but not the center of NEC.
- **Green's functions / propagators** -> useful analogy for forward recursions, but HMM forward-backward is discrete probabilistic recursion.
- **Quantum mechanics / statistical mechanics** -> useful analogy for softmax, partition functions, entropy, and variational free energy.
- **Probability** -> direct foundation for latent-variable models, mixture likelihoods, and regime inference.

The new math you need to learn is less physics-flavored and more statistical ML-flavored: conditional likelihoods, latent variables, gates, EM, regularization, and financial validation.

---

## 4. Depth Map

| Topic | Pre-proposal depth | Thesis-depth after proposal | Priority |
|---|---|---|---|
| Mixture-model probability | derive MoE equation and implement soft MoE | likelihood derivation, EM comparison, log-sum-exp, responsibilities | highest |
| Optimization | backprop, Adam, regularization, gradient flow through gates | nonconvex training, collapse dynamics, penalty methods | highest |
| Load balancing | know formula and implement it | derive auxiliary loss, analyze utilization/entropy tradeoff | highest |
| HMM / dynamic programming | implement forward algorithm and understand Hamilton filter | forward-backward, Viterbi, Baum-Welch, Markov-switching regression | high |
| Financial validation statistics | implement purging, rank-IC, backtest costs | multiple testing, deflated Sharpe, uncertainty of IC | high |
| Information theory | entropy, KL, cross-entropy | gate entropy, mutual information diagnostics if useful | medium |
| Bias-variance / ensembles | ESL-level bias-variance | optional expert-ensemble discussion | low-medium |
| Stochastic calculus | skip | skip unless supervisor requires it | low |
| Measure theory | skip | skip | low |

---

## 5. Math Pillar 1: Linear Algebra and Matrix Calculus

**Connected current modules:** Module 2, Module 4, Module 6.

### Why it matters for NEC

Every model component is linear algebra plus nonlinearities:

- MLP experts are affine maps and activations.
- LSTM gates are matrix-vector recurrences.
- Softmax gates use logits from matrix operations.
- Ridge/logistic baselines are linear algebra baselines.
- PCA or covariance diagnostics may be useful for features.

### Previous-knowledge bridge

This is closest to your existing toolkit. Inner products, basis changes, eigenvectors, and quadratic forms already show up in physics. Here the same objects become regression, covariance, and gradient machinery.

### Readings by section number

- Strang, *Linear Algebra and Its Applications*: Sec 3.2-3.4, 5.1-5.3, 6.1-6.3.
- Matrix Cookbook: Sec 2.1, 2.4, 2.5, 8.1, 8.2.
- ESL: Ch 3.2-3.4.

### Focused exercise set

- Strang Sec 3.3: Exercises 1, 3, and 7.
- Strang Sec 6.3: Exercises 1 and 3.
- ESL Ch 3: Exercises 3.4 and 3.6.
- Applied: derive and implement ridge regression from scratch.
- Applied: derive the softmax Jacobian and verify it numerically.

### Thesis deliverable

You should be able to write the gate as:

```text
pi(x) = softmax(W h(x) + b)
```

and explain how gradients flow from the prediction loss through `pi(x)` into the gate parameters.

---

## 6. Math Pillar 2: Mixture-Model Probability

**Connected current modules:** Module 3, Module 8, Module 10.

### Why it matters for NEC

This is the mathematical heart of the thesis. A proper MoE is a conditional mixture model:

```text
p(y | x) = sum_k p(z = k | x) p(y | x, z = k)
```

The original NEC code is a rough hard-routed version of this idea. The corrected NEC should be a differentiable conditional mixture.

### Previous-knowledge bridge

This extends ordinary probability and Bayes rule. The physics analogy is summing over latent states: the final prediction marginalizes over hidden expert assignments.

### Readings by section number

- Bishop Ch 2.1-2.3 for probability distributions and Gaussians.
- Bishop Ch 9.1-9.4 for mixture models and EM.
- ESL Ch 8.5 for EM from another perspective.

### Focused exercise set

- Bishop Ch 2: Exercise 2.20.
- Bishop Ch 9: Exercises 9.7, 9.12, and 9.15.
- ESL Ch 8: Exercise 8.1 or 8.2.
- Applied: implement a 2-component GMM and compare with sklearn.
- Applied: implement a 3-expert soft MoE on synthetic regime data.

### Thesis deliverable

You should be able to derive:

```text
L(theta) = - sum_i log sum_k pi_k(x_i; theta) p(y_i | x_i, z_i = k; theta)
```

and explain the difference between:

- EM for classical mixtures.
- gradient descent for neural MoE.
- hard routing with argmax.
- soft routing with differentiable gates.

---

## 7. Math Pillar 3: Information Theory for Gates

**Connected current modules:** Module 3, Module 10.

### Why it matters for NEC

You do not need a full information theory course. You need enough to talk about:

- entropy of the gate,
- cross-entropy loss,
- KL divergence,
- expert collapse,
- whether the gate uses all experts or degenerates to one.

### Previous-knowledge bridge

Entropy connects naturally to statistical mechanics. Softmax can be read as a Boltzmann-like distribution over experts, where logits behave like negative energies up to temperature scaling.

### Readings by section number

- Bishop Ch 1.6.
- Goodfellow Ch 3.13.
- Shazeer et al. (2017), Sec 4.

### Focused exercise set

- Bishop Ch 1: Exercises 1.10 and 1.11.
- Applied: compute entropy of MoE gate probabilities over a batch.
- Applied: plot gate entropy over training and identify collapse.
- Applied: add an entropy penalty and compare it with Shazeer load balancing.

### Thesis deliverable

You should be able to report:

```text
H(pi(x)) = - sum_k pi_k(x) log pi_k(x)
```

and interpret low entropy, high entropy, and healthy expert specialization.

---

## 8. Math Pillar 4: Optimization and Training Dynamics

**Connected current modules:** Module 4, Module 6, Module 10.

### Why it matters for NEC

MoE success often depends less on architectural elegance and more on training dynamics. Gates can collapse; experts can fail to specialize; hard routing can block gradient flow.

### Previous-knowledge bridge

This connects directly to calculus and variational methods. A training objective is an energy functional; regularization and load balancing are penalty terms; optimization searches for parameters that reduce the objective.

### Readings by section number

- Prince Ch 6.1-6.5 and Ch 7.1-7.5.
- Goodfellow Ch 8.1-8.5.
- Goodfellow Ch 7.1-7.12 for regularization.

### Focused exercise set

- Prince Ch 7: Problems 7.1 and 7.2.
- Goodfellow Ch 6: Exercises 6.1 and 6.2.
- Applied: derive backprop for a 2-layer MLP and verify with `torch.autograd`.
- Applied: train the same MoE with and without load balancing and compare expert utilization.

### Thesis deliverable

You should be able to explain:

- why hard argmax routing is problematic for gradient flow,
- why soft routing trains end-to-end,
- why MoE loss is nonconvex,
- why auxiliary losses are penalty methods,
- why regularization can improve out-of-sample performance.

---

## 9. Math Pillar 5: Load Balancing and Expert Collapse

**Connected current modules:** Module 10, Thesis Research Design.

### Why it matters for NEC

This deserves its own mini-block because it is one of the most important MoE-specific issues. Without balancing, the gate can route most samples to one expert. Then the model becomes a single predictor with extra unused parts.

### Previous-knowledge bridge

This is a constrained optimization idea. You want low prediction loss, but also a non-degenerate allocation of samples across experts. The auxiliary loss acts like a Lagrange-style penalty.

### Readings by section number

- Shazeer et al. (2017), Sec 4.
- Fedus, Zoph, Shazeer (2022), Sec 2.2-2.3.
- Goodfellow Ch 7.1-7.2 for regularization framing.

### Focused exercise set

- Paper exercise: reproduce Shazeer's auxiliary load-balancing loss from Sec 4.
- Paper exercise: explain why expert utilization and average gate probability both matter.
- Applied: implement the load-balancing term.
- Applied: run three experiments: no balancing, weak balancing, strong balancing.

### Thesis deliverable

You should be able to include a figure/table like:

```text
alpha   rank-IC   expert entropy   expert utilization   turnover
0       ...       ...              ...                  ...
0.01    ...       ...              ...                  ...
0.10    ...       ...              ...                  ...
```

This directly connects math to empirical results.

---

## 10. Math Pillar 6: Dynamic Programming, HMMs, and Hamilton

**Connected current modules:** Module 9, Module 11.

### Why it matters for NEC

The HMM/Hamilton baseline is the classical finance comparison. You need enough dynamic programming to understand latent regime inference.

### Previous-knowledge bridge

Think of this as a discrete-time state evolution problem. Unlike deterministic dynamical systems, the state is hidden and probabilistic. The forward algorithm propagates a probability vector through time.

### Readings by section number

- Bishop Ch 13.1-13.3.
- Tsay Ch 4.1-4.6.
- Hamilton (1989), Sec 1-4.

### Focused exercise set

- Bishop Ch 13: Exercises 13.1, 13.3, and 13.5.
- Tsay Ch 4: two end-of-chapter exercises tied to Markov switching or volatility regimes.
- Applied: implement the HMM forward algorithm.
- Applied: fit a two-state model to SPY returns and compare states with realized volatility.

### Thesis deliverable

You should be able to explain:

```text
p(s_t | y_1, ..., y_t)
```

for Hamilton filtering, and contrast it with MoE gating:

```text
p(z = k | x_t)
```

The key distinction is temporal Markov persistence versus feature-conditioned routing.

---

## 11. Math Pillar 7: Financial Validation Statistics

**Connected current modules:** Module 5, Module 11, Module 12.

### Why it matters for NEC

The thesis will be judged partly on whether the empirical design avoids common finance mistakes. A complicated MoE is not convincing if the validation leaks information.

### Previous-knowledge bridge

This is applied statistics: sampling error, dependence, multiple testing, and estimating uncertainty under non-i.i.d. data.

### Readings by section number

- ESL Ch 7.1-7.12.
- Lopez de Prado AFML Ch 7.1-7.6.
- Bailey and Lopez de Prado (2014), Sec 1-4.

### Focused exercise set

- ESL Ch 7: Exercises 7.1, 7.2, and 7.10.
- Applied: implement purging and embargoing for a forward-return label.
- Applied: compute daily rank-IC and IC information ratio.
- Applied: simulate many random strategies and show why the best Sharpe is biased upward.

### Thesis deliverable

You should be able to justify:

- chronological splits,
- purging and embargoing,
- rank-IC rather than only MSE,
- transaction costs,
- deflated Sharpe or at least multiple-testing caution.

---

## 12. Optional Math: Bias-Variance for Experts and Ensembles

**Connected current modules:** Module 3, Module 10.

### Verdict

Learn standard ESL bias-variance before the proposal. Save non-independent ensemble bias-variance for after the proposal, and only include it if it helps your thesis discussion.

### Readings by section number

- ESL Ch 2.9.
- ESL Ch 8.7 if you want ensemble context.

### Focused exercise set

- ESL Ch 2: Exercise 2.9 or one nearby bias-variance exercise from your edition.
- Applied: train several experts with different seeds and measure prediction correlation.
- Applied: compare an average ensemble to a gated MoE.

### Thesis deliverable

This can support a short discussion:

```text
MoE is not just an ensemble; it is a conditional ensemble whose weights depend on x.
```

Do not make this a central math chapter unless your supervisor asks for more theory.

---

## 13. Before-Proposal Math Plan

This is the minimum mathematical preparation before September 2026.

### Block A: NEC probability and gate math

- ESL Ch 2.1-2.9.
- Bishop Ch 2.1-2.3.
- Bishop Ch 9.1-9.4.
- ESL Ch 8.5.
- Implement GMM and soft MoE on synthetic data.

### Block B: optimization and deep learning

- Prince Ch 6.1-6.5 and 7.1-7.5.
- Goodfellow Ch 8.1-8.5.
- Derive 2-layer backprop.
- Implement PyTorch MLP.
- Verify gradients.

### Block C: HMM / Hamilton baseline

- Bishop Ch 13.1-13.3.
- Tsay Ch 4.6.
- Hamilton (1989), Sec 1-4.
- Implement HMM forward algorithm.
- Fit two-state volatility model to SPY.

### Block D: financial validation

- ESL Ch 7.1-7.12.
- AFML Ch 7.1-7.6.
- Implement purged walk-forward split.
- Implement rank-IC and decile backtest.

### Proposal-level math deliverables

Before proposal, be able to write and defend:

1. MoE conditional mixture equation.
2. Original NEC hard-routing defect.
3. Corrected soft-routing equation.
4. Load-balancing loss at a high level.
5. HMM/Hamilton baseline at a high level.
6. Walk-forward validation and purging logic.

---

## 14. After-Proposal Math Plan

This is where the deeper theory belongs.

### Deepen mixture-model theory

- Responsibilities.
- EM versus gradient descent.
- Gaussian mixture likelihood.
- Mixture density network connection.
- Identifiability and label switching.

### Deepen optimization and load balancing

- Expert collapse dynamics.
- Gate entropy.
- Shazeer versus Switch load balancing.
- Penalty-method interpretation.
- Hyperparameter sensitivity of balancing coefficient.

### Deepen HMM and regime-switching

- Forward-backward smoothing.
- Viterbi path.
- Baum-Welch updates.
- Markov-switching regression.
- Regime persistence and transition matrix interpretation.

### Deepen financial statistics

- Deflated Sharpe.
- Multiple testing.
- Confidence intervals for IC.
- Robustness across market regimes.
- Transaction-cost sensitivity.

### Thesis-level math deliverables

By January 2027, the thesis should include:

1. A formal MoE model section.
2. A short EM/HMM comparison section.
3. A corrected NEC architecture section.
4. A load-balancing and gate-diagnostics section.
5. A validation/statistical testing section.

---

## 15. What Not To Overlearn

Do not spend major time on these unless your supervisor specifically pushes you there:

- stochastic calculus,
- measure-theoretic probability,
- PDE theory,
- advanced random matrix theory,
- optimal transport,
- PAC-Bayes,
- deep information theory.

They are interesting, but they are not the bottleneck for this thesis.

---

## 16. Final Math Thesis Spine

The thesis math can be organized as:

```text
1. Conditional mixture probability:
   p(y | x) = sum_k p(z = k | x) p(y | x, z = k)

2. NEC defect:
   three-class gate -> binary hard route

3. Corrected MoE:
   pi_k(x) = softmax(g_k(x))
   y_hat = sum_k pi_k(x) f_k(x)

4. Training objective:
   prediction loss + alpha * load-balancing loss

5. Classical baseline:
   HMM / Hamilton filter with Markov state persistence

6. Financial evaluation:
   walk-forward validation, purging, rank-IC, transaction-cost backtest
```

If you can explain and implement those six pieces, the math foundation is strong enough for the proposal. The deeper derivations can be added during the thesis phase.
