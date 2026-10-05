# MoE formulation course: plan, feedback and running notes

Started 2026-10-01. Working file for the multi-prompt course on **how to formulate the HMM-gated
mixture of experts** before any training design. At the end this file is the source for one PDF
("how to construct an HMM-gate MoE"). Training details come after the formulation is understood.

Companion documents: `Model_Derivation_BaseCase.md` (2026-09-22 derivation of the full base case),
`Advisor_Questions.md`, `Progress_Tracker.md`.

---

## 1. Tom's feedback on Module 1 (2026-10-01), recorded verbatim in substance

Module 1 ("translating Bishop's MoE to ours") was accepted as fine, with these requested changes:

1. Explain what $w_k^\top\phi(x)$ stands for, and give its **full expansion** in the PDF: basis-function
   regression, then the MLP as learned basis functions, layer by layer. Go very deep on MLP theory.
2. Explain **why the volatility $\sigma_k$ is the noise of the expert** (what the Gaussian around the
   MLP output means, why the MLP does not output it).
3. Explain **where the softmax gate comes from** (derive it, do not just state it).
4. Before any HMM gate: a full derivation of the **generative model for the softmax-gated MoE**: the
   likelihood, and how it would be maximised with **EM**. Even though we train by backpropagation,
   explain **why backprop on the likelihood looks like EM**, and give an in-depth intuition and
   derivation of **backpropagation**.
5. **Before all of that**: Claude poses a question/exercise where Tom derives a MoE with **linear
   regression experts** (EM is tractable because the experts are linear). Tom attempts it; Claude then
   gives a deep worked answer so Tom can check himself.
6. Then: how to put an **HMM gate on the linear-regression MoE**, in depth: transforming the gate into
   an HMM, and how to train it, using the attached paper as the main reference
   (Chamroukhi & Nguyen 2018, "Model-Based Clustering and Classification of Functional Data",
   arXiv:1803.00276).
7. Before Tom attempts the linear MoE and HMM-MoE derivations: Claude recommends which sections of
   the paper to read as a warm-up.
8. Only after the linear versions: the MLP MoE (items 1-4), then the HMM gate on the MLP MoE.
9. Record all of this in an .md file to generate the PDF later (this file).

---

## 2. Revised order of the course

### Part A. Linear-regression experts (EM is exact and closed-form where possible)

| Step | Content | Status |
|---|---|---|
| A0 | Warm-up reading: paper sections 2.1-2.2, 3.1-3.2, 4.3.1-4.3.2 (+ Bishop 14.5.1, 4.3.3-4.3.4) | assigned 2026-10-01 |
| A1 | **Exercise (Tom):** softmax-gated MoE with linear-regression experts: generative model, complete and observed likelihood, E-step, M-step (closed-form experts, IRLS gate), blocks with a shared latent | posed 2026-10-01 |
| A1-ans | Claude's in-depth worked answer to A1 | after Tom's attempt |
| A2 | Warm-up reading for the HMM version: paper section 4.2.1-4.2.2 (+ Bishop 13.2.1-13.2.2, 13.2.4) | pending |
| A3 | **Exercise (Tom):** HMM-gated linear MoE: one regime per date shared by all stocks, Markov chain over dates, likelihood via the forward recursion, EM = Baum-Welch with weighted least squares for the experts; filtered vs predicted vs smoothed probabilities; forecasting | pending |
| A3-ans | Claude's in-depth worked answer to A3, plus "how to train it" following paper 4.2.2 | pending |

### Part B. MLP experts

| Step | Content |
|---|---|
| B1 | What $w_k^\top\phi(x)$ is: linear basis-function regression (fixed $\phi$), then the MLP as learned $\phi$. Full layer-by-layer expansion, activations, parameter count, universal approximation intuition, link to GKX NN1-NN5 |
| B2 | Why $\sigma_k$ is the noise: $y = \mu_k(x) + \sigma_k\varepsilon$, what the MLP models (conditional mean) vs what $\sigma_k$ models (irreducible scatter in regime k); MLE of $\sigma_k^2$ = responsibility-weighted MSE; link to MSE when $\sigma$ is fixed |
| B3 | Where the softmax gate comes from: (i) Bayes' rule with Gaussian class-conditionals sharing a covariance gives linear log-odds (Bishop 4.2); (ii) multinomial logit / GLM canonical link for a categorical variable; (iii) identifiability, one reference class $w_K = 0$ |
| B4 | Softmax-gated MLP MoE: generative model, likelihood, EM view (E-step exact, M-step not closed form, so generalised EM); Fisher's identity: gradient of the log-likelihood equals gradient of Q at the current parameters, hence "backprop looks like EM"; in-depth backpropagation (chain rule on the computational graph, forward pass, backward pass, deltas for every layer, gradient of the mixture NLL, gate gradient $\gamma - \pi$) |
| B5 | HMM gate on the MLP MoE (old Module 3): regime chain, emission on the market series, forward filter, the gate probability used at date t |
| B6 | Full generative model (old Module 4): graphical model with HMM layer + cross-section layer, joint factorisation, observed / latent / parameters |
| B7 | Likelihood (old Module 5): per date vs per row (Q24); joint fit vs two-stage (gate fitted on the market series and frozen) |
| B8 | Prediction and posteriors (old Module 6): forecast, predictive variance, responsibilities, filtered vs smoothed (look-ahead) |
| B9 | Assumptions and design checklist (old Module 7): K, h, gate series, timing, state-to-expert labelling, residual form $f_0 + r_k$, open advisor questions |

### Part C. The PDF

Compile everything into one document with one consistent notation (Section 4 below), worked
derivations in full, and the exercises with solutions.

---

## 3. Material already covered in this conversation (to go into the PDF)

- **Why Bishop's formula has no latent variable:** $p(t\mid x) = \sum_z p(z\mid x)p(t\mid x,z)$; the latent is
  summed out; it reappears in the complete-data likelihood (EM) and as the posterior responsibility.
- **Tom's derivation of the MoE likelihood, with three fixes:** (1) one one-hot $z_n$ per observation, so
  $p(Z\mid X) = \prod_n\prod_k \pi_k(x_n)^{z_{nk}}$; (2) expert density indexed by $n$ and $k$; (3) summing over $Z$
  turns $\prod_k$ into $\sum_k$ (sum moves inside the product because the factors for different $n$ share no
  latent; contrast the HMM). Result $\prod_n\sum_k \pi_k(x_n)\mathcal N(t_n\mid\mu_k(x_n),\sigma_k^2)$. Date-level
  version: $\prod_t\sum_k\pi_k(t)\prod_{i}\mathcal N(\cdot)$ (Q24).
- **Papers that confirm it:** Jacobs, Jordan, Nowlan & Hinton 1991 (eq 1.3, 1.5); Jordan & Jacobs 1994
  (log-likelihood, indicators, complete-data likelihood, $h$); Weigend, Mangeas & Srivastava 1995 (eqs 5,
  9-10, 12, 21-25; MLP experts with 10 tanh units, gate 20 units); Bengio & Frasconi 1996 (IOHMM).
  Ye & Borde 2026 uses MLP experts but MSE on the mixed prediction (eq 7), not the mixture likelihood.
- **Weigend 1995 takeaways:** M-step cost = gate cross-entropy + $h$-weighted Gaussian NLL; closed-form
  variance update and the variance prior $\lambda\sigma_0^2$ (random-walk variance for finance); expert gradient
  $h/\sigma^2$ = weighted regression; fixed variances lose regime discovery (fn 12); $g$ vs $h$ scatter
  diagnostic; start with more experts than needed; seed instability when regime variances differ by < 3x;
  annealing $h' = (1-\kappa)h + \kappa g$; two-stage residual model (Section 6.2).
- **`|` versus `;`:** `|` conditions on a random variable; `;` indexes by a fixed parameter. Bishop uses `|`
  for both. In our model: $z$ after `|`, $(\theta_k,\sigma_k)$ after `;`, $x$ after `|` by convention.
- **Target timing:** $y_{i,t} \ne x_{i,t+1}$ ($x$ is a characteristic vector, $y$ a scalar return). Write the
  forward return explicitly: $r_{i,t+1}$ (or $r_{i,t\to t+h}$). Two consistent conventions exist (label by
  realisation date vs by formation date); the earlier lesson mixed them. Use explicit $r_{i,t+1}$ in the PDF.
- **HMM joint factorisation (Bishop 13.6):** product rule $p(X,Z)=p(Z)p(X\mid Z)$, chain rule on both, then
  the Markov assumption and the emission assumption; equivalently read off the DAG (Bishop 8.5).
  Hamilton AR version breaks the emission assumption on purpose (bug G-2 alignment).
- **Module 1 (Bishop to ours):** three changes: indices $n \to (i,t)$; experts $w_k^\top\phi(x) \to \mu_k(x_{i,t};\theta_k)$
  with per-expert $\sigma_k^2$; gate softmax of $x_n$ $\to$ HMM probability from market info, one latent per date
  shared by all stocks, linked over time. Generative story; reasons for an HMM gate (persistence,
  interpretability, few parameters, regime info only through the gate). HMM state labels fix expert labels.

---

## 4. Notation rules for the PDF

- Target: $r_{i,t+1}$ (forward return), features $x_{i,t}$, market information $\mathcal F_t$, regime $z_t$.
- **Clash to resolve:** `Model_Derivation_BaseCase.md` uses $\sigma_k$ for the **gate's** regime volatility
  of the market series and $s_k$ for the **expert** noise scale; this conversation used $\sigma_k$ for the
  expert noise. The PDF must pick one scheme and state it (proposal: keep the base-case document's
  $s_k$ for experts, $\sigma_k$ for the gate emission).
- Bishop's $t$ is the target; ours is time. Never use $t$ for the target.
- The Chamroukhi-Nguyen paper uses $\pi$ for the HMM **initial** distribution and also for the logistic
  gate $\pi_{kr}(x_j;w_k)$; $\alpha_k$ for constant mixing proportions; $\tau$ for cluster posteriors, $\gamma$ for
  regime posteriors, $\xi$ for pairwise posteriors.

---

## 5. Open item found while writing this file (not decided)

**Filtered or predicted probability as the gate.** `Model_Derivation_BaseCase.md` §1.7 defines the gate
as the filtered probability $\pi_k(t) = \xi_{t\mid t}(k) = p(s_t = k\mid r_{1:t})$, used to weight the forward return
over $(t, t+h]$. In this conversation (2026-10-01) the gate was written as the one-step-ahead predicted
probability $p(z_{t+1}=k\mid\mathcal F_t) = \sum_j P_{jk}\,\xi_{t\mid t}(j)$, on the argument that the regime generating
$r_{i,t+1}$ is the regime at $t+1$. With persistent regimes ($P_{kk}\approx 0.98$) the two are close, but they are
different models. To be settled in step B7/B9 and possibly raised as an advisor question. Nothing changed
in code.

---

## 6. Warm-up reading assigned for A0 (paper page numbers as printed)

1. §2.1-2.2, pp. 8-11: general mixture, complete-data log-likelihood (eq 4), E-step (5)-(6), M-step (7)-(8).
2. §3.1-3.2, pp. 14-16: mixture of linear regressions and its EM: weighted least squares (18)-(19).
   Note: one latent label per **curve** shared by all $m_i$ points of that curve, so each component density
   is a product over the curve's points. That is exactly our "one regime per date shared by all stocks".
3. §4.3.1-4.3.2, pp. 48-51: RHLP = MoE with linear (polynomial) experts and a softmax/logistic gate;
   E-step (54) without forward-backward; gate M-step is a weighted multinomial logistic regression
   solved by IRLS. Read it with $K = 1$ (ignore the outer cluster mixture).
4. Companion: Bishop 14.5.1 (mixtures of linear regression models, EM), 4.3.3-4.3.4 (IRLS, multiclass
   logistic regression).
5. For A2 later: §4.2.1-4.2.2, pp. 41-45 (mixture of HMM regressions: forward-backward E-step, weighted
   updates for initial distribution, transition matrix, $\beta$ and $\sigma^2$, eqs 39-47).
Skip: §3.3-3.6 (regularised mixtures, experiments), §4.1 (piecewise regression), §5 (discriminant analysis).

---

## 7. Exercise A1 (posed 2026-10-01): softmax-gated MoE with linear-regression experts

**Setup.** Data $\mathcal D=\{(x_n,y_n)\}_{n=1}^N$, $x_n\in\mathbb R^d$ with first entry 1 (intercept), $y_n\in\mathbb R$.
$K$ experts, latent one-hot $z_n$.

$$p(z_{nk}=1\mid x_n;w)=\pi_k(x_n;w)=\frac{\exp(w_k^\top x_n)}{\sum_{j=1}^K\exp(w_j^\top x_n)},\qquad w_K=0$$

$$p(y_n\mid x_n,z_{nk}=1;\beta_k,\sigma_k^2)=\mathcal N(y_n;\beta_k^\top x_n,\sigma_k^2)$$

Parameters $\Psi=(w_1,\dots,w_{K-1},\beta_1,\dots,\beta_K,\sigma_1^2,\dots,\sigma_K^2)$.

(a) Generative story (ancestral sampling for one data point) and the graphical model: observed, latent, parameters.
(b) Complete-data likelihood $p(Y,Z\mid X;\Psi)$ in 1-of-K exponent form, and $\log L_c$.
(c) Observed-data log-likelihood $\log p(Y\mid X;\Psi)$, showing the step where $Z$ is summed out; why there is no closed-form maximiser.
(d) E-step: $\tau_{nk}^{(q)}=p(z_{nk}=1\mid y_n,x_n;\Psi^{(q)})$ by Bayes' rule.
(e) $Q(\Psi;\Psi^{(q)})$; show it separates into one gate term (only $w$) and $K$ expert terms (each only $\beta_k,\sigma_k^2$).
(f) Expert M-step in closed form: $\beta_k^{(q+1)}$ (matrix form with $X$, $y$ and a diagonal weight matrix) and $\sigma_k^{2(q+1)}$.
(g) Gate M-step: objective, gradient w.r.t. $w_k$, Hessian blocks, concavity, why no closed form, one Newton-Raphson (IRLS) step.
(h) Special case: gate without inputs ($\pi_k$ constant $=\alpha_k$); show $\alpha_k^{(q+1)}=\frac1N\sum_n\tau_{nk}^{(q)}$.
(i) Panel version: dates $t$, stocks $i=1,\dots,N_t$, one latent $z_t$ per date shared by all stocks, gate input a date-level vector $u_t$,
experts $\mathcal N(r_{i,t+1};\beta_k^\top x_{i,t},\sigma_k^2)$. Redo (b), (c), (d), (f). What changes in $\tau$ and why it sharpens as $N_t$ grows (hint: paper §3).
(j) Optional: show $\log L(\Psi)=Q(\Psi;\Psi^{(q)})+H(\Psi;\Psi^{(q)})$ (entropy-type term) and use it to argue EM never decreases the log-likelihood.


## 7b. Exercise A2 (drafted 2026-10-01 for the PDF): the linear MoE with an HMM gate

**Setup.** Days $t=1..T$, stocks $i\in\mathcal S_t$, market series $m_t$, regime $z_t\in\{1..K\}$:
$p(z_1=k)=\rho_k$, $p(z_t=k\mid z_{t-1}=j)=A_{jk}$, $p(m_t\mid z_t=k)=\mathcal N(m_t;\nu_k,\sigma_k^2)$,
$p(r_{i,t}\mid x_{i,t-1},z_t=k)=\mathcal N(r_{i,t};\beta_k^\top x_{i,t-1},s_k^2)$ independently over $i$ (realisation-date labelling).

(a) Graphical model; observed / latent / parameters / conditioned-on only.
(b) Derive the joint $p(z_{1:T},m_{1:T},R_{1:T}\mid X)$ from product and chain rules, naming every independence assumption.
(c) Day-$t$ emission $e_t(k)$; likelihood as a sum over paths; scaled forward recursion; $O(TK^2)$.
(d) Backward recursion; $\gamma_t(k)$ and pairwise $\xi_t(j,k)$.
(e) $Q$ and M-step for $\rho, A, \nu_k, \sigma_k^2, \beta_k, s_k^2$; compare $\beta_k$ with Chamroukhi-Nguyen eq. 46.
(f) Two-stage version: Baum-Welch on $m$ only, freeze; Stage-2 E and M steps with prior $\xi_{t\mid t-1}$; what information the joint fit uses that two-stage ignores.
(g) Which probability may be used to forecast $r_{i,t+1}$ at the close of $t$; show smoothed $\gamma_t$ is look-ahead.
(h) Map onto Chamroukhi-Nguyen MixHMMR (their $n, K, R_k, m_i, y_{ij}, W_{ikr}$).

Notation decided for the PDF: expert noise $s_k$ (also in the linear exercises), gate emission $\mathcal N(m_t;\nu_k,\sigma_k^2)$,
corrections $c_k$ (code: $r_k$), market series $m_t$ (base-case doc: $r_t$), regime $z_t$ (base-case doc: $s_t$), transition $A$ (base-case doc: $\mathbf P$).

## 7c. Decisions from Tom on 2026-10-01 (scope and notation of the PDF)

- **Notation:** predict $r_{i,t+1}$ from $x_{i,t}$ everywhere (formation-date timing written explicitly). The regime that
  generates day-$(t+1)$ returns is $z_{t+1}$; the market return over day $t+1$ is $m_{t+1}$. Do not use the $r_{i,t}$ / $x_{i,t-1}$ form.
- **Scope:** the PDF derives the **base-case MLP MoE** only. Estimation = plain **maximum likelihood** (gate: EM on the
  market series; experts: maximise the mixture likelihood). No penalties, no tempering, no joint gate-expert fit, no
  alternative losses. Loss choice (Q20) and other refinements are out of scope for this document.
- **Pace:** one step at a time in chat; the PDF is built only when Tom says so.
- **Module 4 must explain the transition matrix A:** what it is, why the Markov chain sits on the latent variables
  (Bishop 13.1-13.2), its role in the HMM gate (prediction step, persistence, durations, stationary distribution,
  h-step forecasts), how it is estimated (counting when regimes are observed, expected counts in Baum-Welch),
  and how it is initialised / used to simulate. Connect every step to Bishop ch. 13.

## 8. Log

- 2026-10-01: file created; feedback recorded; course reordered; A0 reading and A1 exercise posed.
- 2026-10-01 (later): Tom: the warm-up (reading guide + exercises A1 and A2, with solutions at the back)
  goes **into the PDF**, not answered in chat. Exercise A2 (HMM-gated linear MoE, parts a-h) drafted for the
  PDF. Tom then asked to **go over Modules 4-7 in chat one by one before any PDF is built**. PDF build is on
  hold; LaTeX drafts of the front matter, Section 1 (translation), Section 2 (warm-up + A1/A2 statements)
  and Section 3 (w^T phi(x), full MLP expansion, why s_k is noise) exist in the cloud workspace only and
  may be lost; this file is the source of truth.
- 2026-10-01: Module 4 (full generative model) and Module 5 (likelihood) presented in chat (r_{i,t}/x_{i,t-1} timing).
- 2026-10-01: Tom: switch to t+1 notation; base case only (maximum likelihood); explain A in depth with Bishop links;
  one step at a time. Module 4 re-presented with the transition matrix.
- 2026-10-01: Module 5 re-presented in t+1 notation with forward-backward usage; Module 6 presented.
- 2026-10-01: Tom asked to define F_t as a sigma-algebra / filtration -> added (PDF Section 1.4: F_t = sigma(m_1..m_t),
  natural filtration, G_t full information, gate as F_t-measurable conditional probability, no look-ahead = adaptedness).
- 2026-10-01: **PDF built**: `MoE_HMM_Gate_Formulation.pdf` (44 pp) in this folder; LaTeX sources in `MoE_Formulation_tex/`
  (edit there and recompile with pdflatex twice). Contents: notation; S1 translation + filtrations; S2 warm-up reading +
  Exercises A1, A2; S3 w^T phi(x), full MLP expansion, why s_k is noise; S4 softmax origin, softmax-MoE likelihood, EM,
  lower bound, Fisher identity, backprop; S5 HMM gate (A, Bishop 13.1-13.2, predict/update, estimation, alpha/beta uses,
  Baum-Welch, scaling, labelling); S6 generative model; S7 likelihood (two-stage, per date vs per row); S8 prediction and
  posteriors (incl. predicted vs filtered open item); S9 assumptions and design checklist; App A/B solutions; App C refs.
  Key formulas numerically checked (forward-backward identities vs brute force, mixture gradients, moments, KL example).
- 2026-10-01: Target decided: market-neutral (cross-sectionally demeaned) return, recorded as Q25 (resolved) in Advisor_Questions.md and in Progress_Tracker.md. PDF notation and checklist updated to match.

- **2026-10-05.** Q24 decided (per row). PDF revised: Section 7 rewritten so the per-row objective is the one maximised, read as the composite (independence) likelihood of the per-date model; new Appendix D with the full theory (two models and graphs, the one-stock marginal identity with a d-separation and an algebra part, composite likelihood, unbiased score and consistency, Godambe information, responsibilities, a model where A3 fails but marginals are right, the design effect, the frozen-gate argument); checklist, Sections 0, 4, 6, 8 and Appendix A updated. 51 pages.
- **2026-10-05.** Gate weight decided (Q27): window average of the 1- to h-step-ahead regime probabilities. PDF Section 8.9 rewritten (was the open predicted-vs-filtered item), Sections 0, 1, 8.6 and the checklist updated.
