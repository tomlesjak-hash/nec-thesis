# Full derivation of the base-case model

Residual mixture of experts, frozen base, frozen Markov switching gate, MLP experts.
Written 2026-09-22. This is the mathematical core of the methodology chapter.

Scope: the **base case only**. One gate, the Markov switching gate of Hamilton (1989), which is the
simplest of the four and the one every other gate is compared against. The jump model, TVTP and
Wasserstein gates change **only** Stage 1 below; Stages 2 to 5 are identical for all four. That is
the point of the design, and it is worth noticing that the derivation makes it obvious.

Open questions that this document deliberately does not settle: the primary error function (Q20,
both candidates derived side by side in Stage 4), the number of regimes K (Q18), the expert depth
(Q11), and the correction penalty weight alpha.

---

## Stage 0. Notation and the data

**The panel.** Dates $t = 1,\dots,T$. On date $t$ there are $N_t$ firms, indexed $i$.

- $x_{i,t} \in \mathbb{R}^d$ — the characteristic vector for firm $i$, known at date $t$.
- $y_{i,t} \in \mathbb{R}$ — the forward return of firm $i$ over $(t,\,t+h]$. The only
  forward-looking quantity in the data.
- $r_t \in \mathbb{R}$ — a market-level series (daily log total return of the CRSP value-weighted S&P 500 universe index, Q23), used **only** by the gate.

**Why two data objects.** The gate is a date-level object: one regime for the whole market on date
$t$, shared by every firm. The experts are cross-sectional: one prediction per firm per date. The
model joins a univariate time series to a panel, and almost every indexing subtlety below comes from
that join.

**The parameters, in three groups.**

| Group | Symbol | Estimated by | Status during expert training |
|---|---|---|---|
| Gate | $\theta_g = \{\mathbf{P}, \{\mu_k,\sigma_k\}, \boldsymbol{\rho}_1\}$ | EM on the market series | **frozen** |
| Base | $\phi$ | backpropagation on the panel | **frozen** |
| Experts | $\psi = \{\psi_1,\dots,\psi_K\}$ and $\{\log s_k\}$ | backpropagation | trainable |

Note the notation clash to avoid: $\sigma_k$ is the gate's regime volatility in the market series;
$s_k$ is the expert $k$ noise scale in the panel. They are different objects and the literature
calls both sigma.

**The model, in one line.**

$$\hat y_{i,t} \;=\; f_0(x_{i,t};\phi) \;+\; \sum_{k=1}^{K} \pi_k(t)\, r_k(x_{i,t};\psi_k)$$

where $\pi_k(t)$ is the gate weight from Stage 1. **Updated 2026-10-05 (Q27):** it is the average over the
5-day target window of the h-step-ahead regime probabilities, built from the **filtered** probability
(section 1.7); originally it was the filtered probability itself. Stage 1, $f_0$ is the frozen base of Stage 2, and $r_k$ are the expert corrections of Stage 3.

---

## Stage 1. The gate: a Markov switching model, fitted by EM

### 1.1 The generative model

A latent state $s_t \in \{1,\dots,K\}$ follows a first-order Markov chain with transition matrix
$\mathbf{P}$, whose entries are

$$P_{jk} \;=\; \mathbb{P}(s_t = k \mid s_{t-1} = j), \qquad \sum_{k=1}^{K} P_{jk} = 1 \ \ \forall j$$

and initial distribution $\rho_1(k) = \mathbb{P}(s_1 = k)$. Conditional on the state, the market
series is Gaussian:

$$r_t \mid s_t = k \;\sim\; \mathcal{N}(\mu_k,\ \sigma_k^2)$$

Write the conditional density as $\eta_t(k) = \mathcal{N}(r_t \mid \mu_k, \sigma_k^2)$. The full
parameter vector is $\theta_g = \{\mathbf{P}, \mu_{1:K}, \sigma_{1:K}, \rho_1\}$.

This is Hamilton's model with a constant mean per regime. Allowing the variance to switch as well as
the mean is standard in finance, because empirical regimes are separated far more sharply by
volatility than by mean return.

### 1.2 Why the likelihood cannot be maximised directly

If the state path $\mathbf{s} = (s_1,\dots,s_T)$ were observed, the complete-data likelihood would
factorise into a product of simple terms:

$$p(\mathbf{r}, \mathbf{s} \mid \theta_g) \;=\; \rho_1(s_1)\, \eta_1(s_1) \prod_{t=2}^{T} P_{s_{t-1} s_t}\, \eta_t(s_t)$$

and its logarithm would be a sum, each parameter appearing in its own additive block, each with a
closed-form maximiser. But the path is not observed. The observed-data likelihood requires summing
it out:

$$p(\mathbf{r} \mid \theta_g) \;=\; \sum_{\mathbf{s}} p(\mathbf{r}, \mathbf{s} \mid \theta_g)$$

That sum runs over $K^T$ paths. With $K=2$ and sixty years of monthly data it has roughly $10^{217}$
terms. Taking the logarithm leaves a log of a sum, which does not separate, and the closed forms
disappear.

**This is exactly the situation EM was built for**, and it is the same structure as the Gaussian
mixture of Bishop chapter 9 with one extra ingredient: the latent variables are no longer
independent, because $s_t$ depends on $s_{t-1}$.

### 1.3 The EM objective

EM maximises the observed-data log likelihood indirectly. Given current parameters
$\theta_g^{\text{old}}$, define

$$Q(\theta_g,\theta_g^{\text{old}}) \;=\; \sum_{\mathbf{s}} p(\mathbf{s}\mid\mathbf{r},\theta_g^{\text{old}})\, \log p(\mathbf{r},\mathbf{s}\mid\theta_g)$$

the expected complete-data log likelihood under the posterior over paths. The identity that makes
this work is

$$\log p(\mathbf{r}\mid\theta_g) \;=\; Q(\theta_g,\theta_g^{\text{old}}) + \mathbb{H}[q] + \mathrm{KL}\big(q \,\|\, p(\mathbf{s}\mid\mathbf{r},\theta_g)\big)$$

with $q = p(\mathbf{s}\mid\mathbf{r},\theta_g^{\text{old}})$ and $\mathbb{H}[q]$ its entropy, a
constant in $\theta_g$. At $\theta_g = \theta_g^{\text{old}}$ the KL term vanishes, so the bound
touches the likelihood; maximising $Q$ therefore cannot decrease the likelihood. The expectation is
there because Jensen's inequality, which is what moves the logarithm inside the sum, is stated in
terms of expectations.

Substituting the factorisation of 1.2 and taking the expectation term by term:

$$Q \;=\; \sum_{k} \hat\xi_{1}(k)\log \rho_1(k) \;+\; \sum_{t=2}^{T}\sum_{j}\sum_{k} \hat\xi_{t-1,t}(j,k)\log P_{jk} \;+\; \sum_{t=1}^{T}\sum_{k} \hat\xi_{t}(k)\log \eta_t(k)$$

where

$$\hat\xi_{t}(k) = \mathbb{P}(s_t = k \mid \mathbf{r}, \theta_g^{\text{old}}), \qquad \hat\xi_{t-1,t}(j,k) = \mathbb{P}(s_{t-1}=j, s_t=k \mid \mathbf{r}, \theta_g^{\text{old}})$$

**Everything the M step needs is these two quantities.** Note that the expectation of a sum over
$K^T$ paths has collapsed into marginal and pairwise marginal probabilities, which cost $O(TK^2)$ to
compute. That collapse is the whole reason EM is tractable here, and it happens because
$\log p(\mathbf{r},\mathbf{s})$ is **linear** in the state indicators.

### 1.4 The E step, part one: the Hamilton filter

The forward recursion. Write $\xi_{t\mid\tau}(k) = \mathbb{P}(s_t = k \mid r_1,\dots,r_\tau)$.

**Initialise** at $\xi_{1\mid 0}(k) = \rho_1(k)$, or at the chain's stationary distribution, the
normalised solution of $\boldsymbol{\rho}\mathbf{P} = \boldsymbol{\rho}$.

**Prediction step.** Push the previous posterior through the transition matrix:

$$\xi_{t\mid t-1}(k) \;=\; \sum_{j=1}^{K} P_{jk}\, \xi_{t-1\mid t-1}(j)$$

**Update step.** Apply Bayes' rule with the new observation:

$$\xi_{t\mid t}(k) \;=\; \frac{\xi_{t\mid t-1}(k)\,\eta_t(k)}{\sum_{j=1}^{K} \xi_{t\mid t-1}(j)\,\eta_t(j)}$$

**The denominator is the likelihood contribution.** It is the one-step-ahead predictive density,

$$p(r_t \mid r_1,\dots,r_{t-1}, \theta_g) \;=\; \sum_{j=1}^{K} \xi_{t\mid t-1}(j)\,\eta_t(j)$$

so the observed-data log likelihood falls out of the same recursion for free:

$$\log p(\mathbf{r}\mid\theta_g) \;=\; \sum_{t=1}^{T} \log\!\left( \sum_{j=1}^{K} \xi_{t\mid t-1}(j)\,\eta_t(j) \right)$$

This is worth pausing on. The $K^T$ sum of 1.2 has been computed exactly in $O(TK^2)$ operations,
because the Markov property means the past enters only through $\xi_{t-1\mid t-1}$. That is the same
trick as the forward algorithm in Bishop 13.2.2, where $\xi_{t\mid t}$ appears as the scaled forward
variable $\hat\alpha(\mathbf{z}_t)$ of section 13.2.4.

**$\xi_{t\mid t}$ is the object the thesis uses as the gate.** Hold that thought until 1.7.

### 1.5 The E step, part two: the Kim smoother

The filter conditions on data through $t$. EM's $Q$ function needs the posterior given **all** the
data, $\hat\xi_t(k) = \xi_{t\mid T}(k)$. Kim's backward recursion delivers it, running from
$t = T-1$ down to $1$ and starting from $\xi_{T\mid T}$, which the filter already produced:

$$\xi_{t\mid T}(j) \;=\; \xi_{t\mid t}(j) \sum_{k=1}^{K} \frac{P_{jk}\,\xi_{t+1\mid T}(k)}{\xi_{t+1\mid t}(k)}$$

and the pairwise posterior, which the transition matrix update needs:

$$\xi_{t,t+1\mid T}(j,k) \;=\; \frac{\xi_{t\mid t}(j)\, P_{jk}\, \xi_{t+1\mid T}(k)}{\xi_{t+1\mid t}(k)}$$

Every quantity on the right is already available: $\xi_{t\mid t}$ and $\xi_{t+1\mid t}$ from the
forward pass, $\xi_{t+1\mid T}$ from the step just completed.

### 1.6 The M step

Maximise $Q$ over $\theta_g$. Because $Q$ separates into three additive blocks, each parameter has a
closed form. The transition matrix, subject to rows summing to one, uses a Lagrange multiplier and
yields the intuitive ratio of expected counts:

$$\hat P_{jk} \;=\; \frac{\sum_{t=2}^{T} \xi_{t-1,t\mid T}(j,k)}{\sum_{t=2}^{T} \xi_{t-1\mid T}(j)}$$

Read it as: expected number of transitions from $j$ to $k$, divided by expected number of visits to
$j$. The emission parameters are responsibility-weighted sample moments:

$$\hat\mu_k \;=\; \frac{\sum_{t=1}^{T} \xi_{t\mid T}(k)\, r_t}{\sum_{t=1}^{T} \xi_{t\mid T}(k)}, \qquad \hat\sigma_k^2 \;=\; \frac{\sum_{t=1}^{T} \xi_{t\mid T}(k)\,(r_t - \hat\mu_k)^2}{\sum_{t=1}^{T} \xi_{t\mid T}(k)}$$

and $\hat\rho_1(k) = \xi_{1\mid T}(k)$.

The denominator $\sum_t \xi_{t\mid T}(k)$ is the effective number of observations assigned to regime
$k$. It is the exact analogue of $N_k$ in Bishop's Gaussian mixture, and it arises the same way: it
is what appears when $\hat\mu_k$ is factored out of the sum in the first-order condition. It does not
cancel.

Iterate E and M to convergence of the log likelihood from 1.4.

### 1.7 Filtered, smoothed, and the one distinction that decides whether the thesis is valid

Two different quantities have appeared, and confusing them is the most common fatal error in
regime-switching backtests.

| | Conditions on | Used for |
|---|---|---|
| $\xi_{t\mid t}(k)$, **filtered** | $r_1,\dots,r_t$ | **the gate**, at every date, train and test |
| $\xi_{t\mid T}(k)$, **smoothed** | $r_1,\dots,r_T$, the whole sample | **only** inside EM, to estimate $\theta_g$ |

Using the smoothed series inside EM is legitimate. It is in-sample parameter estimation on the
training block, the parameters are then frozen, and no test observation ever enters it.

Using the smoothed series **as the gate** is look-ahead: $\xi_{t\mid T}$ knows what happened after
$t$, so a model gated on it is told in 2007 that a crisis is coming. Results look excellent and mean
nothing.

Two practical consequences.

1. `statsmodels` exposes both as `filtered_marginal_probabilities` and
   `smoothed_marginal_probabilities`. The smoothed one is the more natural-looking name and is the
   one most people reach for.
2. **The filtered series is used on training dates too**, not only on test dates. If the experts
   were trained against smoothed weights and applied to filtered ones, the model would see a
   different kind of input at test time than it learned from. Consistency of the input distribution
   matters more here than using the sharper estimate.

Define, once and for all,

$$\pi_k(t) \;\equiv\; \xi_{t\mid t}(k)$$

**Superseded 2026-10-05 (Advisor_Questions.md, Q27).** The gate weight for an h-day target is the
average of the h-step-ahead probabilities, built from the filtered probability and the transition matrix:

$$\pi_k(t) \;\equiv\; \bar w_k(t) \;=\; \frac{1}{h}\sum_{j=1}^{h}\big[\xi_{t\mid t}^{\top}\mathbf P^{\,j}\big]_k, \qquad h = 5$$

It is still built from the filtered series, never the smoothed one, and is used on training and test
dates alike, so everything above about look-ahead and consistency still applies. It is exact for the
expected 5-day return; the filtered probability alone assumes the regime never changes inside the window.

### 1.8 Freezing, and application beyond the training block

After EM converges on the training block, $\hat\theta_g$ is frozen. To obtain $\pi_k(t)$ for a date
$t$ beyond the training block, run **only the forward recursion of 1.4** forward through the new
observations using $\hat\theta_g$. Do not re-run EM.

This is legal because the filter at $t$ conditions only on $r_1,\dots,r_t$. Re-running EM over the
extended series would not be, because the M step's smoothed weights would then be computed using
observations after $t$.

### 1.9 Two properties worth reporting

**Expected duration.** Time in regime $k$ before leaving is geometric, so

$$\mathbb{E}[\text{duration in } k] \;=\; \frac{1}{1 - \hat P_{kk}}$$

This is the quantity the persistence argument of the thesis is about, and it is one line from a
fitted transition matrix.

**Stationary distribution.** The normalised left eigenvector of $\hat{\mathbf{P}}$ for eigenvalue 1,
the long-run share of time in each regime. A fitted chain whose stationary distribution is extremely
lopsided has effectively found fewer than $K$ regimes.

### 1.10 Two things that go wrong

**Local optima.** The likelihood is multimodal. EM converges to a local maximum that depends on the
starting values, so multiple random starts are run and the best converged likelihood kept. That is a
selection over candidates and enters the multiple-testing accounting like any other.

**Label switching.** The likelihood is invariant to permuting the state labels: relabelling regimes
and permuting $\mathbf{P}$, $\mu$, $\sigma$ accordingly gives an identical likelihood. So "regime 1"
in one walk-forward fold need not be "regime 1" in the next. A canonical ordering must be imposed
after every fit, before anything is compared or pooled across folds. Sorting by fitted $\sigma_k$
ascending, so regime 0 is always the calmest, is the declared rule.

---

## Stage 2. The base network, fitted by backpropagation

### 2.1 Architecture

A multilayer perceptron, that is a stack of affine maps each followed by an elementwise nonlinear
function. With $L$ layers, writing $a^{(0)} = x \in \mathbb{R}^d$:

$$z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}, \qquad a^{(\ell)} = h\big(z^{(\ell)}\big), \qquad \ell = 1,\dots,L-1$$

$$f_0(x;\phi) \;=\; z^{(L)} \;=\; W^{(L)} a^{(L-1)} + b^{(L)} \in \mathbb{R}$$

with $h$ the activation, typically $\mathrm{ReLU}(u) = \max(0,u)$, and $\phi = \{W^{(\ell)},
b^{(\ell)}\}_{\ell=1}^{L}$. The output layer is affine because the target is a real-valued return.

The nonlinearity is not decoration. Without it the composition of affine maps is a single affine
map, and the network could represent nothing a linear regression could not.

### 2.2 Objective

The base is a standalone predictor of $y$. Fitted on the training block by minimising a
per-observation loss, averaged over the panel:

$$\mathcal{L}_{\text{base}}(\phi) \;=\; \frac{1}{M}\sum_{(i,t)\in\mathcal{D}_{\text{train}}} \ell\big(y_{i,t},\, f_0(x_{i,t};\phi)\big) \;+\; \lambda \|\phi\|_2^2$$

with $M$ the number of training observations. Squared error $\ell(y,\hat y) = (y-\hat y)^2$ is the
default; the Huber loss, quadratic near zero and linear in the tails, is the robust alternative that
Gu, Kelly and Xiu prefer because returns are fat-tailed and a handful of extreme observations
otherwise dominate the gradient.

### 2.3 Backpropagation

Backpropagation is the chain rule organised so that each partial derivative is computed once.
Define the **error signal** at layer $\ell$ as the derivative of the loss with respect to that
layer's pre-activation:

$$\delta^{(\ell)} \;\equiv\; \frac{\partial \ell}{\partial z^{(\ell)}}$$

**Output layer.** For squared error on a scalar output,

$$\delta^{(L)} \;=\; \frac{\partial}{\partial z^{(L)}} \big(y - z^{(L)}\big)^2 \;=\; -2\big(y - f_0(x)\big)$$

**Backward recursion.** The pre-activation $z^{(\ell)}$ influences the loss only through
$z^{(\ell+1)} = W^{(\ell+1)} h(z^{(\ell)}) + b^{(\ell+1)}$, so by the chain rule

$$\delta^{(\ell)} \;=\; \Big( W^{(\ell+1)\top} \delta^{(\ell+1)} \Big) \odot h'\big(z^{(\ell)}\big)$$

with $\odot$ the elementwise product. Two operations per layer: multiply by the transpose of the
next layer's weights, then gate by the local derivative of the activation. For ReLU, $h'(u)$ is 1
where $u>0$ and 0 elsewhere, so the second operation simply zeroes the entries of the error signal
belonging to units that were inactive on the forward pass.

**Parameter gradients.** Since $\partial z^{(\ell)}/\partial W^{(\ell)}$ picks out $a^{(\ell-1)}$,

$$\frac{\partial \ell}{\partial W^{(\ell)}} \;=\; \delta^{(\ell)} \, a^{(\ell-1)\top}, \qquad \frac{\partial \ell}{\partial b^{(\ell)}} \;=\; \delta^{(\ell)}$$

Sum these over the minibatch, add the weight decay term $2\lambda W^{(\ell)}$, and hand the result to
the optimiser. Adam is the usual choice: it maintains per-parameter running estimates of the first
and second moments of the gradient and rescales each step by the square root of the second, which
makes it insensitive to the very different scales that different characteristics produce.

### 2.4 Freezing

Once trained, set `requires_grad = False` on every parameter of $\phi$ and put the module in
evaluation mode so that dropout is disabled and any normalisation statistics stop updating. From
here on $f_0$ is a fixed function. **It is fitted once per walk-forward window and seed, and reused
across all four gate arms**, because it never sees regime information. That amortisation is what
makes the full experimental grid affordable.

---

## Stage 3. The experts, fitted by backpropagation

### 3.1 Architecture and initialisation

$K$ multilayer perceptrons $r_k(\cdot;\psi_k)$ of exactly the form in 2.1, each on the same input
$x_{i,t}$, each producing a scalar **correction**.

**The final layer of every expert is initialised to zero**, $W_k^{(L)} = 0$ and $b_k^{(L)} = 0$, so
that at step zero

$$r_k(x;\psi_k) \equiv 0 \quad \forall k \qquad \Longrightarrow \qquad \hat y_{i,t} = f_0(x_{i,t})$$

The model starts exactly at the base. Any measured improvement is therefore measured from zero and
cannot be an artefact of a luckier initialisation.

Two consequences of zero initialisation worth knowing. First, the hidden layers receive no gradient
at step zero, because by 2.3 the error signal propagating back into them is multiplied by
$W_k^{(L)} = 0$; they begin learning one step late, which is harmless. Second, all $K$ experts start
**identical**. What separates them is that the frozen gate hands them different weights. Symmetry is
broken by the gate rather than by initialisation, which is elegant and exactly right for this
thesis, but it means that if the gate is near-uniform the experts never differentiate and the
mixture degenerates to the base. That is a diagnosable outcome, not a bug.

### 3.2 The mixture, written two ways

Per-expert predictions:

$$\mu_{i,t,k} \;=\; f_0(x_{i,t}) \;+\; r_k(x_{i,t};\psi_k)$$

Point prediction, using $\sum_k \pi_k(t) = 1$:

$$\hat y_{i,t} \;=\; \sum_{k} \pi_k(t)\,\mu_{i,t,k} \;=\; f_0(x_{i,t}) \sum_k \pi_k(t) \;+\; \sum_k \pi_k(t)\, r_k(x_{i,t}) \;=\; f_0(x_{i,t}) + \sum_k \pi_k(t)\, r_k(x_{i,t})$$

The residual form and the mixture form are the same object. This is why the implementation needs no
special case: carry $\mu_k = f_0 + r_k$ through the ordinary mixture machinery and the residual
structure appears on its own. **The collapse holds for the prediction; it does not hold for the
gradient**, which is the subject of 3.4.

It relies on $\sum_k \pi_k(t) = 1$. That is true of every prior used here, but it is worth asserting
in code rather than assuming, because an unnormalised prior would silently rescale $f_0$.

### 3.3 The two candidate objectives

**Objective A, the mixture negative log likelihood.** Each expert is a Gaussian emission with its own
noise scale $s_k$:

$$\mathcal{N}_{i,t,k} \;=\; \mathcal{N}\big(y_{i,t} \,\big|\, \mu_{i,t,k},\, s_k^2\big), \qquad \mathcal{L}^{\text{NLL}}_{i,t} \;=\; -\log \sum_{k=1}^{K} \pi_k(t)\, \mathcal{N}_{i,t,k}$$

computed in the log domain with a single `logsumexp`, never by exponentiating densities and taking
the log afterwards, which underflows on fat-tailed returns.

**Objective B, squared error on the point prediction.**

$$\mathcal{L}^{\text{SE}}_{i,t} \;=\; \big(y_{i,t} - \hat y_{i,t}\big)^2 \;=\; \Big(y_{i,t} - f_0(x_{i,t}) - \textstyle\sum_k \pi_k(t) r_k(x_{i,t})\Big)^2$$

Note that under B the base contributes a fixed offset, so minimising squared error of the mixture
against $y$ is identical to minimising squared error of $\sum_k \pi_k r_k$ against the residual
$y - f_0$. Under A that equivalence fails, because $s_k$ then describes a different quantity.

**Both objectives carry the correction penalty:**

$$\Omega_{i,t} \;=\; \alpha \Big( \sum_k \pi_k(t)\, r_k(x_{i,t}) \Big)^2$$

which shrinks the mixture toward the base. It penalises the **combined** correction, not each expert
separately.

### 3.4 The gradients, and why the choice of objective is not cosmetic

**Under objective A.** Define the **responsibility**, the posterior probability that observation
$(i,t)$ came from expert $k$:

$$\gamma_{i,t,k} \;=\; \frac{\pi_k(t)\, \mathcal{N}_{i,t,k}}{\sum_{j} \pi_j(t)\, \mathcal{N}_{i,t,j}}$$

Differentiating the NLL with respect to expert $k$'s output, and using
$\partial \mu_{i,t,k} / \partial r_k = 1$:

$$\frac{\partial \mathcal{L}^{\text{NLL}}_{i,t}}{\partial r_k(x_{i,t})} \;=\; -\,\gamma_{i,t,k}\,\frac{y_{i,t} - \mu_{i,t,k}}{s_k^2}$$

**Under objective B:**

$$\frac{\partial \mathcal{L}^{\text{SE}}_{i,t}}{\partial r_k(x_{i,t})} \;=\; -\,2\,\pi_k(t)\,\big(y_{i,t} - \hat y_{i,t}\big)$$

Put them side by side, because the difference is structural rather than a matter of scaling.

| | Weight on expert $k$ | Residual expert $k$ sees |
|---|---|---|
| A, likelihood | $\gamma_{i,t,k}$, the **posterior**, depends on $y_{i,t}$ | its **own** residual $y - \mu_k$ |
| B, squared error | $\pi_k(t)$, the **prior**, from the frozen gate alone | the **shared** residual $y - \hat y$ |

Under B the frozen gate fully determines which expert learns from which observation, and every
expert is pushed to fix the same collective error. Under A an expert that happens to sit close to
the realised $y$ receives more gradient than its gate weight alone would give it, and each expert
chases its own residual, which is what produces specialisation.

Under a **trainable** gate the posterior weighting is the entire justification for the likelihood,
because $\gamma - \pi$ is the gradient that trains the gate. Under the **frozen** gate of this
thesis that justification is gone, and what remains is that the realised target re-weights the
experts after the regime assignment has already been made. Since the purpose of freezing both the
gate and the base is to leave the regime assignment as the only thing varying, this is worth deciding
deliberately rather than by default. It is Q20 and it is open.

**The noise scales, under A.** Differentiating with respect to $\log s_k$:

$$\frac{\partial \mathcal{L}^{\text{NLL}}_{i,t}}{\partial \log s_k} \;=\; \gamma_{i,t,k}\left(1 - \frac{(y_{i,t}-\mu_{i,t,k})^2}{s_k^2}\right)$$

Setting the sum over the sample to zero gives the stationary point

$$s_k^2 \;=\; \frac{\sum_{i,t} \gamma_{i,t,k}\,(y_{i,t}-\mu_{i,t,k})^2}{\sum_{i,t} \gamma_{i,t,k}}$$

the responsibility-weighted mean squared error of expert $k$: exactly the Gaussian-mixture M step of
Bishop 9.2.2, with the same effective count $\sum \gamma$ in the denominator. In practice $s_k$ is
held frozen for a warm-up period, because a scale that is free from step zero collapses toward zero
on whichever expert momentarily fits best.

**The penalty.**

$$\frac{\partial \Omega_{i,t}}{\partial r_k(x_{i,t})} \;=\; 2\alpha\, \pi_k(t) \sum_j \pi_j(t)\, r_j(x_{i,t})$$

Note that it is prior-weighted under both objectives, so under A the total gradient mixes a
posterior-weighted data term with a prior-weighted penalty term. Nothing is wrong with that, but it
means $\alpha$ is not comparable across a change of objective.

**A property at initialisation.** With every $r_k = 0$, all $\mu_{i,t,k}$ equal $f_0(x_{i,t})$, so
all $\mathcal{N}_{i,t,k}$ are equal and

$$\gamma_{i,t,k} \;=\; \pi_k(t)$$

exactly. Posterior equals prior at step zero, and A and B give proportional gradients on the first
step. They diverge only as the experts differentiate.

### 3.5 From the output gradient into the network

Everything above produces $\partial \mathcal{L}/\partial r_k(x_{i,t})$, a scalar per expert per
observation. That is precisely the output-layer error signal for expert $k$'s network:

$$\delta_k^{(L)} \;=\; \frac{\partial \mathcal{L}}{\partial r_k(x_{i,t})}$$

and the backward recursion of 2.3 runs unchanged from there, inside each expert independently. The
experts do not share parameters, so there is no cross-expert term in the backward pass. The gate and
the base contribute no gradient at all, because both are frozen.

The total objective actually minimised is

$$\mathcal{L}(\psi, \{\log s_k\}) \;=\; \frac{1}{M}\sum_{i,t} \Big[ \mathcal{L}_{i,t} \;+\; \alpha\big(\textstyle\sum_k \pi_k(t) r_k(x_{i,t})\big)^2 \Big] \;+\; \lambda\|\psi\|_2^2$$

### 3.6 The load-balancing term is inert here

Architectures with trainable gates add a load-balancing penalty to stop the gate collapsing onto one
expert. It has the form of a product of the mean gate probability and an assignment fraction, and its
gradient flows through the gate.

**Here it has exactly zero gradient.** The mean gate weight $\bar\pi_k$ is computed from a frozen
model with no trainable parameters, so the term is constant with respect to everything being
optimised. It must be removed from the frozen arms or explicitly documented as inert. The contrast
with the architectures where it does work is a legitimate methodological note.

---

## Stage 4. Where EM is and is not, and why they are the same thing

A natural question: the classical mixture of experts is trained by EM, so why is this model trained
by backpropagation?

**EM appears exactly once, in Stage 1**, to estimate the gate's parameters. It is available there
because the latent variable is discrete, the complete-data log likelihood is linear in the state
indicators, and every M step has a closed form.

**EM does not appear in Stage 3**, for two reasons. The experts are neural networks, so the M step
has no closed form; and the gate is frozen, so there is nothing for a responsibility-based update to
re-estimate on the gate side.

But the connection is closer than "we use a different algorithm". Consider the per-observation NLL
and differentiate directly:

$$\frac{\partial}{\partial \psi} \left[ -\log \sum_k \pi_k \mathcal{N}_k(\psi) \right] \;=\; -\frac{\sum_k \pi_k \,\partial \mathcal{N}_k/\partial\psi}{\sum_j \pi_j \mathcal{N}_j} \;=\; -\sum_k \gamma_k \,\frac{\partial \log \mathcal{N}_k}{\partial \psi}$$

Now write the EM auxiliary function for the same model, with responsibilities held fixed at their
current values $\gamma_k^{\text{old}}$:

$$Q(\psi,\psi^{\text{old}}) \;=\; \sum_k \gamma_k^{\text{old}} \log\big(\pi_k \mathcal{N}_k(\psi)\big) \qquad \Longrightarrow \qquad \frac{\partial Q}{\partial \psi} \;=\; \sum_k \gamma_k^{\text{old}} \frac{\partial \log \mathcal{N}_k}{\partial \psi}$$

At $\psi = \psi^{\text{old}}$ the two responsibilities coincide, so

$$\frac{\partial}{\partial\psi}\Big[-\log p(y\mid\psi)\Big] \;=\; -\frac{\partial Q(\psi,\psi^{\text{old}})}{\partial \psi}\bigg|_{\psi=\psi^{\text{old}}}$$

**Gradient descent on the mixture NLL is generalised EM with a gradient M step**, where the E step is
recomputed at every minibatch. Computing $\gamma$ inside the forward pass *is* the E step; taking
one optimiser step *is* a partial M step. The two descriptions are the same algorithm, and the
implementation that fuses them into one `logsumexp` is computing both at once.

This is the honest answer to the question "where is EM in my model". It is in the gate exactly, and
it is in the experts in the generalised sense of Neal and Hinton (1998), who showed that an M step
which merely increases $Q$ rather than maximising it preserves the monotonicity guarantee.

Under objective B there is no EM interpretation at all, because there is no likelihood and no
posterior. That is a further respect in which Q20 is not cosmetic.

---

## Stage 5. The complete algorithm, one walk-forward fold

Let the fold have a training block of dates $\mathcal{T}_{\text{tr}}$ and a test block
$\mathcal{T}_{\text{te}}$, with a purge gap of $h$ dates between them so that no training label was
computed from test-period prices.

**Step 1, the gate.** Run EM (1.4 to 1.6) on $\{r_t : t \in \mathcal{T}_{\text{tr}}\}$ from several
random starts; keep the highest converged likelihood; log every start as a trial. Apply the
canonical ordering of 1.10. Freeze $\hat\theta_g$.

**Step 2, the gate series.** Run the forward filter of 1.4 with the frozen $\hat\theta_g$ across
$\mathcal{T}_{\text{tr}}$ and then, continuing without re-fitting, across $\mathcal{T}_{\text{te}}$.
Store $\pi_k(t)$ for every date. Filtered only. Assert that the parameters used on the test block are
identical to those from the training fit.

**Step 3, the base.** Fit $f_0$ by backpropagation (Stage 2) on the training block, selecting any
early stopping on a tail slice of the **training** block. Freeze. Cache it keyed by
(base config, window, seed) so the other three gate arms reuse it.

**Step 4, the experts.** Initialise every $r_k$ with a zeroed output layer. Minimise the objective of
3.5 by stochastic gradient descent with Adam over training-block minibatches, with $f_0$ and the
gate frozen in the graph. Freeze $s_k$ for a warm-up period.

**Step 5, prediction.** For each test observation, compute $f_0(x_{i,t})$ and $r_k(x_{i,t})$, take
$\pi_k(t)$ from step 2, and form $\hat y_{i,t} = f_0 + \sum_k \pi_k r_k$.

**Step 6, scoring.** Cross-sectional rank correlation between $\hat y_{\cdot,t}$ and $y_{\cdot,t}$
per test date, aggregated into the mean and the information ratio across dates; and the long-short
portfolio formed on $\hat y$. **Reported beside every result: the base's own out-of-sample score**,
since the quantity of interest is the improvement over the base, not the level, and the correction
magnitude $\sum_k \pi_k r_k$, since corrections that are numerically negligible answer the question
in the negative whatever the metric does.

**What changes for the other three gates.** Step 1 and step 2 only. The jump model replaces EM with
an alternating assign-and-fit procedure penalising state changes; TVTP makes $P_{jk}$ a function of
covariates inside the same filter; the Wasserstein gate replaces the probabilistic model entirely
with clustering of segment distributions. All three deliver the same object, a $(T \times K)$ table
of $\pi_k(t)$, and steps 3 to 6 do not change by a single line. That is the comparison the thesis is
making.

---

## Stage 6. Literature

**The gate, in order of usefulness for this derivation.**

1. **Hamilton, J. D. (1994)**, *Time Series Analysis*, Princeton University Press, **Chapter 22**.
   The filter and the likelihood written out completely. Start here.
2. **Kim, C.-J. and Nelson, C. R. (1999)**, *State-Space Models with Regime Switching*, MIT Press,
   **Chapters 4 and 5**. The smoother of 1.5 is Kim's, and this is the one book that treats the
   filtered-versus-smoothed distinction as a modelling decision rather than an implementation
   detail. The most important item on this list for the validity of the thesis.
3. **Hamilton, J. D. (1989)**, *Econometrica* 57(2), 357 to 384. The original, and the source of the
   expected-duration result.
4. **Hamilton, J. D. (1990)**, *Journal of Econometrics* 45(1 to 2), 39 to 70. EM for this model
   specifically: the M-step formulas of 1.6 are derived here.
5. **Kim, C.-J. (1994)**, "Dynamic linear models with Markov-switching," *Journal of Econometrics*
   60(1 to 2), 1 to 22. The smoothing recursion.
6. **Bishop, PRML, sections 13.1 and 13.2**, especially 13.2.2 and 13.2.4. The same algorithms in
   machine-learning notation, which is useful because the thesis has to speak both languages.
   Notation map: Bishop's $\hat\alpha(\mathbf{z}_t)$ is the filtered probability, his
   $\gamma(\mathbf{z}_t)$ is the **smoothed** one, and his $\pi_k$ is the initial state distribution
   here rather than a mixing weight.

**EM in general.**

7. **Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977)**, "Maximum likelihood from incomplete
   data via the EM algorithm," *JRSS-B* 39(1), 1 to 38. The paper that named it.
8. **Bishop, PRML, chapter 9**, especially 9.2 to 9.4. The Gaussian mixture, the general EM
   treatment, and the lower-bound view used in 1.3.
9. **Neal, R. M. and Hinton, G. E. (1998)**, "A view of the EM algorithm that justifies incremental,
   sparse, and other variants," in Jordan (ed.), *Learning in Graphical Models*, Kluwer. The
   justification for the generalised, gradient M step of Stage 4.

**Mixtures of experts.**

10. **Jacobs, R. A., Jordan, M. I., Nowlan, S. J. and Hinton, G. E. (1991)**, "Adaptive Mixtures of
    Local Experts," *Neural Computation* 3(1), 79 to 87. Ten pages, the original, and the source of
    the competitive-learning argument for why a gated mixture trains differently from an ensemble.
11. **Jordan, M. I. and Jacobs, R. A. (1994)**, "Hierarchical Mixtures of Experts and the EM
    Algorithm," *Neural Computation* 6(2), 181 to 214. EM applied to a gated network, the closest
    published relative of Stage 4.
12. **Bishop, PRML, section 14.5**, conditional mixture models. 14.5.1, mixtures of linear regression
    models, is this architecture with the networks replaced by linear maps and is the best possible
    warm-up. 14.5.3 is the textbook mixture of experts, trained jointly by EM, which is what this
    thesis deliberately does not do.
13. **Chen, Z., Deng, Y., Wu, Y., Gu, Q. and Li, Y.**, "Towards Understanding Mixture of Experts in
    Deep Learning," arXiv:2208.02813. The formal result that a mixture beats a single network
    precisely when the problem has cluster structure and the experts are nonlinear. The theoretical
    anchor of the thesis premise, with regimes supplying the cluster structure.
14. **Ye and Borde**, arXiv:2608.12251. The residual architecture, the zero initialisation and the
    correction penalty of Stage 3.

**Neural network training.**

15. **Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986)**, "Learning representations by
    back-propagating errors," *Nature* 323, 533 to 536. The recursion of 2.3.
16. **Goodfellow, I., Bengio, Y. and Courville, A. (2016)**, *Deep Learning*, **chapters 6 and 8**.
    Feed-forward networks and optimisation, including Adam.

**The finance setting.**

17. **Gu, S., Kelly, B. and Xiu, D. (2020)**, "Empirical Asset Pricing via Machine Learning,"
    *Review of Financial Studies* 33(5), 2223 to 2273. The benchmark for the base network, the
    argument for Huber over squared error, and the magnitude of the signal being chased.
18. **Ang, A. and Timmermann, A. (2012)**, "Regime Changes and Financial Markets," *Annual Review of
    Financial Economics* 4, 313 to 337. Why regime switching is applied to returns at all.

**A suggested route.** Hamilton chapter 22 for the filter, then Kim and Nelson chapter 4 for the
filtered-versus-smoothed distinction, then Hamilton (1990) for the M step. That is the whole of
Stage 1 and roughly six hours. Bishop 9.2 to 9.4 and 14.5 cover Stages 3 and 4 and are already
partly read. Rumelhart et al. for Stage 2 is an afternoon. Everything else is reference.
