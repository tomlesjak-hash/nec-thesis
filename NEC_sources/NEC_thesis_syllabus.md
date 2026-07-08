# Personal Thesis Syllabus

## Machine Learning for Regime-Conditional Return Prediction (NEC / MoE)

**Prepared for:** Tom Koreslesjak  
**Purpose:** personal learning plan before proposal and thesis execution  
**Proposal target:** September 2026  
**First full draft target:** January 2027  
**Created:** June 2026  

---

## 1. Thesis Direction

The thesis should be framed as **machine learning work with finance as the empirical domain**, not as a pure finance thesis. The finance motivation is regime-dependent return predictability; the ML contribution is the design, correction, training, and interpretation of mixture-of-experts models for non-stationary financial prediction.

The current NEC code in `OP model/nec/nec_hybrid_explained.py` is a useful starting point because it is clearly a rough mixture-of-experts idea:

- `C_L`: an LSTM sequence encoder.
- `C_F`: a 3-class softmax classifier head.
- `NEC_MLP`: a feedforward regression expert.
- `NEC`: a wrapper that hard-routes predictions to either an extreme expert or a normal expert.

The key defect is central to the thesis: the classifier produces three classes, but the routing rule only checks whether `argmax(prob) == 1`. Classes 0 and 2 are collapsed into the same branch. So the model computes a 3-way gating distribution and then uses only a binary decision. It also uses hard `argmax` routing, so regression-loss gradients do not naturally train the classifier gate end-to-end.

The corrected thesis version should turn this into a principled MoE:

```text
p(z = k | x) = softmax(g_k(x))
y_hat = sum_k p(z = k | x) f_k(x)
```

where `g_k` is the gate and `f_k` are expert predictors. The thesis can then compare soft routing, hard routing, sparse/top-k routing, and classical regime-switching baselines.

---

## 2. Critical Framing: What Is True, What Needs Care

### Strong part of the argument

Regime-switching is a serious and recognizable finance tradition. Hamilton-style Markov-switching models are canonical; asset-pricing and portfolio-choice papers often ask whether return dynamics, volatility, risk premia, or factor payoffs change across latent market states. This gives the thesis a finance motivation that a committee can understand.

The clean thesis pitch is:

> Classical regime-switching models represent state-dependent return dynamics with latent Markov states and relatively restrictive observation models. Modern MoE models replace the restrictive expert models with flexible neural experts and replace or augment Markov gating with feature-conditioned gating. This thesis tests whether modern MoE methods improve regime-conditional return prediction and whether learned expert assignments correspond to recognizable market regimes.

### Point to avoid overstating

MoE is not automatically a full generalization of Hamilton. Hamilton has a latent state with **Markov persistence**. A standard MoE gate is usually **feature-conditioned** and has no explicit transition matrix. The defensible claim is:

- MoE generalizes the **expert / observation model** side.
- MoE relaxes or replaces the **gating mechanism** with feature-conditioned routing.
- A hybrid model can add Markov structure back if needed.

That distinction should appear in the thesis and proposal.

### Interpretability hypothesis

It is plausible that learned experts correspond to known regimes such as high-volatility, low-volatility, bull, bear, recession, expansion, or risk-on/risk-off. But this must be tested, not assumed. Experts may also split by size, liquidity, sector, missingness, filing frequency, or artifacts in the data.

The thesis should include explicit gate diagnostics:

- Expert utilization over time.
- Gate entropy and collapse checks.
- Correlation with VIX or realized market volatility.
- Alignment with NBER recession dates, if macro data is used.
- Expert exposure to market beta, size, sector, liquidity, and volatility.
- Stability of expert behavior across walk-forward folds.

---

## 3. Data Recommendation

### What NEC actually needs

NEC does not need "SEC data" specifically. It needs a supervised panel:

```text
(date, ticker, feature history) -> future return target
```

The existing code expects a sequence tensor:

```text
x shape = (batch_size, sequence_length, input_dim)
```

The LSTM gate reads recent history. The regressors use last-step features. So the natural dataset is a **cross-sectional stock panel** with a rolling history of price/volume/factor features and a future-return label.

### Best first dataset: price-first, free data

For a quant-trading path, the most relevant first version should be price-first:

- Universe: large, liquid U.S. equities.
- Frequency: daily.
- History: ideally 10+ years if available.
- Features: returns, volatility, volume, momentum, reversal, drawdown, market-relative features.
- Target: future 5-day, 10-day, or 20-day return, preferably cross-sectionally ranked.

This is better than starting with SEC fundamentals because it gives enough observations for ML and aligns more directly with trading-model evaluation. SEC fundamentals update quarterly or annually, so they are useful but sparse.

### Recommended free data stack

Use a staged data plan:

1. **Stage A: synthetic OP pipeline data**
   - Purpose: verify the model, training loop, gate behavior, and metrics.
   - Use before any real-data claim.

2. **Stage B: free daily price data**
   - Use Stooq or another free daily-price source for U.S. equities.
   - Use `yfinance` only as a prototyping fallback because it is unofficial.
   - Include SPY or broad-market ETF data for market-state features.

3. **Stage C: free regime/context data**
   - VIX history for volatility regimes.
   - Kenneth French factor data for market, size, value, profitability, investment, momentum context.
   - FRED macro series and recession indicators if needed.

4. **Stage D: SEC fundamentals extension**
   - Use SEC EDGAR / companyfacts for quarterly and annual fundamentals.
   - Add features such as revenue growth, earnings growth, leverage, book-to-market proxy, profitability, accruals, shares outstanding changes.
   - Align by filing acceptance date, not fiscal period end, to avoid lookahead.

### Why SEC should not be the first price source

SEC filings are not a clean stock-price database. SEC is valuable for fundamentals and filing metadata. Stock prices should come from a market-data source. If the thesis says "SEC data", it should mean SEC-derived fundamentals, not the main source of price history.

### Initial feature set

Start with features that are simple, defensible, and common in quant work:

- Asset returns: 1-day, 5-day, 20-day, 60-day log returns.
- Reversal: short-horizon return reversal, e.g. negative 1-day or 5-day return.
- Momentum: 20-day, 60-day, 120-day cumulative return.
- Realized volatility: rolling 5-day, 20-day, 60-day standard deviation.
- Downside volatility: rolling standard deviation of negative returns.
- Drawdown: distance from recent rolling high.
- Volume features: dollar volume, volume change, rolling volume z-score.
- Market-relative features: asset return minus SPY return, asset volatility divided by market volatility.
- Cross-sectional ranks: rank-normalized versions of key features each date.

### Initial target

Use one main target and keep it stable:

```text
y_{i,t} = log(P_{i,t+h} / P_{i,t})
```

where `h` is 5, 10, or 20 trading days. For a thesis, 5-day and 20-day are the most reasonable starting horizons.

For evaluation, focus on:

- Cross-sectional rank-IC by date.
- IC information ratio.
- Decile long-short return.
- Turnover and transaction-cost drag.
- Sharpe only after costs and with skepticism.

Later, use residual returns or market-neutral returns:

```text
residual_return = asset_return - beta * market_return
```

This prevents the model from just learning market direction.

---

## 4. Books and Core Resources

### Core textbooks

Each entry is tagged as either **Primary** (read the listed chapters carefully) or **Reference** (consult only when the listed topics come up).

1. Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*. — **Primary**
   - Read: Ch 2 (already done), Ch 3, Ch 4, Ch 7, Ch 8 (§8.5 especially).
   - Main use: statistical learning theory, linear models, regularization, model assessment, EM.
   - Local file: `Books/The Elements of Statistical Learning.pdf`.

2. Bishop, *Pattern Recognition and Machine Learning*. — **Primary**
   - Read: Ch 1 (§1.6 information theory), Ch 2 (§2.3 Gaussians), Ch 3 (optional reinforcement), Ch 9 (mixture models and EM — central), Ch 10 (§10.1–10.3 variational inference), Ch 13 (§13.1–13.3 HMMs).
   - Main use: probability, Gaussian models, mixture models, EM, variational inference, graphical models, HMMs.

3. Prince, *Understanding Deep Learning*. — **Primary**
   - Read: Ch 1–8 (deep-learning foundations), Ch 12 (RNN, LSTM, GRU).
   - Skip: Ch 9 (CNNs), Ch 10 (transformers) unless added as later baselines.
   - Main use: modern deep-learning foundations and PyTorch-compatible intuition.

4. Goodfellow, Bengio, Courville, *Deep Learning*. — **Reference**
   - Consult only for: Ch 8 (deeper optimization theory: convergence, second-order methods, batch normalization); Ch 7 (additional regularization perspectives: adversarial training, data augmentation).
   - Not read cover-to-relevant-chapters. Prince is the primary deep-learning reference.
   - Local file: `Books/Deep+Learning+Ian+Goodfellow.pdf`.

5. Lopez de Prado, *Advances in Financial Machine Learning*. — **Primary**
   - Read: Ch 2 (financial data structures), Ch 3 (labeling), Ch 4 (sample weights), Ch 7 (cross-validation in finance).
   - Main use: purged validation, embargoing, backtest overfitting, financial ML methodology.
   - Local file: `Books/advances-in-financial-machine-learning-1nbsped-9781119482109-9781119482116-9781119482086.pdf`.

6. Tsay, *Analysis of Financial Time Series*. — **Primary**
   - Read: Ch 4 (Markov-switching models — especially §4.6). Skim Ch 2–3 (AR/ARMA/GARCH) as background.
   - Main use: financial time-series basics, volatility, Markov switching.

7. Campbell, Lo, MacKinlay, *The Econometrics of Financial Markets*. — **Reference / selective**
   - Read: Ch 2 (§2.1–2.4 predictability of returns), Ch 5 (§5.1–5.3 present-value relations).
   - Not read cover-to-cover; selective for finance framing in the proposal and literature review.

8. Bali, Engle, Murray, *Empirical Asset Pricing*. — **Primary**
   - Read: Ch 1 (§1.1–1.5 introduction), Ch 3 (§3.1–3.4 portfolio sorts), Ch 8 (§8.1–8.5 cross-sectional regressions and Fama-MacBeth).
   - Main use: cross-sectional return prediction, portfolio sorting methodology, factor context.

### Papers to know before proposal

- Hamilton (1989), Markov-switching model.
- Jacobs, Jordan, Nowlan, Hinton (1991), adaptive mixtures of local experts.
- Shazeer et al. (2017), sparsely-gated MoE and load balancing.
- Fedus, Zoph, Shazeer (2022), Switch Transformer.
- Ang and Bekaert (2002), regime switches in interest rates.
- Daniel and Moskowitz (2016), momentum crashes.
- Bailey and Lopez de Prado (2014), deflated Sharpe ratio.

---

## 5. Learning Plan Overview

The plan is split into two tracks:

1. **Proposal-readiness track.** Enough theory, code, and finance context to write a serious proposal by September 2026.
2. **Thesis-execution track.** Deeper material and empirical work through January 2027.

This is intentionally not locked to a fixed weekly hour count. If you have more time, move faster. If you have less time, preserve the milestone order.

Exercise numbers refer to the standard editions of the listed books. If your edition differs, use the closest exercise in the same numbered section.

---

## 6. Part I: Programming Foundations

Part I is the programming literacy phase. The goal is fluency with Python, NumPy, pandas, and basic plotting so that tooling never blocks the rest of the syllabus. All mathematical content is introduced in later parts, in dedicated **Math Foundations** sections placed immediately before the modules that need that math — this keeps math grounded in upcoming applications rather than crammed all at the start.

### Module 1: Python, NumPy, pandas, and reproducible notebooks

**Goal.** Be able to load data, build a panel, compute features, train simple models, and make plots without fighting the tools.

**Topics.**

- NumPy arrays, broadcasting, vectorization.
- pandas indexing, groupby, rolling windows, joins, missing data.
- Matplotlib and seaborn for diagnostics.
- Scikit-learn estimators, pipelines, train/test splits.
- Jupyter notebooks versus scripts.
- Reproducibility: random seeds, file paths, config objects.

**Readings by section number.**

- VanderPlas, *Python Data Science Handbook*, Ch 2.1-2.7 for NumPy.
- VanderPlas, Ch 3.1-3.11 for pandas.
- VanderPlas, Ch 4.1-4.15 for plotting.
- Scikit-learn user guide, sections 1.1, 3.1, 3.3, and 6.1.

**Focused exercise set.**

- VanderPlas Ch 2 notebook checks: reproduce the array slicing, broadcasting, and aggregation examples.
- VanderPlas Ch 3 notebook checks: reproduce the groupby, merge, and time-series indexing examples.
- Applied: build a `(date, ticker, features, target)` panel from daily prices.
- Applied: recreate the OP pipeline's rank-IC calculation on synthetic data.

**Checkpoint.**

You can reproduce the data -> features -> labels -> model -> metrics pipeline on synthetic data.

---

## 7. Part II: Linear Models and Classical ML

Part II begins with two **Math Foundations** sections (A and B) that teach the linear algebra and probability needed for everything downstream. The two ML modules that follow (Module 4 on linear models and Module 5 on model assessment) then use this math throughout. Subsequent parts (III–V) will each introduce only the *new* math needed for their modules in smaller, focused Math Foundations sections.

### Math Foundations A: Linear algebra essentials

**Goal.** Understand the linear algebra behind regression, PCA, embeddings, neural-network layers, and matrix calculus.

**Recommended ordering within this module.**

The module is best worked in four ordered passes, not in parallel. SVD is the pivot — almost everything downstream becomes clearer once SVD is in your toolkit, and in particular ESL's ridge regression material reads as just "regularized least squares" without SVD but as "principled shrinkage of noise-dominated singular directions" with it.

1. **Strang foundations.** Work through Strang §2.3, 3.2–3.4, 5.1–5.3, 6.1–6.3 (vector spaces, projections, orthogonality, least squares, eigenvalues and eigendecomposition). Do the Strang exercises from the focused exercise set as you go. This builds the prerequisites for SVD; you cannot do SVD cleanly without eigendecomposition first. The matrix-calculus identities you need for Modules 4 and 5 (gradient of a quadratic form, gradient of a norm) are derived in the focused exercise set below; the more general matrix-calculus reference (Petersen & Pedersen's *Matrix Cookbook*) is introduced later in Math Foundations C, where the deeper Jacobian and tensor-chain-rule machinery is actually load-bearing for neural networks.
2. **PCA and SVD focus block** (the dedicated subsection below). Take this as one unit in this position. The 3Blue1Brown videos first, then Strang's SVD chapter, then ESL §14.5, then Bishop §12.1–12.2. Do the focus-block exercises here.
3. **ESL Chapter 3 (§3.2–3.4).** With SVD now in your toolkit, work through linear regression, ridge, and lasso. The ridge derivation in the focused exercise set is the natural capstone here — at this point you should be able to derive ridge in closed form and immediately interpret the result as SVD-basis shrinkage.
4. **Remaining focused exercises and the Module 2 checkpoint.** ESL Ch 3 problems, the MAP / Gaussian-prior view of ridge, and the overall module checkpoint to confirm everything is in place.

**Readings by section number.**

- Strang, *Linear Algebra and Its Applications*, Sec 2.3, 3.2-3.4, 5.1-5.3, 6.1-6.3 (Step 1), plus the SVD chapter (Step 2, typically §6.7 or Ch 7 depending on edition).
- ESL, Ch 3.2-3.4 as the applied regression use case for linear algebra (Step 3).
- ESL §14.5 and Bishop §12.1–12.2 (Step 2, inside the PCA/SVD focus block).

The Matrix Cookbook is NOT a reading here. The Strang chapters above give you the matrix-calculus identities Modules 4 and 5 actually require (gradient of a quadratic form, projection geometry of least squares). The full Matrix Cookbook is introduced in Math Foundations C, where the Jacobian and tensor-chain-rule content becomes load-bearing for neural networks.

**Topics.**

- Vector spaces, bases, rank, null space.
- Projections and least squares.
- Orthogonality and QR.
- Eigenvalues and eigendecomposition.
- SVD and PCA.
- Positive semidefinite matrices.
- Norms and conditioning.
- Matrix calculus: gradients of quadratic forms, traces, matrix-vector products.
- Jacobian of softmax.

**Focused exercise set.**

*General linear algebra (Step 1 readings).*

- Strang Sec 3.3: Exercises 1, 3, and 7.
- Strang Sec 6.3: Exercises 1 and 3.
- Analytical: derive Gram–Schmidt orthogonalization for a 3-column matrix. Use it to derive the QR decomposition $A = QR$ and show that the columns of $Q$ form an orthonormal basis for the column space of $A$.
- Analytical: prove that a symmetric matrix $A$ is positive semidefinite if and only if all its eigenvalues are non-negative. Then show that for any matrix $X$, the Gram matrix $X^\top X$ is always PSD. (You will use this fact constantly: it's why ridge regression's $X^\top X + \lambda I$ is always invertible for $\lambda > 0$.)
- Analytical: derive the matrix-calculus identities $\nabla_x (a^\top x) = a$, $\nabla_x (x^\top A x) = (A + A^\top) x$ (and specialize to symmetric $A$ to get $2 A x$), and $\nabla_A \text{tr}(A^\top B) = B$. These are the workhorse identities you will use to derive almost every ML closed-form solution.
- Analytical: derive the Jacobian of the softmax function $\hat{p}_k = e^{z_k} / \sum_j e^{z_j}$ with respect to $z$. Show $\partial \hat{p}_k / \partial z_j = \hat{p}_k (\delta_{kj} - \hat{p}_j)$, so the Jacobian is $\text{diag}(\hat{p}) - \hat{p}\hat{p}^\top$. (This Jacobian reappears as the Hessian of cross-entropy in Module 4.)
- Analytical: define the condition number of a matrix as $\kappa(A) = \sigma_{\max}/\sigma_{\min}$. Show that for ridge regression, the condition number of $X^\top X + \lambda I$ is $(\sigma_{\max}^2 + \lambda)/(\sigma_{\min}^2 + \lambda)$, which approaches 1 as $\lambda$ grows. This is the precise numerical-stability argument for regularization.

*Linear regression and ridge (Step 3 readings).*

- ESL Ch 3: Exercises 3.4 and 3.6.
- Analytical: derive the ridge regression closed-form solution $\hat{\beta} = (X^\top X + \lambda I)^{-1} X^\top y$ from the regularized objective. Then, using the SVD $X = U \Sigma V^\top$, show that ridge shrinks each singular direction by the factor $\sigma_j^2 / (\sigma_j^2 + \lambda)$. This makes the SVD basis the natural "eigenbasis of regularization."
- Analytical: derive ridge regression as MAP estimation under a Gaussian prior on $\beta$; identify exactly which prior variance corresponds to a given $\lambda$.
- Analytical: show that the OLS solution $\hat{\beta} = (X^\top X)^{-1} X^\top y$ is the orthogonal projection of $y$ onto the column space of $X$. Use this to derive the hat matrix $H = X(X^\top X)^{-1} X^\top$ and verify it is symmetric and idempotent ($H^2 = H$).

**PCA and SVD focus block (3–4 days of dedicated work).**

You have not previously encountered PCA or SVD, so treat this as its own learning thread within Module 2. The underlying math is the eigendecomposition you already know from quantum mechanics (energy eigenstates of a Hermitian Hamiltonian, $H = U \Lambda U^\dagger$), but the applied technique, the statistical interpretation, and the data-side intuition all need their own time.

*Step 1: Intuition first (before any reading).* Watch 3Blue1Brown's *Essence of Linear Algebra* series on YouTube — specifically the eigenvalues/eigenvectors video and the SVD video ("But what is the SVD?"). Twenty minutes total, exceptionally good visual intuition for what these decompositions actually do to a point cloud and to a linear map.

*Step 2: Theory.*

- Strang, *Linear Algebra and Its Applications*, the SVD chapter (typically §6.7 or Ch 7 depending on edition). Strang's treatment is direct and connects SVD to the four fundamental subspaces.
- ESL §14.5 (Principal Components, Curves, and Surfaces). PCA in the unsupervised-learning context: when it works, when it fails, relation to factor analysis.
- Bishop Ch 12 §12.1–12.2 (PCA and probabilistic PCA). PPCA recasts PCA as a maximum-likelihood estimator under a Gaussian latent-variable model — the conceptual bridge to the mixture models you will see in Module 8.

*Topics specific to this block.*

- SVD as the general decomposition $A = U \Sigma V^\top$ for any matrix.
- Relationship between SVD and eigendecomposition: they coincide for symmetric positive semidefinite matrices, differ otherwise.
- PCA as eigendecomposition of the sample covariance matrix $\hat{\Sigma} = X^\top X / n$.
- PCA via SVD of the centered data matrix (numerically preferred).
- Explained variance ratio and choosing the number of components.
- Probabilistic PCA: PCA as maximum-likelihood estimation under a Gaussian latent-variable model with isotropic noise.
- Connection to statistical factor models in finance (PCA-derived factors vs. observed Fama-French factors; Connor and Korajczyk's APT-PCA tradition).
- Why PCA can mislead in finance: signal-to-noise issues, non-stationarity of return covariances, and the random-matrix-theory cleaning argument (you'll come back to this if you use spectral methods later).

*Focused exercises for the PCA/SVD block.*

- By hand, compute the SVD of a 2×2 or 3×3 matrix (e.g., $\begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ or another tractable example). Verify $A = U \Sigma V^\top$ explicitly by multiplication.
- Analytical: show that for a symmetric positive semidefinite matrix $A$, the SVD reduces to eigendecomposition with $U = V$ and $\Sigma = \Lambda$ (the diagonal matrix of eigenvalues).
- Analytical: derive PCA from a variance-maximization argument. Show that the first principal direction $v_1 = \arg\max_{\|v\|=1} \text{Var}(X v)$ is the top eigenvector of the sample covariance matrix, and that subsequent directions are obtained by deflation.
- Analytical: prove the equivalence of (a) PCA via eigendecomposition of $X^\top X / n$ and (b) PCA via SVD of the centered data matrix $X$. Show explicitly the relationship between the singular values and the eigenvalues.
- Analytical: derive the maximum-likelihood solution of probabilistic PCA. Show that in the zero-noise limit $\sigma^2 \to 0$, the PPCA solution recovers the standard PCA principal directions.
- Connect to finance, on paper: explain why for a typical U.S. equity return matrix the first principal component is approximately the market factor — i.e., a roughly equal-weighted average of stocks. Sketch what the second and third components typically capture (size and value-ish dimensions). This is an interpretation exercise, no computation needed.

*Checkpoint for the PCA/SVD block.*

You can: explain SVD in one sentence; describe when SVD and eigendecomposition coincide and when they differ; compute PCA two ways (eigendecomposition of covariance vs. SVD of centered data) and verify they agree; interpret the explained variance ratio; explain PPCA as a latent-variable model; explain in plain language why the first principal component of a U.S. equity return matrix is "the market."

**Module 2 overall checkpoint.**

You can explain why ridge regression is a linear smoother and why SVD/PCA matter for noisy financial features. You are fluent enough with eigendecomposition, SVD, and PCA that they no longer require lookup — they are tools you reach for naturally when reasoning about covariance structure, factor models, dimensionality reduction, and weight-matrix analysis.

### Math Foundations B: Probability and statistics essentials

**Goal.** Learn the probability language needed for mixture models, likelihoods, gates, and uncertainty.

**Readings by section number.**

- ESL Ch 2.1-2.9.
- Bishop Ch 1.1-1.6.
- Bishop Ch 2.1-2.3.

**Topics.**

- Random variables, expectation, variance, covariance.
- Conditional probability and Bayes rule.
- Conditional expectation.
- Gaussian distributions, multivariate Gaussians.
- Maximum likelihood and MAP.
- Bias, variance, irreducible error.
- Law of large numbers and central limit theorem intuition.
- Bootstrap and sampling variability.
- Heavy tails and why financial returns are not Gaussian.
- KL divergence, entropy, cross entropy, mutual information.

**Focused exercise set.**

*Textbook problems.*

- ESL Ch 2: Exercises 2.1, 2.3, 2.5, and 2.8.
- Bishop Ch 1: Exercises 1.10 and 1.11.
- Bishop Ch 2: Exercise 2.20.

*MLE and Gaussians.*

- Analytical: derive the MLE for the mean and variance of a univariate Gaussian from $n$ i.i.d. samples. Show that the MLE for variance is $\hat{\sigma}^2_{\text{MLE}} = \tfrac{1}{n} \sum_i (x_i - \bar{x})^2$ and that this estimator is *biased* — specifically, $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \tfrac{n-1}{n} \sigma^2$. Derive the unbiased correction (Bessel's correction).
- Analytical: derive the conditional distribution of a multivariate Gaussian. Given $X = (X_1, X_2) \sim \mathcal{N}(\mu, \Sigma)$ with block structure, derive the conditional distribution $p(X_1 \mid X_2 = x_2)$ in closed form — both the conditional mean (the Schur complement formula) and the conditional covariance. (This formula appears constantly: it's how Kalman filters and Gaussian processes condition on observations.)
- Analytical: derive the differential entropy $h(\mathcal{N}(\mu, \sigma^2))$ of a univariate Gaussian. Then derive the maximum-entropy principle: show that among all distributions on $\mathbb{R}$ with fixed mean $\mu$ and variance $\sigma^2$, the Gaussian uniquely maximizes the differential entropy. This is the deep reason Gaussians appear everywhere in statistics.

*Bias-variance, high dimensions, KL.*

- Analytical: derive the bias-variance decomposition for the expected squared error of a generic estimator $\hat{f}(x)$ of a target $y = f(x) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$ and $\text{Var}(\varepsilon) = \sigma^2$. Identify the three additive terms (irreducible noise, squared bias, variance) and verify the cross-terms vanish.
- Analytical: prove the distance-concentration phenomenon in high dimensions. For two independent uniform random vectors in $[0,1]^d$, derive $\mathbb{E}\|X - Y\|^2$ and $\text{Var}(\|X - Y\|^2)$ as functions of $d$, and show that the coefficient of variation $\text{Var}(\|X - Y\|)^{1/2} / \mathbb{E}\|X - Y\|$ shrinks like $1/\sqrt{d}$. Interpret what this means for k-NN in high dimensions.
- Analytical: derive the KL divergence between two univariate Gaussians $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$ in closed form.
- Analytical: prove Jensen's inequality for a convex function $\varphi$: $\mathbb{E}[\varphi(X)] \geq \varphi(\mathbb{E}[X])$. Use it to prove that KL divergence is always non-negative: $\text{KL}(p \| q) \geq 0$, with equality iff $p = q$ almost everywhere. (This is the inequality underneath everything in Module 8's ELBO derivations.)

*Heavy tails (finance-relevant).*

- Analytical: derive the moments of a Student-$t$ distribution with $\nu$ degrees of freedom. Show that the $k$-th moment exists only for $k < \nu$. Use this to explain why empirical financial returns, which look approximately Student-$t$ with $\nu \approx 3$–$5$, have well-defined means and variances but potentially undefined fourth moments (kurtosis).

**Checkpoint.**

You can explain why a MoE model is a conditional mixture model and why its loss contains a log-sum-exp term.

### Module 4: Linear models, regularization, and classification

**Readings by section number.**

- ESL Ch 3.1-3.4, 3.8.
- ESL Ch 4.1-4.4.
- Bishop Ch 3.1-3.3 as optional reinforcement.

**Topics.**

- Linear regression.
- Ridge and lasso.
- Logistic regression.
- Multiclass softmax regression.
- Regularization as constrained optimization.
- Regularization as Bayesian prior.
- Calibration and class probabilities.

**Focused exercise set.**

*Textbook problems.*

- ESL Ch 3: Exercises 3.2, 3.4, and 3.6.
- ESL Ch 4: Exercises 4.2 and 4.4.
- Bishop Ch 3: Exercise 3.3.

*Softmax and cross-entropy (the NEC gate math).*

- Analytical: derive the gradient of the multi-class cross-entropy loss $\mathcal{L} = -\sum_k y_k \log \hat{p}_k$ with respect to the pre-softmax logits $z_k$, where $\hat{p}_k = e^{z_k} / \sum_j e^{z_j}$. Show that the gradient takes the clean form $\partial \mathcal{L} / \partial z_k = \hat{p}_k - y_k$ (the prediction minus the one-hot target).
- Analytical: derive the Hessian of the multi-class cross-entropy with respect to the logits, $\partial^2 \mathcal{L} / \partial z_k \partial z_j$. Show it equals $\text{diag}(\hat{p}) - \hat{p}\hat{p}^\top$ and prove this matrix is positive semidefinite, so the loss is convex in the logits.
- Analytical: show that binary logistic regression is the $K=2$ special case of multi-class softmax regression. Recover the standard logistic sigmoid form $\sigma(z) = 1/(1 + e^{-z})$ from the softmax formula.

*Regularization as constrained optimization and as Bayesian prior.*

- Analytical: write ridge and lasso as constrained optimization problems — minimize the residual sum of squares subject to $\|\beta\|_2^2 \leq t$ (ridge) or $\|\beta\|_1 \leq t$ (lasso) — and derive the KKT conditions. Show that the constrained form is equivalent to the Lagrangian form (unconstrained with penalty $\lambda$) with a one-to-one correspondence between $t$ and $\lambda$.
- Analytical: derive the lasso solution coordinate-wise. For a single coordinate update, derive the soft-thresholding operator $\hat{\beta}_j = \text{sign}(\tilde{\beta}_j) \cdot \max(|\tilde{\beta}_j| - \lambda, 0)$, where $\tilde{\beta}_j$ is the OLS update. Contrast with ridge, which scales every coordinate by $1/(1 + \lambda)$ but never sets coordinates to exactly zero. This is the math underneath "lasso does feature selection, ridge does not."
- Analytical: derive lasso as MAP estimation under a Laplace (double-exponential) prior on $\beta$. Show that the prior $p(\beta) \propto e^{-|\beta|/b}$ leads to the L1 penalty in the negative log-posterior.

**Checkpoint.**

You can derive the softmax gradient and Hessian from scratch and explain why the gradient form $\hat{p} - y$ is what makes softmax regression numerically well-behaved. You can derive both ridge and lasso as Lagrangian solutions to constrained least-squares and as MAP estimators under Gaussian and Laplace priors respectively. You understand the math of the NEC gate at the level needed to debug it later.

### Module 5: Model assessment and financial validation

**Readings by section number.**

- ESL Ch 7.1-7.12.
- Lopez de Prado, AFML Ch 7.1-7.6.
- Bailey and Lopez de Prado (2014), Sec 1-4.

**Topics.**

- Train/validation/test splits.
- Cross-validation and why standard k-fold fails in finance.
- Walk-forward validation.
- Purging and embargoing.
- Lookahead bias.
- Survivorship bias.
- Multiple testing and selection bias.
- IC, rank-IC, ICIR.
- Long-short decile backtests.
- Transaction costs and turnover.

**Focused exercise set.**

*Textbook problems.*

- ESL Ch 7: Exercises 7.1, 7.2, and 7.10.

*Purging, embargoing, and leakage.*

- Analytical: describe the AFML purging and embargo algorithms (§7.4–7.5) in precise pseudocode, then identify and prove the leakage conditions they each eliminate. Construct a small counterexample where naive k-fold without purging leaks information from the test fold into the training fold via label overlap.
- Analytical: characterize survivorship bias quantitatively. Set up a simple two-period model where survivors and non-survivors have systematically different return distributions; derive the bias in the apparent mean return when the analyst observes only the survivor sample. Show the bias is monotone in the survival hazard.

*IC, rank-IC, and ICIR.*

- Analytical: derive the expectation and variance of date-level rank-IC (Spearman correlation between predictions and realized returns) under the null hypothesis of random predictions. Show that for a cross-section of $N$ stocks, the standard error of the daily rank-IC scales like $1/\sqrt{N-1}$.
- Analytical: derive the IC information ratio (ICIR) as a $t$-statistic on the mean of the daily rank-IC time series. Specifically, ICIR $= \bar{\text{IC}} / \text{SE}(\bar{\text{IC}})$ where $\text{SE}(\bar{\text{IC}}) = \sigma_{\text{IC}}/\sqrt{T}$. Show how this relates to the out-of-sample Sharpe ratio of a strategy that scales positions by predicted IC.

*Backtests, costs, and the Sharpe ratio.*

- Analytical: write the decile long-short portfolio return at date $t$ explicitly as a function of (a) the predicted-rank assignment, (b) the realized returns, (c) the long and short leg weights, and (d) the turnover-adjusted transaction cost. Show how turnover enters as the L1 norm of the weight change between consecutive dates.
- Analytical: derive the deflated Sharpe ratio formula from Bailey and López de Prado (2014). Identify which terms correct for selection bias (the maximum over $N$ trials problem) and which correct for non-normality of returns (skewness and kurtosis adjustments).

*Multiple testing (the factor zoo problem).*

- Analytical: derive the Bonferroni correction for multiple hypothesis testing. Show that testing $N$ hypotheses at level $\alpha/N$ each controls the family-wise error rate (FWER) at $\alpha$. Identify the conservatism of this correction when test statistics are correlated.
- Analytical: derive the Benjamini–Hochberg procedure for false discovery rate (FDR) control. State precisely what FDR controls vs. what FWER controls, and explain why FDR is often the more appropriate criterion for screening large numbers of trading signals.
- Analytical: state Harvey, Liu & Zhu's (2016) "and the cross-section of expected returns" multiple-testing critique: if hundreds of factors have been tested in the literature, what $t$-statistic threshold should we require for a newly proposed factor to be considered statistically significant? Derive a back-of-envelope answer using the Bonferroni-style argument.

**Checkpoint.**

You can defend why your evaluation protocol is not leaking future information. You can articulate why multiple-testing corrections matter for your thesis's claim that the corrected NEC beats baselines, and explicitly state the family of tests over which your significance claim is corrected.

---

## 8. Part III: Deep Learning Foundations

Part III builds the neural-network machinery that Module 7 (sequence models) and ultimately the corrected NEC will use. It opens with a focused Math Foundations section on the optimization-theory and matrix-calculus identities that backpropagation, SGD, Adam, initialization theory, and batch normalization rely on. The linear algebra and probability you already have from Part II are still load-bearing here.

### Math Foundations C: Optimization theory and matrix calculus for neural networks

**Goal.** Build the optimization and matrix-calculus machinery you will need to derive backpropagation, analyze gradient descent's convergence, understand momentum and Adam, and derive principled initialization schemes. Most of the underlying calculus you already have; the new content is its application to high-dimensional, non-convex objectives.

**Readings by section number.**

- Boyd & Vandenberghe, *Convex Optimization*, Ch 1 (introduction), §3.1–3.2 (convex functions), §9.1–9.3 (gradient descent and Newton's method). Free PDF at stanford.edu/~boyd/cvxbook.
- Goodfellow Ch 4 (numerical computation: overflow, conditioning, gradient-based optimization) and §8.1–8.3 (challenges in neural network optimization, ill-conditioning, saddle points).
- Petersen & Pedersen, *Matrix Cookbook* — this is the primary reference for the deep-learning matrix calculus you'll do here and in Module 8. Read §2.1 (derivatives of a determinant), §2.4 (derivatives of matrices and vectors), §2.5 (special matrix derivatives), §2.6 (derivatives of traces), §2.7 (derivatives of vector norms). The Cookbook's §8 (Gaussian distributions, with explicit moment formulas) is a forward reference for Math Foundations D and Module 8's variational mean-field derivations — skim now, return later. Free PDF at the Imm Technical University of Denmark site.

**Topics.**

- Convex vs. non-convex functions; why neural network losses are non-convex.
- Gradient descent: linear convergence for strongly convex objectives, $O(1/k)$ for general convex.
- Stochastic gradient descent and the variance of stochastic gradients.
- Momentum and its interpretation as a damped second-order ODE.
- Adaptive methods: Adam, RMSProp, AdaGrad, and bias correction.
- Conditioning of the Hessian and its effect on gradient-descent speed.
- Saddle points and the (more practical) issue of poorly-conditioned plateaus.
- Matrix calculus for backpropagation: chain rule on tensor expressions, Jacobian-vector products.
- Initialization theory: forward and backward variance preservation arguments.

**Focused exercise set.**

- Analytical: prove that for a strongly convex quadratic $f(\beta) = \tfrac{1}{2} \beta^\top A \beta - b^\top \beta$, vanilla gradient descent with constant step size $\eta < 2/\lambda_{\max}(A)$ achieves linear convergence: $f(\beta_t) - f^* \leq (1 - \eta \lambda_{\min}(A))^t (f(\beta_0) - f^*)$.
- Analytical: derive the condition number's role in gradient descent. Show that the number of iterations to reach $\varepsilon$-suboptimality scales as $\kappa \log(1/\varepsilon)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$.
- Analytical: derive the SGD variance term. Show that under unbiased stochastic gradients $\mathbb{E}[\tilde{g}] = \nabla f$ with $\text{Var}(\tilde{g}) = \sigma^2$, SGD converges in expectation to a neighborhood of the optimum whose size is proportional to $\eta \sigma^2 / \lambda_{\min}(A)$. Identify the bias-variance tradeoff in step-size selection.
- Analytical: derive momentum as the discretization of $\ddot{\beta} + \gamma \dot{\beta} + \nabla f(\beta) = 0$ (a damped second-order ODE). Identify the friction $\gamma$ and time step $h$ corresponding to a given momentum coefficient $\beta_{\text{momentum}}$ and learning rate $\eta$.
- Analytical: derive matrix chain rule for backprop. For $\mathcal{L}(W) = g(Wx)$ where $g: \mathbb{R}^m \to \mathbb{R}$ and $W \in \mathbb{R}^{m \times n}$, show that $\nabla_W \mathcal{L} = (\nabla_y g(y)) x^\top$ where $y = Wx$. Generalize to deeper compositions.

**Checkpoint.**

You can analyze gradient-descent convergence, derive Adam from scratch, and explain why initialization matters in terms of variance propagation. You are equipped to derive backpropagation and analyze its failure modes in Module 6.

### Module 6: Neural networks and optimization

**Primary reading.**

- Prince Ch 1.1-1.4, 2.1-2.5, 3.1-3.6, 4.1-4.5, 5.1-5.6, 6.1-6.5, 7.1-7.5, 8.1-8.5.

**Reference (consult as needed for specific topics).**

- Goodfellow Ch 8 for deeper optimization theory: convergence analysis, second-order methods, batch normalization mechanics.
- Goodfellow Ch 7 for additional perspectives on regularization, especially adversarial training and data augmentation.

**Topics.**

- MLPs as function approximators.
- Loss functions.
- Backpropagation.
- SGD, momentum, Adam.
- Learning rates and schedulers.
- Initialization.
- Batch normalization.
- Dropout.
- Weight decay.
- Early stopping.
- Gradient clipping.

**Focused exercise set.**

*Textbook problems.*

- Prince Ch 3: Problems 3.1 and 3.2.
- Prince Ch 5: Problems 5.1 and 5.2.
- Prince Ch 7: Problems 7.1 and 7.2.
- Goodfellow Ch 6: Exercises 6.1 and 6.2.

*Backpropagation.*

- Analytical: derive backpropagation explicitly for a 2-layer MLP $\hat{y} = W_2 \, \sigma(W_1 x + b_1) + b_2$ with squared-error loss. Express $\partial \mathcal{L} / \partial W_1$ and $\partial \mathcal{L} / \partial W_2$ as products of forward-pass quantities and their Jacobians. Then generalize to $L$ layers by induction.
- Analytical: derive the cross-entropy + softmax composition's gradient. Starting from the result that $\nabla_z \mathcal{L}_{\text{CE}}(\text{softmax}(z), y) = \hat{p} - y$, work backward through one linear layer to derive the gradient with respect to the layer weights. Verify this matches the structure of the standard implementation.

*Optimization.*

- Analytical: derive the convergence rate of vanilla SGD for a strongly convex quadratic $\mathcal{L}(\beta) = \tfrac{1}{2} \beta^\top A \beta - b^\top \beta$. Show that with constant step size $\eta < 2/\lambda_{\max}(A)$, the expected suboptimality decreases by a factor $(1 - \eta \lambda_{\min}(A))^t$ per step.
- Analytical: derive the Adam update with bias correction. Starting from the first-moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ and second-moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, derive the bias-corrected estimates $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$. Show why bias correction matters in the early iterations ($t$ small) and becomes negligible later.
- Analytical: derive momentum as a discretization of a continuous-time ODE with friction. Identify the friction coefficient that corresponds to a given momentum hyperparameter $\beta$ and step size $\eta$.

*Initialization and normalization.*

- Analytical: derive Xavier (Glorot) initialization for a fully-connected layer $h = Wx$. For input variance $\text{Var}(x_j) = \sigma_x^2$, find the weight initialization variance $\text{Var}(W_{ij})$ such that the output variance equals the input variance (forward-pass preservation). Repeat for backward-pass preservation. Take the average for the standard Xavier formula.
- Analytical: derive He initialization for a layer with ReLU activation. Account for the fact that ReLU zeros out half the inputs on average; show that this halves the effective variance and requires doubling the initialization variance: $\text{Var}(W_{ij}) = 2/n_{\text{in}}$.
- Analytical: derive the gradient of batch normalization. For a mini-batch $x_1, \ldots, x_B$ normalized to $\hat{x}_i = (x_i - \mu)/\sigma$ with $\mu, \sigma$ computed from the batch, derive $\partial \mathcal{L} / \partial x_i$ in terms of $\partial \mathcal{L} / \partial \hat{x}_i$ and the batch statistics.

*Practical milestone.*

- Applied: implement a 2-layer PyTorch MLP and verify gradients against `torch.autograd`. Compare your hand-derived gradients (from the backprop exercise above) numerically against `autograd` on a small input.

**Checkpoint.**

You can derive backprop, SGD convergence, Adam bias correction, Xavier/He initialization, and the batch-norm gradient from scratch. You can build, train, debug, and evaluate a small PyTorch model without copying a training loop blindly.

### Module 7: Sequence models for NEC gating

**Readings by section number.**

- Prince Ch 12.1-12.6.
- Goodfellow Ch 10.1-10.10.
- Olah, "Understanding LSTM Networks," sections 1-5.

**Topics.**

- RNNs.
- Backpropagation through time (BPTT) and truncated BPTT.
- Vanishing and exploding gradients (the eigenvalue argument on the recurrent Jacobian).
- GRU.
- LSTM gates.
- Sequence encoders versus sequence-to-sequence models.
- Hidden state interpretation.
- Why financial sequence models overfit easily.

**Focused exercise set.**

*Textbook problems.*

- Prince Ch 12: Problems 12.1 and 12.2.
- Goodfellow Ch 10: Exercises 10.1 and 10.2.

*BPTT and vanishing gradients.*

- Analytical: derive backpropagation through time for a 3-step vanilla RNN with hidden update $h_t = \tanh(W h_{t-1} + U x_t)$. Express the gradient $\partial \mathcal{L}_T / \partial W$ as a sum over $t = 1, \ldots, T$ of products of recurrent Jacobians $\prod_{s=t+1}^{T} \partial h_s / \partial h_{s-1}$. Then generalize to length $T$.
- Analytical: prove the vanishing/exploding gradient theorem for vanilla RNNs. Let $J_t = \partial h_t / \partial h_{t-1}$ and assume its spectral radius is $\rho$. Show that $\|\prod_{s} J_s\|$ grows or shrinks geometrically as $\rho^T$, leading to exploding ($\rho > 1$) or vanishing ($\rho < 1$) gradients over long sequences. State the implication for training RNNs on long financial sequences.
- Analytical: derive truncated BPTT and show that it introduces bias in the gradient estimate. Quantify the bias as a function of the truncation length $k$ vs. the true sequence length $T$.

*Gating mechanism analysis.*

- Analytical: derive the GRU cell's update equation $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ from the reset gate $r_t$, update gate $z_t$, and candidate state $\tilde{h}_t$. Show that when the update gate $z_t \to 0$, the gradient of $h_t$ with respect to $h_{t-1}$ approaches the identity, suppressing vanishing gradients.
- Analytical: derive the LSTM cell-state update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ and show that when the forget gate $f_t \to 1$, the cell-state gradient pathway is approximately the identity. This is the formal statement of "LSTM gates protect gradient flow."
- Analytical: contrast LSTM and GRU formally. Identify which parameters and which gates are dropped in moving from LSTM to GRU. Argue at the level of parameter count and computational cost why GRU is often comparable or better than LSTM on small sequences.

*Practical milestone.*

- Applied: implement a GRU cell from scratch in PyTorch and validate against `torch.nn.GRU`. Verify your forward-pass output matches and that gradients agree under autograd.
- Applied (reading, not coding): re-read `OP model/nec/nec_hybrid_explained.py` and map every tensor shape. Identify exactly where the LSTM encoder hidden state $h_T$ is consumed by the classifier head and how it interacts with the routing wrapper.

**Checkpoint.**

You can derive BPTT, the vanishing-gradient eigenvalue argument, the GRU/LSTM gating gradient pathways, and explain exactly what `C_L` and `C_F` are doing in the current NEC code at the math level.

---

## 9. Part IV: Mixture Models, HMMs, and MoE

Part IV is the conceptual core of the thesis: mixture models, EM, HMMs, and modern mixture-of-experts theory. Two focused Math Foundations sections appear in this part — D, on information theory and variational analysis (needed for Module 8's ELBO), and E, on special distributions and reparameterization (needed for Module 10's Gumbel-Softmax and routing tricks).

### Math Foundations D: Information theory and variational analysis

**Goal.** Deepen the information-theoretic and variational tools you'll need to derive the ELBO, understand EM as coordinate ascent on a free-energy functional, and reason about latent-variable models in general.

**Readings by section number.**

- MacKay, *Information Theory, Inference, and Learning Algorithms*, Ch 2 (probability and inference) and Ch 4 (the source coding theorem). Free PDF at inference.org.uk.
- Bishop §1.6 (information theory — re-read with EM in mind) and §10.1 (variational inference setup).
- Murphy, *Probabilistic Machine Learning: An Introduction*, §8.4 (the EM algorithm) and §10.1 (variational inference) for an alternative perspective. Free chapters at probml.github.io.
- Petersen & Pedersen, *Matrix Cookbook*, §8 (Gaussian distributions and explicit moment formulas) — primary reference for the Gaussian-moment computations that appear in variational mean-field updates with Gaussian components. Used heavily in Bishop §10.2's variational mixture of Gaussians.

**Topics.**

- Entropy, joint entropy, conditional entropy, mutual information.
- The chain rule of mutual information.
- KL divergence as a Bregman divergence; its asymmetry and what it means.
- Forward vs. reverse KL: mean-seeking vs. mode-seeking variational approximations.
- Jensen's inequality and convex/concave function arguments.
- The ELBO as Helmholtz free energy: $-\mathcal{F}(q, \theta) = \mathbb{E}_q[\log p(X,Z|\theta)] + H(q)$.
- Exponential families and conjugate priors as the natural setting for closed-form variational updates.
- Reparameterization vs. score-function gradient estimators (preview for Module 10).

**Focused exercise set.**

- Analytical: derive the chain rule of mutual information: $I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y)$. Use it to motivate the data-processing inequality $I(X; Z) \leq I(X; Y)$ for $X \to Y \to Z$ Markov chains.
- Analytical: prove that for any joint distribution $p(X, Z)$ and any variational distribution $q(Z)$, $\log p(X) = \mathcal{L}(q) + \text{KL}(q(Z) \| p(Z \mid X))$ where $\mathcal{L}(q) = \mathbb{E}_q[\log p(X, Z) - \log q(Z)]$. This is the ELBO decomposition that underlies all variational methods.
- Analytical: derive the variational update for mean-field $q(Z) = \prod_i q_i(Z_i)$. Show that the optimal $q_i^*$ given the other factors fixed is $q_i^*(Z_i) \propto \exp(\mathbb{E}_{-i}[\log p(X, Z)])$. (This is Bishop's equation 10.9.)
- Analytical: contrast forward KL $\text{KL}(p \| q)$ with reverse KL $\text{KL}(q \| p)$. Show that minimizing forward KL leads to a "mean-seeking" approximation (covers all modes of $p$) while reverse KL leads to "mode-seeking" (locks onto one mode). Connect to why VI typically uses reverse KL.
- Analytical: state and prove that the Gaussian distribution maximizes entropy among all distributions with given mean and covariance. (You already derived the univariate case in Math Foundations B; generalize to multivariate.)

**Checkpoint.**

You can derive the ELBO three ways (Jensen's inequality, KL decomposition, free energy), explain when forward vs. reverse KL is appropriate, and motivate why mean-field variational inference is sometimes adequate and sometimes not. You are equipped to follow Module 8's variational EM derivation cold.

### Module 8: Mixture models, EM, and variational inference

**Readings by section number.**

- Bishop Ch 9.1-9.4.
- Bishop Ch 10.1-10.3 (variational inference, the evidence lower bound, variational mixture of Gaussians).
- ESL Ch 8.5.

**Topics.**

- Latent-variable models.
- Gaussian mixture models.
- Responsibilities.
- EM algorithm.
- Log-sum-exp.
- Local optima and initialization.
- Mixture collapse.
- Evidence lower bound (ELBO) and EM as ELBO maximization.
- Variational inference for mixture models.
- KL divergence between distributions on latent variables.
- Connection to Bayesian neural networks and variational autoencoders (forward-looking optional context).

**Focused exercise set.**

- Bishop Ch 9: Exercises 9.7, 9.12, and 9.15.
- Bishop Ch 10: Exercise 10.1 (ELBO derivation) and one of 10.5, 10.10, or 10.12 (variational mixture of Gaussians).
- ESL Ch 8: Exercise 8.1 or 8.2.
- Analytical: derive the M-step closed-form updates for a Gaussian mixture model from scratch. Show that the optimal mean is the responsibility-weighted average $\mu_k = \sum_n \gamma(z_{nk}) x_n / N_k$ where $N_k = \sum_n \gamma(z_{nk})$, derive the analogous formula for the covariance, and show the mixing coefficients become $\pi_k = N_k / N$.
- Analytical: prove the monotonicity property of EM — show that each iteration of E-step + M-step monotonically increases (or leaves unchanged) the observed-data log-likelihood. Use the ELBO decomposition $\log p(X) = \mathcal{L}(q, \theta) + \text{KL}(q \| p(Z|X))$.
- Analytical: derive EM explicitly as coordinate ascent on the ELBO. Write out the E-step as $q^* = p(Z | X, \theta^{\text{old}})$ and verify this closes the KL gap; then write the M-step as maximizing the expected complete-data log-likelihood under $q^*$.
- Analytical: characterize the principal failure modes of EM — (a) component collapse (a Gaussian centered on a single data point with variance shrinking to zero), (b) identifiability under label permutation, (c) local-optimum dependence on initialization. For each, identify the structural property of the GMM likelihood that produces the failure mode.
- Analytical: derive the log-sum-exp trick for numerical stability. Show that $\log \sum_k e^{a_k} = a^* + \log \sum_k e^{a_k - a^*}$ where $a^* = \max_k a_k$. Explain why this prevents overflow/underflow when computing mixture likelihoods $\log \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ for inputs far from any component mean.
- Analytical: derive the responsibility update of the E-step explicitly. Starting from Bayes' rule, show $\gamma(z_{nk}) = p(z_n = k \mid x_n, \theta) = \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k) / \sum_j \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)$. Verify this matches the variational $q^*$ from the ELBO derivation.

**Checkpoint.**

You can write the MoE likelihood, explain how it relates to GMMs, state EM as a coordinate ascent on the ELBO, derive M-step closed forms, apply the log-sum-exp trick to prevent numerical issues, and articulate why the variational view matters for any future Bayesian or stochastic-gate extension of the MoE.

### Module 9: HMMs and classical regime switching

**Readings by section number.**

- Bishop Ch 13.1-13.3.
- Tsay Ch 4.1-4.6, with Ch 4.6 as the Markov-switching focus.
- Hamilton (1989), Sec 1-4.

**Topics.**

- Hidden Markov models.
- Forward algorithm.
- Backward algorithm.
- Forward-backward smoothing.
- Viterbi decoding.
- Baum-Welch.
- Hamilton filter.
- Markov-switching mean and volatility.
- Markov-switching regression.

**Focused exercise set.**

- Bishop Ch 13: Exercises 13.1, 13.3, and 13.5.
- Tsay Ch 4: do two end-of-chapter problems tied to Markov switching or volatility regimes.
- Analytical: derive the forward recursion for an HMM in full. Starting from $\alpha_t(j) = p(o_1, \ldots, o_t, s_t = j)$, show that $\alpha_t(j) = \big(\sum_i \alpha_{t-1}(i) a_{ij}\big) b_j(o_t)$, where $a_{ij}$ is the transition probability and $b_j$ the emission distribution. Compute the complexity: $O(NT^2)$ via the forward recursion versus $O(T^N)$ for a naive joint-distribution sum.
- Analytical: derive the backward recursion $\beta_t(i)$ symmetrically and show how forward-backward smoothing recovers the posterior marginals $\gamma_t(i) = p(s_t = i \mid o_{1:T}) = \alpha_t(i) \beta_t(i) / p(o_{1:T})$.
- Analytical: derive the Hamilton filter (1989) as the special case of HMM filtering applied to a Markov-switching mean and volatility model with Gaussian observations. Identify what changes between the general HMM filter and Hamilton's filter in terms of the observation model.
- Analytical: state precisely and prove the structural difference between Hamilton-style Markov-switching gating (latent state has Markov persistence governed by a transition matrix $P$) and feature-conditioned MoE gating (latent state given by a deterministic function of features $x_t$ with no transition structure). Show that the two reduce to the same model in the degenerate case where the transition matrix encodes only the marginal state distribution.
- Analytical: derive the Viterbi algorithm as the max-product analog of the forward recursion. Define $\delta_t(j) = \max_{s_1, \ldots, s_{t-1}} p(o_{1:t}, s_{1:t-1}, s_t = j)$ and show it satisfies the recursion $\delta_t(j) = \big(\max_i \delta_{t-1}(i) a_{ij}\big) b_j(o_t)$. Use backpointer reconstruction to recover the most-likely state sequence $\arg\max_{s_{1:T}} p(s_{1:T} \mid o_{1:T})$.
- Analytical: derive the Baum-Welch EM updates for HMM parameters. Using the forward-backward posteriors $\gamma_t(i) = p(s_t = i \mid o_{1:T})$ and pairwise marginals $\xi_t(i,j) = p(s_t = i, s_{t+1} = j \mid o_{1:T})$, derive the M-step closed-form updates for the transition matrix $\hat{a}_{ij} = \sum_t \xi_t(i,j) / \sum_t \gamma_t(i)$ and the emission parameters.

**Checkpoint.**

You can derive the full HMM machinery (forward, backward, Viterbi, Baum-Welch) from scratch and state precisely how a Markov-switching model differs from a feature-conditioned MoE.

### Math Foundations E: Special distributions and reparameterization

**Goal.** Cover the discrete-distribution sampling and reparameterization machinery you'll need for Module 10's Gumbel-Softmax, straight-through estimator, and load-balancing derivations.

**Readings by section number.**

- Maddison, Mnih, Teh (2017), *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables*, ICLR — Sec 1–3. Companion paper to Jang et al. (2017) Gumbel-Softmax.
- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*, Sec 2 — for the original reparameterization trick.
- Wikipedia article on the Gumbel distribution — for the CDF, PDF, and max-stability property.

**Topics.**

- Extreme value distributions: Gumbel as the limiting distribution of the maximum of i.i.d. samples from a wide class of distributions (Fisher-Tippett theorem).
- The Gumbel-Max trick: how Gumbel-perturbed logits sample exactly from a categorical distribution.
- The Gumbel-Softmax relaxation: trading bias for differentiability via a temperature parameter.
- Reparameterization trick more generally: when an expectation $\mathbb{E}_{z \sim q_\phi}[f(z)]$ can be rewritten with a fixed base distribution as $\mathbb{E}_{\varepsilon}[f(g_\phi(\varepsilon))]$ to enable gradient flow.
- Score-function (REINFORCE) gradient estimator as the high-variance alternative for non-reparameterizable distributions.
- Straight-through estimators: forward-pass discreteness with backward-pass continuity.

**Focused exercise set.**

- Analytical: derive the CDF and PDF of the standard Gumbel distribution from its definition $G = -\log(-\log U)$ where $U \sim \text{Uniform}(0, 1)$. Verify $F_G(g) = e^{-e^{-g}}$.
- Analytical: prove the Gumbel-Max trick. Let $g_1, \ldots, g_K$ be logits and $G_1, \ldots, G_K \sim \text{Gumbel}(0, 1)$ i.i.d. Show $P(\arg\max_k (g_k + G_k) = j) = e^{g_j} / \sum_k e^{g_k}$. The proof uses the max-stability of the Gumbel distribution.
- Analytical: derive the Gumbel-Softmax relaxation. Replace $\arg\max$ with $\text{softmax}_\tau$ at temperature $\tau$, write $y_k = e^{(g_k + G_k)/\tau} / \sum_j e^{(g_j + G_j)/\tau}$, and show that as $\tau \to 0$, $y \to$ one-hot at $\arg\max_k(g_k + G_k)$. Identify the bias-variance tradeoff: small $\tau$ means low bias (samples are nearly one-hot) but high gradient variance.
- Analytical: derive the reparameterization gradient. For $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$, rewrite as $z = \mu_\phi + \sigma_\phi \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 1)$ fixed. Show that $\nabla_\phi \mathbb{E}_z[f(z)] = \mathbb{E}_\varepsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi \varepsilon)]$, swapping the gradient and expectation operators.
- Analytical: derive the REINFORCE / score-function gradient estimator. For categorical $z \sim \text{Cat}(\pi_\phi)$, show $\nabla_\phi \mathbb{E}_z[f(z)] = \mathbb{E}_z[f(z) \nabla_\phi \log \pi_\phi(z)]$. Compare its variance to the reparameterized estimator and explain why Gumbel-Softmax is preferred when applicable.

**Checkpoint.**

You can derive the Gumbel-Max trick from scratch, explain when Gumbel-Softmax vs. REINFORCE is the right gradient estimator, and articulate the role of temperature in the bias-variance tradeoff. You're equipped to derive the routing-and-load-balancing machinery in Module 10.

### Module 10: Mixture-of-experts theory

**Readings by section number.**

- Jacobs, Jordan, Nowlan, Hinton (1991), Sec 1-4.
- Shazeer et al. (2017), Sec 2-4.
- Fedus, Zoph, Shazeer (2022), Sec 2-3.
- Jang, Gu, Poole (2017), Sec 1-3.
- Jiang et al. (2024), *Mixtral of Experts* — read for the current state of the art in open-source MoE, with focus on top-2 routing, expert-parallel training, and the auxiliary-loss choices used in practice.
- Dai et al. (2024), *DeepSeekMoE* — read for fine-grained expert specialization, shared expert isolation, and the load-balancing refinements that distinguish it from Switch / Mixtral.

**Topics.**

- Conditional mixture decomposition.
- Soft routing.
- Hard routing.
- Sparse top-k routing.
- Gumbel-softmax and straight-through estimators.
- Load-balancing loss.
- Expert collapse.
- Gate entropy.
- Expert specialization.
- Mixture density networks.

**Focused exercise set.**

*Paper-derivation exercises.*

- Paper exercise: reproduce the MoE objective from Jacobs et al. Sec 2 in your own notation. Show how it arises as a conditional mixture decomposition $p(y \mid x) = \sum_k p(z = k \mid x) \, p(y \mid x, z = k)$ and identify what each factor parameterizes.
- Paper exercise: reproduce the Shazeer auxiliary load-balancing loss from Sec 4 of Shazeer et al. (2017). Derive it as $\mathcal{L}_{\text{aux}} = \alpha \cdot K \cdot \sum_k f_k \cdot P_k$ where $f_k$ is the fraction of inputs routed to expert $k$ and $P_k$ is the average gate probability. Show that this is minimized when both $f_k$ and $P_k$ are uniform across experts.

*Routing and reparameterization.*

- Analytical: derive the Gumbel-Max trick. Let $g_1, \ldots, g_K$ be gate logits and $G_1, \ldots, G_K \sim \text{Gumbel}(0, 1)$ i.i.d. Prove that $\arg\max_k (g_k + G_k)$ is distributed as a categorical sample with probabilities $\hat{p}_k = e^{g_k} / \sum_j e^{g_j}$. (This requires deriving the Gumbel distribution's max statistic.)
- Analytical: derive the Gumbel-Softmax relaxation as a continuous, differentiable approximation to argmax. Define $y_k = e^{(g_k + G_k)/\tau} / \sum_j e^{(g_j + G_j)/\tau}$ and show that as $\tau \to 0$, $y \to$ one-hot at $\arg\max_k (g_k + G_k)$. Explain the role of $\tau$ as a bias-variance tradeoff for gradient estimation.
- Analytical: derive the straight-through estimator. Show how it allows backpropagation through hard top-$k$ routing by passing the gradient through as if routing were the identity, even though the forward pass returns a discrete selection.

*Load balancing and expert dynamics.*

- Analytical: derive the gradient of the Shazeer auxiliary loss $\mathcal{L}_{\text{aux}}$ with respect to the gating logits. Identify which term ($f_k$ vs $P_k$) provides the gradient signal and which acts as a stop-gradient scaling factor.
- Analytical: analyze the expert-collapse failure mode formally. Set up a small model with $K = 2$ experts and a softmax gate; show that without any load-balancing penalty, the gradient dynamics of standard MSE training naturally concentrate all routing mass on whichever expert has the lower loss at initialization, regardless of whether the data is actually homogeneous. State the rate of collapse as a function of learning rate.

*Mixture density connection.*

- Analytical: derive a mixture density network (MDN) training objective. Set up a model that outputs $K$ mixture means $\mu_k(x)$, variances $\sigma_k^2(x)$, and mixing weights $\pi_k(x)$, all as neural-network outputs. Write the negative log-likelihood loss $-\log \sum_k \pi_k(x) \mathcal{N}(y \mid \mu_k(x), \sigma_k^2(x))$ and contrast its log-sum-exp structure with the soft-routed MoE loss for point prediction. Explain when MDN is the right choice (heteroscedastic, multimodal predictive distributions) and when MoE-for-prediction is sufficient.

*Practical milestone (the thesis prototype).*

- Applied: implement a soft-routed MoE on synthetic regression data with built-in regime structure ($y = f_1(x)$ when $z = 0$, $y = f_2(x)$ when $z = 1$). Verify the gate learns to recover the latent regime.
- Applied: add load balancing and plot expert utilization $f_k$ and gate entropy $H(p(z \mid x))$ over training. Verify load balancing prevents the collapse failure mode analyzed above.

**Checkpoint.**

You can derive the modern MoE machinery from scratch — soft routing, Gumbel-Softmax, straight-through estimator, load balancing, and the expert-collapse dynamics. You can build the corrected NEC prototype:

```text
sequence features -> gate -> K experts -> weighted prediction
```

---

## 10. Part V: Finance Domain Knowledge Needed for the Thesis

Part V supplies the finance-domain knowledge the thesis sits in: empirical asset pricing, factor models, and the practical data-engineering realities of financial datasets. It opens with a brief Math Foundations section on the econometric tools (time-series vs. cross-sectional regression, errors-in-variables, the structure of factor models) that you'll use throughout Module 11 and the thesis's evaluation chapter.

### Math Foundations F: Financial econometrics primer

**Goal.** Cover the regression-and-inference tools specific to finance — particularly the asymmetry between time-series and cross-sectional regressions, the errors-in-variables structure of two-pass estimation, and the GMM framing that underlies most empirical asset pricing. Most of the math is OLS specialized to two different sampling structures; the key insight is that *which dimension you sample over* changes which estimators are consistent.

**Readings by section number.**

- Cochrane, *Asset Pricing*, Ch 11 (GMM in explicit detail) and Ch 12 (regression-based tests). Sections 11.1–11.3 and 12.1–12.3 are the load-bearing pages. Free PDF available via Cochrane's University of Chicago page.
- Campbell, Lo, MacKinlay Ch 5 (present-value relations) §5.1–5.3 for the basic econometric setup.
- Greene, *Econometric Analysis*, Ch 8 (asymptotic distributions of OLS) §8.1–8.3 — re-read with finance-panel applications in mind.

**Topics.**

- Time-series regression: OLS of $r_{i,t}$ on factors $f_t$ over $t = 1, \ldots, T$ for a fixed asset $i$. Yields per-asset betas.
- Cross-sectional regression: OLS of $r_{i,t}$ on $\hat{\beta}_i$ over $i = 1, \ldots, N$ for a fixed (or average) date. Yields factor risk premia.
- Why the two-pass (Fama-MacBeth) procedure is needed: time-series-only or cross-section-only fails to estimate risk premia consistently.
- Errors-in-variables: the second-pass regressor $\hat{\beta}_i$ is itself an estimate, with sampling error that biases the risk-premium estimate toward zero.
- The Shanken correction: closed-form adjustment for errors-in-variables in the second-pass standard errors.
- GMM as the unifying framework: factor models as moment conditions $\mathbb{E}[(r_{i,t} - \alpha_i - \beta_i f_t) f_t] = 0$.
- Heteroskedasticity and serial correlation in financial residuals; Newey-West standard errors.
- The cross-section vs. time-series $R^2$ distinction.

**Focused exercise set.**

- Analytical: derive the OLS estimator for a single factor regression $r_{i,t} = \alpha_i + \beta_i f_t + \varepsilon_{i,t}$. Show $\hat{\beta}_i = \text{Cov}(r_i, f) / \text{Var}(f)$. Derive its asymptotic distribution under standard regularity conditions.
- Analytical: derive the Fama-MacBeth two-pass estimator formally. State the regularity conditions under which the second-pass risk-premium estimator $\hat{\lambda}$ converges to the true $\lambda$ as $T \to \infty$ for fixed $N$.
- Analytical: derive the errors-in-variables bias in the second-pass regression. Show that when $\hat{\beta}_i = \beta_i + u_i$ with $u_i$ measurement noise, OLS of returns on $\hat{\beta}_i$ produces a slope estimate biased toward zero. Identify the attenuation factor in terms of $\text{Var}(\beta) / (\text{Var}(\beta) + \text{Var}(u))$.
- Analytical: derive the GMM objective for a factor model. Write the moment conditions $\mathbb{E}[g_t(\theta)] = 0$ where $g_t(\theta) = (r_{i,t} - \alpha_i - \beta_i f_t) \otimes (1, f_t^\top)^\top$. Show how minimizing the sample-moment quadratic form recovers OLS with the appropriate weighting matrix.
- Analytical: derive Newey-West standard errors as a kernel-smoothed estimate of the long-run variance of residuals. Show why ordinary OLS standard errors are too small in the presence of serial correlation in returns.

**Checkpoint.**

You can derive the Fama-MacBeth estimator and its errors-in-variables correction, write a factor model as a GMM problem, and explain why Newey-West standard errors are the standard choice for financial panels. You're equipped to engage with Module 11's empirical asset-pricing material from first principles rather than as a list of techniques.

### Module 11: Empirical asset pricing for ML people

**Readings by section number.**

- Bali, Engle, Murray Ch 1.1-1.5, Ch 3.1-3.4, Ch 8.1-8.5.
- Campbell, Lo, MacKinlay Ch 2.1-2.4 and Ch 5.1-5.3.
- Fama-French factor documentation, sections "Daily Factors" and "Momentum Factor".
- Daniel and Moskowitz (2016), Sec 1-5.

**Topics.**

- Cross-sectional versus time-series prediction.
- Factor models.
- Market beta.
- Size, value, momentum, profitability, investment.
- Return predictability and low signal-to-noise.
- Factor crashes and conditional performance.
- Fama-MacBeth regression.
- Residual returns and neutralization.

**Focused exercise set.**

*Textbook problems.*

- Bali, Engle, Murray Ch 3: Exercises 1 and 2.
- Campbell, Lo, MacKinlay Ch 2: Exercises 1 and 2.

*Factor models and Fama-MacBeth.*

- Analytical: derive the Fama-MacBeth two-pass procedure from first principles. In the first pass, run time-series regressions of asset returns on factors to estimate per-asset factor betas. In the second pass, run cross-sectional regressions of returns on the estimated betas to estimate factor risk premia. State the assumptions under which the two-pass estimator is consistent and identify the errors-in-variables problem introduced by using estimated betas as regressors.
- Analytical: derive the formula for the cross-sectional R-squared of a factor model and explain why time-series R-squared and cross-sectional R-squared can diverge.
- Analytical: derive the OLS estimator for market beta $\hat{\beta}_i = \text{Cov}(r_i, r_m) / \text{Var}(r_m)$ from a time-series regression $r_{i,t} = \alpha_i + \beta_i r_{m,t} + \varepsilon_{i,t}$. Show how using stale (window-lagged) beta estimates avoids look-ahead bias when the beta is itself an input to a forecast.

*Residual returns and neutralization.*

- Analytical: derive the formula for the market-residual return $r_i^{\text{residual}} = r_i - \hat{\beta}_i r_m$. Show that the residual has zero correlation with the market by construction. Explain why residualizing eliminates market-factor exposure but does not eliminate cross-asset correlations driven by other (non-market) factors.
- Analytical: derive multi-factor residualization. Given a set of $K$ factors $f_1, \ldots, f_K$, derive the residualized return as the residual from a multivariate regression of $r_i$ on the factors. Show that the residualized cross-section is orthogonal to the factor space.

*Factor crashes.*

- Analytical: state Daniel and Moskowitz's (2016) momentum-crash mechanism informally and then formally. Show that when the past-winners portfolio loaded short the market in the prior period (bear-market lookback), the momentum strategy implicitly takes a short market position right when the market is rebounding. Identify the conditional-beta structure that produces the crash.

*Practical milestones (data-required).*

- Applied: download Fama-French factors from Kenneth French's data library and compute the market excess return series. This is operationally required for the thesis baseline.
- Applied: run a cross-sectional regression of future returns on momentum and volatility features on a small panel. This sets up the methodology you will use for the thesis evaluation.

**Checkpoint.**

You can explain why a quant fund usually cares about cross-sectional ranking, not only time-series price prediction.

### Module 12: Financial data engineering

**Topics.**

- Ticker universes.
- Delistings and survivorship bias.
- Corporate actions and adjusted prices.
- Splits and dividends.
- Point-in-time fundamentals.
- Filing-date alignment.
- Missing data.
- Liquidity filters.
- Rebalancing frequency.

**Readings by section number.**

- Lopez de Prado, AFML Ch 2.1-2.5, Ch 3.1-3.4, Ch 4.1-4.4.
- SEC EDGAR API documentation, sections "Submissions" and "Company Facts".
- SEC fair-access guidance, sections 1-3.

**Focused exercise set.**

- Build a large-cap ticker universe.
- Pull daily price data for the universe and verify adjusted prices.
- Build rolling features without lookahead.
- Pull SEC companyfacts for 10 tickers and align by filing date.

**Checkpoint.**

You have a real-data panel ready for a baseline model.

---

## 11. Proposal-Readiness Milestones

Before writing the September proposal, have these artifacts:

1. **Corrected NEC on synthetic data**
   - Soft-routed MoE with K = 2 or K = 3.
   - Gate utilization and entropy plots.
   - Synthetic hidden-regime recovery test.

2. **Classical baseline**
   - Single MLP or LightGBM model.
   - Simple HMM or Hamilton-style regime baseline.

3. **Real price-data baseline**
   - Free daily-price panel.
   - Simple technical/factor features.
   - Forward-return target.
   - Rank-IC and decile backtest.

4. **Evaluation protocol**
   - Walk-forward validation.
   - Purging and embargoing.
   - Transaction costs.
   - No random shuffling across time.

5. **Literature map**
   - Classical regime switching.
   - Classical MoE.
   - Modern MoE.
   - Financial ML validation.
   - Cross-sectional return prediction.

6. **One-page thesis claim**
   - What exact ML method is being tested?
   - What finance prediction problem is used?
   - What are the baselines?
   - What would count as success?
   - What would a negative result still teach?

---

## 12. Thesis Research Design

### Main research question

Do modern mixture-of-experts routing methods improve out-of-sample cross-sectional return prediction relative to single-model baselines and classical regime-switching baselines?

### Secondary research question

Do learned expert assignments correspond to recognizable financial regimes or do they split the data along other dimensions?

### Candidate models

Start simple and expand only after the basics work:

1. Ridge or elastic-net regression.
2. LightGBM.
3. Single MLP.
4. Original NEC hard-routed model.
5. Corrected soft-routed NEC / MoE.
6. Sparse top-k MoE.
7. HMM / Markov-switching regression baseline.

### Main evaluation metrics

- Mean daily rank-IC.
- Rank-IC information ratio.
- Decile long-short return.
- Turnover.
- Transaction-cost-adjusted Sharpe.
- Maximum drawdown.
- Expert utilization.
- Gate entropy.
- Gate-regime alignment.

### Minimum ablations

- Number of experts: K = 2, 3, 4, 8.
- Soft versus hard routing.
- With versus without load balancing.
- Price-only features versus price + market-context features.
- Raw returns versus market-residual returns.
- Different prediction horizons: 5-day and 20-day.

### What a good negative result looks like

If MoE does not beat strong baselines, the thesis can still be strong if it shows:

- Single-model baselines are hard to beat in noisy financial data.
- Expert gates collapse without regularization.
- Learned experts do not align with intuitive regimes.
- Apparent improvements disappear after purged validation or transaction costs.
- Classical regime labels explain little about cross-sectional returns at the chosen horizon.

That is still a useful ML-for-finance result.

---

## 13. Thesis Execution Timeline

### June to July 2026: foundations and code fluency

- Finish Python/pandas/NumPy/PyTorch foundations.
- Work through ESL Ch 2, Ch 3, Ch 4, Ch 7.
- Implement linear/logistic regression and MLP from scratch.
- Understand the existing NEC code completely.
- Run the OP pipeline tests and demos.

### July to August 2026: mixtures, HMMs, and corrected NEC

- Finish Bishop mixture-model material.
- Implement GMM and EM.
- Implement small HMM/forward algorithm.
- Implement soft-routed MoE on synthetic data.
- Add load-balancing and gate diagnostics.
- Write the first corrected NEC prototype.

### August to September 2026: real-data baseline and proposal

- Build free daily-price panel.
- Build simple price/volume features.
- Train ridge, LightGBM, MLP baselines.
- Add HMM/regime baseline.
- Run walk-forward evaluation.
- Write proposal with realistic scope.

### September to October 2026: thesis model experiments

- Train corrected NEC/MoE variants.
- Compare soft, hard, and sparse routing.
- Tune number of experts.
- Add gate diagnostics.
- Decide whether SEC fundamentals are feasible as an extension.

### October to November 2026: robust evaluation

- Run full walk-forward experiments.
- Add transaction costs and turnover.
- Run ablations.
- Compare raw and residual targets.
- Stress-test across horizons and regimes.

### November to December 2026: interpretation and writing

- Interpret expert behavior.
- Compare gate assignments to VIX, realized volatility, SPY drawdowns, and recession indicators if used.
- Write methods, data, and results chapters.

### December 2026 to January 2027: full draft

- Finish literature review.
- Finish results tables and figures.
- Write limitations.
- Write conclusion.
- Prepare first full draft.

---

## 14. Core Exercise Spine

If time gets compressed, preserve these:

1. ESL Ch 2: Exercises 2.1, 2.5, and one bias-variance simulation.
2. ESL Ch 3: Exercises 3.4 and 3.6, plus ridge regression from scratch.
3. ESL Ch 4: Exercise 4.2, plus softmax regression with a gradient check.
4. Bishop Ch 9: Exercises 9.7 and 9.12, plus a GMM implementation.
5. Bishop Ch 13: Exercises 13.1 and 13.3, plus the HMM forward algorithm.
6. Prince Ch 7: Problems 7.1 and 7.2, plus a PyTorch MLP gradient check.
7. Shazeer et al. Sec 4: reproduce the load-balancing loss and implement it.
8. AFML Ch 7.4-7.5: implement purging and embargoing.
9. Build a real cross-sectional price panel and compute rank-IC.
10. Compare corrected NEC against a single MLP and LightGBM baseline.

---

## 15. Final Personal Rule

Do not let the thesis become "I used a complicated neural network on stocks." The defensible thesis is:

```text
I studied conditional computation for non-stationary financial prediction.
I corrected a flawed hard-routed NEC architecture into a principled MoE.
I compared routing mechanisms under proper financial validation.
I tested whether learned experts correspond to market regimes.
```

That is ML-first, finance-relevant, and realistic for a January 2027 first draft.

---

## 16. Useful Source Links

- SEC EDGAR APIs and data: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC fair access guidance: https://www.sec.gov/os/accessing-edgar-data
- Stooq historical data: https://stooq.com/
- Kenneth French data library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- FRED API: https://fred.stlouisfed.org/docs/api/fred/
- CBOE VIX historical data: https://www.cboe.com/tradable_products/vix/vix_historical_data/
- Shazeer et al. MoE paper page: https://research.google/pubs/pub45929/
- Switch Transformer JMLR: https://www.jmlr.org/papers/v23/21-0998.html
