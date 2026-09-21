# Code change brief, nec_baseline
Date: 2026-09-21. Scope: settled items only. Nothing in this brief depends on an open design question.

Repository: `Quant Model/nec_baseline`. Package: `nec_moe`.

---

## 0. Do not change these

These are already correct and several are load bearing. Preserve their behaviour and their tests.

- **Filtering, never smoothing.** `HMMRegimePrior.forward` is the forward filter's predict step only, and `_fold_predictions` warms the filter through the training block before the test block. The prior at t conditions on data through t-1, the posterior on data through t. This is the single most common fatal error in regime backtests and this codebase already gets it right.
- **Log domain end to end.** `mixture_nll` uses one fused `logsumexp`; `expert_log_likelihood` never exponentiates a density.
- **The `RegimePrior` abstraction and `PRIOR_REGISTRY`.** A gate is a config string. Every addition below plugs in here rather than forking the model.
- **Purged expanding walk forward**, fit once per window, rank IC, ICIR, deflated Sharpe, the append only `TrialRegistry`, and `canonical_expert_order` for label switching.
- **`LoadBalanceBuffer`** scoped across many batches rather than one. The reasoning in the docstring is correct and cited.
- **Point in time universe** with an honest published coverage column.

---

## 1. Diagnostics, pure additions, no behaviour change

### 1.1 Expected regime duration and implied stationary distribution
Add to `nec_moe/diagnostics.py`:

- `expected_durations(A) -> Tensor (K,)` returning `1 / (1 - A[k,k])`, the expected number of consecutive periods spent in regime k for a first order chain.
- `stationary_distribution(A) -> Tensor (K,)`, the normalised left eigenvector of A for eigenvalue 1 (equivalently the solution of `pi A = pi`, `sum pi = 1`).

Log both every eval interval for any prior exposing a transition matrix, and write them into the `TrialRegistry` metrics.

Why: persistence is the property that distinguishes the gate families in this thesis, and the transition matrix is already available as `HMMRegimePrior.transition_matrix`. Right now nothing reports how sticky the fitted chain actually is. A fitted chain with expected duration near one period is a degenerate regime process and should be visible immediately rather than inferred from returns.

Literature: Hamilton, J. D. (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle," *Econometrica* 57(2), 357 to 384, for the expected duration result. Kim, C.-J. and Nelson, C. R. (1999), *State-Space Models with Regime Switching*, MIT Press, for the filtering and smoothing treatment.

Test: on a synthetic chain built with a known stay probability, recovered expected duration matches `1/(1-stay)` within tolerance.

### 1.2 Variance decomposition of gate weights (ICC)
New function in `nec_moe/diagnostics.py`:

`gate_variance_decomposition(gate_probs, date_codes, entity_codes) -> dict`

Run a one way ANOVA on each expert's gate weight with date as the grouping factor, and report the intraclass correlation

    ICC = sigma2_date / (sigma2_date + MS_within)

together with the two variance components and the residual entity level share.

Why: this measures directly whether the gate is behaving as a *regime* variable (a date level quantity shared by the whole cross section) or as a per stock characteristic. The premise of the whole thesis is that it is the former. An ICC near 1 confirms the date level reading; a low ICC means the gate has found something cross sectional and the regime interpretation does not hold. There is currently no way to tell these apart.

Note that for `HMMRegimePrior` the prior is constant within a date by construction, so ICC is 1 trivially; run the decomposition on the **filtered posterior** (`log_filtered.exp()`), which does vary within a date because the likelihood term is per entity, and on the memoryless priors where the prior itself varies.

Test: synthetic data with regime injected purely at date level yields ICC close to 1; data with the signal injected per entity yields ICC close to 0.

### 1.3 Gate permutation test
New module `nec_moe/permutation.py`:

`gate_permutation_test(trainer_factory, panel, n_permutations, seed)`

Procedure: take the fitted gate weight series, randomly permute it **across dates** while leaving the per date cross sections intact, retrain the expert stage against the permuted gate, and record the out of sample metric. Repeat n times to build a null distribution, and report the empirical p value of the unpermuted result.

Why: permuting across dates preserves the marginal distribution of the gate weights and destroys only their alignment with time. If the real gate does not beat the permuted null, the mixture is gaining from having K experts rather than from regime timing, and that is the honest finding. This is the strongest single piece of evidence the thesis can produce and it needs no additional data.

Register every permutation as a trial under its own tag so `n_trials` and the deflated Sharpe stay correct.

### 1.4 Remove the silent stub
`diagnostics.wasserstein_template_tracking` is exported in `__all__` but is a stub. Either implement it or make it raise `NotImplementedError` with a pointer. An importable no op is a trap.

---

## 2. Guard against the parameter count confound

In `HMMRegimePrior`, `gate_logits` is used only for shape inference; the docstring says so. Therefore under `prior.kind="hmm"` together with `experts.input_mode="snapshot"`, the GRU encoder and the gate head receive **zero gradient** and are dead parameters. Under `prior.kind="soft"` the same parameters are live. Any comparison across prior kinds is therefore a comparison across different effective model sizes unless this is controlled and reported.

Add to `nec_moe/train.py`:

- After the first backward pass, walk `model.named_parameters()` and record which parameters have `grad is None` or an all zero gradient. Emit a single structured warning naming the dead parameter groups.
- Compute `live_param_count` (parameters that actually receive gradient) and write it into the `TrialRegistry` metrics for every trial.

Report `live_param_count` beside every result in the thesis. This is a real confound, it is cheap to measure, and stating it explicitly is stronger than hoping nobody asks.

---

## 3. Priors that are fitted rather than backpropagated: interface change

Two of the four gate families in this thesis cannot be trained by backpropagation. The statistical jump model is fitted by an alternating assign and fit procedure with a jump penalty, and the Wasserstein gate is fitted by clustering. Both are fitted once on a training window and then applied causally.

The current `RegimePrior` contract assumes a differentiable module trained jointly. Extend it before adding either gate:

1. Add an optional method to the `RegimePrior` base:

   `def fit(self, train_panel: Panel) -> None: ...`

   Default is a no op. The walk forward harness calls it **once per fold, on the training block only**, before the expert stage trains. This is the natural place and it inherits the existing purge discipline.

2. Add a `PrecomputedRegimePrior` family: holds a `(n_dates, K)` table of log priors keyed by date code, produced by `fit`, and returns rows by lookup in `forward`. It contributes no gradient. Assert at lookup time that the requested date is inside the fitted range and that no date beyond the fold's information set is ever read.

3. In `NECConfig.validate`, require that a precomputed prior is paired with `sequence_ordered=False` unless the prior is also stateful, and that `fit` has been called before `forward` (a fitted flag).

This change is required regardless of anything still undecided, because neither gate can exist without it, and it keeps the "config string selects the mechanism" property that the rest of the codebase is built around.

---

## 4. The three missing gates

The registry currently holds `soft`, `uniform`, `hard`, `topk`, `gumbel`, `hmm`. The thesis compares four structurally distinct regime processes. Three are absent.

### 4.1 TVTP, time varying transition probabilities
Implement behind the existing guarded `PriorConfig.tvtp` flag, which currently raises `NotImplementedError` in `config.py`.

Mechanism: the transition matrix becomes a function of observed covariates,

    A_t[j,k] = softmax_k( L[j,k] + w[j,k] . c_t )

with `c_t` the date level context vector already produced by `nec_moe/context_data.py` (VIX, French factors). Everything else in the forward filter is unchanged, so this is a localised change to `HMMRegimePrior`.

Keep the identifiability caution already cited in `config.py` (arXiv:2605.14976) in the docstring, and start from the static solution as the initialisation.

Literature: Diebold, F. X., Lee, J.-H. and Weinbach, G. C. (1994), "Regime Switching with Time-Varying Transition Probabilities," in Hargreaves (ed.), *Nonstationary Time Series Analysis and Cointegration*, Oxford University Press. Filardo, A. J. (1994), "Business-Cycle Phases and Their Transitional Dynamics," *Journal of Business and Economic Statistics* 12(3), 299 to 308.

### 4.2 Statistical jump model gate
New `JumpModelRegimePrior`, a `PrecomputedRegimePrior`. Fitted by the alternating procedure: given state assignments, fit per state parameters; given parameters, assign states by dynamic programming over the sequence with a fixed penalty `lambda` charged at every state change. Iterate to convergence. The jump penalty is what produces persistence, replacing the transition matrix.

Expose `jump_penalty` in config and select it by time series cross validation on the training block only.

Literature: Bemporad, A., Breschi, V., Piga, D. and Boyd, S. (2018), "Fitting jump models," *Automatica* 96, 11 to 21. Nystrup, P., Lindström, E. and Madsen, H. (2020), "Learning hidden Markov models with persistent states by penalizing jumps," *Expert Systems with Applications* 150, 113307. Shu, Y. and Mulvey, J. M., "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach," arXiv:2402.05272. For the equity factor application closest to this thesis, "Dynamic Factor Allocation Leveraging Regime-Switching Signals," arXiv:2410.14841.

### 4.3 Wasserstein gate
New `WassersteinRegimePrior`, a `PrecomputedRegimePrior`. Segment the training window and cluster the segment level empirical distributions under the Wasserstein distance, then assign each date a soft membership over the K clusters. In one dimension the Wasserstein distance has a closed form via quantile functions, so no optimal transport solver is needed for the univariate case; use the sliced variant for the multivariate case.

Literature: Horvath, B., Issa, Z. and Muguruza, A., "Clustering Market Regimes using the Wasserstein Distance," arXiv:2110.11848. For hyperparameter behaviour and initialisation sensitivity, which will matter at implementation time, "Automated regime classification in multidimensional time series data using sliced Wasserstein k-means clustering," arXiv:2310.01285.

Note that `diagnostics.wasserstein_template_tracking` (item 1.4) and this gate are different things: one tracks regime identity across folds, the other is a gate. Keep the names distinct.

---

## 5. Residual mixture mode

The model currently predicts `y_hat = sum_k pi_k mu_k`. The design this thesis is built on is residual: a base predictor is fitted first and frozen, and the mixture models only the correction, with expert heads initialised at zero so the model starts exactly at the base.

    y_hat = f0(x) + sum_k pi_k r_k(x)

Add this as a config gated mode, default off, matching the codebase's existing convention for optional machinery:

- `ExpertConfig.residual: bool = False` plus a base model spec.
- Fit `f0` on the training block of each fold, freeze it (`requires_grad_(False)`), and verify it is frozen in the dead parameter check from item 2.
- Zero initialise the final layer of every expert head so that at step 0 the correction is identically zero and `y_hat == f0(x)`.
- Add the correction magnitude penalty `alpha * (sum_k pi_k r_k)^2` as a config gated auxiliary loss in `losses.py`, following the existing pattern of `load_balance_aux` and `expert_decorrelation_aux`.

Note that `features.py` already uses the word "residual" for a market neutralised *target*, which is a different concept. Pick a distinct name (`correction_mode`, or similar) so the two never get confused in config or in the write up.

Test: with `residual=True` at initialisation, predictions equal the base model's predictions exactly.

Literature: the residual mixture of experts paper by Ye and Borde, in `Quant Model/MoE papers`. Take the architecture and the loss form from it; note in the docstring that it publishes no code and that `alpha` is not reported, so the value used here is selected on the training block and logged as a trial.

---

## 6. Reference baseline for the Markov gate

The current `HMMRegimePrior` learns its transition matrix by backpropagation through the forward filter, jointly with the emissions, and its transitions are homogeneous and independent of any covariate. This is a legitimate model but it is **not** Hamilton's Markov switching model as the econometrics literature estimates it, and the thesis must not describe it as though it were.

Two changes:

1. **Docstring and README correction.** Name the current object accurately: a jointly fitted latent Markov mixture with homogeneous data independent transitions, estimated by gradient descent through the forward recursion. State that the encoder and gate head are dormant under this prior.

2. **Add a conventionally estimated reference.** In `nec_moe/baselines.py`, add a Markov switching baseline fitted by standard maximum likelihood (`statsmodels.tsa.regime_switching.MarkovRegression`), and report side by side the transition matrix, the expected durations from item 1.1, and the filtered probability series from both estimators. If the backpropagated filter recovers what the standard estimator recovers, that is a defensible sentence in the methodology chapter. If it does not, that is something to find out now rather than at the defence.

   Critical: pull `filtered_marginal_probabilities` from statsmodels, **never** `smoothed_marginal_probabilities`. The smoothed series conditions on the whole sample including the future and is the default attribute most people reach for.

Literature: Hamilton (1989) as above. Hamilton, J. D. (1990), "Analysis of time series subject to changes in regime," *Journal of Econometrics* 45(1 to 2), 39 to 70, for the EM estimator. Kim and Nelson (1999) for the filtering versus smoothing distinction stated as a modelling decision rather than an implementation detail.

---

## 7. Data source seam

The confirmed database for the thesis is CRSP. Access is not yet in place, so do not write CRSP specific code now. Instead:

- Define the loader interface explicitly against the existing cache CSV contract (`Date,Open,High,Low,Close,Volume`, one file per symbol under `data_cache/`), so that switching sources is a single file change.
- Keep yfinance as the development source and keep the survivorship warning in `README.md` exactly as it stands.
- Add a `source` field to every `TrialRegistry` entry, so no result can later be mistaken for a CRSP result.

---

## 8. References to add to the literature file

The codebase asserts in several docstrings that routing adds value over a single network. There is now a formal result for this and it is the theoretical anchor for the thesis premise, so cite it rather than asserting it:

- Chen, Z., Deng, Y., Wu, Y., Gu, Q. and Li, Y., "Towards Understanding Mixture of Experts in Deep Learning," arXiv:2208.02813. Establishes that the mixture of experts layer outperforms a single network precisely when the underlying problem has **cluster structure**, and that expert non linearity is what prevents the mixture collapsing to a single model. This is the formal statement of this thesis's premise, with regimes supplying the cluster structure.
- "Mixture of Experts Provably Detect and Learn the Latent Cluster Structure in Gradient-Based Learning," arXiv:2506.01656. Sample and runtime complexity for the same setting; relevant to the question of how much data is needed per regime.
- Shazeer, N. et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR, as the source of the load balancing form already implemented in `losses.py`.
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J. and Hinton, G. E. (1991), "Adaptive Mixtures of Local Experts," *Neural Computation* 3(1), 79 to 87.
- Jordan, M. I. and Jacobs, R. A. (1994), "Hierarchical Mixtures of Experts and the EM Algorithm," *Neural Computation* 6(2), 181 to 214.

---

## Suggested order

1. Items 1.1, 1.4 and 2. Small, self contained, no design surface.
2. Item 6.1, the docstring correction. Costs nothing and prevents a wrong sentence propagating into the thesis.
3. Item 3, the `fit` and `PrecomputedRegimePrior` interface. Everything in section 4 depends on it.
4. Items 1.2 and 1.3, the ICC and permutation diagnostics.
5. Item 4.1 TVTP, then 4.2 jump model, then 4.3 Wasserstein.
6. Items 5, 6.2, 7.

Every item keeps the existing conventions: config gated, registry built, default off where it changes behaviour, with a test that pins the property being claimed.
