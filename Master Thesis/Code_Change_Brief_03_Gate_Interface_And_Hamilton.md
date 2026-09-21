# Code change brief 03: the gate interface, and the Hamilton gate

Date: 2026-09-21. Depends on Q19 and Q7 (both RESOLVED 2026-09-21) in `Advisor_Questions.md`, and
on section 3 and section 6 of `Code_Change_Brief_2026-09-21.md` (brief 01), which this brief
supersedes and expands for those two sections only.

Repository `Quant Model/nec_baseline`, package `nec_moe`. As in brief 02: **every numeric quantity
is a config field with a documented default and no hardcoded constant.** Nothing here is chosen yet.

---

## 0. Read first

Do not write code until these are read.

**The model and its estimation**

1. **Hamilton, J. D. (1989)**, "A New Approach to the Economic Analysis of Nonstationary Time Series
   and the Business Cycle," *Econometrica* 57(2), 357 to 384. The origin of the model being
   implemented. Read for the filter recursion, the expected duration result `1/(1-p_kk)`, and the
   economic framing of a regime.
2. **Hamilton, J. D. (1994)**, *Time Series Analysis*, Princeton University Press, **Chapter 22**.
   The textbook treatment: the filter written out step by step, the likelihood it evaluates, and the
   estimation.
3. **Hamilton, J. D. (1990)**, "Analysis of time series subject to changes in regime," *Journal of
   Econometrics* 45(1 to 2), 39 to 70. The EM estimator, which is what a maximum-likelihood fit of
   this model is doing underneath.
4. **Kim, C.-J. and Nelson, C. R. (1999)**, *State-Space Models with Regime Switching*, MIT Press,
   **Chapter 4**. The one source that treats **filtering versus smoothing as a modelling decision**
   rather than an implementation detail. This is the single most important item on the list for this
   task, because the whole causal-application requirement below rests on it.

**The bridge to what has already been read**

5. **Bishop, PRML, sections 13.2.2 and 13.2.4.** Forward-backward, and the scaling factors where the
   filtered probability appears as the scaled forward variable. Notation map: Bishop's
   `alpha-hat(z_n)` is Hamilton's filtered probability; Bishop's `gamma(z_n)` is the **smoothed**
   probability, conditioning on all of X, and is the one that must never reach the gate. Bishop's
   `pi_k` is the initial state distribution here, not a mixing weight.

**The API**

6. **statsmodels documentation** for `statsmodels.tsa.regime_switching.MarkovRegression` and
   `MarkovAutoregression`. Read the `fit`, `filter`, `apply` and `smooth` methods and the results
   attributes, specifically `filtered_marginal_probabilities`,
   `smoothed_marginal_probabilities`, `regime_transition`, `expected_durations` and `params`.

**Context on regimes in finance**

7. **Ang, A. and Timmermann, A. (2012)**, "Regime Changes and Financial Markets," *Annual Review of
   Financial Economics* 4, 313 to 337. Why regime switching is applied to returns at all, and what a
   regime is usually taken to mean empirically.
8. **Shu, Y. and Mulvey, J. M.**, arXiv:2402.05272. Read only for how it handles selection of the
   regime model's hyperparameters on training data alone; the same discipline applies here.

**Label switching**

9. **Stephens, M. (2000)**, "Dealing with label switching in mixture models," *Journal of the Royal
   Statistical Society Series B* 62(4), 795 to 809. Needed for section 4 below.

---

## 1. `RegimePrior.fit` and the precomputed family

Extend the existing `RegimePrior` contract in `nec_moe/priors.py`:

    def fit(self, train_panel: Panel) -> None: ...

Default implementation is a no-op, so every existing prior is unaffected. The walk-forward harness
calls it **once per fold, on the training block only**, before the base and the experts are trained.
This inherits the purge discipline the harness already enforces.

Add `PrecomputedRegimePrior(RegimePrior)`:

- holds a `(n_dates, K)` table of **log** priors, produced by `fit`, keyed by date code;
- `forward` returns rows by lookup, expanded to the batch; contributes **no gradient**;
- `stateful = False`, since the recursion has already been run at fit time;
- carries a `fitted` flag; `forward` before `fit` raises;
- **asserts at lookup that every requested date lies within the fitted information set.** A date
  beyond the fold's training block is legal for *application* (the filter runs forward causally) but
  must be produced by the causal application path of section 3, never by a table built from a fit
  that saw it.

In `NECConfig.validate`, require that a precomputed prior carries its `fitted` flag before any
forward pass, and that its `K` matches `experts.n_experts`.

---

## 2. What the Markov switching gate is fitted on

Not specified in brief 01, and it must be explicit because it is a modelling choice, not plumbing.

Hamilton's model is **univariate**: it needs one series, and the gate is a date-level variable, so
the series is a market-level one, never the cross-section. Make it a config field with a registry of
candidates rather than a hardcoded choice:

    series: str = "market_excess_return"     # registry key
    k_regimes: int = 2                       # ties to experts.n_experts; open (Q18)
    trend: str = "c"
    switching_variance: bool = True
    switching_trend: bool = True
    order: int = 0                           # 0 -> MarkovRegression; >0 -> MarkovAutoregression
    search_reps: int = 20                    # random starts; see section 5
    maxiter: int = 500

`switching_variance=True` is the usual finance setting, because empirical regimes are distinguished
more by volatility than by mean, but it is a default and not a decision: register both and let the
sweep decide. The market excess return series is already available through
`nec_moe/context_data.py` (Kenneth French daily factors), so no new data source is needed.

---

## 3. Causal application, which is where this goes wrong silently

**Fit on the training block. Apply to the test block with the frozen parameters. Never re-fit.**

The mechanics matter:

1. Fit on the training block's series. Store `params`.
2. To obtain probabilities for the test block, construct the model over **train plus test** and run
   the **filter** with the stored `params` (`.filter(params)`, or `.apply(endog, ...)` on the fitted
   results, whichever the installed statsmodels version supports). Do **not** call `fit` again.
3. Take `filtered_marginal_probabilities` over the test dates.
4. Assert that the parameters used for the test block are element-wise identical to those from the
   training fit. If they are not, the model was re-estimated on data it must not see.

Filtering over the extended series is legal because the filter at date t conditions only on the
series through t. Re-fitting is not, because maximum likelihood over the extended series uses every
observation including the future.

**Never `smoothed_marginal_probabilities`, anywhere, for any purpose.** It conditions on the whole
sample. It is also the attribute most people reach for first, and it is the more natural-looking
name. Add a test that fails if the string appears in the gate path, and a comment at the call site
saying why.

**Horizon check.** The target is a forward return over `t` to `t+h`. The filtered probability at `t`
uses the series through `t`. That is legal. Assert the alignment explicitly so that a later change
to the horizon cannot silently shift it.

---

## 4. Regime identity across folds

New problem created by fitting per fold, and not yet handled anywhere.

Each fold fits its own Markov switching model, and the state labelled 1 in fold 3 need not be the
state labelled 1 in fold 4. The likelihood is invariant to permutation of the states, so nothing in
the estimator pins the labels. `canonical_expert_order` solves this for the **experts** by sorting
on sigma; the **gate's own regimes** now need the same treatment and do not have it.

Implement a declared canonical ordering for the fitted gate, applied immediately after each fit and
before the table is built: sort the regimes by the fitted conditional variance, ascending, so that
regime 0 is always the calmest. Record the applied permutation in the fold result so it is
auditable, and make the ordering rule a config field rather than an assumption, since sorting by
mean is the alternative.

This is the practical form of the label switching problem; Stephens (2000) is the reference for why
it cannot be waved away. The escalation path, if a scalar sort proves unstable across folds, is the
template tracking already cited in `diagnostics.py` as arXiv:2603.04441, which is currently a stub.

---

## 5. Local optima are a selection event

The Markov switching likelihood is multimodal and maximum likelihood lands in different places from
different starting values. `search_reps` controls how many random starts are tried and the best is
kept. That is a selection over candidates, exactly like "best of 20 initialisations" elsewhere in
this project.

Log every start's converged log-likelihood to the `TrialRegistry` under its own tag, so the
multiplicity reaches the deflated Sharpe and the corrections. Record the chosen start's seed and the
count. Report convergence failures rather than silently falling back.

---

## 6. The existing HMM prior becomes a baseline

Keep `HMMRegimePrior` exactly as it is behaviourally. Correct its docstring and the README to say
what it actually is:

> A jointly fitted latent Markov mixture with homogeneous, covariate-independent transitions,
> estimated by gradient descent through the forward recursion rather than by maximum likelihood.
> The encoder and gate head are dormant under this prior, which ignores the gate logits and uses
> them for shape inference only. Under Q19 this is a **baseline arm**, not one of the four compared
> regime mechanisms; the Hamilton arm is `MarkovSwitchingRegimePrior`, fitted separately and frozen.

The comparison between the two is itself worth reporting: if the backpropagated filter recovers
transition matrices and expected durations close to the maximum-likelihood ones, that is a defensible
sentence in the methodology. If it does not, better to find out now.

---

## 7. Tests

1. `fit` is called exactly once per fold, with a panel whose maximum date is strictly less than the
   fold's first test date.
2. A `PrecomputedRegimePrior` forward pass before `fit` raises.
3. Parameters used for the test block are identical to the training fit's parameters.
4. The gate path never touches `smoothed_marginal_probabilities` (string-level assertion is
   acceptable here; the failure mode is a careless edit, not a subtle one).
5. On data simulated from a known two-state Markov switching process, the fitted transition matrix
   and expected durations recover ground truth within tolerance, after canonical reordering.
6. Canonical ordering is stable: fitting the same data twice from different seeds yields the same
   regime labelling after the sort.
7. Filtered probabilities for a test date computed by the frozen-apply path equal those computed by
   running the filter manually with the same parameters.
