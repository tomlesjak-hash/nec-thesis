# Code state: what `nec_baseline` actually contains

Recorded 2026-09-21, from a full read of the package. This file is a **record of facts**, not a
task list. The task list is `Code_Change_Brief_2026-09-21.md`. Keep the two separate: this file
says what is true today, that file says what to change.

Location: `Quant Model/nec_baseline`. Package `nec_moe`, 26 modules, roughly 5,950 lines, plus a
further 4,100 lines across 18 test modules. Last touched 16 July 2026. The repository is under git.

---

## 1. The architecture as built

    x_seq --> GRUEncoder --> h_T --> GateHead --> gate logits --> RegimePrior --> log pi (B,K)
    x_snap ------------------------------------> Emission ------> mu (B,K), log sigma (K,)
    y_hat = sum_k pi_k mu_k

The objective is a mixture negative log likelihood computed in the log domain with one fused
`logsumexp`. The Bayes update that produces the responsibilities lives in the loss, not the model,
which is what keeps the recursive path causal.

Two orthogonal plug axes, both selected by a config string:

- **Prior** (`PRIOR_REGISTRY`): `soft`, `uniform`, `hard`, `topk`, `gumbel`, `hmm`.
- **Emission** (`EMISSION_REGISTRY`): `mlp` (K small MLP regressors), `classical` (per regime
  constant Gaussians). `classical` x `hmm` is a Gaussian HMM and is parameter-recovery tested.

**The gate is trained jointly with the experts, by backpropagation, in every registered variant.**
There is no pre-fitted or frozen gate mode anywhere in the package.

## 2. What the HMM prior actually is

`HMMRegimePrior` implements the forward filter's predict step,

    log pi_t[k] = logsumexp_j ( log r_{t-1}[j] + log A[j,k] )

with `A = row_softmax(L)` over a learnable `(K,K)` logit matrix, initialised with a positive
diagonal bias so training starts sticky. It is fitted **by backpropagation through the recursion**,
jointly with the emissions, not by EM or Baum-Welch.

Three properties of it that were not previously written down anywhere:

1. The transitions are **homogeneous and independent of any covariate**. TVTP exists only as a
   guarded config flag that raises `NotImplementedError`.
2. **`gate_logits` is ignored** by this prior; it is used for shape inference only. The module
   docstring states this.
3. It follows that under `prior.kind="hmm"` together with `experts.input_mode="snapshot"`, the GRU
   encoder and the gate head **receive zero gradient**. Under `prior.kind="soft"` the same
   parameters are live. Comparisons across prior kinds are therefore comparisons across different
   effective model sizes unless this is measured and reported. Under
   `input_mode="snapshot_plus_hidden"` the encoder stays live through the expert path, so the
   effect is config dependent rather than universal.

This is a legitimate model but it is **not** Hamilton's Markov switching model as the econometrics
literature estimates it, and must not be described as though it were.

## 3. What is already correct and should not be disturbed

- **Filtering, never smoothing.** The prior at t conditions on data through t-1, the posterior on
  data through t, and `_fold_predictions` warms the filter through the training block before
  scoring the test block. The look-ahead trap that ruins most regime backtests is already closed
  and is covered by tests.
- **Purged expanding walk-forward**, fit once per window, with the purge set to the target horizon.
- **Rank IC, ICIR, long-short portfolio summary** with turnover and cost drag.
- **Selection-aware inference**: append-only `TrialRegistry` where picking a winner is itself a
  logged event carrying the candidate count, feeding deflated Sharpe and Benjamini-Hochberg.
- **Label switching** handled by `canonical_expert_order`, an ascending sigma sort applied after
  each fit before any cross-fit aggregation. Experts are indexed, never named.
- **Load balancing scope**: `LoadBalanceBuffer` estimates the assignment fraction over a running
  buffer spanning many batches rather than one, because regime is a date level variable and
  per-batch balancing would destroy specialisation. Reasoning cited to arXiv:2501.11873, which is
  in `MoE papers`.
- **Point in time universe** with a published coverage column, and an explicit refusal to claim
  survivorship freedom.
- **Anti-leakage in features**: every feature at date t computed from data at or before t, verified
  by `test_no_lookahead`; snapshot features rank normalised cross-sectionally per date.

## 4. What is absent

- **Three of the four gate families.** Only Hamilton-style Markov switching exists, in the form
  described in section 2. TVTP is guarded off. The **statistical jump model** and the
  **Wasserstein** gate do not exist at all.
- **The interface those two gates need.** Neither can be trained by backpropagation: both are
  fitted once on a training window by their own algorithm and then applied causally. `RegimePrior`
  has no `fit(train_panel)` hook and there is no precomputed-prior family.
- **The residual architecture.** The model predicts `y_hat = sum_k pi_k mu_k`. There is no base
  predictor `f0`, no freezing, no zero-initialised corrections, and no correction magnitude
  penalty. Note that `features.py` already uses the word "residual" for a market neutralised
  *target*, which is an unrelated concept; the two must not share a name.
- **A conventionally estimated Markov switching reference** to sit beside the backpropagated
  filter.
- **Persistence diagnostics.** The transition matrix is available but nothing reports expected
  regime duration `1/(1-A_kk)` or the implied stationary distribution.
- **The ICC variance decomposition** of gate weights (date level versus entity level), which is the
  diagnostic that tests whether the gate is behaving as a regime variable at all.
- **The gate permutation test.**
- **CRSP.** Data is yfinance or Stooq through a cache CSV contract. The README already states that
  delisting returns need CRSP and that the default universe is pipeline-verification grade only.

`diagnostics.wasserstein_template_tracking` is exported in `__all__` but is an empty stub. It is
regime *identity tracking across folds*, a different thing from the Wasserstein gate.

## 5. A correction to an earlier claim

It was asserted in conversation on 20 September that the load balancing term has zero gradient and
that this was a finding worth writing up. **That is false of this codebase.** The gate here is a
trainable `LayerNorm -> Linear` head and `load_balance_aux` routes gradient through `gate_probs`
correctly, exactly as in Shazeer et al. (2017). The zero-gradient argument holds only for a
two-stage design with a frozen pre-fitted gate, which is not what is implemented. Do not carry that
claim into the thesis as a statement about *the code*. As a statement about the *design* it is
true again as of the Q19 decision of 21 September, because a frozen gate has no trainable
parameters for the term to act on. Keep the two carefully apart when writing it up.

## 6. The design fork this exposed, now decided

The code trains the gate **jointly** with the experts. The proposal describes four structurally
distinct regime processes **fitted separately and used as the gate**.

**Decided 2026-09-21: fitted separately and frozen, for all four gates.** Recorded in full as Q19
in `Advisor_Questions.md`, which carries the motivation, the accepted cost and the consequences.
The short version: the object being measured is what the experts add on top of a fixed regime
assignment, and only a frozen gate makes that attribution possible.

The code therefore implements a model the thesis is no longer making. What this means concretely:

- The `fit(train_panel)` hook and the precomputed-prior family (section 3 of the change brief) are
  now the **primary** path for every gate, not an accommodation for the two that cannot be
  backpropagated. Build them first.
- `soft`, `hard`, `topk` and `gumbel` become **baseline arms**, not compared mechanisms.
- `GateHead` becomes vestigial on the gate path.
- The load-balancing term becomes **inert**, since the mean gate weight would come from a model
  with no trainable parameters. See section 5, which is now a statement about the code only.

Note that this is a distinct question from Q6 (residual or standard MoE) and Q7 (base pre-trained
and frozen, or trained jointly), both of which concern the base and the experts. Q19 concerned the
gate. Parameters (K, depth, alpha, jump penalty, per-gate hyperparameters) remain open.

## 7. Papers the code already cites

Present in `Quant Model/MoE papers` and referenced from docstrings:

- arXiv:2501.11873, Qiu et al., "Demons in the Detail: On Implementing Load Balancing Loss for
  Training Specialized Mixture-of-Expert Models" — the batch scope argument in `losses.py`.
- arXiv:2205.01565, Li, "Recursive Score and Hessian Computation in Regime-Switching Models" —
  cited in `priors.py` as what backprop through the log-domain recursion obtains automatically.
- arXiv:2603.19136 — cited for the frozen-uniform-gate routing-value ablation.
- arXiv:2603.04441 — cited as the escalation path for Wasserstein template tracking.
- arXiv:2605.14976 — cited as the identifiability caution guarding the TVTP flag.
- arXiv:2410.02241, MIGA — mixture of experts with group aggregation for stock market prediction.
