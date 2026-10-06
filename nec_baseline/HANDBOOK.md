# The NEC Baseline Handbook

**Code companion to `NEC_sources/NEC_baseline_design.md`.** The design doc records *why*
each choice was made (defect ledger, textbook modules, signed-off decisions); this
handbook explains *how the machinery actually works* — down to the tensor shapes and
gradient formulas — and then serves as the operating manual, the roadmap for each
supervisor scenario, and an honest evaluation of what the code is and is not.

**How to use this document.**

- Learning the code? Read **Part I** top to bottom once; it is written to be readable
  without the source open, then to make the source obvious when you do open it.
- Running an experiment? Jump into **Part II** — every recipe is runnable as written
  from `nec_baseline/` with the system `python3.14`.
- Supervisor meeting happened? **Part III** is the work plan for either outcome.
- Wondering whether to trust a number? **Part IV**.

Conventions: `B` batch size, `T` window length, `N` panel rows, `L` number of dates in a
sequence, `d_seq`/`d_snap` feature widths, `n` encoder hidden size, `K` number of
experts. Shapes are written `(B, K)`. Files are in `nec_moe/`. "M4/M5/M7/M8" cite the
textbook modules; "§n" cites the design doc; "Decision A–D" are the signed-off design
decisions (A: experts see snapshot only; B: gate unsupervised; C: end-to-end
differentiable forward filter; D: σ-sorted expert identity, fit-once-per-window).

---

# Part I — The Machinery

## I.0 The big picture

### I.0.1 One equation, four kinds of code

The model is a **conditional mixture**. For one stock-date sample — a trailing window
`x_seq` and a snapshot `x_snap` — the model asserts:

```
p(y | x) = Σ_k  π_k · N(y | μ_k(x_snap), σ_k²)
           ────   ─────────────────────────────
           prior      per-expert likelihood
```

Read it as a two-stage story: *first* decide how much to believe each regime is active
(the prior `π`, produced by the gate), *then* let each regime's specialist predict
(the experts' Gaussians), and mix. Training maximizes the likelihood of observed `y`
under this mixture — nothing else. Every file in the package is one of four kinds of
code:

1. **Prior producers** — `encoder.py` → `gate.py` → `priors.py`. This is the *pluggable*
   half: six mechanisms (`soft`, `uniform`, `hard`, `topk`, `gumbel`, `hmm`) behind one
   interface. The thesis's routing comparison lives here.
2. **Likelihood producers** — `experts.py` (neural or classical emissions) →
   `likelihood.py`. This is the *shared core*: every prior plugs into the same
   likelihood, which is what makes "same model, different routing" literally true.
3. **The Bayes combine** — `losses.py::mixture_nll`. One fused `logsumexp` produces the
   training objective *and* the posterior over experts.
4. **Everything that makes results defensible** — data contracts (`data.py`,
   `features.py`, `crsp.py`, `universe.py`, `context_data.py`), the judging
   machinery (`evaluation.py`, `baselines.py`), diagnostics (`diagnostics.py`,
   `alignment.py`, `calibration.py`, `plots.py`), the honesty layer (`registry.py`,
   `multiple_testing.py`), and the protocol layer that composes them into experiments
   (`sweep.py`, `tuning.py` — I.18).

The two plug axes give a 2×2 of models from one skeleton, all config-selected:

|                        | memoryless prior            | Markov prior (`hmm`)      |
|------------------------|-----------------------------|---------------------------|
| **neural emission**    | the corrected NEC           | HMM-gated NEC (Var. 3)    |
| **classical emission** | feature-gated mixture       | classical Hamilton        |

### I.0.2 Shape walkthrough with real numbers

Take the real Stage-B defaults: `T=20`, `d_seq=4`, `d_snap=14`, `n=32`, `K=2`, batch
`B=256`. One forward pass:

| step | module | in | out | what it is |
|---|---|---|---|---|
| 1 | `GRUEncoder` | `x_seq (256, 20, 4)` | `h_T (256, 32)` | learned rolling summary of each window |
| 2 | `GateHead` | `h_T (256, 32)` | `z (256, 2)` | regime *logits* (never probabilities) |
| 3 | `RegimePrior` | `z (256, 2)` [+ state] | `log π (256, 2)` | normalized log prior, `logsumexp_k = 0` |
| 4 | `Emission` | `x_snap (256, 14)` | `μ (256, 2)`, `log σ (2,)` | per-expert mean + noise scale |
| 5 | `NECModel` | 3 + 4 | `ŷ = Σ_k π_k μ_k (256,)` | mixture-mean point prediction |
| 6 | `expert_log_likelihood` | `μ, log σ, y` | `log N (256, 2)` | how well each expert explains what happened |
| 7 | `mixture_nll` | `log π + log N` | `NLL (scalar)`, `r (256, 2)` | objective + Bayes posterior over experts |

Parameter budget at these sizes: GRU ≈ 3·32·(32+4+1) = 3,552 (+biases), gate LayerNorm
64 + Linear 66, each expert 14→64→32→1 ≈ 3.1k. Total ≈ 11k parameters — versus the
prototype's ~4M. That is deliberate (M7's effective-sample arithmetic): with a few
thousand effective observations, parameters are variance.

### I.0.3 The life of one memoryless training step

What `Trainer.train_step(batch)` actually does, in order:

1. `model.train()`; apply the σ schedule (freeze `log_sigma` if
   `step < sigma_freeze_steps`).
2. Build `PriorContext(step=step_count)` — even memoryless priors get the step, because
   the Gumbel prior schedules its temperature off it.
3. Forward (I.0.2 steps 1–5) → `NECOutput`.
4. `expert_log_likelihood(μ, log σ, y)` → `log N (B, K)`.
5. Pick the training weights: `prior.log_train_weights` if the prior set them (only
   `hard` does), else `prior.log_prior`.
6. `mixture_nll` → per-sample NLL, mean NLL, attached `log_filtered`, detached
   `responsibilities`.
7. Update the load-balance buffer with the responsibilities (always — it feeds the
   running-utilization dashboard even when the aux loss is off).
8. Compose the loss: `NLL (+ α_bal·L_bal) (+ α_div·L_div) (+ prior.aux_loss)` — the aux
   terms only if config-enabled (default off).
9. `zero_grad → backward → clip_grad_norm(grad_clip) → opt.step()`; `step_count += 1`.
10. Compute the metrics dict from detached tensors (II.3 lists every key) and append to
    `trainer.history`.

The gradients that flow in step 9 (derived in I.9): gate logits receive `π − r`, expert
k's mean receives `r_k(μ_k − y)/σ_k²`, `log σ_k` receives `r_k[1 − (y−μ_k)²/σ_k²]`.
One SGD step on the mixture NLL **is** a generalized-EM step (M8): the E-step
(responsibilities) is implicit in the forward, the M-step nudge is the update.

### I.0.4 The life of one HMM sequence chunk

`Trainer.train_step_sequence(chunk, init_state)` — `chunk` is a list of per-date
`Batch`es in chronological order (batch dimension = entities):

1. `state ← init_state` (log-domain filtered posterior from the previous chunk,
   **detached** there; `None` at a sequence start → the prior uses `π₀`).
2. For each date-batch `t` in the chunk:
   a. Forward with `PriorContext(prev_filtered=state)`. The HMM prior computes the
      **predict step** `log π_t = logsumexp_j(log r_{t−1,j} + log A[j,·])` — this is the
      *prediction-time* prior, using only data through t−1.
   b. Likelihood + `mixture_nll` → per-sample NLL and `log_filtered` — the **update
      step**, now conditioning on `y_t`.
   c. `state ← log_filtered` **attached**: gradient will flow backward through time.
3. Loss = mean of all per-sample NLLs in the chunk; one optimizer step; return the final
   state for the caller to **detach** and carry into the next chunk.

This is truncated BPTT over the regime posterior. Backprop through the recursion is the
exact recursive score computation of the regime-switching literature
(arXiv:2205.01565), obtained automatically — the substance of Decision C. Chunking
bounds memory/compute per step; detaching at chunk boundaries truncates gradient flow,
not the filter itself (the *values* still thread exactly).

## I.1 `config.py` — one tree, loud failures

All configuration is one frozen dataclass tree; modules take their sub-config, never
loose ints, so there is exactly one place a dimension can be written.

**`DataConfig`** — `d_seq`, `d_snap`, `seq_len` (required); optional
`sequence_features`/`snapshot_features` name tuples (lengths cross-checked) and
`target`. Build it from a real panel with `data_config_from_panel(panel)`.

**`EncoderConfig`** — `hidden_dim=32`, `num_layers=1`, `dropout=0.0` (inter-layer only;
validation demands `num_layers ≥ 2` if set — a silent no-op dropout is exactly the kind
of thing this codebase refuses).

**`GateConfig`** — `zero_init=True`. The gate's *input* width is deliberately absent:
it is `encoder.hidden_dim` by construction, so the prototype's 256-vs-64 mismatch
(ledger defect 8) is unrepresentable, not just checked.

**`ExpertConfig`** — `n_experts=2` (must be ≥ 2 — a "mixture" of one is a lie),
`hidden_dim=64`, `dropout=0.05`, `input_mode="snapshot"` (Decision A;
`"snapshot_plus_hidden"` concatenates `h_T` — the pre-registered ablation),
`kind="mlp"` (`"classical"` = per-regime constant Gaussians).

**`PriorConfig`** — `kind="soft"` (registry key: soft/uniform/hard/topk/gumbel/hmm).
HMM-only: `transition_diag_bias=2.0` (persistence-biased init), `learn_pi0=True`,
`tvtp=False` (**guarded**: selecting it raises `NotImplementedError` pointing at §9bis's
identifiability caution). Top-k-only: `top_k=2` (validation enforces `2 ≤ k ≤ K`; see
I.8 for why k=1 is forbidden). Gumbel-only: `tau_init=1.0`, `tau_min=0.1`,
`tau_anneal_steps=0` (0 = constant temperature).

**`TrainConfig`** — the knobs you will actually touch:

| field | default | meaning / when to change |
|---|---|---|
| `lr` | 1e-3 | AdamW learning rate. 3e-3 works for the small synthetic models; 3e-2 for the tiny classical Hamilton. |
| `weight_decay` | 1e-4 | decay on weight matrices (encoder, experts). |
| `gate_weight_decay` | 1e-3 | separate decay for the gate's linear weight — *structural* (M4 separation: an unpenalized softmax head diverges on near-separable vol features). |
| `batch_size` | 128 | memoryless minibatch size. 256–512 fine at these scales. |
| `steps` | 500 | default for `fit`/`fit_sequence` when not overridden. |
| `grad_clip` | 5.0 | global norm clip; `None` disables. |
| `sigma_init` | 1.0 | initial per-expert σ. **Set near `std(y)`** — for 5-day real returns that's ~0.05, for the synthetic default ~1.5. |
| `sigma_freeze_steps` | 100 | freeze `log_sigma` first — let the means move before σ can explain everything as noise. |
| `aux_load_balance` | False | Shazeer-form load balancing (buffered scope). An ablation lever, not a default. |
| `aux_load_balance_weight` | 1e-2 | its α. |
| `load_balance_buffer_batches` | 32 | the *scope* of `f̄` — must span many dates (I.9). |
| `aux_expert_decorrelation` | False | RAVEN-style output decorrelation. Ablation lever. |
| `aux_decorrelation_weight` | 1e-2 | its weight. |
| `seed` | 0 | expert-diversification and shuffling seed. |
| `sequence_ordered` | False | **must be True for `prior.kind="hmm"`** — validation enforces the coupling. |
| `log_every` | 50 | reserved for future periodic logging. |

`NECConfig.validate()` runs inside `NECModel.__init__`; every error carries the expected
and actual values. `to_dict()`/`from_dict()` round-trip the tree — that is what the
trial registry stores, so any run is reconstructible from its registry row.

## I.2 `utils.py` — two deliberate trivialities

`assert_shape(t, (None, T, d), "x_seq")` — `None` is a wildcard; failure raises
`ValueError: shape mismatch for 'x_seq': expected (*, 20, 4), got (256, 20, 3)`. Every
`forward` in the package opens with one; shape bugs die as sentences, not broadcast
surprises. `set_seed(s)` seeds python/numpy/torch together.

## I.3 `data.py` — the contract everything stands on

### The `Panel`

```
x_seq  (N, T, d_seq) float32   trailing windows, oldest → newest inside the window
x_snap (N, d_snap)   float32   prediction-date snapshot features
y      (N,)          float32   forward return target (the ONLY forward-looking column)
date   (N,)          int64     date codes 0..D-1
entity (N,)          int64     stable name ids (weight alignment across dates)
schema               FeatureSchema — names every column; order == channel order
date_labels / entity_labels    optional code→calendar-date / code→ticker maps
```

Three invariants worth internalizing:

- **Date-major row order.** All of date 0's entities, then date 1's, etc. Example with
  3 entities: rows are `(d0,e0)(d0,e1)(d0,e2)(d1,e0)…`. The stateful trainer's per-date
  threading and the harness's stateful scoring both *assume* this; boolean-mask
  subsetting preserves it, which is why every subset operation is a mask.
- **No reserved columns, no in-model slicing** (defect 7). The prototype hid its target
  in the trailing feature column and sliced it off with `x[:, -1, :-1]`. Here the target
  is its own tensor, the schema names every feature, and `Batch.validate_schema`
  width-checks — a smuggled column *changes the width* and fails loudly.
- **Subsetting via `_take(idx)`** = `dataclasses.replace(self, **self._row_kwargs(idx))`.
  Subclasses extend `_row_kwargs`, so `SyntheticPanel`'s extra row-aligned fields
  (`regime`, `y_clean`) subset automatically and can never desynchronize.

Methods you'll use: `full_batch()`, `minibatches(bs, generator)` (shuffled — memoryless
training only), `time_sequence()` (chronological per-date `Batch` list — the stateful
trainer's input), `subset_dates(dates)`, `split_by_date(frac)` (chronological; there is
deliberately no random split in the package).

### The synthetic truth machine

`SyntheticRegimePanel(SyntheticSpec).generate(n_dates, n_entities)` produces a
`SyntheticPanel` with the *truth attached*: `regime (N,)`, `y_clean` (noiseless
conditional mean), `betas (K, d_snap)`, `transition (K,K)` (markov mode).

The generating process, exactly:

1. A date-level regime path `z_d`: iid draws from `marginal` (default uniform) or a
   sticky Markov chain with `transition_stay` on the diagonal. *Date-level* — all
   entities share a date's regime — mirrors reality and is the premise behind the
   load-balancing scope rule (I.9).
2. Betas: for K=2, `β₁ = −β₀` (norm `beta_scale`) — **sign-flipped**, so the pooled
   cross-sectional relationship is exactly zero. This is the pivotal design: a model
   without regime structure is *structurally* blind here, which is what makes the
   MoE-vs-baseline comparison (I.14) evidential.
3. `y = x_snap·β_{z} + ε`, `ε ~ N(0, noise_std²)`; `x_snap ~ N(0, I)`.
4. The window: channel 0 of `x_seq` is `N(0, vol_levels[z]²)` — the regime is inferable
   from window volatility, but noisily. Other channels are pure noise.

`SyntheticSpec` knobs: `n_regimes, d_seq=2, d_snap=3, seq_len=8, regime_process
("iid"|"markov"), marginal, transition_stay=0.95, vol_levels=(0.5, 2.0),
beta_scale=2.0, noise_std=0.5, seed`. Two levers matter for experiments:
`vol_levels` spacing sets how identifiable the regime is from one window (close levels
⇒ the HMM's persistence advantage appears); `noise_std` vs `beta_scale` sets the SNR of
the prediction task itself.

## I.4 `encoder.py` — a small GRU, on purpose

`GRUEncoder: x_seq (B,T,d_seq) → h_T (B,n)` — the final hidden state of the last layer.
The GRU cell per step (M7):

```
z_t = σ(W_z x_t + U_z h_{t-1})          update gate
r_t = σ(W_r x_t + U_r h_{t-1})          reset gate
h̃_t = tanh(W x_t + U (r_t ∘ h_{t-1}))   candidate
h_t = (1 − z_t) ∘ h_{t-1} + z_t ∘ h̃_t   convex blend
```

When `z_t → 0` the state copies through — the gradient highway that suppresses
vanishing gradients over the 20-step window. `h_T` is best understood as a **learned
rolling statistic**: the model's own version of "realized vol over the window", except
the kernel is learned.

Choices and their reasons: GRU over LSTM = `3n(n+d+1)` vs `4n(n+d+1)` params/layer —
25% variance saved at identical function class quality on short windows (M7).
`bidirectional` is not even an option — a backward pass reads the future side of the
window and the baseline refuses to open that leakage question. No manual `h0` (PyTorch
zero-initializes). Strict `assert_shape` on the way in *and* out.

## I.5 `gate.py` — ten lines, three defect fixes

`GateHead: h_T (B,n) → LayerNorm(n) → Linear(n,K) → z (B,K)`, zero-initialized.

**Why LayerNorm and never BatchNorm** (defect 4): BatchNorm's train-mode statistics
couple every sample to its batch-mates. With date-sliced batches that is *cross-date
information flow directly upstream of the regime decision* — a leakage channel and a
train/eval behavior split. LayerNorm normalizes each sample by itself. The guard is
behavioral, not textual: `test_no_cross_batch_coupling` replaces every other row of a
train-mode batch and asserts row i's output is bit-identical (and demonstrates a
BatchNorm module failing the same probe).

**Why logits and never probabilities** (defect 3): write the loss as a function of
probabilities and differentiate: `∂(−log p̂_c)/∂p̂_k = −y_k/p̂_k` — unbounded exactly when
an early, confused gate assigns the true class low probability. Compose through the
softmax Jacobian `∂p̂_k/∂z_j = p̂_k(δ_kj − p̂_j)` and the singular factors cancel
*algebraically*: `∂L/∂z = p̂ − y`, bounded in [−1,1] (M4 §4.10). The fused
`log_softmax`-inside-the-loss computes the cancelled form; an in-model `nn.Softmax`
followed by a log later computes the explosive chain. Hence: no `nn.Softmax` anywhere in
the model tree (tested).

**Why zero-init**: `z ≡ 0` ⇒ π exactly uniform ⇒ gate entropy exactly `log K` at step 0
⇒ no expert is born dead. "A confident random gate at t=0 is a collapsed mixture before
training starts" (M8 capstone). The smoke test asserts the `log K` starting entropy.

## I.6 `experts.py` — the emission axis

`Emission` is the second plug axis: `x (B,d) → (μ (B,K), log σ (K,))`, built from
`EMISSION_REGISTRY[cfg.experts.kind]` exactly like the priors.

**`ExpertBank`** (`"mlp"`): K structurally identical `ExpertMLP`s — the funnel
`d → h → h/2 → 1` with ReLU and dropout — each **re-initialized from its own seeded
generator** (`seed + 7919·(k+1)`, uniform `±1/√fan_in`, the PyTorch default law):
symmetry-breaking level 1. Plus one `(K,)` learnable `log_sigma` — homoscedastic per
expert. σ is a *statistical parameter*, not a weight: it enters the likelihood, is
excluded from weight decay (I.11), and is frozen early (`sigma_freeze_steps`).

Experts are **indexed, never named**. Label switching (M8 failure mode 2) means "expert
0" carries no stable meaning across runs or folds; naming one "extreme" in code would be
a lie waiting to happen. Semantics are assigned post hoc: `canonical_expert_order`
(σ-sort) for cross-fit alignment, `alignment.py` for interpretation.

**`ClassicalGaussianEmission`** (`"classical"`): learned scalars `(μ_k, σ_k)`; the input
is ignored except for its batch size. With the HMM prior this *is* Hamilton's
observation model; with a memoryless prior it is a feature-gated classical mixture.

**`warmstart_slices(x, y, slices)`** is deliberately polymorphic — each emission breaks
symmetry its own way: the neural bank pre-trains expert k by MSE on sample-slice k
(Adam, ~150 steps); the classical emission sets `(μ_k, log σ_k)` to slice moments in
closed form (quantile initialization, exactly the GMM init trick). Why a warm-start
exists at all is an empirical finding — see I.11.

## I.7 `likelihood.py` — the shared core, and the cliff it avoids

```
log N(y | μ_k, σ_k²) = −½·log 2π − log σ_k − (y − μ_k)² / (2σ_k²)
```

computed *directly in the log domain* as `-0.5*log2π - logσ - 0.5·resid²·exp(-2 logσ)`.
The arithmetic of the cliff (M8 Ex. 8.1): float64 dies below ~`e^−745` (smallest
subnormal), so the naive pipeline — exponentiate each density, multiply by weights, sum,
take log — returns exactly 0.0 whenever every component's squared standardized residual
exceeds ~1490 (a ~39σ point). Fat-tailed return panels produce such points routinely;
the naive mixture then takes `log(0)`. The fused log-domain path never exponentiates a
density at all: `logsumexp` does its own max-shift stabilization.

The function is pure — no parameters, no state, no module. That is the point: both
thesis variations consume *this exact function*, so "same likelihood, different prior"
is a property of the code, not a description of it.

## I.8 `priors.py` — six ways to form the prior

The contract: a `RegimePrior` maps gate logits (and, if `stateful`, the previous
filtered posterior in `PriorContext.prev_filtered`) to
`PriorOutput.log_prior (B,K)` — normalized (`logsumexp_k = 0`), with `−inf` permitted
for sparse mechanisms (those entries contribute `e^{−inf} = 0` to the mixture's
`logsumexp` and receive exactly zero responsibility; no special-casing anywhere).
`PriorOutput.log_train_weights` optionally carries *different* weights for the training
objective (only `hard` uses it); `PriorOutput.info` carries diagnostics (selected
indices, temperature, `log A`).

### `soft` — the baseline, and the gradient that fixes defect 1

`log π = log_softmax(z)`. Derive its gradient once and the whole corrected NEC makes
sense. Per sample, the loss is `L = −logsumexp_k(a_k)` with `a_k = log π_k + log N_k`;
by the LSE gradient, `∂L/∂a_k = −softmax(a)_k = −r_k` (the responsibility). Chain
through `log π = z − logsumexp(z)`:

```
∂L/∂z_j = Σ_k (−r_k)(δ_kj − π_j) = π_j − r_j
```

**The gate is doing cross-entropy toward soft labels that the experts negotiate.** When
expert 1 explains today's returns better than its prior share (r₁ > π₁), the gate is
pushed to route more to expert 1 *on days that look like today*. The prototype's argmax
made this gradient identically zero — that single fact is defect 1, and `π − r` is its
repair. (Verified analytically: `test_gate_gradient_is_pi_minus_r`.)

### `uniform` — the ablation that measures routing itself

`log π ≡ −log K`, constant. Because it does not depend on `z`, the gate and encoder
receive *no gradient by construction* — which is the ablation's meaning: experts +
mixture, zero learned routing. On the sign-flip panel it scores IC ≈ 0 (the frozen
50/50 mixture of opposite experts has mean ≈ 0) — the cleanest demonstration that on
regime-structured data, *routing is where the value is* (echoing arXiv:2603.19136's
ablation, where removing the detection module cost the most).

### `hard` — top-1 that actually trains

Two weight sets, and the distinction is the mechanism:

- **Predictive** `log_prior`: one-hot (0 at the argmax, −inf elsewhere) ⇒
  `ŷ = μ_selected`, and held-out NLL is the selected expert's density. Honest hard
  routing.
- **Training** `log_train_weights`: `log softmax(z)_sel` at the argmax, −inf elsewhere
  ⇒ objective `−(log π_sel + log N_sel)`. This is the hard-EM / Viterbi surrogate
  (responsibilities hardened to the argmax indicator; an upper bound on the mixture
  NLL). Its gate gradient: with only the selected coordinate active,
  `∂L/∂z_j = π_j − 1[j = sel]` — **self-training toward the gate's own argmax**, scaled
  into the objective exactly the way Switch Transformer scales the selected expert by
  its gate probability. That is what makes top-1 *trainable*, in precise contrast to
  the prototype's gradient-dead bare argmax.

Known pathology, visible in the demo sweep and worth citing in the thesis: from a
zero-init gate the argmax is arbitrary, and self-training *reinforces* the arbitrary
choice (cold start). Mitigations are Part III.1 work (soft-then-hard warm switch, or
logit noise).

### `topk` — and the k = 1 theorem

Mask all but the k largest logits to −inf, then `log_softmax`. Gradient flows to the
gate only through *competition among the kept entries* (the renormalization). Set k = 1
and the single kept weight is `log(1) = 0` — a constant: **top-1-renormalized is
gradient-dead, i.e. ledger defect 1 reborn in sparse clothing.** This was discovered by
a failing test and is now a loud config error (`top_k ≥ 2`, message pointing to `hard`
for trainable top-1). `k = K` reduces exactly to `soft` (tested to 1e-7).

### `gumbel` — stochastic routing with an annealed temperature

Train mode: `log π = log_softmax((z + g)/τ)` with `g = −log(−log U)`, `U ~ Uniform` —
i.i.d. Gumbel noise. The Gumbel-max trick: `argmax_k(z_k + g_k)` is an *exact* sample
from `Categorical(softmax(z))`; finite τ trades that exactness for differentiability
(Foundations E), and `test_gumbel_max_trick_sampling_frequencies` verifies the sampling
law empirically at τ = 0.01. Temperature anneals geometrically from `tau_init` to
`tau_min` over `tau_anneal_steps`, driven by `PriorContext.step`. Eval mode is
deterministic `log_softmax(z)` — the noise is a training device, and eval must be
reproducible.

### `hmm` — the forward filter's predict step (Variation 3)

State: an unconstrained logit matrix `L (K,K)`; `A = row_softmax(L)` is *always* a valid
stochastic matrix under unconstrained optimization — no projections, no constrained
optimizers. Init adds `transition_diag_bias` to the diagonal: `softmax([2,0]) ≈
[0.88, 0.12]`, so training **starts sticky** (financial regimes are assumed persistent —
the transition-matrix analogue of initializing an LSTM forget gate to 1). `π₀ =
softmax(pi0_logits)` serves sequence starts.

The predict step, per entity `b`:

```
log π_t[b,k] = logsumexp_j ( log r_{t−1}[b,j] + log A[j,k] )
```

The state `r` is carried **by entity**, not by row position (audit M-5): the trainer
keeps each date's posteriors with their entity codes (`FilterState`), looks every row's
previous posterior up by its entity, starts a new entrant from `π₀` and drops an exit.
So a point-in-time panel whose membership changes runs, and the row order within a
date does not matter. (A date-level regime would be a different model; not decided.)

**A worked numeric pass** (K=2, `A = [[0.9, 0.1], [0.1, 0.9]]`, start `π₀ = [0.5, 0.5]`,
expert densities at t=1: `N₁ = 0.05` (calm expert, poor fit), `N₂ = 0.30` (stress
expert, good fit)):

```
t=1 predict: π₁ = π₀ᵀA           = [0.5, 0.5]              (uniform in, uniform out)
t=1 update:  r₁ ∝ π₁ ∘ [N₁, N₂]  = [0.025, 0.150] → [0.143, 0.857]
t=2 predict: π₂ = r₁ᵀA           = [0.143·0.9 + 0.857·0.1, …] = [0.214, 0.786]
```

The stress evidence at t=1 *persists into t=2's prior* — that is the entire advantage of
Markov gating, and why it wins exactly when per-date evidence is weak but regimes are
sticky (the pinned test: `test_hmm_recovers_persistent_regime`).

Three structural facts:

- **Filtering, never smoothing.** The prior at t conditions on data through t−1 only.
  Smoothing (forward–backward) conditions on the *future* — lookahead leakage, the same
  bug class as the whole defect ledger. `test_filtering_is_causal_no_lookahead` perturbs
  a future date and asserts every earlier filtered posterior is bit-identical, and shows
  a smoothing implementation failing the same probe.
- **With a static `A`, the encoder and gate are dormant** — the prior ignores the
  logits; they re-enter only via the guarded TVTP extension. This is faithful Hamilton
  gating: in Variation 3 the *neural* upgrade lives in the emission, not the gate.
- **`gate_logits` is still an argument** (used for batch-size inference) so the
  interface is uniform — swapping prior families changes no call sites.

### `markov` — the Hamilton gate, and the gate weight it serves (Q27)

`nec_moe/markov_gate.py`: Hamilton's Markov switching model, fitted by maximum likelihood
on each fold's training block, canonically ordered, frozen, and applied to later dates by
the filter alone (brief 03). The table it serves is `MarkovGateConfig.gate_weight`, built
from the filtered probability `ξ_t` and the frozen canonical transition matrix `A` by
`gate_weight_probs`:

| `gate_weight` | row served at t | reading |
|---|---|---|
| `"filtered"` | `ξ_t` | today's regime |
| `"predicted"` | `ξ_t A` | tomorrow's regime |
| `"window"` (default, decided) | `(1/h) Σ_{j=1..h} ξ_t A^j` | expected share of the target window `(t, t+h]` in each regime |

`h` is the panel target's horizon (`DataConfig.horizon_periods`, read from the target
name), never a second field; the gate refuses a panel whose horizon differs from the one
it was built for. With `h = 1` the window weight is the predicted one; with `A = I` it is
the filtered one. The same rule runs on training and test dates; `prob_floor` and the
renormalisation come after it. The backprop-HMM baseline (`hmm` above) is **not**
changed: its latent is per target window, so its `h`-step predict is already the
consistent weight for it. Gates still to come (jump model, Wasserstein, TVTP) must supply
a transition matrix before they can serve `"predicted"` or `"window"`. Every trial row
records `gate_weight` (`None` without a Hamilton gate).

## I.9 `losses.py` — the fused objective, and the two levers

### `mixture_nll` — five lines that carry everything

```
a            = log_prior + log_lik          # (B, K); −inf-safe
per_sample   = −logsumexp_k(a)              # the NLL
log_filtered = a − logsumexp(a)             # Bayes posterior, log domain
nll          = per_sample.mean()
```

The **attached/detached distinction** is the subtlest correctness decision in the
package:

- `log_filtered` returns **attached to the graph**, because for the HMM it *is* the
  recursion's state — gradient must flow through it to earlier timesteps (that is BPTT
  over the posterior, i.e. Decision C). If someone "tidies" this with a `.detach()`, the
  transition matrix silently stops learning from multi-step evidence; the guard is
  `test_hmm_recovers_persistent_regime` (the learned `A` would stay at its init) and the
  gradient-flow assertions in the filter tests.
- `responsibilities` returns **detached** — dashboards and the LB buffer only. Letting
  diagnostics leak gradient would make monitoring alter training, the quietest possible
  bug.

Gradient table (all verified against autograd in `test_loss_math.py`):

| parameter | per-sample gradient | stationarity reading |
|---|---|---|
| gate logits | `π_k − r_k` | gate predicts the responsibilities |
| expert mean | `r_k (μ_k − y)/σ_k²` | responsibility-weighted least squares |
| `log σ_k` | `r_k [1 − (y−μ_k)²/σ_k²]` | σ_k² = responsibility-weighted MSE |

SGD on this NLL **is generalized EM** (M8 §8.6): tight E-step at the current parameters
(the responsibilities in the forward), partial M-step (the gradient step). That
equivalence is why EM's vocabulary — responsibilities, effective counts, collapse — is
the right diagnostic language for a model trained by Adam.

**Per row is the per-date model's composite likelihood (Q24, decided 2026-10-05).** The
model has one regime per date, shared by all stocks; integrating out the other stocks
leaves each stock's own distribution `Σ_k π_k(t) N(y_i; μ_k(x_i), s_k²)`, which is exactly
one row's term. The mean over rows is therefore that model's independence (composite)
likelihood: consistent, robust to stocks co-moving beyond the regime, and it keeps the
frozen gate in charge of which data each expert learns from (one stock's evidence is
small). Costs: efficiency and no Hessian standard errors, neither used here. The
objective is unchanged; Appendix D of `Master Thesis/MoE_HMM_Gate_Formulation.pdf` has
the theory. The softer specialisation it allows is watched by
`diagnostics.expert_weight_diagnostics` (I.12).

### `LoadBalanceBuffer` + `load_balance_aux` — and the scope theorem

Shazeer form `L_bal = K · Σ_k f̄_k · P̄_k`: `P̄` (batch-mean gate probability) carries
the gradient; `f̄` (argmax-assignment fractions) is a stop-grad scale. The entire design
question is **the scope over which `f̄` is estimated** (arXiv:2501.11873): our batches
are date-sliced and regime is date-level, so a per-batch `f̄` demands that *one calm
day's* stocks be spread over the calm and stress experts — the exact
specialization-killer. `f̄` therefore comes from a **running buffer** over the last
`load_balance_buffer_batches` batches (spanning many dates).

The two-line theorem the tests pin: with a balanced buffer, `f̄_k = 1/K` ⇒
`L_bal = K·Σ_k (1/K)·P̄_k = Σ_k P̄_k ≡ 1` — *identically constant in the routing*, zero
gradient. Balanced-marginal load balancing never fights sharp within-batch routing;
per-batch scope always does. The buffer updates every step even with the aux off — it
feeds the running-utilization dashboard.

### `expert_decorrelation_aux` — the other failure mode

`L_div = ‖R − I‖_F²`, `R` = correlation matrix of the experts' outputs over the batch.
It targets **homogenization** — all experts drift to the pooled regression, outputs
correlate to 1, responsibilities pin at uniform, the gate starves (`π − r ≡ 0`). Note
the duality: load balancing fights *collapse onto one expert* (utilization → one-hot);
decorrelation fights *everyone-learns-the-same-thing* (utilization stuck uniform). Both
default **off**: each biases the likelihood (regimes may genuinely agree on part of the
cross-section), so both are pre-registered ablation levers, not baseline terms.

## I.10 `model.py` — assembly, and the predict/update split

`NECModel.__init__` validates the config, then assembles encoder / gate /
`build_prior(cfg)` / `build_emission(cfg)`. `forward(x_seq, x_snap, ctx)` returns:

| `NECOutput` field | shape | meaning |
|---|---|---|
| `gate_logits` | `(B,K)` | raw gate scores (pre-prior) |
| `prior` | `PriorOutput` | `log_prior`, optional `log_train_weights`, `aux_loss`, `info` |
| `mu` | `(B,K)` | expert means |
| `log_sigma` | `(K,)` | expert noise scales |
| `y_hat` | `(B,)` | `Σ_k π_k μ_k` — the mixture-mean point prediction |
| `h_t` | `(B,n)` | encoder state (diagnostics; Decision A's concat variant) |

**The forward never sees `y`.** That is a design statement, not an accident: forward =
the *predict* half (prior formation + expert means + `ŷ`); the Bayes *update*
(responsibilities / filtered posterior) lives in `mixture_nll`, which does see `y`. For
the HMM this split is exactly what keeps predictions causal — `ŷ_t` is computed from
`r_{t−1}` before `y_t` enters anything. For every prior it means `y_hat` is always a
legitimate out-of-sample prediction.

`expert_input(x_snap, h_t)` implements Decision A: return `x_snap`, or concatenate
`h_T` (`input_mode="snapshot_plus_hidden"`). Construction, never slicing.

## I.11 `train.py` — the Trainer's two worlds

### Optimizer hygiene (`_param_groups`)

| group | members | decay | why |
|---|---|---|---|
| gate | `gate.linear.weight` | `gate_weight_decay` | structural: separation-proofing (M4) — near-separable vol features push an unpenalized softmax head to ‖w‖→∞ |
| no-decay | every 1-D parameter (biases, LayerNorm scales, `log_sigma`, classical `μ`, `π₀`) **and all `prior.*` parameters** | 0 | 1-D params are statistical/normalization quantities, not weight matrices; decaying transition logits would pull `A` toward uniform — i.e. silently fight persistence |
| default | remaining weight matrices (GRU, expert MLPs) | `weight_decay` | ordinary capacity control |

`log_sigma.requires_grad` is toggled by the σ schedule: frozen while
`step < sigma_freeze_steps` (let the means commit before σ can absorb everything as
noise — the heteroskedastic-head caution).

### The memoryless path

`train_step(batch)` is I.0.3. `fit(panel, steps)` cycles shuffled minibatches from a
seeded generator. It **raises** if the prior is stateful — shuffled batches would
silently skip the forward recursion, and silent is the one thing this codebase refuses
to be.

### The stateful path

`train_step_sequence(chunk, init_state)` is I.0.4;
`fit_sequence(sequence, steps=…, chunk_len=50)` drives it:

```
sequence:  [b₀ b₁ … b₄₉ | b₅₀ … b₉₉ | …]        chunks of chunk_len dates
            └── chunk 0 ──┘└── chunk 1 ──┘
state:     π₀ ─▶ r₄₉ ─detach─▶ r₉₉ ─detach─▶ …   values thread exactly;
                                                  gradients truncate at chunk edges
```

Each optimizer step consumes one chunk; the carried state is detached between chunks
(truncated BPTT over the regime posterior) and reset to `π₀` at each pass over the
sequence start. `chunk_len` trades gradient horizon against step cost; ~50 dates works
for the tests, and there is no free lunch — a longer horizon back-propagates persistence
evidence further but costs linearly more per step.

### The warm-start, and why it exists

`warmstart_experts(batch, sort_key, steps=150, lr=1e-2)`:

1. Compute the expert inputs once (encoder under `no_grad` — the gate is untouched).
2. `argsort(sort_key)` → split into K contiguous quantile slices.
3. Delegate to `emission.warmstart_slices(x, y, slices)` — MSE pre-training per expert
   (neural) or closed-form slice moments (classical).

The empirical finding that forced this (worth knowing because you will meet it on any
symmetric task): on the sign-flip panel, **seed diversity alone never escapes the pooled
local optimum.** Both experts converge to the marginal regression (μ ≈ 0, σ absorbing
the bimodal variance), responsibilities never separate, and the gate has nothing to
learn — gate AUC ≈ 0.5 across every σ schedule tried. Sorting by an *observable* regime
proxy (window vol here; VIX/realized vol on real data) and giving each expert one slice
breaks the symmetry deliberately and interpretably. Decision B survives intact: **only
the emission is touched — the gate stays unsupervised** and must still learn to route
via `π − r`. Measured effect: responsibility→regime AUC ≈ 0.99 immediately after
warm-start; gate AUC ≈ 1.0 after joint training; ≈ 0.5 without.

### Evaluation

`evaluate(batch)` (memoryless) and `evaluate_sequence(sequence, init_state) →
SequenceEval(nll, log_filtered (L,B,K), log_prior (L,B,K), y_hat (L,B))` — a no-grad,
strictly causal filtering pass. Both evaluate under the prior's **predictive** weights
(`train_objective=False`), so hard routing's held-out NLL is its honest
selected-expert density, not its training surrogate.

## I.12 `diagnostics.py` — failure modes as dashboards

M8's capstone monitoring, as pure functions over detached tensors:

| function | formula | healthy | pathological |
|---|---|---|---|
| `gate_entropy(π)` | `mean_b −Σ_k π log π` | starts at `log K`, declines gradually | stuck at `log K` = dead gate; ≈0 instantly = collapse |
| `utilization(r)` | `mean_b r_bk` per expert | per-*batch* extremes are fine on one-regime dates | running scope near 0 = dead expert (its gradients vanish with its responsibilities) |
| `sharpness(r)` | `mean_b max_k r_bk` | 1/K → up gradually | 1.0 immediately = collapse |
| `expert_output_correlation(μ)` | corr matrix of expert outputs | off-diag well below 1 | ≈1 = homogenization (I.9's decorrelation target) |

`binary_auc` is a dependency-free Mann–Whitney implementation (tie-averaged ranks);
`regime_recovery_auc` reports `max(auc, 1−auc)` because *which* learned expert maps to
*which* true regime is a permutation the likelihood cannot identify — orientation is
never part of a claim. `canonical_expert_order(log_sigma)` is Decision D's declared
statistic: sort experts by ascending σ after *every* fit, before any cross-fit table.
`wasserstein_template_tracking` is a documented stub — the escalation path if σ-sorting
proves unstable across refits (near-ties), per arXiv:2603.04441.

`expert_weight_diagnostics(responsibilities, gate_prob, abs_residual)` (Q24): on the
training rows, per expert (canonical order), the correlation of its responsibility with
the gate's probability (`ew_corr_resp_gate_k`, 1 when they are equal) and with the
absolute residual `|y − ŷ|` (`ew_corr_resp_absres_k`), and its mean responsibility in
bins of the gate probability cut at `TrainConfig.expert_weight_quantiles` (quintiles;
`ew_resp_gate_bin{j}_k`). `evaluation.expert_weight_report` runs it at the end of every
fold's training; the numbers go to each fold's metrics, the sweep's pooled metrics and the
quick mode's metrics. An expert whose weight follows the residual rather than the gate is
absorbing outlying stock-days. An undefined correlation (a constant gate) is omitted, not
written as 0.

## I.13 `evaluation.py` — the judging machinery

### Purged walk-forward splits

`walk_forward_folds(date, n_folds, test_dates_per_fold, purge_dates, min_train_dates)`:
the last `n_folds × test_dates_per_fold` dates become consecutive test blocks; each fold
trains on **all** earlier dates minus the `purge_dates` immediately before its block.

The purge arithmetic, worked: horizon h=5, test block starting at date 100. A training
sample dated 97 carries the label `log(C₁₀₂/C₉₇)` — computed from prices *inside the
test block*. Purging the last 5 dates (95–99) enforces `d + h < 100` for every training
date: no train label touches test-period prices (AFML §7.4). Embargo (protecting train
data *after* a test block) does not arise in expanding walk-forward — no training date
follows a test date — and the docstring says so, so nobody adds it cargo-cult style.

### Rank-IC and its summary

`rank_ic_by_date(pred, y, date)`: per date, Spearman = Pearson correlation of the
double-argsort ranks. Worked micro-example: `pred = [3.1, −0.2, 0.7]` → ranks
`[3,1,2]`; `y = [0.02, −0.01, 0.03]` → ranks `[2,1,3]`; centered dot / norms → IC =
0.5. Dates with < `min_names` or a degenerate cross-section are skipped — and the
"no scoreable cross-section" error message explains the classical-emission case
(constant per-date predictions have no ranking; compare Hamilton on NLL/regime recovery
instead). `IcSummary`: `icir = mean/std` of the *daily* series; `t_stat = icir·√T`
(the syllabus's `IC̄/SE(IC̄)`); feed it to `ic_pvalue` for the claim family.

### The long-short backtest

`long_short_by_date(pred, y, date, entity, n_quantiles)` per date: rank by prediction,
long the top `⌊n/q⌋` names at `+1/leg`, short the bottom leg at `−1/leg` — dollar
neutral, unit gross per side. Turnover is one-sided `0.5·Σ_names |w_t − w_{t−1}|` with
weights aligned across dates **by entity**; the first date's book counts as full entry
(turnover 1.0). Worked: 4 names, halves ⇒ weights ±½. Same ranks next day ⇒ turnover 0;
fully reversed ranks ⇒ every weight flips by 1 ⇒ Σ|Δw| = 4 ⇒ turnover 2.
`portfolio_summary`: `net_t = gross_t − cost_rate·(2·turnover_t)` (cost per unit
*traded notional*), per-period IRs, **no annualization** — synthetic date codes carry no
calendar, and pretending otherwise manufactures Sharpe ratios.

That daily book is right only for a one-period target. With an `h`-day target its gross
is an `h`-day return while turnover is charged daily, and the overlapping holdings
autocorrelate the series (audit E-3). The harness therefore uses
`long_short_book(..., horizon=h, scheme=...)`, `portfolio_scheme` in the harness, sweep
and settings block (the default `"nonoverlapping"` is **not a decision**; every registry
row records it):

- `"nonoverlapping"`: form a portfolio every `h` dates, hold it `h` dates, charge
  turnover once per rebalance; each period is `h` dates (`PortfolioSummary.period_dates`).
- `"staggered"` (Jegadeesh-Titman 1993): a cohort formed every date, `h` cohorts live at
  weight `1/h`, each date's return the cohorts' **daily** returns, turnover charged
  daily on the combined book. It reads `Panel.y_daily`, each row's `h` daily forward
  returns built under the target's timing rule (they sum to the target).

With `h = 1` both equal the daily book exactly.

### The two harnesses, one grader

`walk_forward_evaluate` (NEC) and `walk_forward_evaluate_baseline` (fit/predict/nll
models) differ *only* in how a fold's model is produced; both feed `_FoldAccumulator`,
so every row of a comparison table is graded by byte-identical code. Per fold: a
**fresh** model (fit-once-per-window — Decision D: no intra-window refits ⇒ no
intra-window label permutation), optional warm-start from `warmstart_key(train_panel)`,
train, freeze, score causally. Stateful (HMM) folds are scored by a no-grad filtering
pass whose state is **warmed through the past training window** — legal because every
training date precedes the test block; the warmed filter then walks the test block
causally. `FoldResult.expert_order` records the σ-sort per fold (`()` for baselines).

## I.14 `baselines.py` — the single-model side

`RidgeBaseline`: closed form `β = (XᵀX + λ diag(1…1,0))⁻¹ Xᵀy` on `[x_snap | 1]` — the
identity is zeroed on the intercept coordinate (M4: *never* penalize the intercept;
tested by shifting y by +5 under λ=10⁶ and asserting the mean survives). Noise scale =
training residual std ⇒ a proper Gaussian predictive NLL.

`MLPBaseline`: **the same `ExpertMLP` class** as one NEC expert + a learnable scalar
`log σ`, trained on the same Gaussian NLL by AdamW. Two deliberate consequences:
held-out NLLs are directly comparable with the NEC's mixture NLL, and "MoE vs single
model" isolates *routing structure at matched capacity* rather than comparing
architecture families.

Validated in both directions (the flagship comparison tests): on the sign-flip panel
the pooled relationship is ≈0 ⇒ both baselines are structurally blind (|IC| < 0.15)
while the NEC routes (IC > 0.3); on regime-free data ridge nails the single true map
(IC > 0.8). The baselines lose where regimes are real *for structural reasons* — that,
and only that, is what makes the headline comparison evidence.

## I.15 The data stack

### `crsp.py` — CRSP daily data, the licensed source (Stage B)

**Licence first.** CRSP is licensed to the university; redistribution is prohibited.
`Quant Model/Data/` is gitignored. Everything derived from it (extracts, parquet files,
built panels, coverage tables, registries of CRSP runs) is written to `Data/derived/`
and nowhere else: never `data_cache/`, `results/`, `figs/` or `Master Thesis/`. Test
fixtures are CIZ-format files with the real headers and **invented numbers**
(`tests/crsp_fixture.py`); tests that read the real files are marked `crsp_data` and skip
without `Data/`. Committed reports quote aggregate statistics only.

- `CRSPSpec`: `crsp_dir` (default `Quant Model/Data/`), `release="ciz202512"`,
  `stock_file="StkDlySecurityData"` (brief 08: it has open, high, low, close, bid and ask;
  `DlyNumTrd` is not used), `market_indno=1000500` (CRSP VW index of the S&P 500 universe,
  `DlyTotRet`; `1000200` is the wider-market alternative), `membership_indno=1000500`
  (the S&P 500 spells; `1000502` has no constituents, see the docstring's evidence),
  `start/end` 2015-01-01..2024-12-31, `post_delisting_return` (`"cash"` | `"market"`, not
  a decision), the member-count band `500..510`, the extract's lookback/lead,
  `cap_unit_dollars` (1000: `DlyCap` in $ thousands, unverified, see below) and block
  size. `data_source` is `"crsp_<release>"`.
- **Lookback** (`lookback_days`): `extract_lookback_days` if set, else derived from the
  longest feature window, `price_history_trading_days(FeatureSpec())` = 1,281 trading days
  (seasonality's five years, beta's 1,260-day correlation), turned into calendar days by
  `trading_to_calendar_days` plus `lookback_margin_days` (30): 1,887 days, so the extract
  starts in late 2009. The training window does not move.
- **Extract identity.** The folder name carries the stock file and lookback
  (`crsp_extract_ciz202512_<start>_<end>_StkDlySecurityData_lb1887d/`; the brief 06
  extract, primary file and 550 days, keeps its old name), `extract.json` records both,
  and `load_extract` refuses an extract whose stock file, lookback or bounds differ from
  the spec (`ExtractMismatch`, listing every difference).
- File access (`crsp_file_source`, any release, CIZ or CCM): the extracted
  `Data/crspdata/<release>_ascii/<file>.dat` if present, else streamed from
  `Data/<release>_ascii.zip`; never unzipped whole.
- `extract_crsp(spec)` (script `scripts/extract_crsp_v2.py`): streams the stock file and
  `StkDlyCumulativeAdjFactor` once each in blocks, keeps the window's ever-members **and
  every other share class of their companies** (PERMCO; for market equity only) from
  `start - lookback_days` to `end + extract_lead_days`, writes parquet parts, and records
  every block's byte offset, so a rerun resumes. Membership, market series, delisting
  records, security-info history and the `MetaItemInfo` rows of `DlyCap`, `DlyShrOut`
  and `DlyVol` (for the unit check) go alongside.
- **Units.** Prices are dollars and `DlyVol` shares. `DlyCap` and `DlyShrOut` are taken
  to be in thousands (CRSP's convention) until checked against `MetaItemInfo`; every Q26
  input that uses them is a rank of a ratio or a log, so a constant unit error moves no
  input.
- `build_crsp_panel(spec, stage_b, compustat=..., compustat_spec=...)` →
  `CRSPBuild(panel, report, coverage)`: daily frames per PERMNO on the market calendar
  (with `retx`, `open`..`ask`, `cap`); for the Q26 set also each PERMNO's company ME,
  GICS sector, report dates and point-in-time fundamentals
  (`compustat.attach_q26_inputs`; the default window's Compustat extract is loaded when
  none is passed); `assemble_panel` with the universe, then the point-in-time filter
  (with re-rank for the legacy set; a Q26 panel is built over the members already). Refuses an extract that is too short for the
  features' warm-up or the horizon, and a member count outside the band. The report is
  aggregate only (rows, dates, members per date, delistings, fill rows, dual-class
  count).

Construction rules, each checked against `MetaColumnInfo`/`MetaFlagInfo`/`MetaSIZtoCIZ`
and tested: daily log return `log(1 + DlyRet)`, missing stays missing; **`DlyRet`
already includes the delisting return** (legacy `DLRET` maps to both `DelRet` and
`DlyRet`; on the real data they agree on every delisting row), so the delisting row is
kept as the stock's final return and is never a panel row; a forward window that runs
past the final return is **completed with the post-delisting fill**, so no row is
dropped because of a future delisting; dollar volume `|DlyPrc| x DlyVol`; drawdown
compounded inside its own window; share volume put on one share basis with ratios of
`DlyCumFacShr` inside the window (the factor is anchored at the end of the sample);
membership bounds inclusive; entities are PERMNOs and `TickerLookup` gives a
date-aware ticker for report labels only.

### `compustat.py` — CCM link, GICS, point-in-time fundamentals (brief 08 C)

Release `cfz202607` (CRSP/Compustat Merged), same licence and file conventions as CRSP.
`CompustatSpec`: `release`, `keyset=1` (industrial, consolidated, standardised;
keyset 8 "PRE" only counted), `link_types=("LC","LU")`, `link_prims=("P","C")`,
`filing_types=("10-Q","10-K")`, `availability_lag_trading_days=1`,
`fallback_lag_days=90`, `history_quarters` (derived: 12) and
`history_margin_quarters=2`. Columns are read by name, case-insensitively.

- **Link** (`linkhistory`): valid on d when `LINKDT ≤ d ≤ LINKENDDT`; `gvkey_on_dates`
  gives a PERMNO's GVKEY per date, none when two are linked at once
  (`link_ambiguities` reports those, never drops them silently).
- **Sectors** (`gicshistory`, `GSECTORH`, dated by `INDFROM`/`INDTHRU`): through the link
  on the date (`lpermno` only checked, `gics_lpermno_disagreements`); `sector_dummies`
  gives 11 columns, all 0 without a sector.
- **Quarters** (`quarterly_fundamentals`): keyset 1 rows of the period descriptor, income
  statement, balance sheet and YTD cash flow (`ytd_to_quarterly`: Q1 its YTD value, Qk
  the difference within the fiscal year, missing without the predecessor). Period ends
  (`period_end_dates`): a `DATADATE` column if the release has one, else computed from
  `FYEARQ`, `FQTR`, `fyrq` (Compustat's convention), cross-checked against
  `fiscalmarketdataquarterly`.
- **Availability**: one trading day after the later of `RDQ` and the first 10-Q/10-K
  `FILEDATE` for that period end (`RDQ` alone or the filing alone when one is missing;
  period end + 90 days with neither); the surprises use `RDQ` + 1 trading day
  (`avail_rdq`). `pit_events` gives the quarters known at each availability date;
  `carry_forward` holds values until the next one, for at most
  `fundamental_max_staleness_days` (365) after the newest quarter's availability.
- `fundamental_inputs(quarters, dates, fs)`: the 16 fundamental inputs of one GVKEY
  (book equity, trailing earnings, net debt and the 13 ratios), point in time.
- `extract_compustat(spec, permnos)` (script `scripts/extract_compustat.py`): the linked
  GVKEYs' rows from `fundamentals_start` (2011-07-01 for the default window), resumable,
  into `Data/derived/compustat_cfz202607_<start>_<end>/` with an aggregate report (link
  ambiguities, the period-end check, the keyset-8 count, the filing types).

### `characteristics.py` — the Q26 characteristics (brief 08 D)

`stock_characteristics(daily, m, fs)` computes one stock's 40 characteristics (below)
from its daily frame; `cross_section(rows, fs)` turns a date's universe rows into the 57
inputs. Every window is a `FeatureSpec` field; a rolling statistic needs
`ceil(min_obs_frac · window)` days (0.8), otherwise it is missing.

| # | input | definition |
|---|---|---|
| 1-2 | `ret_5d`, `ret_20d` | sum of `r` over 5 / 20 days |
| 3 | `mom_12_1` | sum of `r` from t−251 to t−21 |
| 4 | `prc_highprc_252d` | `P_t / max(P, 252)`, `P` = cumulative product of `1 + DlyRetx` |
| 5 | `rvol_21d` | std of `r`, 21 days |
| 6 | `beta_bab` | corr of 3-day overlapping sums of `r_i` and `m` over 1,260 days (min 750) × std(`r_i`, 252) / std(`m`, 252) |
| 7 | `log_me` | log of company ME |
| 8 | `ami_126d` | mean of `|R|` / dollar volume ($M), 126 days, zero-volume days excluded |
| 9 | `seas_2_5an` | mean simple return in the month of t+1, years 2-5 back |
| 10 | `coskew_21d` | `mean(x y²) / (√mean(x²) · mean(y²))`, demeaned `r_i`, `m` |
| 11-26 | `be_me` … `saleq_su` | fundamentals (`compustat.fundamental_inputs`), ratios to ME on the day |
| 27 | `ivol_capm_21d` | residual std of OLS `r_i` on `m`, 21 days |
| 28 | `vol_shock` | std(`r`, 5) / std(`r`, 60) |
| 29-30 | `rskew_21d`, `rmax1_21d` | sample skewness of `r`; max of `R`; 21 days |
| 31 | `volume_z` | split-invariant volume z-score, 20 days |
| 32 | `qspread_21d` | mean `(ask − bid) / mid`, days with `0 < bid ≤ ask` |
| 33 | `overnight_20d` | sum of `r − log(close/open)`, 20 days |
| 34 | `ma50_gap` | `P_t / mean(P, 50) − 1` |
| 35-36 | `earn_next5`, `days_since_earn` | report expected within 5 business days (past report + 364 days); trading days since report + 1 |
| 37 | `var_ratio_60d` | var of 5-day sums / (5 · var of `r`), 60 days |
| 38 | `ret_vol_corr_60d` | corr of `r` and log split-adjusted volume, 60 days |
| 39-40 | `ind_mom_12_1`, `ret_20d_ind_rel` | the sector's `mom_12_1`; `ret_20d` minus the sector's (value-weighted) |

Then per date (D.4): rank each characteristic over the rows that have it (ties averaged),
`x = 2 (rank − 1)/(n − 1) − 1` (0 when n = 1); `be_me`, `ni_me`, `niq_be`, `ocf_at` also
within their sector (universe rank below `min_sector_names` = 5 values); missing → 0; the
flags `flag_price_missing` (any CRSP characteristic filled) and `flag_fund_missing` (any
of 11-26 filled); 11 GICS dummies. `earn_next5`, the dummies and the flags are not
ranked. 26 + 12 + 2 + 4 + 11 + 2 = 57 inputs.

### `features.py` — daily returns → Panel, under one timing rule

**The rule** (stated once, tested mechanically): a row (date t, entity i) is a
prediction made *at the close of t*. Every feature uses information through t; the
forward target is the **only** forward-looking column; cross-sectional ranking uses only
date-t's own cross-section.

**The target** (`StageBSpec.target_kind`, Q25): `"market_neutral"` (default) is
`fwd_mn_ret_{h}d = fwd_ret_{h}d − mean_j fwd_ret_{h}d[j]`, the equal-weighted mean over the
date's universe rows with a valid target on a tradable day, taken in `assemble_panel`
before any row is dropped for a missing feature; `y_daily` is demeaned day by day over
the same rows, so it still sums to the target. It sums to zero on every date and a
return common to every stock cancels from it. The long-short book is dollar neutral, so
it is unchanged by a common return (tested). `"raw"` (`fwd_ret_{h}d`) and `"residual"`
(trailing-beta residual) remain options.

**The feature set** (`StageBSpec.features.feature_set`, Q26): `"q26"` (default, the 57
inputs of `characteristics.py` above) or `"legacy14"` (the table below, for comparison
and the older tests). `test_no_lookahead` truncates the series at t and asserts every feature
at t is identical — copy its pattern whenever you add a feature.

The input is one **daily frame** per entity (`DAILY_COLUMNS`: `ret`, `volume`,
`dollar_volume`, `share_factor`, `tradable`, `fill_ret`) plus the market's daily log
return; nothing in `features.py` knows the source. `stock_features` computes one
entity's features and target; `market_frame` the market columns; `assemble_panel` the
`Panel`.

The named features (order = channel order; all trailing):

| snapshot column | formula |
|---|---|
| `ret_1d/5d/20d/60d` | sum of the last k daily log returns |
| `mom_120d` | 120-day log return |
| `vol_5d/20d/60d` | rolling std of daily log returns |
| `downside_vol_20d` | rolling std of `min(r, 0)` |
| `drawdown_60d` | `I_t / max(I_{t−59..t}) − 1`, `I` compounded from returns inside the window |
| `dollar_vol_20d` | `log(mean₂₀(|P|·V))`, raw price times raw volume |
| `volume_z_20d` | `(V_t − mean₂₀V')/std₂₀V'`, `V'_s = V_s·F_s/F_t` (split factors inside the window) |
| `rel_ret_20d` | `ret_20d − mkt_ret_20d` |
| `rel_vol_20d` | `vol_20d / mkt_vol_20d` |

Sequence channels (`d_seq=4`, natural units, per-row trailing window of `seq_len`):
`ret_1d`, `rel_ret_1d`, `vol_20d`, `volume_z_20d` — the observable regime signals the
encoder reads.

Assembly mechanics worth knowing: for `q26`, `_valid_rows` demands a tradable day and the
target only (and, for `StageBSpec.input_mode="snapshot_plus_hidden"`, a full sequence
window; a panel built for the other mode is refused at training); with the universe
given, only the date's members are rows, so every rank and sector aggregate is over the
universe, and an incomplete sequence window is zero-filled and counted. For `legacy14`,
`_valid_rows` demands a tradable day, a full trailing window, all snapshot features and
the target; windows are built with
`sliding_window_view` over each entity's rows (the CRSP layer puts every stock on the
market calendar, so a day without a CRSP row is a missing return and invalidates the
windows that hold it); dates keep only cross-sections with ≥ `min_names_per_date` names;
snapshot features are per-date **rank-normalized to [−0.5, 0.5]** when `cs_rank=True`
(point-in-time safe by construction, and the natural normalization for a rank-IC
target); final sort is date-major. `StageBSpec(seq_len=20, horizon=5, cs_rank=True,
min_names_per_date=5)`; `spec.target` names the target column.
`data_config_from_panel(panel)` derives the matching `DataConfig`.

### `universe.py` — point-in-time membership

`SpellUniverse` holds index membership as spells (`entity`, `start`, `end`), **both
bounds inclusive** — the shape of CRSP's `StkIndMembership`, which
`crsp.membership_universe` loads with PERMNOs as entities. `members_asof(d)`,
`members_union(start, end)` and `count_by_date(dates)` answer the obvious questions.
`filter_point_in_time(panel, universe)` keeps a row only if its entity was a member on
the row's date (requires `date_labels`/`entity_labels`; preserves date-major order) and
re-ranks a rank-normalized panel among the survivors (audit D-1); it accepts anything
with `members_asof` (the `Universe` protocol). `universe_coverage_report` gives members /
with-data / coverage per date. CRSP carries departed names and their delisting returns,
so both survivorship components are covered by the data itself; the coverage report
still says how many members lack usable rows.

### `context_data.py` — VIX and French factors (Stage C)

Official free sources, cache-first, parsers pure (offline-testable): VIX from CBOE's
history CSV (`MM/DD/YYYY` dates, close column); daily factors from Kenneth French's
zips — the parser survives the real files' quirks (text preamble; copyright footer; the
momentum file's **trailing comma on every line**; percent values → decimals; `−99.99`
missing markers → NaN, not −99% returns). `build_context(vix, factors)` joins and
derives `abs_mkt` and `mkt_vol_20d` (annualized 20-day realized market vol).
**Diagnostics only, never training inputs** (Decision B): the moment VIX enters
training, "do learned experts correspond to volatility regimes?" stops being a testable
question. One exception, by brief 03 §2: the Hamilton gate's registered series
`market_excess_return` is French daily Mkt-RF, so that series is the gate's input (never
a target or an expert feature). These are the only free downloads left; CRSP has no
equivalent of either.

## I.16 `alignment.py` — the interpretability tooling

`gate_utilization_by_date(trainer, panel) → (date_codes (D,), util (D,K))`: memoryless
priors → per-date mean of the predictive `π(x)`; HMM → per-date mean **filtered**
posterior (the classic Hamilton filtered-regime-probability series, from the same
causal pass as the NLL). `regime_alignment(panel, dates, util, context)` joins the
utilization series to a context frame — via `date_labels` for real panels, integer
codes for synthetic — and reports per expert: Pearson correlation with every context
column, plus mean utilization in the top vs bottom VIX terciles.

Validated by a planted-truth test: build a fake "VIX" from the true synthetic regime
and the report must recover it (|corr| > 0.6, terciles ordered) for both prior
families. First real run (30 names, 2015–2024, unsupervised soft gate): the high-σ
expert's utilization correlated **+0.81 with VIX** and +0.85 with realized market vol,
≈0 with every directional factor — a volatility-regime split found by the mixture
likelihood alone. (Single seed, toy universe, all-dates window: a tooling demo, not a
claim — the thesis version runs on held-out folds, multi-seed.)

## I.17 `registry.py` + `multiple_testing.py` — the honesty layer

**`TrialRegistry(path)`** — append-only JSONL; the registry *is the file* (reopen
anywhere). One row per trial:

```json
{"trial_id": 7, "timestamp": "2026-07-03T14:22:11+00:00", "tag": "routing_sweep",
 "metrics": {"mean_ic": 0.031, "p": 0.002}, "config": {"prior": "soft"}, "seed": 3,
 "notes": ""}
```

`n_trials(tag)` is the selection multiplicity N. `best(tag, metric)` **never silently
picks**: it appends a `tag#selection` row recording the candidate count and winner id
(and the reserved suffix cannot be logged directly). `metric_values(tag, metric)` feeds
the cross-trial variance below.

**Deflated Sharpe** (Bailey & López de Prado 2014), the max-selection correction:

```
SR      = mean(r)/std(r)                                    (per period)
PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )
E[maxSR] = √V[SR] · ((1−γ)·z(1−1/N) + γ·z(1−1/(N·e)))       γ = 0.5772…
DSR     = PSR(SR* = E[maxSR])
```

The PSR denominator prices *non-normality* (negative skew γ₃ and fat tails γ₄ widen it
and deflate confidence); the hurdle prices *selection* (grows like `√V·z(1−1/N)`).
`n_trials` and `sr_variance` come from the registry. The factor-zoo test pins the
behavior: best of 60 pure-noise strategies → naive PSR > 0.95 but DSR < 0.75; genuine
skill of the same apparent size survives with margin.

**Families**: `ic_pvalue(IcSummary)` = two-sided normal p from the IC t-stat;
`bonferroni(pvals, α)` (FWER: `p·N ≤ α` — small family, singular claim);
`benjamini_hochberg(pvals, α)` (FDR step-up with monotone q-values — the factor-zoo
screening criterion). Worked BH example: sorted p = (.005, .01, .03, .04) vs thresholds
`k/N·α` = (.0125, .025, .0375, .05) at α=.05 ⇒ all four rejected. Property-tested:
BH's rejections always contain Bonferroni's.

## I.18 The protocol layer: sweep, tuning, calibration

Three modules sit *above* the harness and turn single runs into defensible experiments.
They share one design idea: **every convenience function is a thin composition of
already-tested machinery**, so the protocol layer adds discipline, not new split logic.

**`sweep.py` — seeds as first-class citizens.** A sweep is named **arms** × a **seed
grid**. `nec_arm(name, cfg, warmstart_key=)` plants each seed in *both* the global RNG
(encoder/gate init) and `TrainConfig.seed` (expert diversification, shuffling) via
`dataclasses.replace` — seeds are genuinely independent initializations, each exactly
reproducible. `baseline_arm` runs ridge/MLP through the same folds and the same grading
code (a deterministic baseline shows seed-std exactly 0, which is itself information).
`run_sweep` logs every `(arm, seed)` trial to the registry with the full config dict and
reports mean ± std per arm. The statistical subtlety lives in `corrected_claims`: **the
claim family is the arms, not the seeds** — seed replicates are repeated measurements of
one hypothesis, so each arm's p-values are combined by their **median** (min-over-seeds
would be selection inside the family) and BH runs across arms.

**`tuning.py` — tune once, freeze, log the pick.** `validation_tail` *is* a one-fold
walk-forward on the training window (it reuses the outer split's tested purge
arithmetic); `tune` scores candidates (ordinary sweep arms) on that tail via
`run_sweep(n_folds=1)`, picks the winner by seed-mean (`mode="min"` for NLL), and logs
the full provenance chain: per-run trials under `<tag>`, arm aggregates under
`<tag>.arms`, one `#selection` event carrying the candidate count. Per-fold re-tuning is
deliberately not offered — it multiplies researcher degrees of freedom.

**`calibration.py` — is the gate lying about its probabilities?** For a latent-regime
gate the honest yardstick is the one it was trained toward: the gate is calibrated iff
`E[r_k | π_k = q] = q` on **held-out** data, where `r` is the Bayes posterior computed
from the held-out `y`. `gate_reliability` bins the pooled `(sample, expert)` claims and
reports the reliability curve + ECE; `fit_temperature` fits the single-scalar repair
`softmax(z/T)` by held-out mixture NLL on a validation tail (T > 1 softens an
overconfident gate). Scope: softmax-predictive gates only — hard/top-k have no smooth
confidence to rescale, and the HMM's regime probabilities come from the filter, not the
gate (rejected with guidance). Anchored by two planted truths: identical experts force
`r ≡ π` (ECE exactly 0), and ×4-sharpened gate logits force a fitted T > 1.5 that
repairs NLL *and* ECE on the untouched test block.

## I.19 The test suite as a map

| file | what it guards |
|---|---|
| `test_defects.py` | all 8 ledger defects, each with a test that fails on the prototype |
| `test_loss_math.py` | fused NLL vs float64 reference; analytic `π−r` and σ-stationarity; the underflow cliff; logit-gauge invariance; the LB scope theorem; decorrelation math |
| `test_gate_abstraction.py` | shared-core claim; forward filter vs reference; **causality** (future perturbation ⇒ bit-identical past); row-stochastic + persistence-biased `A`; HMM beats soft under planted persistence; learned `A` tracks true persistence |
| `test_routing_variants.py` | per-mechanism contracts and *gradient behavior*; Gumbel-max sampling law; the k≥2 rule; every mechanism trains end-to-end |
| `test_hamilton.py` | the four-corner grid; **Gaussian-HMM parameter recovery by autograd-through-filter** (μ,σ,A ≈ truth); closed-form moment warm-start |
| `test_evaluation.py` | fold chronology + purge arithmetic; rank-IC properties (perfect/anti/monotone-invariant/random); long-short, turnover, and cost-drag math; both harnesses end-to-end |
| `test_baselines.py` | ridge exact recovery + unpenalized intercept; MLP determinism; **the two-direction headline comparison** |
| `test_selection_inference.py` | registry persistence + selection events; PSR/E[maxSR]/DSR properties; **the factor-zoo deflation**; BH/Bonferroni |
| `test_sweep.py` | end-to-end mini-sweep on planted truth (arms separate correctly; claims family rejects only the signal); seed reality + determinism; arm validation |
| `test_tuning.py` | validation-tail arithmetic; planted-winner selection with full registry provenance; no test-window contamination; min-mode |
| `test_calibration.py` | ECE = 0 for identical experts; **planted overconfidence detected and repaired out of sample**; honest gate fits T ≈ 1; non-softmax priors rejected |
| `test_residual_target.py` | rolling β recovers truth and is trailing (no-lookahead probe); market clone ⇒ zero residual target; β=2 name ⇒ market-neutral target |
| `test_plots.py` | every figure renders headlessly and saves a real PNG; input guards |
| `test_smoke.py` | end-to-end training + regime recovery (AUC > 0.8) + a broken-gradient guard that must fail |
| `test_stage_b.py`, `test_stage_c.py`, `test_universe.py` | feature values by hand, the no-lookahead probe, panel contract; context parsers against fixtures that reproduce real-file quirks; inclusive membership spells; coverage math |
| `test_crsp.py` | every CRSP construction rule on an invented-number CIZ fixture (delisting return reaches the target, the post-delisting fill, missing never zero, split-invariant volume, inclusive bounds, PERMNO labels, provenance, market INDNO, resumable extract); three `crsp_data` tests on the real files that skip without `Data/` |
| `test_market_neutral_target.py` | brief 08 A: the target averages to zero per date, a common return cancels, the mean counts rows dropped for a feature and only the date's members, `y_daily` sums to it, the long-short book is blind to a common return, `target_kind` in every row |
| `test_gate_weight.py` | brief 08 B: the brief's example matrix by hand (and the corrected calm value), `h = 1` and `A = I` limits, the horizon read from the panel, the same rule on training and test dates, `gate_weight` in every row; the Q24 diagnostic |
| `test_compustat.py` | brief 08 C on invented CIZ and cfz fixtures: the derived lookback, extract identity and refusal, the new columns, links over time and ambiguity reporting, GICS switching on `INDFROM`, period ends, YTD differencing, availability, staleness, keyset-8 count |
| `test_q26_features.py` | brief 08 D: every characteristic by hand, ranks, fill, flags, sector fallback, earnings timing on past report dates only, no look-ahead (prices and fundamentals), the build on both fixtures, `feature_set` in every row |
| `test_feature_coverage.py` | brief 08 E: the data-layer registry fields, the coverage report (aggregate only, refuses per-security values), the scripts compile; one `crsp_data` test |

`python3.14 -m pytest tests/ -q` — the whole offline suite, deterministic, ~38 s
(132 tests + 3 network-gated skips). Lint and types:
`python3.14 -m ruff check nec_moe/ tests/ scripts/` and
`python3.14 -m mypy nec_moe/` — both clean, configured in `pyproject.toml`.

---

# Part II — The Manual

Everything below runs as written from `nec_baseline/` with the system `python3.14`.
Where output is shown, it is the output you should approximately see.

## II.0 The workflow at a glance

**The shortcut: `run_experiment.py` — the control panel.** One file, all the knobs in a
single `Experiment(...)` settings block (data source, model, prior, training length,
seeds, evaluation split, figures on/off), everything downstream automatic:

```bash
# edit the EXPERIMENT = Experiment(...) block at the top, then
python3.14 run_experiment.py
```

`mode="quick"` trains ONE model on a chronological split and produces the full
diagnostic picture (dashboard, utilization, IC/L-S curves, calibration, optional VIX
alignment) — the iterate-fast loop. `mode="evaluate"` runs the honest protocol
(multi-seed purged walk-forward, baseline rows, BH-corrected claims) — the
show-to-someone loop. Every run lands in the run store, `results/<campaign>/<run_id>/`
(II.0a). It is a thin driver over everything below — the rest of Part II documents the
pieces it drives, for when you outgrow the panel.

**Long runs are interruptible.** Ctrl+C once checkpoints and stops; continue with

```bash
python3.14 run_experiment.py --resume            # latest interrupted/crashed run of the tag
python3.14 run_experiment.py --resume <run_id>   # a specific run
```

Completed work is skipped (sweep runs via the registry, folds via persisted fold
files), an interrupted fit continues from its last checkpoint on the *exact* same
trajectory, and every write is atomic. A resume on changed settings or data is
refused (`--force` overrides and records it). **RUNBOOK.md** is the day-to-day guide;
II.0a below is the reference, and II.6 the underlying `Trainer.save/load` API.

## II.0a The run store and resume (brief 07)

**Layout.** `nec_moe/runstore.py` gives every run, from any entry point
(`run_experiment.main`, `run_sweep(run=...)`, `walk_forward_evaluate(run=...)`, the
smoke script), one folder and one index row:

```
results/
  INDEX.csv                 one row per run: appended at launch, updated at exit
  <campaign>/<run_id>/      run_id = YYYYMMDD-HHMMSS_<tag>_<8-char settings hash>
    run.json                identity: campaign, tag, purpose, mode, created, git commit +
                            dirty flag, data source, data fingerprint, versions, device,
                            touches_test, forced resumes
    settings.json           the full Experiment snapshot
    status.json             state, current arm/seed/fold/step, heartbeat, exit reason
    trials.jsonl            the trial registry
    metrics/                aggregate results: quick.json | report.csv,
                            <arm>_seed<s>_folds.csv, <arm>_seed<s>_pooled.json
    figures/
    logs/run.log            everything printed, timestamped      (gitignored)
    checkpoints/            resume state                         (gitignored)
    SUMMARY.md              at exit: settings diff against defaults, status, metrics
Data/derived/runs/<run_id>/ per-security outputs (fold predictions; licensed, gitignored)
```

`results/` holds aggregate outputs only; anything per security goes to the
`Data/derived/runs/` folder (licence rules, brief 06). `purpose` is a free-text label.
`touches_test` is true for a run that scored a test block on a real panel (Q15's
accounting); it is recorded. What is enforced (brief 09 F.3) is the pre-registration lock:
a run that scores the test period (2010 on) needs the pilot's selection file (II.0b). The settings hash ignores `resume`,
`resume_run_id` and `force`; the data fingerprint hashes the panel's shape, date
range, entity set, target checksum, data source and CRSP release.
`scripts/runs.py` lists, shows, diffs, finds and resumes runs; nothing deletes.

**Resume** composes four levels, all under `checkpoints/`:

| level | file | restored on resume |
|---|---|---|
| sweep | the registry row of each finished (arm, seed) | the run is skipped, its metrics reused |
| fold | `<arm>_seed<s>/fold_<i>.pt` | the fold is skipped, its scored payload reused |
| fit | `fold_<i>_trainer.pt` (+ `.1`, ...) | model, optimizer, sampler, RNG: the fit continues exactly |
| fold inputs | `fold_<i>_gate.pt`, `fold_<i>_base.pt` | the fitted gate table (G-3) and the frozen base, not refitted |

Quick mode keeps `trainer.pt` and `gate.pt` at the top of `checkpoints/`.

- **Clean interruption** (`nec_moe/interrupt.py`): the first SIGINT/SIGTERM requests a
  stop; the trainer finishes the current optimizer step, checkpoints, raises
  `RunInterrupted`, and the run is recorded `interrupted`. A second signal exits at
  once.
- **Safe writes**: checkpoints, registry rows, metrics, figures and reports go through a
  temporary file and a rename. `TrainConfig.checkpoint_keep` (default 2) checkpoints
  are kept per fit; `Trainer.load_latest` loads the newest that loads cleanly and says
  when it fell back.
- **Crash detection**: `status.json` carries a heartbeat every
  `TrainConfig.heartbeat_every` steps (default 50); `runs.py list` shows a `running` run
  whose heartbeat is older than `stale_after` seconds (default 1800) as `crashed`, which
  resumes like `interrupted`.
- **Refusal**: resume compares the stored settings hash and data fingerprint with the
  current ones and refuses on any difference, listing it (`ResumeRefused`); `--force`
  overrides with a loud warning and appends the change to `run.json`'s
  `forced_resumes`.

The property that matters, pinned by `tests/test_resume.py` with expert dropout on: an
interrupted and resumed run equals an uninterrupted one **exactly**, interrupted mid-fit
inside a fold, between folds, in the middle of a sweep, or with its newest checkpoint
torn by a crash during the write.

## II.0b The training scheme (brief 09; Q16)

Decided 2026-10-05; the reasoning is in `Master Thesis/Advisor_Questions.md` Q16 and
`Training_Window_and_Weighting_Theory.md` sections 4-8. RUNBOOK section 8c is the how-to.

**Sample and folds** (`FoldConfig`, `nec_moe.evaluation`). The sample is 2000-2024
(`CRSPSpec.start`, `Experiment.start`); the extracts reach further back for the feature
windows only. `fold_scheme="calendar_year"` (the default) gives one main fold per test
year 2010-2024 (`calendar_year_folds`): train on every date from the sample start to
the end of the previous year minus the purge (the target horizon, 5), test on the year.
The pilot's folds (`pilot_folds`) use the same rule with validation years 2007-2009, on
`pilot_slice(panel)`: the dates before 2010 minus the last 5 (their labels are 2010
returns). Every pilot fold builder asserts on the panel it receives that no date is in
2010 or later (`PreTestViolation`). `fold_scheme="count"` keeps the earlier
count-based split for development.

**No held-out tail, fixed budgets, fresh starts, paired seeds.** In main and pilot folds
the base trains on its whole training block (`BaseConfig.val_fraction = 0`, the default;
a tail or early stopping is refused there), and the experts train their fixed
`TrainConfig.steps` on every trading day's rows. Every fold starts from fresh weights
(the harness reseeds the global RNG before building each fold's trainer). A seed fixes
the base, the experts' initialisation and the minibatch order whatever the arm, so arms
on one seed list are paired.

**Decay weights** (`nec_moe.decay`; theory notes sections 4-6). The base's rows are
weighted `2^(-age/H_base)` (`BaseConfig.decay_half_life_days`, age in trading days to
the fold's last training date); the experts' rows by the regime clock,
`w(s) = sum_k xi_s(k) rho^n_k(s,T)` with `n_k` the later regime-k experience by the
frozen gate's **filtered** probabilities (`TrainConfig.expert_decay="regime_clock"`,
`expert_decay_half_life` in regime-days; the Hamilton gate only). Each loss is the
weighted mean `sum(w l) / sum(w)`; `None` or an infinite half-life skips the weighting
and is bit-identical to an unweighted run. Per fold, `FoldResult.weights` (and the
per-fold metrics CSV) report the Kish ESS overall and per regime and the weight mass by
calendar year.

**The gate's memory** (`nec_moe.baum_welch`, `MarkovGateConfig`). `fit_backend="native"`
fits the order-0 Gaussian Markov switching model with a scaled Baum-Welch (validated
against statsmodels: the same log-likelihood, parameters and filtered probabilities, on
synthetic series and on the real market series 2000-2008). `memory="regime_clock"` adds
a weighted second pass (each regime's emission on its own clock, transitions out of `j`
on `j`'s; `gate_half_life` in regime-days; a pass that does not settle is reported as
`gate_weighted_settled = 0`). Only the estimate changes: the filter, `apply_causal` and
the Q27 gate weight are as before.

**The pilot** (`nec_moe.pilot`, `scripts/run_pilot.py`, `configs/pilot_grid.json`).
Stage 1 scores each gate half-life by the one-step predictive log-likelihood of the
market series over the validation years (frozen filter), reports the 20 worst days apart
and gives ties to the longer memory. Stage 2 scores every other setting by the mixture's
regime-balanced validation loss (`v(s) = sum_k wbar_k(s) / M_k`, `wbar` the window-average
gate weight), averaged over the 3 folds and the pilot seeds; the step budgets are read as
milestones of one run. It writes `results/pilot/pilot_selection.json` and its SHA-256.

**The lock** (`run_experiment.selection_lock`). A run that scores any date from
`first_test_year` on needs `Experiment.pilot_selection_file`; settings that differ from
the chosen ones need `force_deviation=True`. The hash and the deviations are in
`run.json`; the hash and the forced flag in every trial row.

**The campaign runner** (`nec_moe.campaign`, `scripts/run_campaign.py`). One job per
(arm, depth, seed, fold), a pool of spawned one-thread workers, the panel's tensors saved
once to `Data/derived/panel_cache/` and memory-mapped, bases cached on disk behind a lock
file, each job resumable in the run's `checkpoints/jobs/`, combined afterwards exactly as
a sequential run (tested equal). RUNBOOK section 8b.

**What every trial row records** (`PROVENANCE_KEYS`, brief 09 I.1), besides the brief 06
and 08 keys: `fold_scheme`, `protocol` (main or pilot), `test_years`, `sample_start` and
`sample_end` (the panel's first and last date), `base_decay_half_life_days`,
`expert_decay`, `expert_decay_half_life`, `gate_memory`, `gate_half_life`,
`gate_fit_backend` (`None` where a key does not apply) and `pilot_selection_hash`; the
seed is the row's own field.

**Not decided here** (the code keeps its defaults and picks nothing): K (Q18), the
widths and depths inside the envelope (Q8/Q17), the error function (Q20), the gate series
(Q23), whether the filter runs through the purge gap (Q22), leave-one-episode-out
(Q16 c), the per-expert form of the regime-clock decay, the fallback to early stopping.

## II.0c The compute envelope (brief 09 H)

Decided 2026-10-05 (Q8 update; `Master Thesis/Progress_Tracker.md` section 1c). The
whole design, run twice, fits the laptop (Apple M4 Pro, 10 one-thread processes) in
about 2-6 days of machine time if it stays inside:

| | envelope |
|---|---|
| depth scan | 1, 2, 3 hidden layers, widths from `pyramid_dims(first_width, depth)` |
| first width | at most 128 |
| K (experts, regimes) | at most 4 |

Depths 4-5 or a first width of 256 take 1-1.5 months (and Gu, Kelly & Xiu find depths
4-5 add nothing). `nec_moe.config.COMPUTE_ENVELOPE` holds the three values as
**documented defaults, not hard limits**: `scripts/run_campaign.py` prints a note for an
arm outside them, and nothing refuses. The envelope says where the design is
affordable, not which K or first width to use: K is Q18 and the widths Q8/Q17, both
open, and the code keeps its current defaults (K = 2, widths (64, 32)).

**The real pipeline's cost per step** (`scripts/time_real_pipeline.py`, timing only,
`results/benchmark/real_pipeline_timing.json`). One job on the real 2000-2024 panel:
Hamilton gate, K = 3, widths 64-32, batch 4,096, 2,000 steps, the 2009 pilot fold
(training 2000-2008, 1.13M rows), one thread: **24.4 ms per step against the
benchmark's 4.19 ms, an overhead factor of 5.8** (the budget in section 1c assumed
2.5). Gate fit (statsmodels, K = 3, 24 starts) 29 s, base fit 0.2 s (its default 1,000
steps of 128), expert training 49 s, peak memory 3.9 GB; the panel loads in 0.3 s from
a warm page cache. A profile of the same step on a synthetic panel of the same shape
(25 ms per step) puts about 60% in the GRU encoder over the 20-day sequence, whose
output the frozen Hamilton gate with snapshot-only experts does not use (the
dead-parameter warning), and about 14% in the precomputed gate's per-row date lookup
(a Python loop over the batch); the experts themselves take about 10%. Those two are
engineering, not design: removing them would bring the factor near 1.5.

Both are now removed. `NECModel.encoder_is_dead` is true when the prior is
precomputed, the experts read the snapshot only, and the encoder draws no random
numbers (no inter-layer dropout); the forward pass then skips the GRU and passes the
gate head zeros it only shape-checks. `skip_dead_encoder = False` restores the full
path. The precomputed gate looks dates up by binary search over its sorted date codes,
and still refuses a date outside the fitted and applied populations.
`tests/test_fast_paths.py` shows that a frozen Hamilton-gate model trained either way
is bit-identical: the same parameters, predictions, RNG state, training history and
gradient audit. On the synthetic same-shape panel (this container, one thread) the
step falls from 61.3 ms to 23.8 ms, 2.6 times faster: 29.4 ms with the encoder
skipped, the rest from the lookup. **Re-timed on the real panel (same job, same
machine, one thread): 5.57 ms per step against the benchmark's 4.19 ms, an overhead
factor of 1.33** (was 24.4 ms and 5.8), now inside the 2.5 the section 1c budget
assumed. Expert training 11.1 s (was 49 s); gate fit 29 s, base fit 0.2 s and peak
memory 3.9 GB are unchanged. The real speed-up (4.4 times) is larger than the
synthetic one measured in a Linux container (2.6 times); the two machines were not
profiled side by side, so the gap is not explained here.

Under the hood, every experiment is the same seven moves:

```
1 DATA      build or load a Panel        (synthetic | Stage B real | your own)
2 CONFIG    NECConfig(...)               (prior kind, emission kind, train knobs)
3 TRAIN     Trainer → warmstart → fit / fit_sequence
4 MONITOR   trainer.history + diagnostics          (is the mixture healthy?)
5 EVALUATE  walk_forward_evaluate(_baseline)       (purged, fit-once-per-window)
6 INTERPRET gate_regime_alignment + figures        (what did the gate learn?)
7 LOG       TrialRegistry + corrections            (make the claim defensible)
```

The rest of Part II is one section per move, plus data acquisition, extension guides,
and a dictionary of the package's error messages.

## II.1 Setup and health checks

```bash
cd nec_baseline
python3.14 -m pytest tests/ -q                    # full offline suite (~38 s) — green?
python3.14 -m pytest tests/test_smoke.py -q       # just the end-to-end smoke
python3.14 -m pytest tests/ -q -k "hmm"           # any subset by keyword
NEC_NETWORK_TESTS=1 python3.14 -m pytest tests/test_stage_b.py tests/test_stage_c.py -q
                                                  # + live endpoints (needs network)
python3.14 -m ruff check nec_moe/ tests/ scripts/ # lint (clean; config in pyproject)
python3.14 -m mypy nec_moe/                       # types (clean; config in pyproject)
```

CI (`.github/workflows/ci.yml`, repo root) runs the same three checks on CPU torch and
goes live the day the repo gets a remote.

Dependencies: `torch`, `numpy`, `pandas` are load-bearing. Optional, imported lazily:
`pyarrow` (the CRSP extract, extra `crsp`), `statsmodels` (the Hamilton gate),
`matplotlib` (your figures). Everything is CPU; typical times on this machine: smoke test ~8 s, a
600-step real-panel fit ~1–2 min, the full suite ~38 s.

Git protects the work: the repo root is the project folder; commit early and often
(`git add -A && git commit -m "..."`). The registry files (`*.jsonl`) and figures
belong in commits too — they are results.

## II.2 First training run (synthetic), with expected output

```python
from nec_moe import (DataConfig, NECConfig, NECModel, SyntheticRegimePanel,
                     SyntheticSpec, Trainer, TrainConfig, set_seed)

set_seed(0)
spec = SyntheticSpec(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=0)
panel = SyntheticRegimePanel(spec).generate(n_dates=300, n_entities=8)
train, test = panel.split_by_date(0.8)                 # chronological, always

cfg = NECConfig(
    data=DataConfig(d_seq=spec.d_seq, d_snap=spec.d_snap, seq_len=spec.seq_len),
    train=TrainConfig(sigma_init=1.5, sigma_freeze_steps=30, lr=3e-3, batch_size=256),
)
trainer = Trainer(NECModel(cfg))

# Symmetry breaking (I.11). Sort key = observable regime proxy: window vol here.
trainer.warmstart_experts(train.full_batch(),
                          sort_key=train.x_seq[:, :, 0].std(dim=1))

history = trainer.fit(train, steps=400)
print("held-out NLL:", round(trainer.evaluate(test.full_batch()), 3))
```

Expected: held-out NLL ≈ 0.9–1.1 (down from ≈ 2.1 untrained); with the warm-start the
gate recovers the regime at AUC > 0.95 (check via II.8's utilization). **If you skip
the warm-start on this panel, nothing visibly fails — the NLL still falls — but the
gate stays at AUC ≈ 0.5.** That silent failure mode is why II.3's dashboards exist.

Rules of thumb: `sigma_init ≈ std(y)` (≈1.5 here; ≈0.05 for 5-day real returns);
`sigma_freeze_steps` 30–100; warm-start *always* on regime-structured data; reseed
(`set_seed` or `torch.manual_seed`) immediately before model construction for
reproducibility.

## II.3 Reading the monitoring history

`trainer.fit` returns (and `trainer.history` accumulates) one dict per step:

| key | definition | healthy | act when |
|---|---|---|---|
| `step`, `loss`, `nll` | loss = nll + enabled aux terms | nll trends down, noisily (SGD on a mixture is not monotone) | plateaus at the *marginal* NLL: experts never specialized → warm-start |
| `gate_entropy` | mean per-sample `H(π)`, nats | starts ≈ `log K`, declines gradually | stuck at `log K` for hundreds of steps → dead gate (see below); ≈0 within ~50 steps → collapse; raise `gate_weight_decay` |
| `sharpness` | `mean_b max_k r_bk` | 1/K rising gradually | 1.0 immediately = collapse |
| `min_batch_utilization` | min over experts of this batch's mean responsibility | **may hit ~0 on one-regime dates — healthy** | (never alarm on this alone) |
| `min_running_utilization` | same, over the LB buffer (many batches) | comfortably > 0.05 | → 0 = an expert died; warm-start, or ablate `aux_load_balance=True` |
| `max_offdiag_expert_corr` | max |corr| between expert outputs | well below 1 | ≈1 = homogenization; warm-start, or ablate `aux_expert_decorrelation=True` |

Dead-gate triage, in order: (1) is `max_offdiag_expert_corr` ≈ 1? Homogenized experts —
warm-start with a better sort key. (2) Do the windows actually carry regime signal?
(On synthetic: are `vol_levels` distinguishable?) (3) Only then reach for the aux
losses — and if you enable one, that run is an *ablation arm*, logged as such.

## II.4 Getting real data (Stage B), step by step

The real data is CRSP (release `ciz202512`) and the CRSP/Compustat Merged release
(`cfz202607`), licensed to the university. **Licence rules:** `Quant Model/Data/` is
gitignored; everything derived from them (extracts, parquet, built panels, coverage
tables, registries of real-data runs) goes to `Data/derived/` and nowhere else; test
fixtures are invented; committed reports quote aggregate statistics only.

### Build the panel (once)

From `nec_baseline/`, with the `crsp` extra installed (`pip install -e '.[crsp]'`, i.e.
pyarrow). RUNBOOK section 8a has the same four commands with what each writes:

```bash
python3.14 scripts/extract_crsp_v2.py           # CRSP StkDlySecurityData, resumable
python3.14 scripts/extract_compustat.py         # CCM link, GICS, quarterly fundamentals
python3.14 scripts/build_pit_panel.py           # the Q26 PIT panel, coverage, build report
python3.14 scripts/feature_coverage_report.py   # aggregate coverage -> results/feature_coverage/
```

The CRSP extract (`Data/derived/crsp_extract_ciz202512_2015-01-01_2024-12-31_StkDlySecurityData_lb1887d/`)
holds the rows of the PERMNOs that were S&P 500 members in 2015-2024 and of their
companies' other share classes, from late 2009 (the feature windows' five years) to two
months after the window, plus their membership spells, delisting records,
security-info history and the market series. The Compustat extract
(`Data/derived/compustat_cfz202607_2015-01-01_2024-12-31/`) holds the linked GVKEYs' link
and GICS rows and their quarterly files from mid-2011. The build writes
`pit_panel_crsp_2015-01-01_2024-12-31_q26_market_neutral.pt`, the coverage `.csv` and an
aggregate `.report.json` next to it; the brief 06 files
(`crsp_extract_ciz202512_2015-01-01_2024-12-31/`, `pit_panel_crsp_2015-01-01_2024-12-31.pt`,
legacy features and raw target) are left in place. The scripts take an optional
`[start] [end]`; the build reads the default window's extracts, which cover every
sub-window.

### Use it

```python
from nec_moe import (NECConfig, EncoderConfig, ExpertConfig, TrainConfig, NECModel,
                     Trainer, CRSPSpec, StageBSpec, build_crsp_panel, data_config_from_panel)

spec = StageBSpec(seq_len=20, horizon=5)    # market-neutral target, Q26 inputs
build = build_crsp_panel(CRSPSpec(), spec)  # loads the default Compustat extract
panel = build.panel     # or torch.load(CRSPSpec().panel_path_for(spec), weights_only=False)
print(build.coverage)                        # PUBLISH this next to any result

cfg = NECConfig(
    data=data_config_from_panel(panel),
    encoder=EncoderConfig(hidden_dim=32),
    experts=ExpertConfig(hidden_dims=(64, 32)),
    train=TrainConfig(sigma_init=0.05,       # ≈ std of 5-day returns!
                      sigma_freeze_steps=100, lr=1e-3, batch_size=512),
)
```

In the control panel: `data="crsp"` (build from the extracts, with `start`, `end`,
`post_delisting_return`, `target_kind`, `feature_set` and `input_mode`) or
`data="panel_file"` (the prebuilt panel; the default path is the Q26 market-neutral
file). Every trial the panel produces carries `data_source="crsp_ciz202512"`, the
`post_delisting_return`, `target_kind`, `feature_set`, `crsp_stock_file`,
`compustat_release` and `sector_source` it was built with, and the model's
`gate_weight`.

### `StageBSpec` reference

| field | default | meaning |
|---|---|---|
| `seq_len` | 20 | trailing window length (the encoder's T) |
| `horizon` | 5 | forward-return target in trading days (5 or 20 per the syllabus); **purge by exactly this number in evaluation** |
| `cs_rank` | True | per-date rank-normalize snapshot features (Q26: to [−1, 1], required; legacy: [−0.5, 0.5]) |
| `min_names_per_date` | 5 | drop dates with a thinner valid cross-section |
| `target_kind` | `"market_neutral"` | Q25: forward return minus the date's cross-sectional mean (`fwd_mn_ret_{h}d`); `"raw"` (`fwd_ret_{h}d`); `"residual"` = trailing-beta residual `fwd − β_t·mkt_fwd`, not the decision |
| `beta_window` | 250 | trailing days for the rolling market beta (residual only) |
| `features` | `FeatureSpec()` | the feature set (`"q26"` or `"legacy14"`) and every Q26 window, threshold and cap |
| `input_mode` | `"snapshot"` | the experts' input mode the panel is built for: `"snapshot_plus_hidden"` drops rows with an incomplete sequence window (Q26 only) |

### What the CRSP build does, and the choices it records

- **Universe:** S&P 500 membership spells of `membership_indno=1000500` (bounds
  inclusive; 502-508 members per day). Features are built for every ever-member, then
  `filter_point_in_time` keeps a row only on its PERMNO's member dates and re-ranks the
  snapshot features among the members. Dual-class companies stay two PERMNOs.
- **Returns:** `log(1 + DlyRet)`; a missing return is missing, and it invalidates every
  row whose features or target need it (up to 120 days, for `mom_120d`).
- **Delistings:** `DlyRet` already includes the delisting return (checked on every
  delisting row). The delisting row is the stock's final return, never a panel row. A
  forward window that runs past it is completed with `post_delisting_return`: `"cash"`
  (0, the default, not a decision) or `"market"`. A delisting without a delisting return
  leaves its targets missing; nothing is imputed.
- **Volume:** dollar volume `|DlyPrc| x DlyVol`; share volume put on today's share basis
  with `DlyCumFacShr` ratios inside the 20-day window, so a split does not move the
  z-score.
- **Market:** `log(1 + DlyTotRet)` of `market_indno=1000500` (the like-for-like SPY
  replacement; `1000200` for a wider universe) feeds the market features, the beta and
  the residual target.
- **Checks:** the build refuses an extract too short for the features' warm-up or the
  horizon, an extract built with another stock file or lookback, and a member count
  outside `member_count_min..member_count_max` (500..510).
- **Q26 inputs (brief 08):** each PERMNO's company market equity (every share class),
  GICS sector, report dates and point-in-time fundamentals come from the Compustat
  extract through the CCM link on each date. A row is never dropped for a missing
  characteristic: it is ranked without it, filled with 0 and flagged. In `taccruals_at`
  and `noa_at`, `IVAOQ`, `IVSTQ`, `MIBQ` and `PSTKQ` count as 0 when blank
  (`FeatureSpec.zero_if_missing_items`; decided 2026-10-05 after the first coverage report
  showed `taccruals_at` present on 5% of rows under the strict rule); every other item stays
  strictly missing, so a definition that does not apply (a bank's current assets) is still
  missing.

## II.5 Bringing your own data

Three routes, by decreasing convenience:

**Route 1 — another CRSP window.** `dataclasses.replace(CRSPSpec(), start=..., end=...)`
with `build_crsp_panel(spec, stage_b, extract=load_extract(CRSPSpec()))` for any
sub-window of the default extract (the Q26 build also loads the default window's
Compustat extract); for a window outside it, run `extract_crsp_v2.py` and
`extract_compustat.py` with the new dates first.

**Route 2 — you have daily returns and volumes.** One daily frame per entity
(`DAILY_COLUMNS`: `ret` = daily log return with NaN for missing, `volume`,
`dollar_volume`, and optionally `share_factor`, `tradable`, `fill_ret`) plus the market's
daily log return:

```python
from nec_moe import StageBSpec, assemble_panel, market_frame
spec = StageBSpec(seq_len=20, horizon=5)
panel = assemble_panel({"10001": daily1, "10002": daily2, ...},  # per-entity frames
                       market_frame(mkt_log_ret, spec), spec,
                       data_source="my_source")                 # recorded on every trial
```

**Route 3 — you have your own features.** Construct a `Panel` directly. Complete
runnable miniature (2 dates × 2 entities):

```python
import torch
from nec_moe import Panel, FeatureSchema

schema = FeatureSchema(sequence_features=("my_ret", "my_vol"),
                       snapshot_features=("f1", "f2", "f3"),
                       target="fwd_ret_5d")
panel = Panel(
    x_seq=torch.randn(4, 20, 2),            # (N=4 rows, T=20, d_seq=2)
    x_snap=torch.randn(4, 3),
    y=torch.randn(4),
    date=torch.tensor([0, 0, 1, 1]),        # date-major order! (all of date 0 first)
    entity=torch.tensor([0, 1, 0, 1]),
    schema=schema,
    date_labels=("2024-01-02", "2024-01-03"),   # needed for PIT filter + alignment
    entity_labels=("tick1", "tick2"),
)
```

Contract checklist for route 3: float32 features, int64 codes, **date-major row
order**, widths matching the schema (checked at construction), features at t computed
from information ≤ t only, the target the *only* forward column. Then prove the timing
rule for your features by copying `test_no_lookahead`'s pattern: rebuild the features
from a series truncated at t and assert the row at t is identical.

## II.6 Training — the full manual

### Choosing and training each prior

```python
from nec_moe import PriorConfig
# Variation 2 family (memoryless) — trained with trainer.fit(panel, steps):
PriorConfig(kind="soft")                                    # baseline
PriorConfig(kind="uniform")                                 # no-routing ablation
PriorConfig(kind="hard")                                    # top-1, hard-EM objective
PriorConfig(kind="topk", top_k=2)                           # sparse; k in [2, K]
PriorConfig(kind="gumbel", tau_init=1.0, tau_min=0.1, tau_anneal_steps=300)
# Variation 3 (Markov) — trained with trainer.fit_sequence(...):
PriorConfig(kind="hmm", transition_diag_bias=2.0, learn_pi0=True)
```

Per-mechanism notes:

- **soft** — the reference. Everything in II.2/II.3 applies unchanged.
- **uniform** — expect `gate_entropy ≡ log K` and zero gate grads *by design*; only the
  experts train. Use it to price the value of routing on your panel.
- **hard** — watch `sharpness` (≡1.0 by construction) and utilization: from a cold gate
  it will glue itself to an arbitrary expert (I.8). Give it a warmed gate (train soft
  first, then rebuild with `kind="hard"` reusing the state dict — II.6 "checkpoints")
  or accept the cold-start row as a finding. Held-out `evaluate()` scores the honest
  selected-expert density.
- **topk** — behaves like soft with a sparsity floor; at K=2, `top_k=2` *is* soft
  (sanity: identical numbers).
- **gumbel** — train-mode metrics are noisy by construction (fresh noise per forward);
  judge on eval passes. Anneal roughly over the first half of training
  (`tau_anneal_steps ≈ steps/2`), then freeze the schedule choice and pre-register it.
- **hmm** — requires `TrainConfig(sequence_ordered=True)` (validation enforces), data
  through `time_sequence()`:

  ```python
  trainer.fit_sequence(train.time_sequence(), steps=300, chunk_len=50)
  ev = trainer.evaluate_sequence(test.time_sequence())      # causal filtering pass
  print(trainer.model.prior.transition_matrix)              # learned A (prob domain)
  ```

  `chunk_len` ≈ 40–60 dates is a good default (I.11). To score a test block with a
  filter warmed through the (past) training window, pass
  `init_state=trainer.evaluate_sequence(train.time_sequence()).log_filtered[-1]` — the
  harness does this for you. Inspect `π₀` via `model.prior.pi0_logits.softmax(-1)`.

### Emissions and the Hamilton baseline

```python
from nec_moe import ExpertConfig
ExpertConfig(kind="mlp")         # neural experts (default)
ExpertConfig(kind="classical")   # per-regime constant Gaussians

# classical × hmm == Hamilton. Warm-start = quantile moment init on y itself:
cfg = small = NECConfig(data=..., experts=ExpertConfig(kind="classical"),
                        prior=PriorConfig(kind="hmm"),
                        train=TrainConfig(sequence_ordered=True, lr=3e-2))
trainer = Trainer(NECModel(cfg))
trainer.warmstart_experts(train.full_batch(), sort_key=train.y)   # legal: train data
trainer.fit_sequence(train.time_sequence(), steps=250, chunk_len=50)
```

Compare Hamilton on **held-out NLL and regime recovery, never rank-IC** (constant
per-date predictions have no cross-sectional ranking; the error message will remind
you).

### Checkpointing and resume (surviving a 100-hour run)

`Trainer.save(path)` / `Trainer.load(path)` persist the **complete** training state:
model, optimizer, `step_count` (the sigma freeze and Gumbel tau anneal key off it, so
resume continues the schedules — never resets to 0), the load-balance buffer, the
metrics history, the minibatch sampler frozen mid-epoch (generator state + current
permutation + position), the stateful path's chunk cursor + carried filter state, and
the *global* torch RNG (dropout draws from it). The consequence, pinned by
`tests/test_checkpoint.py`: **interrupt + reload + finish reproduces the uninterrupted
trajectory bit for bit**, given the same panel/sequence — the checkpoint stores the
sampler, not the data.

```python
trainer.fit(train, steps=100_000, checkpoint_path="runs/soft.pt")  # long run …
# … process dies at step 61_430. Later:
trainer = Trainer.load("runs/soft.pt")
trainer.fit(train, steps=100_000 - trainer.step_count)             # same trajectory
```

- `TrainConfig(checkpoint_every=N)` sets the save cadence when a `checkpoint_path` is
  given (plus always one final save at the end of the call, and one on a requested
  stop); writes are atomic (tmp + rename), so a crash mid-write cannot corrupt the
  last good checkpoint.
- `steps` means *additional steps for this call* — `fit(60)` then `fit(40)` equals one
  `fit(100)`; compute the remainder from `trainer.step_count` as above.
- `fit_sequence` resumes mid-pass too, and validates that the resumed call uses the
  same sequence length and `chunk_len` (a chunking mismatch is a loud `ValueError`).

The layers above compose with this: `walk_forward_evaluate(..., resume_dir=...)`
persists each completed fold (scored payload + trained model state) as `fold_<i>.pt`
and skips it on restart, while a fold interrupted mid-fit resumes from its
`fold_<i>_trainer.pt` (rotated copies `.1`, ... per `checkpoint_keep`) with its gate
and base restored from `fold_<i>_gate.pt` and `fold_<i>_base.pt`;
`run_sweep(..., resume_dir=...)` additionally skips every (arm, seed) pair that
already has a registry row, reusing its logged metrics. `Trainer.save_checkpoint`
rotates and `Trainer.load_latest` falls back past a torn file. From the control panel,
checkpoints are always on and Ctrl+C stops cleanly; relaunch with
`python3.14 run_experiment.py --resume` (II.0a, RUNBOOK.md).

For **inference-only** artifacts (shipping a fitted model, no optimizer), the plain
recipe still works and is smaller:

```python
import torch
torch.save({"state_dict": trainer.model.state_dict(),
            "config": trainer.model.cfg.to_dict()}, "runs/nec_soft_final.pt")
ckpt = torch.load("runs/nec_soft_final.pt", weights_only=False)
model = NECModel(NECConfig.from_dict(ckpt["config"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

### Determinism rules

Seed *immediately before* model construction (expert init draws from the global RNG);
keep `dropout=0` if you need bit-reproducible train-mode forwards; `model.eval()`
before any evaluation you report (dropout off, Gumbel deterministic); the data
generators take explicit seeds and are independent of the global RNG.

## II.7 Evaluation — the full manual

### `walk_forward_evaluate`, parameter by parameter

```python
result = walk_forward_evaluate(
    panel,
    make_trainer,             # () -> Trainer with a FRESH model; reseed inside!
    n_folds=4,                # how many consecutive test blocks at the end
    test_dates_per_fold=60,   # block width in dates
    purge_dates=5,            # ALWAYS = the label horizon (StageBSpec.horizon)
    steps=600,                # training steps per fold
    warmstart_key=lambda p: p.x_seq[:, -1, 2],   # or None to skip
    min_train_dates=1,        # guard for degenerate requests
    backtest_quantiles=5,     # None disables the long-short backtest
    cost_rate=0.001,          # cost per unit traded notional (10 bps)
    portfolio_scheme="nonoverlapping",  # | "staggered": the h-day book (I.13, audit E-3)
    resume_dir="runs/wf_soft",  # None = off; fold-level resume (II.6)
)
```

The two easy mistakes: (1) `make_trainer` that reuses one model — each fold must be a
fresh optimization (fit-once-per-window; the function's contract), so construct inside
the lambda and reseed there for determinism; (2) `purge_dates ≠ horizon` — the purge is
the label horizon, nothing else.

`resume_dir` makes a long harness run interruptible: each completed fold is persisted
there and skipped on restart, and a fold killed mid-fit resumes from its trainer
checkpoint (written every `TrainConfig.checkpoint_every` steps) — the restarted run
reproduces the uninterrupted one exactly, provided `make_trainer` reseeds (as above)
and the settings/panel are unchanged. `run_sweep` takes the same parameter and
additionally skips (arm, seed) runs already logged in the registry. Full mechanics:
II.6 "Checkpointing and resume".

`make_trainer` pattern:

```python
def make_trainer():
    torch.manual_seed(0)          # identical init per window; vary for seed sweeps
    return Trainer(NECModel(cfg))
```

### Reading the results

```
result.pooled_ic         IcSummary over ALL folds' test dates
    .mean_ic  .ic_std  .icir (= mean/std, per period)  .t_stat (= icir·√T)  .n_dates
result.pooled_portfolio  PortfolioSummary | None
    .mean_gross .mean_net .gross_std .ir_gross .ir_net .mean_turnover .cost_rate
result.mean_fold_nll     average held-out predictive NLL
result.folds[i]          FoldResult:
    .fold .nll .ic .n_train .n_test .portfolio .expert_order   (σ-sorted; () for baselines)
```

Interpretation guardrails: with 30 names/date, a daily IC has standard error
≈ 1/√29 ≈ 0.19 — pooled t-stats below ~2 are noise; per-period IRs are **not**
annualized Sharpe ratios (multiply by √252 only when the dates are real trading days
and say so); `expert_order` is your key for aligning per-expert statistics across folds
(Decision D) — never average "expert 0" across folds without it.

### Baselines and fair comparisons

```python
ridge = walk_forward_evaluate_baseline(panel, lambda: RidgeBaseline(l2=1.0), **common)
mlp   = walk_forward_evaluate_baseline(
    panel, lambda: MLPBaseline(input_dim=panel.x_snap.shape[1], hidden_dim=32,
                               steps=600), **common)
```

Fairness checklist for a comparison table: identical `common` split parameters
(byte-identical folds and grading — guaranteed if you pass the same dict), matched
training budgets, same panel object, per-model seeds swept identically, and one
registry tag for the whole family (II.9). The classical Hamilton row reports NLL and
regime metrics only.

### Getting a fold's raw predictions (custom analyses)

The harness doesn't return predictions; reproduce any fold directly:

```python
from nec_moe.evaluation import walk_forward_folds
folds = walk_forward_folds(panel.date, n_folds=4, test_dates_per_fold=60, purge_dates=5)
f = folds[2]
train, test = panel.subset_dates(f.train_dates), panel.subset_dates(f.test_dates)
trainer = make_trainer(); ...fit as the harness does...
model.eval()
with torch.no_grad():
    pred = model(test.x_seq, test.x_snap).y_hat        # memoryless
```

## II.8 Regime alignment (the interpretability analysis)

Quick version (demo-grade — in-sample dates included):

```python
from nec_moe import load_vix, load_french_factors, build_context, gate_regime_alignment
ctx = build_context(load_vix("data_cache"), load_french_factors("data_cache"))
report = gate_regime_alignment(trainer, panel, ctx)
print(report.to_frame().round(3))
#            vix  mkt_rf   smb  ...  abs_mkt  mkt_vol_20d  high_vix_util  low_vix_util
# expert_0  0.81   -0.06  0.01  ...     0.53         0.85          0.216         0.065
# expert_1 -0.81    0.06 -0.01  ...    -0.53        -0.85          0.784         0.935
```

Thesis-grade version — held-out folds, multi-seed:

```python
rows = []
for f in walk_forward_folds(panel.date, n_folds=4, test_dates_per_fold=60, purge_dates=5):
    train, test = panel.subset_dates(f.train_dates), panel.subset_dates(f.test_dates)
    for seed in range(5):
        trainer = make_trainer(seed); ...warmstart + fit on train...
        rep = gate_regime_alignment(trainer, test, ctx)     # HELD-OUT dates only
        order = canonical_expert_order(trainer.model.experts.log_sigma)
        k_hi = int(order[-1])                                # highest-σ expert
        rows.append({"fold": f.fold, "seed": seed,
                     "vix_corr": rep.per_expert[k_hi].correlations["vix"]})
```

Report the distribution of `vix_corr` across folds × seeds — that is the honest form of
the +0.81 headline. Both prior families work (`hmm` uses the filtered posterior). For
synthetic panels, index the context frame by integer date codes.

### Gate calibration (reliability + temperature scaling)

A gate can rank regimes correctly and still lie about probabilities — and gate
probabilities are *mixture weights*, so overconfidence misallocates experts on every
forward pass (M4 §4.9). For a latent-regime gate, "calibrated" means: among held-out
samples where the gate claimed `π_k ≈ q`, the realized **responsibility** `r_k` averages
`q` (the responsibilities are the gate's training target, so they are its honest
yardstick).

```python
from nec_moe import gate_reliability, fit_temperature, plot_reliability, validation_tail

report = gate_reliability(trainer, test_panel)        # held-out data only!
print(report.ece, report.to_frame())

tail = validation_tail(train_panel.date, val_dates=40, purge_dates=5)
fit = fit_temperature(trainer, train_panel.subset_dates(tail.test_dates))
print(fit.verdict)     # e.g. "gate overconfident (T > 1 softens): T = 2.31"

fixed = gate_reliability(trainer, test_panel, temperature=fit.temperature)
plot_reliability(report, path="figs/rel_before.png")
plot_reliability(fixed,  path="figs/rel_after.png")
```

Protocol: fit `T` on a **validation tail of the training window** (never the test
block); it is one scalar chosen by held-out mixture NLL, so it cannot meaningfully
overfit — that is the charm of temperature scaling. Scope: softmax-predictive gates
(soft, Gumbel-at-eval); hard/top-k have no smooth confidence to rescale, and the HMM's
regime probabilities come from the forward filter, not the gate — the functions reject
those with guidance. Planted-truth tested: identical experts ⇒ ECE exactly 0; a gate
with ×4-sharpened logits ⇒ fitted T > 1.5 improving NLL *and* ECE out of sample, while
an honestly trained gate fits T ≈ 1.

## II.9 Multi-seed sweeps, the registry, and defensible claims

The protocol layer is `run_sweep` (`nec_moe/sweep.py`): named **arms** × a **seed
grid**, every run through the identical harness, every trial logged, mean ± std per
arm. This is the intended shape of every thesis experiment:

```python
from nec_moe import (NECConfig, PriorConfig, RidgeBaseline, MLPBaseline,
                     TrialRegistry, baseline_arm, corrected_claims, nec_arm,
                     run_sweep)

reg = TrialRegistry("results/routing_sweep.jsonl")
vol_key = lambda p: p.x_seq[:, -1, 2]                 # observable regime proxy

arms = [
    nec_arm("soft",    NECConfig(data=data_cfg, train=train_cfg),
            warmstart_key=vol_key),
    nec_arm("uniform", NECConfig(data=data_cfg, prior=PriorConfig(kind="uniform"),
                                 train=train_cfg), warmstart_key=vol_key),
    nec_arm("hard",    NECConfig(data=data_cfg, prior=PriorConfig(kind="hard"),
                                 train=train_cfg), warmstart_key=vol_key),
    nec_arm("topk",    NECConfig(data=data_cfg, prior=PriorConfig(kind="topk", top_k=2),
                                 train=train_cfg), warmstart_key=vol_key),
    nec_arm("gumbel",  NECConfig(data=data_cfg,
                                 prior=PriorConfig(kind="gumbel", tau_anneal_steps=300),
                                 train=train_cfg), warmstart_key=vol_key),
    baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)),
    baseline_arm("mlp",   lambda seed: MLPBaseline(input_dim=d_snap, hidden_dim=32,
                                                   steps=600, seed=seed)),
]
report = run_sweep(panel, arms, seeds=range(5), registry=reg, tag="routing_sweep",
                   steps=600, n_folds=4, test_dates_per_fold=60, purge_dates=5,
                   backtest_quantiles=5, cost_rate=0.001)

print(report.to_frame().round(3))    # arms × {mean_ic, mean_ic_std, icir, nll, …}
print(corrected_claims(report, alpha=0.10))   # {arm: (reject, q_value)}
```

What the pieces guarantee:

- `nec_arm(name, cfg, warmstart_key=)` plants each seed in the global RNG *and* in
  `TrainConfig.seed` via `dataclasses.replace`, so seeds vary the expert
  diversification and shuffling too; each seed remains fully deterministic. Every
  trial's registry row carries the complete `NECConfig` dict — reconstructible.
- `baseline_arm(name, build)` runs baselines through the same folds and the same
  grading code; deterministic baselines show a seed std of exactly 0 (itself
  informative in the table).
- **The claim family is the arms, not the seeds.** Seed replicates are repeated
  measurements of one hypothesis, not new hypotheses; `corrected_claims` combines each
  arm's p-values by their **median** and runs BH across arms. Pre-register that choice.
  (Do *not* take the min over seeds — that is anti-conservative selection inside the
  family.)

### Tuning before sweeping (`tune` — the honest hyperparameter protocol)

Hand-picked hyperparameters are researcher degrees of freedom; tuning on test dates is
leakage. The protocol: **tune once on a purged validation tail inside the first fold's
training window, freeze the winner, pre-register the candidate list.**

```python
from nec_moe import nec_arm, tune, walk_forward_folds

outer = walk_forward_folds(panel.date, n_folds=4, test_dates_per_fold=60, purge_dates=5)
train_panel = panel.subset_dates(outer[0].train_dates)   # NEVER the full panel

candidates = [
    nec_arm("lr1e-3", NECConfig(data=data_cfg, train=TrainConfig(lr=1e-3, ...)),
            warmstart_key=vol_key),
    nec_arm("lr3e-3", NECConfig(data=data_cfg, train=TrainConfig(lr=3e-3, ...)),
            warmstart_key=vol_key),
    nec_arm("wide",   NECConfig(data=data_cfg, encoder=EncoderConfig(hidden_dim=64),
                                train=train_cfg), warmstart_key=vol_key),
]
result = tune(train_panel, candidates, seeds=(0, 1, 2), registry=reg, tag="hp_tune",
              steps=600, val_dates=40, purge_dates=5)   # purge = label horizon, again
print(result.winner, result.scores)   # freeze this; run the real sweep with it
```

What it does under the hood: the validation tail *is* a one-fold walk-forward on the
training panel (`validation_tail` wraps the same tested purge arithmetic), candidates
*are* sweep arms scored by the same harness, the pick is by **seed-mean** of the metric
(`mode="min"` for NLL), and the selection is **logged**: per-run trials under the tag,
arm aggregates under `<tag>.arms`, and one `#selection` event carrying the candidate
count — tuning is a selection with multiplicity, and the registry remembers it.
Deliberately not offered: per-fold re-tuning (multiplies degrees of freedom).

Selecting a winner and deflating its Sharpe (unchanged from the registry workflow):

```python
import statistics
from nec_moe import deflated_sharpe_ratio
winner = reg.best("routing_sweep", "net_ir")          # LOGS the selection event
d = deflated_sharpe_ratio(
    winner_ls_returns,                                # per-date net L/S series (II.7)
    n_trials=reg.n_trials("routing_sweep"),
    sr_variance=statistics.pvariance(reg.metric_values("routing_sweep", "net_ir")))
print(f"naive PSR {d.psr_zero:.3f} → DSR {d.dsr:.3f} (hurdle {d.expected_max_sr:.3f})")
```

Registry API in one breath: `log(tag, metrics, config=, seed=, notes=)`, `trials(tag)`,
`n_trials(tag)`, `metric_values(tag, metric)`, `best(tag, metric, mode=)` (logs the
pick), `selection_events(tag)`. Metrics must be finite; the `#selection` suffix is
reserved.

## II.10 Making figures

`nec_moe/plots.py` provides the six standard figures as functions — figures are
artifacts of a run, saved next to the registry rows they illustrate. Every function
returns the matplotlib `Figure` and takes `path=` to save (dpi 150, parents created);
matplotlib is imported lazily (optional dependency; set `MPLBACKEND=Agg` for headless).

```python
from nec_moe import (plot_training_dashboard, plot_gate_utilization, plot_ic_series,
                     plot_long_short_curve, plot_transition_matrix, plot_sweep_report)

plot_training_dashboard(history, n_experts=2, path="figs/dash.png")
    # 4 panels: NLL, entropy (log K line), min running utilization (alarm), expert corr

plot_gate_utilization(trainer, panel, context=ctx, path="figs/util_vs_vix.png")
    # the interpretability headline: one expert's per-date gate share (default:
    # highest-σ), filled; memoryless -> mean predictive π, HMM -> the FILTERED
    # posterior (the Hamilton figure); optional context overlay on a twin axis

plot_ic_series(pred, test.y, test.date, path="figs/ic.png")
plot_long_short_curve(pred, test.y, test.date, test.entity,
                      n_quantiles=5, cost_rate=0.001, path="figs/ls.png")
plot_transition_matrix(trainer, path="figs/A.png")     # Variation 3; rejects memoryless
plot_sweep_report(report, metric="mean_ic", path="figs/sweep.png")
    # mean ± seed-std bars per arm — "nothing is a result until it has a seed std",
    # as a picture
plot_reliability(gate_reliability(trainer, test_panel), path="figs/rel.png")
    # the calibration diagram (II.8): realized responsibility vs claimed π + ECE
```

The raw matplotlib recipes below remain for *custom* figures — they show how to work
directly against the returned data structures.

**1. Training dashboard** (from `history = trainer.fit(...)`):

```python
import math
import matplotlib.pyplot as plt

K = trainer.cfg.experts.n_experts
s = [h["step"] for h in history]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), tight_layout=True)
ax[0].plot(s, [h["nll"] for h in history]); ax[0].set_title("train NLL")
ax[1].plot(s, [h["gate_entropy"] for h in history]); ax[1].set_title("gate entropy")
ax[1].axhline(math.log(K), ls="--", c="gray", lw=1)
ax[2].plot(s, [h["min_running_utilization"] for h in history])
ax[2].axhline(0.05, ls="--", c="r", lw=1); ax[2].set_title("min running utilization")
fig.savefig("figs/dashboard.png", dpi=150)
```

**2. Gate utilization vs VIX** (the interpretability headline):

```python
import pandas as pd
from nec_moe import gate_utilization_by_date
from nec_moe.diagnostics import canonical_expert_order

dates, util = gate_utilization_by_date(trainer, panel)
k_hi = int(canonical_expert_order(trainer.model.experts.log_sigma)[-1])
idx = pd.to_datetime([panel.date_labels[int(d)] for d in dates])
fig, ax1 = plt.subplots(figsize=(11, 3.5), tight_layout=True)
ax1.plot(idx, util[:, k_hi].numpy(), lw=1, label=f"expert {k_hi} (high-σ) share")
ax2 = ax1.twinx(); ax2.plot(idx, ctx["vix"].reindex(idx).values, c="gray", alpha=.5)
ax1.set_ylabel("gate share"); ax2.set_ylabel("VIX"); ax1.legend(loc="upper left")
fig.savefig("figs/util_vs_vix.png", dpi=150)
```

**3. Filtered regime path** (HMM/Hamilton — the classic figure):

```python
ev = trainer.evaluate_sequence(panel.time_sequence())
prob = ev.log_filtered.exp()[:, :, k_hi].mean(dim=1)     # (L,) mean over entities
fig, ax = plt.subplots(figsize=(11, 2.8), tight_layout=True)
ax.fill_between(range(len(prob)), prob.numpy(), color="tab:red", alpha=.4)
ax.set_ylabel(f"P(regime {k_hi} | data ≤ t)"); ax.set_ylim(0, 1)
fig.savefig("figs/filtered_path.png", dpi=150)
```

**4. Daily IC series** (the honest picture behind a pooled IC):

```python
from nec_moe import rank_ic_by_date
d, ics = rank_ic_by_date(pred, test.y, test.date)
fig, ax = plt.subplots(figsize=(11, 2.8), tight_layout=True)
ax.bar(range(len(ics)), ics.numpy(), width=1.0, alpha=.6)
ax.plot(pd.Series(ics.numpy()).rolling(20).mean().values, c="k", lw=1.5)
ax.axhline(0, c="gray", lw=1); ax.set_ylabel("daily rank-IC")
fig.savefig("figs/ic_series.png", dpi=150)
```

**5. Net long-short equity curve**:

```python
from nec_moe import long_short_book
_, gross, tno = long_short_book(pred, test.y, test.date, test.entity, n_quantiles=5,
                                horizon=test.horizon, scheme="nonoverlapping")
net = gross - 0.001 * 2.0 * tno   # per period: h dates under "nonoverlapping"
fig, ax = plt.subplots(figsize=(11, 2.8), tight_layout=True)
ax.plot(net.cumsum(0).numpy()); ax.set_ylabel("cum. net L/S return (per-period)")
fig.savefig("figs/ls_curve.png", dpi=150)
```

**6. Learned transition matrix** (Variation 3):

```python
A = trainer.model.prior.transition_matrix.detach().numpy()
fig, ax = plt.subplots(figsize=(3, 2.6), tight_layout=True)
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center")
ax.set_xlabel("to regime"); ax.set_ylabel("from regime"); fig.colorbar(im)
fig.savefig("figs/transition.png", dpi=150)
```

## II.11 Extending the machinery

**A new routing mechanism** — skeleton:

```python
class MyPrior(RegimePrior):
    stateful: ClassVar[bool] = False
    def forward(self, gate_logits, ctx=None):
        assert_shape(gate_logits, (None, self.n_experts), "gate logits")
        log_prior = ...            # (B, K), logsumexp_k == 0; −inf allowed
        return PriorOutput(log_prior=log_prior, info={...})

PRIOR_REGISTRY["mine"] = lambda cfg: MyPrior(cfg.experts.n_experts)
```

Then: config fields + `validate()` rules if any; a contract test (normalization, eval
determinism, and above all *gradient behavior* — does the gate train, and should it?);
a row in `test_variant_trains_end_to_end`. Set `log_train_weights` only if the training
objective genuinely differs from the predictive weights (study `hard` first). Set
`stateful=True` only if you thread state — the trainer switches paths on it.

**A new emission**: subclass `Emission` (`forward → (μ (B,K), logσ (K,))` and
`warmstart_slices` — decide how *your* emission breaks symmetry); register in
`EMISSION_REGISTRY`; extend the four-corner test. Per-regime *linear* emissions
(Markov-switching regression) are the next natural rung.

**A new baseline**: subclass `BaselineModel` (`fit/predict/nll`; make `nll` a real
density for comparability) — the whole harness comes free via
`walk_forward_evaluate_baseline`.

**A new feature**: add the column in `features.py::stock_features` (trailing
information only, from the daily frame's returns and volumes), append the name to
`SEQUENCE_FEATURES`/`SNAPSHOT_FEATURES` (order = channel order), extend
`test_no_lookahead` and its CRSP-fixture twin in `test_crsp.py`. The no-lookahead test is the gatekeeper —
a feature that fails it does not exist.

**A new aux loss**: pure function in `losses.py`, config flag default-off, compose in
`Trainer.train_step`, test the math *and* the gradient path, and treat "on" as a
registered ablation arm.

## II.12 Error-message dictionary

The package fails loudly and specifically; the message usually *is* the fix.

| message (abridged) | meaning | fix |
|---|---|---|
| `shape mismatch for 'x_seq': expected (*, 20, 4), got …` | tensor/config disagreement at a module boundary | check `DataConfig` vs your panel; use `data_config_from_panel` |
| `prior.kind='hmm' requires train.sequence_ordered=True` | HMM needs chronological batches | set the flag; train via `fit_sequence` |
| `the model's prior is stateful (HMM): … use train_step_sequence/fit_sequence` | called `fit()` on an HMM model | use the sequence path |
| `prior.top_k must be in [2, n_experts]: … gate is gradient-dead (ledger defect 1)` | top-1-renormalized is untrainable | use `kind="hard"` for top-1 |
| `PriorConfig.tvtp … guarded-off extension` | TVTP not built (identifiability caution) | see Part III.2 before building |
| `auxiliary losses are supported on the memoryless path only` | aux + HMM | disable aux for the HMM arm |
| `n_experts must be >= 2 (a mixture)` | K=1 requested | use `MLPBaseline` — that *is* the single-model |
| `no complete CRSP extract at …: run scripts/extract_crsp_v2.py first` | the extract is missing or was interrupted | run (or rerun: it resumes) `scripts/extract_crsp_v2.py` |
| `the extract at … does not match the spec: stock_file …` | the extract was built with another stock file or lookback | run `scripts/extract_crsp_v2.py` (it writes a new folder), or set the spec to the extract's values |
| `no complete Compustat extract at …` | the Q26 build needs it | run `scripts/extract_compustat.py` (after the CRSP extract) |
| `this panel was built for input_mode=…` | a Q26 panel built for the other expert input mode | rebuild with `StageBSpec(input_mode=...)` matching `ExpertConfig.input_mode` |
| `members per date range … outside the band …` | the membership INDNO or the window is wrong | check `CRSPSpec.membership_indno` (1000500) and the band |
| `trial config lacks the provenance keys …` | a registry row without one of `PROVENANCE_KEYS` (`data_source`, `post_delisting_return`, `hidden_init`, `portfolio_scheme`, `target_kind`, `gate_weight`, `feature_set`, `crsp_stock_file`, `compustat_release`, `sector_source`) | pass `**trial_provenance(panel, cfg)` into its config |
| `only N dates have >= min_names valid names` | thin panel after validity filtering | widen dates/universe; lower `min_names_per_date` knowingly |
| `no date had a scoreable cross-section — constant per-date predictions …` | rank-IC on a classical emission | compare on NLL/regime recovery |
| `no overlap between panel dates and context index` | alignment join failed | real panels need `date_labels`; synthetic contexts index by integer codes |
| `sort_key length … != batch size` | warm-start key misaligned | compute the key from the same panel you pass |
| `not enough dates for the requested splits` | fold request exceeds the calendar | fewer/narrower folds, or less purge |
| `metric 'x' is not finite` | NaN/inf into the registry | fix upstream; the registry refuses to store lies |

## II.13 Practical notes

- **Timing** (this machine, CPU): full suite ~38 s; synthetic 400-step fit ~2 s;
  real-panel 600-step fit ~1–2 min; a 5-mechanism × 5-seed sweep with 4 folds ≈ 1–2 h —
  start it and walk away, the registry accumulates.
- **Memory**: the real panel is ~70k × (20×4 + 14) floats ≈ 25 MB; full-batch forwards
  are fine on CPU.
- **Reproducibility gotchas**: expert init consumes the global RNG → seed immediately
  before `NECModel(cfg)`; `minibatches` uses its own seeded generator (from
  `TrainConfig.seed`) → `fit` is reproducible given the seed; dropout makes train-mode
  forwards stochastic → tests and comparisons use `dropout=0` or `eval()`.
- **The cache is a lab notebook**: `data_cache/` + `results/*.jsonl` + `figs/` +
  committed code = a fully reconstructible experiment. Commit all four together. The
  exception is anything derived from CRSP (the extract, the panel, registries of CRSP
  runs): it lives in `Data/derived/`, is gitignored, and is never committed.

---

# Part III — The Roadmap

## III.1 If the supervisor picks **Variation 2** (routing-mechanism comparison)

Everything mechanical exists — six priors, the harness, the registry. The remaining
work is *experimental design and execution*:

1. **Pre-registration document first** (hours, not days — the form exists). Copy
   `PREREGISTRATION_TEMPLATE.md` to `results/prereg_<tag>.md`, fill the grid
   (mechanisms × K ∈ {2,3,4} × ≥5 seeds), metrics, claim family (BH at α=0.10,
   median-combined replicates), split parameters, and the success/negative-result
   criteria; commit before the first trial. The registry then makes deviations visible.
2. ✅ **Multi-seed sweep runner** — built (`run_sweep`, II.9); the grid is now a list of
   `nec_arm`/`baseline_arm` entries.
3. **Hard routing's cold start needs a fair shake** (1–2 days). Give top-1 its
   literature-standard aids — a soft-warmed gate (train soft, switch mechanism via the
   checkpoint recipe) and/or train-time logit noise — or the comparison indicts the
   cold start, not the mechanism. Reviewers will notice.
4. **Gumbel schedule tuning** (½ day). Sweep `tau_anneal_steps` once on synthetic,
   freeze, pre-register.
5. **Load-balancing / decorrelation ablation arms** (½ day). Both levers are wired and
   default-off; the grid should include soft±LBL and soft±decorr (the syllabus's "with
   vs without load balancing" row).
6. **The real-data grid on the PIT panel** (compute time) — the CRSP panel of II.4,
   coverage table published alongside.

## III.2 If the supervisor picks **Variation 3** (HMM-gated NEC)

The filter, transition machinery, causality tests, Hamilton baseline, and
filtered-path tooling exist. In order:

1. **Pre-registration** with the Variation-3 comparison set: HMM-NEC vs soft-NEC vs
   classical Hamilton vs single models — metrics split by capability (rank-IC/backtest
   for feature-conditioned models; NLL, regime recovery, filtered paths for all).
2. **Horizon thinking** (½ day design). Persistence pays when per-date evidence is weak
   and regimes are sticky; consider h=20 targets / weekly aggregation as pre-registered
   secondary settings — the synthetic result (HMM wins at weak windows × stay=0.97)
   says where to look.
3. **TVTP only if needed, with the caution attached** (2–4 days). Implement transition
   logits `L₀ + W·covariate` (VIX or `h_T`) behind the existing `prior.tvtp` guard.
   Exogenous-covariate form only — never score-driven/GAS (arXiv:2605.14976:
   non-identifiable) — and budget ~1000+ effective observations at K=3. A reproduced
   identifiability failure is itself a reportable result.
4. **Wasserstein template tracking — trigger-based** (1–2 days *if triggered*). Only if
   σ-sorting shows cross-refit instability (near-ties in σ_k). The stub and paper
   recipe are in `diagnostics.py`.
5. **Per-regime linear emissions** (1 day). The rung between constant Gaussians and
   MLPs — Markov-switching regression, Hamilton's econometric workhorse; a third
   `Emission` subclass making emission complexity a proper ablation axis.
6. **OPG standard errors for the Hamilton table** (optional, 1 day) — arXiv:2205.01565's
   outer-product-of-score recipe, if the thesis wants econometric error bars on Â.

## III.3 Variation-agnostic completion list (research rigor)

Ordered by value per effort; ✅ exists, ◻ to do:

1. ✅ **Multi-seed protocol** — `run_sweep` (`sweep.py`, II.9): arms × seeds through the
   shared harness, registry-logged, mean ± std per arm, BH-corrected claim family
   across arms with median-combined replicate p-values.
2. ✅ **Full point-in-time panel, on CRSP** — `scripts/extract_crsp.py` (resumable)
   then `scripts/build_pit_panel.py` (brief 06). Built 2015–2024 from CRSP `ciz202512`:
   736 ever-members, **1,264,598 PIT rows** across 2,516 dates, 502–508 members per day,
   delisting returns included; coverage by year 98.3%–99.8%, in
   `Data/derived/pit_coverage_crsp_2015-01-01_2024-12-31.csv` (licensed, gitignored,
   re-buildable). The earlier free-data panel (`data_cache/pit_panel_2015_2024.pt`,
   survivorship-biased) is obsolete and unused.
3. ✅ **Residual-return target** — `StageBSpec(target_kind="residual", beta_window=250)`
   (`rolling_beta`, trailing OLS β; market-clone ⇒ zero target, β=2 name ⇒
   market-neutral target, no-lookahead β — all planted-truth tested). The "raw vs
   residual" ablation is now two specs on the same prices.
4. ✅ **Hyperparameter discipline** — `tune` (`tuning.py`, II.9): purged validation
   tail inside the training window, candidates scored as sweep arms by the shared
   harness, winner by seed-mean, selection event logged with its multiplicity;
   tune-once-freeze protocol documented, per-fold re-tuning deliberately not offered.
5. ✅ **`plots.py`** — the six standard figures as tested functions (headless-safe,
   lazy matplotlib, `path=` saving); II.10 leads with them.
6. ✅ **Gate calibration** (M4 §4.9) — `calibration.py`: `gate_reliability` (reliability
   vs held-out responsibilities, ECE) + `fit_temperature` (held-out-NLL-fitted scalar on
   the validation tail) + `plot_reliability`; planted-truth tested both ways
   (identical experts ⇒ ECE 0; ×4-sharpened gate ⇒ T > 1.5 repairs NLL and ECE out of
   sample). The thesis-quality diagnostic nobody else will have — use it in the
   interpretability chapter.
7. ✅ **Lint + type-check + CI** — `ruff` (E/F/W/I/B/UP, clean) and `mypy` (clean over
   all 26 source files) configured in `pyproject.toml`; run them with
   `python3.14 -m ruff check nec_moe/ tests/ scripts/` and `python3.14 -m mypy nec_moe/`.
   The pass hardened real things: `zip(strict=True)` throughout, no
   call-in-argument-defaults, and `log_sigma` promoted to a declared field of the
   `Emission` contract (the σ-schedule and canonical ordering depend on it).
   CI: `.github/workflows/ci.yml` (repo root) runs ruff + mypy + the offline suite on
   CPU torch — live the day the repo gets a remote.
8. ✅ **Pre-registration template** — `PREREGISTRATION_TEMPLATE.md`: identity, frozen
   data + coverage posture, the complete candidate grid, tuning plan, evaluation
   protocol, claim family + corrections (BH over arms, median-combined replicates,
   deflated winner), success/negative-result criteria committed in advance, and an
   append-only deviations log. Copy to `results/prereg_<tag>.md`, fill, commit
   **before the first trial**.
9. ✅ Purged walk-forward, cost-aware backtest, registry + DSR/BH, PIT membership +
   coverage, alignment diagnostics, defect regression tests.

**All eight items are done.** The completion list is closed: seeds, the PIT panel, the
residual target, tuning discipline, figures, gate calibration, lint/type-check/CI, and
the pre-registration template. What remains is not infrastructure — it is the science:
copy `PREREGISTRATION_TEMPLATE.md`, fill it for the supervisor-approved variation,
commit it, and run the grid.

---

# Part IV — The Evaluation

## IV.1 As research code

Graded against the honest reference class — published quant-ML research code and
typical thesis repositories:

| dimension | grade | justification |
|---|---|---|
| Correctness assurance | **A** | 100+ tests: analytic gradients, reference filters, causality probes, planted-truth recoveries, defect regressions. Far above field norm. |
| Methodology | **A** | Purged WF, fit-once-per-window, logged selection events, DSR/FDR, capacity-matched baselines, one grading path — plus the protocol layer: multi-seed sweeps with arm-level corrections, tune-once-freeze validation tails, gate calibration, and a pre-registration template. The tooling is complete; what remains is *using* it on the pre-registered runs. |
| Reproducibility | **A−** | Deterministic seeds planted end-to-end, config round-trip in every registry row, cache-first data, append-only registry, figures as code, ruff+mypy clean, CI workflow committed. Missing: a pinned environment (lockfile) and an actual remote for the CI to run against. |
| Data rigor | **B+** | Timing contract *tested*; CRSP daily data with PIT S&P 500 membership and delisting returns, each construction rule checked against the release metadata and pinned by an invented-number fixture; licence-safe layout. Not yet: fundamentals (Compustat) and a universe wider than the S&P 500. |
| Architecture | **A−** | Two plug axes proven by tests; the protocol layer composes tested pieces rather than duplicating split logic; extension checklists are short because the seams are real. Remaining debt: conftest path hack (package not installed editable), no device/GPU handling. |
| Documentation | **A−** | Design doc with decision traceability; docstrings with shapes *and reasons*; this handbook. |

**Overall: strong research code whose distinguishing feature is that the
claims machinery was built before the claims.** Most research code does the opposite.

## IV.2 As quant-desk code

A desk pipeline has roughly six layers; where this code stands in each:

| layer | desk requirement | this code | gap |
|---|---|---|---|
| **Data** | vendor PIT (CRSP/Compustat/Refinitiv), corporate actions, delisting returns, 3000+ names, ongoing ingestion + QA | CRSP daily (CIZ), S&P 500 PIT membership, delisting returns, split-aware volume; one frozen release | medium — no fundamentals, ~500 names, no ongoing ingestion |
| **Signal research** | leakage-proof backtests, multiplicity control, registered experiments | purged WF, DSR/FDR, registry, causality tests | **small** — genuinely desk-grade *thinking*; a desk reviewer would recognize the discipline |
| **Portfolio construction** | optimizer + risk-model neutralization (Barra/Axioma), constraints, capacity | equal-weight quantile L/S | large — a research proxy, not a book |
| **Transaction costs** | spread + impact (√participation), borrow, realistic fills | linear cost on traded notional | medium — fine for ranking models, not for sizing capital |
| **Execution & ops** | scheduled retrains, drift monitoring, model versioning, kill criteria, audit | research trainer + JSONL registry | large — deliberately out of scope |
| **Scale** | GPU/distributed, big hyperparameter searches | single CPU, minutes | medium — models are tiny by design |

**Bottom line:** a signal-research prototype with production-grade methodology hygiene
and prototype-grade everything else — the correct shape for its purpose. A desk quant
would trust the honesty of its evaluation numbers (rare) and would not trade it as-is
(correct). Path to desk-usable, in order: (1) a vendor PIT data layer behind the
existing `Panel` contract — now in place for CRSP prices and membership (`crsp.py`),
fundamentals still to come; (2) a
real cost model + constrained optimizer replacing quantile L/S; (3) ops — scheduled
refits, drift monitors (gate entropy/utilization are already the right drift signals),
model versioning atop the registry; (4) scale-out, least urgent.

## IV.3 The one-paragraph verdict

The machinery is finished for its declared purpose — and so, now, is the protocol
around it. Both thesis variations are one config string away, the comparison table has
both of its sides, every number that could lie has a test or a correction standing over
it, and the completion list (III.3) is closed: seeds, the point-in-time panel, the
residual target, tuning discipline, figures, calibration, lint/type-check/CI, and the
pre-registration template all exist and are tested. Nothing left between here and
thesis-grade results is infrastructure: fill `PREREGISTRATION_TEMPLATE.md` for the
supervisor-approved variation, commit it, and run the grid through `tune` → `run_sweep`
→ `corrected_claims` → deflated winner. What separates the codebase from a desk is
still a data vendor and a portfolio layer, not a rewrite — the contracts were drawn so
those bolt on. The one risk to keep in view is unchanged: every real-data number shown
*so far* was a single-seed pipeline demonstration on measured-coverage data — the first
numbers that deserve belief are the pre-registered, multi-seed, corrected ones this
tooling now exists to produce.

---

*Handbook version 3 — 2026-07-15, matching suite state 132 offline + 3 network-gated
tests, ruff clean, mypy clean. When the code moves, move this document: it is committed
next to what it describes.*
