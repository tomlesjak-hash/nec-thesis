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
   `features.py`, `market_data.py`, `universe.py`, `context_data.py`), the judging
   machinery (`evaluation.py`, `baselines.py`), diagnostics (`diagnostics.py`,
   `alignment.py`), and the honesty layer (`registry.py`, `multiple_testing.py`).

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

### `market_data.py` — cached daily OHLCV (Stage B)

**The cache CSV is the interface**: `data_cache/{symbol}.us.{start}.{end}.csv` with
header `Date,Open,High,Low,Close,Volume`. Everything downstream reads only this format,
so the pipeline is source-agnostic and fully offline once cached — which is also the
manual-download workflow and how the tests run.

- `load_ohlcv(symbol, start, end, cache_dir, source, refresh)` → DataFrame
  (DatetimeIndex; columns `open/high/low/close/volume`; duplicate dates deduped;
  non-positive closes rejected).
- `source="stooq"` (syllabus primary): `stooq_url()` builds the endpoint; the fetcher
  detects Stooq's JavaScript anti-bot wall and **raises with instructions** (browser URL
  + exact cache filename) rather than circumventing it — a deliberate ethics/robustness
  stance. `source="yfinance"` (the syllabus's sanctioned prototyping fallback) fetches
  auto-adjusted bars and writes the *identical* cache format.
- `load_universe(tickers, …, min_tickers=5)` loads the whole list plus the market
  symbol (`spy`), collecting and reporting per-ticker failures; too few successes is an
  error (a 3-name "panel" would be meaningless silently).
- `DEFAULT_UNIVERSE` (~30 current large caps) exists to verify the pipeline; its
  docstring is a survivorship warning, not an endorsement.

### `features.py` — prices → Panel, under one timing rule

**The rule** (stated once, tested mechanically): a row (date t, ticker i) is a
prediction made *at the close of t*. Every feature uses information through t; the
forward target `fwd_ret_{h}d = log(C_{t+h}/C_t)` is the **only** forward-looking
column; cross-sectional rank-normalization uses only date-t's own cross-section.
`test_no_lookahead` truncates the series at t and asserts every feature at t is
identical — copy its pattern whenever you add a feature.

The named features (order = channel order; all trailing):

| snapshot column | formula |
|---|---|
| `ret_1d/5d/20d/60d` | `log C_t − log C_{t−k}` |
| `mom_120d` | 120-day log return |
| `vol_5d/20d/60d` | rolling std of daily log returns |
| `downside_vol_20d` | rolling std of `min(r, 0)` |
| `drawdown_60d` | `C_t / max(C_{t−59..t}) − 1` |
| `dollar_vol_20d` | `log(mean₂₀(C·V))` |
| `volume_z_20d` | `(V − mean₂₀V)/std₂₀V` |
| `rel_ret_20d` | `ret_20d − mkt_ret_20d` |
| `rel_vol_20d` | `vol_20d / mkt_vol_20d` |

Sequence channels (`d_seq=4`, natural units, per-row trailing window of `seq_len`):
`ret_1d`, `rel_ret_1d`, `vol_20d`, `volume_z_20d` — the observable regime signals the
encoder reads.

Assembly mechanics worth knowing: `_valid_rows` demands a full trailing window + all
snapshot features + the target; windows are built with `sliding_window_view` over each
ticker's *own* rows (positions, not calendar — a missing day shifts the window, it does
not create NaNs); dates keep only cross-sections with ≥ `min_names_per_date` names;
snapshot features are per-date **rank-normalized to [−0.5, 0.5]** when `cs_rank=True`
(point-in-time safe by construction, and the natural normalization for a rank-IC
target); final sort is date-major. `StageBSpec(seq_len=20, horizon=5, cs_rank=True,
min_names_per_date=5)`; `spec.target` names the target column.
`data_config_from_panel(panel)` derives the matching `DataConfig`.

### `universe.py` — approximately point-in-time membership (partial Module 12)

Survivorship has two components, and the module is explicit about which it fixes:

1. **Backward-looking selection** — *fixed*. Membership at any date is reconstructed
   from Wikipedia's S&P 500 constituent-change table by **reverse-chronological event
   undo**: start from today's membership; for each event *after* the as-of date, newest
   first, un-add the added and re-add the removed. Worked example: today = {A, C};
   events: 2023 (+C, −B), 2021 (+A, −Z). `members_asof(2022)`: undo 2023 ⇒ {A, B};
   result {A, B}. Undoing in reverse order makes leave-and-rejoin resolve correctly.
2. **Missing departed names** — *not fixed, made measurable*. Free sources rarely serve
   delisted tickers and never delisting returns. `universe_coverage_report(universe,
   dates, available)` publishes members / with-data / coverage per date — the honest
   number to print next to any backtest.

Reliability is *measured*, not assumed: the change table is ~complete only from the
2010s (`EARLIEST_RELIABLE = "2011-01-01"`; `members_asof` warns below it).
`members_union(start, end)` is the candidate list to *attempt* downloading;
`stable_members` the untouched-throughout subset; `filter_point_in_time(panel,
universe)` drops rows whose entity wasn't a member on that row's date (requires
`date_labels`/`entity_labels`; preserves date-major order). The defensible posture this
buys: *"survivorship-mitigated with documented residual coverage"* — never
"survivorship-free" (that requires CRSP).

### `context_data.py` — VIX and French factors (Stage C)

Official free sources, cache-first, parsers pure (offline-testable): VIX from CBOE's
history CSV (`MM/DD/YYYY` dates, close column); daily factors from Kenneth French's
zips — the parser survives the real files' quirks (text preamble; copyright footer; the
momentum file's **trailing comma on every line**; percent values → decimals; `−99.99`
missing markers → NaN, not −99% returns). `build_context(vix, factors)` joins and
derives `abs_mkt` and `mkt_vol_20d` (annualized 20-day realized market vol).
**Diagnostics only, never training inputs** (Decision B): the moment VIX enters
training, "do learned experts correspond to volatility regimes?" stops being a testable
question.

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

## I.18 The test suite as a map

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
| `test_smoke.py` | end-to-end training + regime recovery (AUC > 0.8) + a broken-gradient guard that must fail |
| `test_stage_b.py`, `test_stage_c.py`, `test_universe.py` | data parsers against fixtures that reproduce real-file quirks; the no-lookahead probe; point-in-time rollback across known epochs; coverage math; opt-in live tests |

`python3.14 -m pytest tests/ -q` — the whole offline suite, deterministic, ~30 s.

---

# Part II — The Manual

Everything below runs as written from `nec_baseline/` with the system `python3.14`.
Where output is shown, it is the output you should approximately see.

## II.0 The workflow at a glance

Every experiment in this codebase is the same seven moves:

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
python3.14 -m pytest tests/ -q                    # full offline suite (~30 s) — green?
python3.14 -m pytest tests/test_smoke.py -q       # just the end-to-end smoke
python3.14 -m pytest tests/ -q -k "hmm"           # any subset by keyword
NEC_NETWORK_TESTS=1 python3.14 -m pytest tests/test_stage_b.py tests/test_stage_c.py -q
                                                  # + live endpoints (needs network)
```

Dependencies: `torch`, `numpy`, `pandas` are load-bearing. Optional, imported lazily:
`yfinance` (price fallback), `lxml` (Wikipedia universe tables), `matplotlib` (your
figures). Everything is CPU; typical times on this machine: smoke test ~8 s, a
600-step real-panel fit ~1–2 min, the full suite ~30 s.

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

### The quick path (default universe — pipeline-grade only)

```python
from nec_moe import (NECConfig, EncoderConfig, ExpertConfig, TrainConfig, NECModel,
                     Trainer, build_stage_b_panel, StageBSpec, data_config_from_panel)

panel = build_stage_b_panel(
    "2015-01-01", "2024-12-31", "data_cache",
    source="yfinance",                       # see "sources" below
    spec=StageBSpec(seq_len=20, horizon=5),
)
print(len(panel), "rows,", len(panel.date_labels), "dates,",
      len(panel.entity_labels), "tickers")   # ≈ 71,700 / 2,490 / 30

cfg = NECConfig(
    data=data_config_from_panel(panel),
    encoder=EncoderConfig(hidden_dim=32),
    experts=ExpertConfig(hidden_dim=32),
    train=TrainConfig(sigma_init=0.05,       # ≈ std of 5-day returns!
                      sigma_freeze_steps=100, lr=1e-3, batch_size=512),
)
trainer = Trainer(NECModel(cfg))
trainer.warmstart_experts(panel.full_batch(),
                          sort_key=panel.x_seq[:, -1, 2])   # channel 2 = vol_20d
```

First run downloads (~1–2 min for 31 symbols); every rerun is fully offline from the
cache. **The default universe is survivorship-biased** — treat every number from it as
a pipeline demonstration.

### `StageBSpec` reference

| field | default | meaning |
|---|---|---|
| `seq_len` | 20 | trailing window length (the encoder's T) |
| `horizon` | 5 | forward-return target in trading days (5 or 20 per the syllabus); **purge by exactly this number in evaluation** |
| `cs_rank` | True | per-date rank-normalize snapshot features to [−0.5, 0.5] — leave on |
| `min_names_per_date` | 5 | drop dates with a thinner valid cross-section |
| `target_kind` | `"raw"` | `"residual"` = market-neutral target `fwd − β_t·mkt_fwd` (β from a *trailing* window — no lookahead, planted-truth tested); the syllabus's "raw vs residual" ablation is these two specs on the same prices |
| `beta_window` | 250 | trailing days for the rolling market beta (residual only); windows > 120 cost extra warm-up rows beyond `mom_120d`'s |

### The point-in-time path (what any shown result should use)

```python
from nec_moe import (load_sp500_universe, filter_point_in_time,
                     universe_coverage_report, build_stage_b_panel, StageBSpec)

u = load_sp500_universe("data_cache")                     # Wikipedia, cached
candidates = sorted(u.members_union("2015-01-01", "2024-12-31"))   # ~600 tickers
panel = build_stage_b_panel("2015-01-01", "2024-12-31", "data_cache",
                            tickers=tuple(candidates), source="yfinance",
                            spec=StageBSpec(seq_len=20, horizon=5))
# expect skip-reports: departed names often have no data — that is the residual bias
panel = filter_point_in_time(panel, u)                    # kill component 1

coverage = universe_coverage_report(
    u, ["2016-01-04", "2018-01-02", "2020-01-02", "2022-01-03", "2024-01-02"],
    available=set(panel.entity_labels))
print(coverage)     # PUBLISH this table next to any result from this panel
```

Notes: the first candidate download is slow (hundreds of symbols; polite 0.5 s pauses)
— let it run once, it caches. Membership history is reliable from ~2011 only
(`EARLIEST_RELIABLE`; earlier as-of dates warn). The posture this buys:
*survivorship-mitigated with documented residual coverage* — say exactly that, never
"survivorship-free".

### Sources, the cache, and manual downloads

- Cache anatomy: `data_cache/aapl.us.2015-01-01.2024-12-31.csv` —

  ```
  Date,Open,High,Low,Close,Volume
  2015-01-02,24.32,24.75,23.87,24.10,212818400
  ...
  ```

  A present file short-circuits *all* network access. `refresh=True` re-downloads.
- **Stooq** (syllabus primary) currently serves a JavaScript anti-bot wall. The code
  detects it and raises with the exact browser URL
  (`https://stooq.com/q/d/l/?s=aapl.us&d1=20150101&d2=20241231&i=d`) and the exact
  filename to save into the cache — after which everything runs offline. Deliberately
  not circumvented.
- **yfinance** (syllabus's sanctioned prototyping fallback): auto-adjusted bars,
  identical cache format, `pip install yfinance` if missing.
- VIX + factors for Stage C land in the same cache: `vix_history.csv`,
  `ff_factors_daily.zip`, `ff_momentum_daily.zip`, `sp500_wiki.html`.

## II.5 Bringing your own data

Three routes, by decreasing convenience:

**Route 1 — you have daily OHLCV CSVs.** Name them into the cache
(`{symbol}.us.{start}.{end}.csv`, header exactly `Date,Open,High,Low,Close,Volume`) and
call `build_stage_b_panel(...)` with matching dates — zero network, full feature
pipeline, all timing rules applied for you.

**Route 2 — you have price DataFrames.** DatetimeIndex, columns
`open/high/low/close/volume`:

```python
from nec_moe import build_panel, StageBSpec
panel = build_panel({"tick1": df1, "tick2": df2, ...},   # per-ticker OHLCV
                    market_px=spy_df,                    # the market symbol's OHLCV
                    spec=StageBSpec(seq_len=20, horizon=5))
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

### Saving and loading a trained model

There is deliberately no bespoke checkpoint format — plain torch + the config
round-trip:

```python
import torch, json
torch.save({"state_dict": trainer.model.state_dict(),
            "config": trainer.model.cfg.to_dict(),
            "step": trainer.step_count}, "runs/nec_soft_seed0.pt")

ckpt = torch.load("runs/nec_soft_seed0.pt", weights_only=False)
model = NECModel(NECConfig.from_dict(ckpt["config"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

(Any `Trainer` wrapped around a loaded model starts a fresh optimizer — fine for
evaluation and fine-tuning; exact optimizer-state resumption isn't built.)

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
)
```

The two easy mistakes: (1) `make_trainer` that reuses one model — each fold must be a
fresh optimization (fit-once-per-window; the function's contract), so construct inside
the lambda and reseed there for determinism; (2) `purge_dates ≠ horizon` — the purge is
the label horizon, nothing else.

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
from nec_moe import long_short_by_date
_, gross, tno = long_short_by_date(pred, test.y, test.date, test.entity, n_quantiles=5)
net = gross - 0.001 * 2.0 * tno
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

**A new feature**: add the column in `features.py::ticker_features` (trailing
information only), append the name to `SEQUENCE_FEATURES`/`SNAPSHOT_FEATURES` (order =
channel order), extend `test_no_lookahead`. The no-lookahead test is the gatekeeper —
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
| `Stooq served its JavaScript anti-bot challenge …` | source wall | browser-download to the printed cache name, or `source="yfinance"` |
| `only N dates have >= min_names valid names` | thin panel after validity filtering | widen dates/universe; lower `min_names_per_date` knowingly |
| `no date had a scoreable cross-section — constant per-date predictions …` | rank-IC on a classical emission | compare on NLL/regime recovery |
| `no overlap between panel dates and context index` | alignment join failed | real panels need `date_labels`; synthetic contexts index by integer codes |
| `sort_key length … != batch size` | warm-start key misaligned | compute the key from the same panel you pass |
| `not enough dates for the requested splits` | fold request exceeds the calendar | fewer/narrower folds, or less purge |
| `metric 'x' is not finite` | NaN/inf into the registry | fix upstream; the registry refuses to store lies |

## II.13 Practical notes

- **Timing** (this machine, CPU): full suite ~30 s; synthetic 400-step fit ~2 s;
  real-panel 600-step fit ~1–2 min; a 5-mechanism × 5-seed sweep with 4 folds ≈ 1–2 h —
  start it and walk away, the registry accumulates.
- **Memory**: the real panel is ~70k × (20×4 + 14) floats ≈ 25 MB; full-batch forwards
  are fine on CPU.
- **Reproducibility gotchas**: expert init consumes the global RNG → seed immediately
  before `NECModel(cfg)`; `minibatches` uses its own seeded generator (from
  `TrainConfig.seed`) → `fit` is reproducible given the seed; dropout makes train-mode
  forwards stochastic → tests and comparisons use `dropout=0` or `eval()`.
- **The cache is a lab notebook**: `data_cache/` + `results/*.jsonl` + `figs/` +
  committed code = a fully reconstructible experiment. Commit all four together.

---

# Part III — The Roadmap

## III.1 If the supervisor picks **Variation 2** (routing-mechanism comparison)

Everything mechanical exists — six priors, the harness, the registry. The remaining
work is *experimental design and execution*:

1. **Pre-registration document first** (½ day). Before any real-data run, commit: the
   exact grid (mechanisms × K ∈ {2,3,4} × ≥5 seeds), the metrics (pooled IC, ICIR, net
   IR, NLL), the claim family and its correction (BH at α=0.10 over per-mechanism
   p-values), the split parameters, and what a negative result would look like. The
   registry then makes deviations visible.
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
6. **The real-data grid on the PIT panel** (compute time) — II.4's point-in-time path,
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
2. ✅ **Full point-in-time panel** — `scripts/build_pit_panel.py` (resumable:
   cache-first, rerun to retry failures). Built 2015–2024: 640 candidates attempted,
   590 loaded, **1,089,688 PIT rows** across 2,390 dates; measured coverage 83.4%
   (2016) → 97.2% (2024), published at `results/pit_coverage_2015_2024.csv`; panel
   checkpoint at `data_cache/pit_panel_2015_2024.pt` (~412 MB, gitignored,
   re-buildable). The ~50 failures are the departed names (TWTR, YHOO, XLNX, WFM…) —
   survivorship component 2, measured not silent.
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
6. ◻ **Gate calibration** (M4 §4.9) — reliability diagrams of gate probabilities on
   held-out folds; temperature-scale if rank-good/overconfident. ~1 day; a
   thesis-quality diagnostic nobody else will have.
7. ◻ **Lint + type-check + CI** — `ruff`, `pyright`, a CI job running the offline
   suite once a remote exists. ~½ day.
8. ◻ **Pre-registration template** in the repo. ~½ day.
9. ✅ Purged walk-forward, cost-aware backtest, registry + DSR/BH, PIT membership +
   coverage, alignment diagnostics, defect regression tests.

Items 1–5 are done: seeds, the PIT panel, the residual target, tuning discipline, and
figures — the line to "these numbers can enter a thesis" is crossed on the
infrastructure side. What remains is polish (6: gate calibration, 7: lint/CI, 8: the
pre-registration template) and then the experiments themselves.

---

# Part IV — The Evaluation

## IV.1 As research code

Graded against the honest reference class — published quant-ML research code and
typical thesis repositories:

| dimension | grade | justification |
|---|---|---|
| Correctness assurance | **A** | 100+ tests: analytic gradients, reference filters, causality probes, planted-truth recoveries, defect regressions. Far above field norm. |
| Methodology | **A−** | Purged WF, fit-once-per-window, logged selection events, DSR/FDR, capacity-matched baselines, one grading path. Missing: multi-seed *practice*, tuning protocol. |
| Reproducibility | **B+** | Deterministic seeds, config round-trip, cache-first data, append-only registry. Missing: CI, pinned env, plots-as-code. |
| Data rigor | **C+** | Timing contract *tested*; PIT membership + measured coverage; but current panels are ≤~600 attempted names with no delisting returns, yfinance-grade prices. Honest about every limit — worth half a grade itself. |
| Architecture | **A−** | Two plug axes proven by tests; extension checklists are short because the seams are real. Debt: no plots module, conftest path hack, no device handling. |
| Documentation | **A−** | Design doc with decision traceability; docstrings with shapes *and reasons*; this handbook. |

**Overall: strong research code whose distinguishing feature is that the
claims machinery was built before the claims.** Most research code does the opposite.

## IV.2 As quant-desk code

A desk pipeline has roughly six layers; where this code stands in each:

| layer | desk requirement | this code | gap |
|---|---|---|---|
| **Data** | vendor PIT (CRSP/Compustat/Refinitiv), corporate actions, delisting returns, 3000+ names, ongoing ingestion + QA | free daily bars, ≤~600 names, PIT *membership* only, measured coverage | **the big one** — this layer is bought, not coded |
| **Signal research** | leakage-proof backtests, multiplicity control, registered experiments | purged WF, DSR/FDR, registry, causality tests | **small** — genuinely desk-grade *thinking*; a desk reviewer would recognize the discipline |
| **Portfolio construction** | optimizer + risk-model neutralization (Barra/Axioma), constraints, capacity | equal-weight quantile L/S | large — a research proxy, not a book |
| **Transaction costs** | spread + impact (√participation), borrow, realistic fills | linear cost on traded notional | medium — fine for ranking models, not for sizing capital |
| **Execution & ops** | scheduled retrains, drift monitoring, model versioning, kill criteria, audit | research trainer + JSONL registry | large — deliberately out of scope |
| **Scale** | GPU/distributed, big hyperparameter searches | single CPU, minutes | medium — models are tiny by design |

**Bottom line:** a signal-research prototype with production-grade methodology hygiene
and prototype-grade everything else — the correct shape for its purpose. A desk quant
would trust the honesty of its evaluation numbers (rare) and would not trade it as-is
(correct). Path to desk-usable, in order: (1) a vendor PIT data layer behind the
existing `Panel` contract — designed so only `market_data`/`features` change; (2) a
real cost model + constrained optimizer replacing quantile L/S; (3) ops — scheduled
refits, drift monitors (gate entropy/utilization are already the right drift signals),
model versioning atop the registry; (4) scale-out, least urgent.

## IV.3 The one-paragraph verdict

The machinery is finished for its declared purpose: both thesis variations are one
config string away, the comparison table has both of its sides, and every number that
could lie has a test or a correction standing over it. What separates the current state
from thesis-grade *results* is protocol, not code — seeds, tuning discipline, the full
point-in-time panel, pre-registration (Part III.3, items 1–4). What separates it from a
desk is a data vendor and a portfolio layer, not a rewrite — the contracts were drawn so
those bolt on. The risk to keep in view: every real-data number so far is single-seed on
a survivorship-heavy universe; treat them all as pipeline demonstrations until III.3 is
done.

---

*Handbook version 2 — 2026-07-03. When the code moves, move this document: it is
committed next to what it describes.*
