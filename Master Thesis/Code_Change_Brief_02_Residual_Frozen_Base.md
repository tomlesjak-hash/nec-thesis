# Code change brief 02: residual mixture with a frozen base

Date: 2026-09-21. Follows `Code_Change_Brief_2026-09-21.md` (brief 01) and
`Code_State_2026-09-21.md`. Repository `Quant Model/nec_baseline`, package `nec_moe`.

Scope: settled design only. **Every numeric quantity introduced here is a config field with a
documented default and no hardcoded constant anywhere in the package.** None of the values are
chosen yet; they will be set by experiment, so they must all be sweepable from
`run_experiment.py` and recordable in the `TrialRegistry`.

**Out of scope: the primary error function.** Which objective the model is trained against is an
open question (Q20 in `Advisor_Questions.md`) and is **not** decided here. Do not choose one, do not
argue for one, and do not change the objective that `losses.py` currently implements. Everything in
this brief must compose with whatever primary objective is selected later, which means the residual
machinery is written to be objective-agnostic and the one structural change asked for below (section
3.0) is the seam that makes a later choice a registration rather than a rewrite.

---

## 0. Read first

Do not write code until these are read. The architecture, the initialisation and the loss form all
come from the first item, and the second is what the design choice has to be defended against.

1. **Ye and Borde, arXiv:2608.12251**, in `Quant Model/MoE papers`. The residual mixture of experts
   this thesis builds on. Take from it: the residual form `y_hat = f0(x) + sum_k pi_k r_k(x)`, the
   zero-initialisation of the expert heads, the correction magnitude penalty, the two-hidden-layer
   block used for every network, and the ablation design (K in {2,4,6}, zero-init versus
   random-init versus no-base). Note what it does **not** report: the value of the correction
   penalty weight. That is why the weight is a config field selected by experiment here.
2. **DeepSeekMoE, arXiv:2401.06066**. Shared expert isolation: an always-on shared expert plus
   routed experts, optimised jointly. This is the configuration this thesis is deliberately *not*
   using, and the reason the base is frozen instead is attribution rather than performance. The
   docstring for the residual mode should state that contrast explicitly. See Q7 in
   `Advisor_Questions.md`.
3. **Chen et al., arXiv:2208.02813**, "Towards Understanding Mixture of Experts in Deep Learning".
   Establishes that a mixture beats a single network when the problem has cluster structure and the
   experts are non-linear. This is the premise the residual design is testing, with regimes as the
   cluster structure.
4. **Gu, Kelly and Xiu**, for the feed-forward network as the benchmark model class on a
   characteristic panel, and for the magnitude of the signal being chased (monthly out-of-sample
   R-squared of roughly 0.4 per cent). The base is expected to capture most of it; the residual the
   experts work on is small by construction.

Then re-read `nec_moe/model.py`, `nec_moe/experts.py`, `nec_moe/losses.py`, `nec_moe/train.py` and
`nec_moe/evaluation.py` before changing any of them.

---

## 1. The design being implemented

    y_hat = f0(x)  +  sum_k pi_k(t) * r_k(x)

- `f0` is a **base MLP** on the characteristic snapshot. Fitted on the training block of each
  walk-forward fold, then **frozen** (`requires_grad_(False)`, `eval()` mode so dropout and any
  normalisation statistics are fixed).
- `r_k` are **K expert MLPs** on the same characteristic snapshot, producing **corrections**, with
  the final layer **zero-initialised** so that at step 0 the correction is identically zero and
  `y_hat == f0(x)` exactly.
- `pi_k(t)` comes from the frozen gate (Q19, brief 01 section 3). Not trained here.

**Only the experts and their noise scales are trainable.** The base is frozen, the gate is frozen.
This is intentional: the quantity being measured is what a regime-conditional correction adds to a
fixed baseline, and freezing both ends is what makes that attribution clean.

---

## 2. Configuration surface

Everything below is a field, with the stated default, and nothing is hardcoded. All must be
reachable from the settings block in `run_experiment.py` and written into every `TrialRegistry`
entry.

### 2.1 New `BaseConfig`

    enabled: bool = False          # residual mode off by default, existing behaviour preserved
    hidden_dims: tuple[int, ...] = (64, 32)   # depth and width both sweepable; see note below
    dropout: float = 0.0
    activation: str = "relu"       # registry-selected, not a hardcoded nn.ReLU
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 1000
    batch_size: int = 128
    early_stopping_patience: int | None = None
    val_fraction: float = 0.2      # tail of the TRAINING block only
    seed_offset: int = 0           # base seed derived from the run seed plus this

`hidden_dims` as a tuple carries depth and width together, so the depth sweep of Q11 is a sweep over
tuple lengths and needs no separate depth field. Provide a helper that builds a pyramid tuple from a
first-layer width and a depth, since that is the rule Q8 settled, but keep the explicit tuple as the
authoritative field so any shape remains expressible.

### 2.2 Extensions to `ExpertConfig`

    correction_mode: bool = False      # experts emit corrections rather than full forecasts
    zero_init_head: bool = True        # final layer zeroed; only meaningful with correction_mode
    hidden_dims: tuple[int, ...] = (64, 32)   # replaces the scalar hidden_dim, same rationale

Keep backward compatibility for the existing scalar `hidden_dim` or migrate every call site in one
pass, but do not leave both live with different meanings.

**Naming.** `features.py` already uses "residual" for a market-neutralised *target*. Do not reuse
the word. `correction_mode` and `BaseConfig` keep the two concepts separate in config, in code and
in the write-up.

### 2.3 Extensions to `TrainConfig`

    aux_correction_penalty: bool = False
    correction_penalty_weight: float = 0.0    # the alpha of the design; value unknown, set by experiment

---

## 3. The objective seam, and the correction magnitude penalty

### 3.0 Make the primary objective a registry, and register only what exists today

The package currently trains against a mixture negative log likelihood with Gaussian emissions.
**Leave that behaviour exactly as it is.** The only change asked for is structural, matching the
convention already used for priors and emissions:

- add an `OBJECTIVE_REGISTRY` and a `TrainConfig.objective: str = "mixture_nll"` field;
- register the existing mixture NLL as `"mixture_nll"`, with identical numerics, verified by a test
  that the registry path and the current path produce bit-identical losses on the same input;
- register nothing else.

The objective is an open question. This seam means answering it later is a registration, not a
rewrite of the trainer, and it costs nothing now. Do not add MSE, Huber, rank or IC objectives, and
do not write a docstring that recommends one.

### 3.1 The correction magnitude penalty

Add to `losses.py`, following the existing pattern of `load_balance_aux` and
`expert_decorrelation_aux` (config gated, default off, tested):

    correction_penalty_aux(pi, r) -> mean over the batch of ( sum_k pi_k * r_k )^2

This penalises the magnitude of the *combined* correction, not each expert separately, which is the
form in Ye and Borde. It shrinks the mixture toward the base.

It is an **auxiliary term added to whichever primary objective is in force**, and it must be written
that way: a function of the prior weights and the corrections only, with no dependence on how the
primary error is computed. It composes; it does not presuppose.

`correction_penalty_weight` is load bearing: it directly controls how much correction the model is
allowed to make, and therefore how large the measured improvement can be. Ye and Borde do not report
their value. Consequently:

- select it on the **training block only**, never on out-of-sample data;
- log every candidate value as a trial, so the selection multiplicity reaches the deflated Sharpe;
- report a **sensitivity curve** of the headline result against it. A result that only exists at one
  value of an unreported knob is not a result.

Note that the weight is not comparable across primary objectives, since it trades against whatever
the primary error's scale happens to be. Record the objective alongside the weight in every trial so
the two are never compared across a change of objective.

## 4. Training flow changes

`train.py` and the walk-forward harness in `evaluation.py`.

**Per fold, in order:**

1. Fit the gate on the training block and freeze it (brief 01 section 3).
2. Fit `f0` on the training block, using its own optimiser and its own config, with validation on a
   tail slice of the **training** block only. Then freeze it.
3. Train the experts against the residual, with the frozen base and frozen gate in the graph.
4. Score the fold out of sample.

**Amortisation, which matters for the compute budget.** The base never sees regime information, so
it depends only on `(base config, window, seed)` and **not on the gate**. Cache it and reuse the
same fitted base across all four gate arms within a window and seed. Q11 costed the programme on
this assumption. Implement the cache with an explicit key covering every field of `BaseConfig` plus
the window identifier and the seed, so a config change can never silently reuse a stale base.

**Assertions to add, because they are cheap and the failure is silent:**

- at step 0 with `correction_mode=True` and `zero_init_head=True`, `y_hat` equals `f0(x)` to
  floating point tolerance;
- after freezing, no parameter of the base or the gate appears in the optimiser's parameter groups;
- after the first backward pass, no base or gate parameter has a non-zero gradient. Reuse the dead
  parameter audit already added for brief 01 section 2, and invert the expectation for these.

**Sigma, conditional on the objective.** *While* the likelihood objective is the one in force, the
per-expert `log_sigma` describes the noise scale of the **residual** rather than of the raw target,
so `sigma_init` should default to the standard deviation of the training-block residual rather than
of `y`. Make this the documented default rather than a hidden adjustment, and scope it to the
likelihood objective rather than writing it as though it were universal: an objective with no noise
parameter would not have this field at all. This is a consequence of the residual design, not a
vote for the likelihood objective.

---

## 5. Reporting requirements

These are not optional extras; without them the headline number cannot be interpreted.

- **The base's own out-of-sample performance**, on every metric the mixture is scored on, reported
  beside every result. The mixture's number is meaningless without the floor it is measured from.
- **The improvement over the base**, per fold and pooled, as the primary quantity.
- **Live parameter count** per arm (brief 01 section 2), now including the note that base and gate
  parameters are frozen and therefore excluded.
- **The correction magnitude actually used**: mean and distribution of `sum_k pi_k r_k` out of
  sample. A model whose corrections are numerically negligible has answered the question in the
  negative regardless of what the R-squared does.

---

## 6. Ablations to make expressible by config, not by code change

All of these already fall out of the fields above, which is the point of putting them there:

- `base.enabled=False` — standard mixture, no base. The Q6 comparison.
- `zero_init_head=False` — random initialisation of the correction heads. Ye and Borde's ablation.
- `correction_mode=False` with a base present — experts emit full forecasts alongside a base.
- `correction_penalty_weight=0` — no shrinkage toward the base.
- `prior.kind` set to a trainable prior with `base.enabled=True` — the jointly trained baseline arm.

Each must be reachable by editing the settings block alone.

---

## 7. Tests

One test per property claimed:

1. At initialisation with `correction_mode` and `zero_init_head`, predictions equal the base's
   predictions exactly.
2. After freezing, base and gate parameters are unchanged by a training step (compare tensors
   before and after, do not merely check `requires_grad`).
3. The base cache returns the same object for two gate arms sharing a window and seed, and a
   different object when any `BaseConfig` field changes.
4. `correction_penalty_aux` is zero when all corrections are zero, and increases with correction
   magnitude.
5. On synthetic data with a known regime-conditional component added on top of a known
   unconditional component, the residual mixture recovers the regime-conditional part and the base
   recovers the unconditional part.

Test 5 is the important one: it is the only check that the whole arrangement can detect the thing it
is built to detect.
