# Corrected NEC Baseline — Design Plan

**Build status (2026-07-03):** IMPLEMENTED and passing. Package at `nec_baseline/nec_moe/`,
54-test suite green (~16 s CPU, deterministic). Sign-off went with every recommendation:
snapshot-only experts (A), unsupervised gate (B), end-to-end differentiable forward filter
(C), σ-sort identity handling (D), K=2, aux losses off by default. One thing surfaced in
implementation that the plan had anticipated: the symmetric sign-flip task needs the
expert warm-start device (§4.3) to escape the pooled local optimum — now implemented as
`Trainer.warmstart_experts`, gate still unsupervised. The HMM forward-filter predict step
ships with causality/correctness tests as promised (§12.3).

**Increment 7 (2026-07-03) — single-model baselines (the comparison's other side).**
`baselines.py`: `RidgeBaseline` (closed form on `x_snap`, intercept unpenalized per
Module 4; residual-std Gaussian NLL) and `MLPBaseline` (deliberately the *same*
`ExpertMLP` class as one NEC expert + a learnable scalar σ, trained on the same Gaussian
NLL — held-out NLLs directly comparable, capacity matched to one expert, so "MoE vs
single model" isolates *routing structure*). `walk_forward_evaluate_baseline` runs them
through the identical folds/purging/scoring via a shared `_FoldAccumulator` (both model
families are graded by the same code path; `FoldResult.expert_order` defaults to empty
for baselines). **The headline comparison is pinned as tests in both directions:** on
sign-flip regime data the pooled cross-sectional relationship averages to ~zero, so
single models are *structurally* blind (NEC pooled IC > 0.3 with warm-start vs |IC| <
0.15 for ridge and MLP); on regime-free data ridge scores IC > 0.8 — the baselines lose
where regimes are real for structural reasons, not because they are broken. LightGBM
(candidate #2) deliberately not wired (heavy optional dep; separate LGBM pipeline exists
in the repo) — add behind an import guard if the thesis wants the row. This completes
proposal milestone 2's single-model side (the classical side was increment 3's
Hamilton). Suite: 101 offline tests + 2 opt-in network, ~27 s.

**Increment 6 (2026-07-03) — Module 5's honesty machinery: trial registry +
selection-aware inference.** `registry.py`: append-only JSONL `TrialRegistry` — one row
per fitted/evaluated configuration (tag = experiment family, config, seed, metrics);
`best()` does not silently pick, it **records the selection event** with the candidate
count (`<tag>#selection` rows, excluded from the trial family itself); `n_trials(tag)`
is the multiplicity that feeds the corrections. `multiple_testing.py`: the deflated
Sharpe (Bailey & LdP 2014 — PSR with skew/kurtosis non-normality correction, expected-max
hurdle `sqrt(V[SR])·((1−γ)z(1−1/N)+γz(1−1/Ne))`, N and V[SR] sourced from the registry),
`ic_pvalue` (from `IcSummary.t_stat`), Bonferroni (FWER) and Benjamini–Hochberg (FDR —
the factor-zoo screening criterion; BH ⊇ Bonferroni property-tested). **Factor-zoo
validation pinned as a test:** the best of 60 pure-noise strategies earns naive
PSR > 0.95 and DSR < 0.75 (the deflator kills the selection artifact), while genuine
skill of the same apparent size survives with a clear margin. This closes the last
§13 methodology deferral and unblocks the thesis experiment grid: sweeps can now log
every trial and correct their claim families as Module 5's checkpoint demands
("explicitly state the family of tests over which your significance claim is
corrected"). Suite: 95 offline tests + 2 opt-in network tests, ~26 s.

**Increment 5 (2026-07-03) — STAGE C, context data + gate–regime alignment.**
`context_data.py`: VIX history (CBOE's official CSV) and Kenneth French daily factors
(zip parsing incl. the momentum file's trailing-comma quirk; percent→decimal; −99.99
missing markers → NaN), cache-first, parsers pure/offline-testable, live-verified.
`alignment.py`: the secondary-research-question tooling — `gate_utilization_by_date`
(memoryless: mean predictive π per date; HMM: mean *filtered* posterior — the Hamilton
filtered-regime-probability series) and `regime_alignment` (per-expert Pearson corr with
VIX/factors/realized vol + high/low-VIX tercile utilization). Context data is
**diagnostics only, never a training signal** (Decision B) — that is what keeps the
interpretability hypothesis testable. Validated by a planted-truth test (fake VIX built
from the true synthetic regime → report must recover the alignment, both prior families).
**First real run** (30-name panel 2015–2024, unsupervised soft gate): high-σ expert's
utilization corr **+0.81 with VIX**, +0.85 with realized market vol, ≈0 with directional
factors (MKT/SMB/HML/MOM); utilization 6.5%→21.6% from calm to stressed VIX terciles — a
volatility-regime split discovered from the mixture likelihood alone. Caveats: single
seed, survivorship-biased universe, alignment over all dates (thesis version: held-out
folds only). Suite: 81 offline tests + 3 opt-in network tests. Still deferred from the
data plan: FRED/NBER recession context, SEC fundamentals (Stage D), Module 12 universe.

**Increment 4 (2026-07-03) — STAGE B, the real data pipeline** (user-authorized; the
original "don't build yet" staging lifted). `market_data.py` + `features.py`:
cache-first daily OHLCV (the Stooq-format cache CSV is the source-agnostic interface;
downloads never touch the feature layer), the syllabus's named feature set (returns
1/5/20/60d, momentum 120d, vol 5/20/60d, downside vol, drawdown, dollar volume, volume z,
market-relative vs SPY), forward-log-return target as the only forward-looking column,
per-date cross-sectional rank normalization, and panel assembly into a general `Panel`
(refactor: `Panel` base + `SyntheticPanel` subclass; `PanelData` aliased) with
date/entity labels — the harness works unchanged on real data. **Anti-leakage is a
test**: `test_no_lookahead` truncates the input at t and asserts every non-target
feature at t is bit-identical. **Source reality:** Stooq now gates its CSV endpoint
behind a JS anti-bot challenge — deliberately NOT circumvented; the error explains the
manual browser-download-into-cache workflow, and `source="yfinance"` (the syllabus's
sanctioned fallback) is wired to the same cache format and verified live. Live run:
30-name default universe (SURVIVORSHIP-BIASED, pipeline-grade only) 2015–2024 → 71,700
rows / 2,390 dates; 3-fold purged walk-forward end-to-end on real data gives pooled
IC ≈ −0.02 (t ≈ −1.0) — statistically zero, as an honest first shot on a 30-name toy
universe should be; the deliverable is the verified chain, not alpha. Suite: 73 offline
tests + 1 opt-in network test (`NEC_NETWORK_TESTS=1`).

**Increment 2 (same date):** the §13 deferrals partially closed. (i) The memoryless
routing variants now ship: `uniform` (frozen-uniform-gate ablation), `hard` (top-1 with
the hard-EM/Switch-style objective via `PriorOutput.log_train_weights` — its *predictive*
weights stay one-hot, its *training* weights are `log π_sel`, giving the gate the
`π − onehot` self-training gradient the prototype's argmax lacked), `topk` (log-domain
masked renormalization; **k ≥ 2 enforced** — a single renormalized entry is constant 1
and gradient-dead, i.e. ledger defect 1 reborn, so top-1 users are redirected to `hard`),
and `gumbel` (annealed temperature off `ctx.step`; eval-deterministic). (ii) The
walk-forward evaluation harness ships (`evaluation.py`): expanding-window purged splits
(label-overlap purging, AFML §7.4), per-date Spearman rank-IC + ICIR/t-stat, and
fit-once-per-window `walk_forward_evaluate` serving both variations (HMM folds are scored
by a causal filtering pass warmed through the past training window) with per-fold
canonical expert order recorded (Decision D). The routing comparison is now literally a
config sweep; a demo run behaves as theory predicts (uniform → IC≈0 since routing *is*
the value on the sign-flip panel; hard shows the known top-1 cold-start pathology;
HMM is weak on iid regimes and wins on persistent ones per §12.3's test).

**Increment 3 (same date):** (i) The **emission axis is config-wired**
(`ExpertConfig.kind ∈ {mlp, classical}`, `EMISSION_REGISTRY` mirroring the prior
registry) — all four corners of the (emission × prior) grid build from config, and the
classical × Markov corner **is the Hamilton baseline**, validated by a
parameter-recovery test: a Gaussian HMM fit by autograd through the log-domain forward
filter recovers true `(μ_k, σ_k, A)` to within ~0.07 and tracks the regime path at AUC
0.99 — the strongest available confirmation of Decision C. Each emission owns its
symmetry-breaking warm-start (`Emission.warmstart_slices`: MSE pre-training for neural,
closed-form slice moments for classical); optimizer no-decay now uses the standard
1-D-parameter rule so classical `μ_k` is never decayed toward zero. (ii) **Portfolio
metrics** join the harness: per-date quantile long-short returns, entity-aligned
one-sided turnover, cost drag (`net = gross − cost_rate·traded_notional`), per-fold and
pooled summaries via `backtest_quantiles`/`cost_rate`; `PanelData` gained a stable
`entity` id column for cross-date weight alignment. Honest note recorded in the code:
a classical emission predicts one value per date (no cross-sectional ranking), so the
Hamilton baseline is compared on held-out NLL/regime recovery, not rank-IC — the error
message says so. Suite after increment 3: **64 tests, ~21 s**.

**Status (design):** decisions flagged in §8, §9, §9bis.
**Scope:** research *baseline*, not the thesis's final model. It is architected to serve
**either** of two candidate thesis variations without a rewrite — whichever the supervisor
approves:
- **Variation 2 (routing-mechanism comparison):** soft vs. hard vs. top-k vs.
  Gumbel-softmax routing, all memoryless (prior is a function of the current encoder
  output only).
- **Variation 3 (HMM-gated NEC):** the gate's prior over "which expert is active" becomes
  recursive — a learned `K×K` transition matrix applied through forward filtering, giving
  Hamilton-style Markov persistence with flexible neural experts.

**Unifying principle (the load-bearing idea of this revision).** Both variations need the
*same* per-expert predictive likelihood at each timestep — `log N(y_t | μ_k(x_t), σ_k²)`,
already required by the mixture-NLL objective. They differ **only** in how the prior over
the active expert is formed *before* combining with that likelihood via Bayes' rule
(§7). So the gate is decomposed into a **shared expert-likelihood core** plus a
**pluggable prior/dynamics component**: memoryless-parametric (Variation 2) or
Markov-transition (Variation 3). The HMM gate is a strict generalization — add a
transition matrix and a recursive filter — not a separate incompatible path.

**Authoritative inputs:** the Module 7 capstone defect ledger (8 defects + keep/change/decide
box), Module 8's mixture-NLL / generalized-EM capstone (and Module 9's HMM/forward-algorithm
material for Variation 3), Module 4's gate math (fused softmax cross-entropy, calibration,
separation), and the thesis syllabus.

---

## 1. What is kept, what is changed

**Kept** (the ledger's "keep"): the architectural thesis — a sequence encoder informing a
regime gate over specialized expert regressors. Sound MoE skeleton, badly executed in
`OP model/nec/nec_hybrid_explained.py`.

**Changed**, each traceable to a ledger defect:

| # | Defect in prototype | Fix in baseline |
|---|---|---|
| 1 | Hard `argmax` routing — zero gradient to gate | Soft routing trained by the mixture NLL (§4) |
| 2 | 3 classes collapsed to one bit (`== 1`) | All `K` responsibilities enter the loss; `K` is config, no class has hardcoded meaning |
| 3 | `nn.Softmax` inside the model, feeding a second loss | Gate emits **logits**; log-domain fused loss (`log_softmax` + `logsumexp`), no in-model softmax |
| 4 | `BatchNorm1d` in the head (cross-sample coupling / leakage) | `LayerNorm` in the gate head |
| 5 | 8-layer LSTM, n=256 (~4M params) vs. effective-sample arithmetic | 1-layer GRU (2 max), hidden n=32 default, n ∈ [16, 64] |
| 6 | Experts read raw last-step slice; encoder value flowed only through untrainable routing | **Decision A** (§8): snapshot-only default, config-switchable to snapshot ⊕ h_T |
| 7 | Silent `x[:, -1, :-1]` feature slice, undocumented reserved column | Named feature schema; model never slices feature tensors; target never lives inside the feature tensor (§6) |
| 8 | Encoder default n=256 vs. head default n=64 — silent mismatch | Single config object with cross-validation at construction + shape assertions at every module boundary (§5) |

**Decided and documented** rather than silently picked: expert inputs (§8) and gate
supervision (§9).

---

## 2. Architecture

```
x_seq (B,T,d_seq) ─► GRUEncoder ─► h_T (B,n) ─► GateHead: LayerNorm→Linear ─► gate logits z (B,K)
                                      │                                              │
                                      │                      ┌───────────────────────┘
                                      │                      ▼
                                      │            RegimePrior (pluggable, §7)  ◄── ctx: prev filtered
                                      │            memoryless:  log π = log_softmax(z)      posterior r_{t-1}
                                      │            HMM:          log π_t = logΣ_j A_jk r_{t-1,j}   (Var. 3)
                                      │                      │
                                      │            log prior over experts  log π (B,K)
                                      ▼                      │
x_snap (B,d_snap) ─► ExpertBank ─► μ (B,K), log σ (K,)      │
   (Decision A: ± h_T)               │                      │
                                     ▼                      ▼
              log-likelihood core  log N_k = log N(y|μ_k,σ_k²)   ┐
                                     │                            ├─► combine (Bayes, §4):
                                     └────────────┬───────────────┘   a_k = log π_k + log N_k
                                                  ▼
                        ┌─────────────────────────┼──────────────────────────┐
                        ▼                         ▼                          ▼
          prediction ŷ = Σ_k π_k μ_k    NLL = −LSE_k(a_k)      responsibilities / filtered
                                        (training, §4)          posterior r_k = softmax_k(a_k)
                                                                         │
                                                          (HMM: r_t threads to next step's ctx;
                                                           memoryless: discarded after loss)
```

The dashed anatomy: everything from `log N_k` rightward — the log-likelihood core, the
Bayes combine, the NLL, and the responsibilities — is **shared across both variations**.
Only the `RegimePrior` box changes. In the memoryless case the prior ignores
`prev filtered posterior`; in the HMM case it *is* the recursion over it. This is the
whole point of the abstraction.

Key structural points:

- **Encoder** — `nn.GRU(d_seq, n, num_layers=L, batch_first=True)`, defaults `L=1`,
  `n=32`. Deliverable is the final hidden state of the last layer, `h_T = h_n[-1]`.
  GRU over LSTM per the Module 7 parameter-count argument (3n(n+d+1) vs 4n(n+d+1);
  25% fewer parameters is pure variance reduction at this data scale). Inter-layer
  dropout only meaningful for `L ≥ 2` (config validates this). `bidirectional=False`
  always — the encoder-only usage makes a backward pass a leakage question we don't open.
  No manual `h0` allocation (PyTorch zero-initializes; removes the device/shape fuss of
  the prototype).
- **Gate head** — `LayerNorm(n) → Linear(n, K)`, **output is logits**, no activation.
  The `Linear` is zero-initialized (weights and bias) so the gate starts exactly uniform:
  entropy `log K` at step 0, no expert starts dead (Module 8 capstone: a confident random
  gate at t=0 is a collapsed mixture before training starts).
- **RegimePrior (the pluggable gate component)** — maps gate logits (and, for stateful
  priors, the previous filtered posterior) to a **log prior over experts** on the simplex.
  Baseline ships `SoftRegimePrior` (= `log_softmax(z)`, memoryless); the interface (§7) is
  where hard/top-k/Gumbel (memoryless family) and the `HMMRegimePrior` (recursive, Var. 3)
  plug in. Named `RegimePrior` rather than `Router` deliberately: its job is to form the
  *prior* `log π`, which the shared likelihood core then combines via Bayes — the framing
  that makes the HMM gate a generalization rather than a fork. (`Router` retained as an
  alias in the routing module for MoE-literature familiarity.)
- **Expert-likelihood core (shared, both variations)** — given expert means `μ_k`, scales
  `σ_k`, and target `y`, computes the per-expert log-density `log N_k`. This is the single
  component every prior plugs into; it is what makes "same likelihood, different prior"
  literally true in code.
- **ExpertBank** — `K` structurally identical small MLP regressors
  (`d_exp → h → h/2 → 1`, ReLU, dropout; default `h=64`), held in a `ModuleList`
  (K ≤ 8, no need to vectorize with vmap for a baseline). Each expert `k` also owns a
  **scalar learnable** `log σ_k` (a `(K,)` parameter on the bank) — homoscedastic per
  expert, *not* input-dependent, which is the minimal noise model the mixture NLL needs.
  Experts are indexed, never named: no "extreme"/"normal" in code — regime semantics are
  assigned post hoc by diagnostics, because label switching makes fixed names meaningless
  across runs/folds (Module 8 §failure mode 2).
- **Expert inputs** — `x_snap` arrives as its **own tensor** from the data interface;
  the model never derives it by slicing `x_seq` (kills defect 7's failure mode at the
  interface level). Whether `h_T` is concatenated on is Decision A (§8).
- **K** — config `n_experts`, **default 2**. Justification: it is the smallest K that
  expresses the regime hypothesis; the prototype was effectively binary anyway (its 3-way
  posterior collapsed to one bit); the effective-sample arithmetic of Module 7 §overfit
  favors minimal capacity; and the syllabus's proposal milestone specifies "soft-routed
  MoE with K = 2 or K = 3", with K ∈ {2, 3, 4, 8} as a later thesis ablation. Nothing in
  the code may assume K=2 or K=3 — every loop, shape, and test is parameterized by `n_experts`.

### Shape table (ledger format), defaults `d_seq`, `d_snap`, `T`, `n=32`, `K=2`

| Stage | Module | Input shape | Output shape | Mathematics |
|---|---|---|---|---|
| Encoder | `GRUEncoder` | `x_seq (B, T, d_seq)` | `h_T (B, n)` | GRU eq. (7.4.1), last layer final state |
| Gate norm | `LayerNorm(n)` | `(B, n)` | `(B, n)` | per-sample normalization — no cross-batch coupling |
| Gate scores | `Linear(n, K)` | `(B, n)` | `z (B, K)` | softmax regression head in logits (4.8) |
| Prior (memoryless) | `SoftRegimePrior` | `z (B, K)` | `log π (B, K)` | `log_softmax` — log prior over experts (8.6) |
| Prior (HMM, Var. 3) | `HMMRegimePrior` | `z`, `r_{t-1} (B, K)` | `log π_t (B, K)` | `logΣ_j A_jk r_{t-1,j}`, forward filter (9.x) |
| Experts | `ExpertBank` | `x_snap (B, d_exp)` | `μ (B, K)`, `log σ (K,)` | K ReLU funnels + per-expert noise scale |
| Likelihood core | `ExpertLikelihood` | `μ, log σ, y` | `log N (B, K)` | `log N(y|μ_k,σ_k²)` — shared, both variations |
| Prediction | `NECModel` | above | `ŷ (B,)` | `Σ_k π_k μ_k` = mixture mean `E[y|x]` |
| Loss | `MixtureNLL` | `log π, log N` | scalar | `−LSE_k(log π_k + log N_k)` (§4) |
| Responsibilities / filtered | `MixtureNLL` | `log π, log N` | `r (B, K)` | `softmax_k(a_k)`, Bayes posterior (8.3.1); HMM: threads to next step |

`d_exp = d_snap` (Decision A default) or `d_snap + n` (concat variant).
Every module asserts its input dims against config in `forward` and raises with an
informative message; the config cross-validates derived dims at construction (§5), so a
defect-8-style mismatch dies at build time, not silently at runtime.

---

## 3. What the model computes (probabilistic statement)

The model is the conditional mixture of Module 8 §8.6:

$$p(y \mid x) \;=\; \sum_{k=1}^{K} \pi_k(x)\, \mathcal{N}\!\big(y \mid \mu_k(x),\, \sigma_k^2\big),
\qquad \pi(x) = \mathrm{softmax}\big(z(x)\big),$$

with `z(x)` the gate logits read from the encoder state and `μ_k(x)` the expert means.
The GMM↔MoE dictionary applies verbatim: constant `π_k` → feature-conditioned gate;
Gaussian components over `x` → Gaussian conditionals over `y | x`. Point prediction for
rank-IC evaluation is the mixture mean `ŷ = Σ_k π_k(x) μ_k(x)`; the full predictive
distribution (means, sigmas, weights) is exposed for later calibration/uncertainty work.

**The generalization (Variation 3), stated as one substitution.** Everything above holds
if `π` is any prior on the simplex. The two variations are exactly two choices of prior:

$$\text{memoryless (Var. 2):}\quad \pi_{t,k} = \mathrm{softmax}_k\big(z(x_t)\big);
\qquad
\text{HMM (Var. 3):}\quad \pi_{t,k} = \underbrace{\textstyle\sum_j A_{jk}\, r_{t-1,j}}_{\text{predict step}},$$

where `A` is the learned `K×K` transition matrix (rows = "from" state) and
`r_{t-1,j} = p(z_{t-1}=j \mid \text{data}_{1:t-1})` is the previous step's **filtered**
posterior. The per-timestep predictive density `p(y_t | data_{1:t-1}) = Σ_k π_{t,k} N_{t,k}`
and the Bayes update `r_{t,k} ∝ π_{t,k} N_{t,k}` (the filter's **update step**) are
identical in form to the memoryless mixture — the forward algorithm is literally the
mixture NLL summed over time with the prior fed recursively. Summing
`−log p(y_t | data_{1:t-1})` over `t` is the HMM's negative log-likelihood; backprop
through this recursion trains `A`, the experts, and the encoder jointly (Decision C, §9bis).

**Correctness requirement — filtering, not smoothing (not a footnote).** The HMM prior
uses the **filtered** posterior `p(z_t | data_{1:t})`, which conditions only on data up to
`t`. It must **never** use the smoothed posterior `p(z_t | data_{1:T})`, which conditions
on the whole sequence including the future. Smoothing here is lookahead leakage — the same
class of bug the defect ledger and Module 5's purging discipline exist to prevent — and it
would silently inflate every backtest. Concretely: the forward recursion is legal; the
backward/forward–backward pass (and Viterbi over the full sequence, and Baum-Welch's
E-step smoothed responsibilities) are **prediction-time-illegal** and may appear only in
offline diagnostics that are never fed back into a prediction for a date inside the window.
This is a hard invariant with a dedicated causality test (§12).

---

## 4. Training objective and loss math

### 4.1 The loss (log domain, fused — fixes defects 1–3)

Per sample, with gate logits `z ∈ R^K`:

$$a_k \;=\; \underbrace{z_k - \mathrm{LSE}_j(z_j)}_{\log \pi_k \;(\texttt{log\_softmax})}
\;+\; \underbrace{\Big[-\tfrac{1}{2}\log 2\pi - \log\sigma_k - \frac{(y-\mu_k)^2}{2\sigma_k^2}\Big]}_{\log \mathcal{N}(y \mid \mu_k, \sigma_k^2)}$$

$$\boxed{\;\mathcal{L} \;=\; -\,\mathrm{LSE}_k(a_k)\;}
\qquad\text{batch loss} = \tfrac{1}{B}\textstyle\sum_n \mathcal{L}_n .$$

Numerics rules (Module 8, Ex. 8.1's float64 cliff; Module 4 §4.10's fused-op lesson):

- **Never** evaluate a density out of the log domain. The naive pipeline
  (exp densities → multiply by weights → sum → log) underflows whenever every component's
  squared standardized residual exceeds ~1490 — routine for fat-tailed returns.
- Exactly one `torch.logsumexp` over `k`; gate log-weights via `F.log_softmax`. There is
  **no** `nn.Softmax` anywhere in the model and no separate classification loss.
- Responsibilities `r_k = softmax_k(a_k)` are computed **detached**, for diagnostics and
  tests only — they are never a second training signal.
- Gauge: adding a constant to all gate logits changes nothing (LSE cancellation); weight
  decay on the gate selects the centered gauge.

### 4.2 Gradients (what each part learns, and why this fixes defects 1–2)

With responsibilities `r_k = softmax_k(a_k)` (Bayes posterior over experts):

| Parameter | Per-sample gradient | Reading |
|---|---|---|
| gate logits | `∂L/∂z_k = π_k − r_k` | cross-entropy toward soft labels the experts negotiate (4.10.4 / Ex. 8.5(b)); replaces the prototype's exactly-zero argmax gradient |
| expert mean | `∂L/∂μ_k = r_k (μ_k − y)/σ_k²` | responsibility-weighted regression — weighted least squares iterated (Bishop 3.3 / Ex. 8.5(c)) |
| expert noise | `∂L/∂log σ_k = r_k [1 − (y−μ_k)²/σ_k²]` | stationarity: σ_k² → responsibility-weighted MSE |

SGD/AdamW on this NLL **is** generalized EM (Module 8 §8.6, Ex. 8.5(d)): each gradient
step implicitly recomputes responsibilities (E-step) and nudges gate and experts toward
their weighted M-step targets. EM's vocabulary — responsibilities, effective counts
`N_k = Σ_n r_nk`, collapse, label switching — is the diagnostic language (§4.4).

**HMM prior (Var. 3): the same three gradients, plus one for `A`.** The expert-mean and
expert-noise gradients are *unchanged* (they depend on `r_k` and the likelihood, both
shared). The gate-logit gradient `π − r` becomes a gradient that flows through the forward
recursion into the transition-matrix logits and (via `r_{t-1}`) back through earlier
timesteps — this is BPTT (Module 7) over the regime posterior. We do **not** hand-derive
it: autograd through the log-domain forward recursion computes it exactly, and this is
precisely the recursive score computation of arXiv:2205.01565 done automatically (that
paper's manual prediction/update-step score recursion = what `logsumexp` + autograd give
for free; its per-step re-scaling = our LSE stabilization). This is the technical reason
Decision C (§9bis) leans toward differentiable-forward over Baum-Welch.

### 4.3 Optimization, regularization, initialization

- **Optimizer:** AdamW. Parameter groups so the gate's weight decay is set independently —
  gate weight decay is *structural*, not cosmetic: near-separating vol features drive an
  unpenalized softmax head to infinity (Module 4 §separation); decay makes the optimum
  exist at all.
- **Initialization** (Module 8 capstone, failure-mode 3 operationalized):
  gate `Linear` zero-init → exactly uniform mixing, entropy `log K`;
  experts diversified by per-expert seeds (symmetry breaking);
  `log σ_k` initialized from config (`sigma_init`, default 1.0 — set to target std when
  known) and **frozen for the first `sigma_freeze_steps` steps** (heteroskedastic-head
  caution: letting σ move first lets the model explain everything as noise).
- **Expert warm-start on regime-sorted slices** (implemented as
  `Trainer.warmstart_experts`). Seed diversity alone does **not** escape the symmetric-
  mixture local optimum on a genuinely symmetric task: with sign-flipped per-regime
  coefficients the pooled optimum is both experts ≈ the marginal regression (μ≈0, σ
  absorbing the bimodal variance), and SGD from a uniform zero-init gate stays there
  (empirically confirmed during implementation — both experts converge to identical σ,
  responsibilities never separate). The prescribed stronger device: sort samples by an
  observable regime proxy (realized/window vol, or VIX on real data), partition into `K`
  quantile slices, and briefly pre-train expert `k` as a plain MSE regressor on slice `k`.
  This breaks symmetry *deliberately and interpretably* and — crucially — leaves the gate
  **unsupervised** (only expert means are touched; the gate must still learn to route via
  `π − r`), so Decision B is intact. Verified: after warm-start, responsibility→regime
  AUC ≈ 0.99 and joint training then reaches gate→regime AUC ≈ 1.0; without it the gate
  recovers nothing (AUC ≈ 0.5). Used by the smoke test and both HMM recovery tests.
- **Transition matrix (HMM prior only).** Parameterized as a **row-wise softmax over an
  unconstrained `K×K` logit matrix** `L`: `A[i,:] = softmax(L[i,:])`, so `A` is a valid
  stochastic matrix under unconstrained optimization automatically (no simplex projection,
  no constrained optimizer) — the log-domain analogue used directly in the forward filter
  as `logΣ_j exp(logA_jk + log r_{t-1,j})`. **Persistence-biased init:** diagonal logits
  set above off-diagonal (a config `transition_diag_bias`, default e.g. +2.0) so training
  *starts* sticky — financial regimes are assumed persistent, and this is the transition-
  matrix analogue of the LSTM forget-gate `+1/+2` bias trick (Module 7). Initial regime
  distribution `π_0`: a learned `K`-logit vector (or the stationary distribution of `A`);
  minor, config-flagged. Caution carried from arXiv:2605.14976: even *static* transition
  parameters need enough data to identify at `K=3` (their `K=3` filtered-probability MSE
  is an order of magnitude worse than `K=2`) — reinforcing the `K=2` default (§2) for the
  HMM variant especially.
- **Optional auxiliary losses**, config-gated, **default off** for the baseline
  (objective purity; "with vs. without load balancing" is a pre-registered thesis
  ablation). The trainer composes `total = NLL + Σ enabled aux terms`, so adding terms
  later (including a supervised gate auxiliary, §9) is additive, not surgical. Two aux
  terms ship implemented-but-off:

  1. **Load balancing** — Shazeer form `L_bal = α · K · Σ_k f̄_k · P̄_k`, where `P̄_k` is
     the mean gate probability (carries the gradient) and `f̄_k` the fraction of samples
     whose argmax responsibility is `k` (**stop-grad** scaling factor).
     **Scope pitfall (arXiv:2501.11873, adopted):** `f̄_k` must be estimated over a
     **running buffer of many recent batches**, never within a single batch. Our
     natural batching for cross-sectional rank-IC training is date-sliced, and regime is
     largely a *date-level* variable — so per-batch balancing would demand that stocks
     from one calm day be spread uniformly over the calm *and* stress experts, which is
     exactly the specialization-killing failure mode that paper demonstrates (their
     micro-batch ≈ our single-date batch, at maximal severity). Implementation: an
     accumulation buffer of argmax-assignment counts over the last `bal_buffer_batches`
     batches (config, default sized to span ≥ several weeks of dates); the penalty pushes
     the *marginal* utilization toward uniform while leaving any single date free to
     route sharply. Corollary from the same paper: router preferences fix early in
     training, so with/without-LBL ablations must be from-scratch runs, never a
     mid-training toggle; and a wrong scope cannot be rescued by shrinking the weight `α`.
  2. **Expert output decorrelation** — RAVEN-style diversity penalty
     (arXiv:2606.24062): `L_div = ‖R − I_K‖_F²` with `R_jk` the batch correlation
     (cosine of centered outputs) between expert means `μ_j, μ_k`. Targets the failure
     mode load balancing does *not*: **expert homogenization** — all experts converge to
     the pooled regression, responsibilities stay uniform, the gate has nothing to learn
     (Module 8's "permanent uniformity = dead gate" symptom, reached via redundancy
     rather than imbalance). Load balancing fights collapse-onto-one-expert;
     decorrelation fights everyone-learns-the-same-thing. Caveat kept in the docstring:
     it biases the likelihood objective (regimes may genuinely agree on part of the
     cross-section), which is why it is an ablation lever, not part of the baseline
     objective.

### 4.4 Monitoring (failure modes as dashboards — Module 8 capstone)

Logged every eval interval; all cheap, all derived from detached responsibilities:

- train/val NLL (monotone only under full EM — don't panic at SGD noise);
- gate entropy per batch, `mean H(π(x))` (uniform forever = dead gate; instant sharpness = collapse);
- expert utilization at **two scopes** (the batch-scope lesson of §4.3 applied to
  diagnostics): per-batch `ū_k = mean_n r_nk` — which may legitimately be extreme on a
  single date if that date sits in one regime — and **running** utilization over the
  balance buffer, which is what the near-zero alarm watches (collapse twin — a dead
  expert's gradients vanish with its responsibilities);
- responsibility sharpness `mean_n max_k r_nk` over training (healthy specialization
  sharpens gradually);
- pairwise correlation of expert outputs `R_jk` (homogenization watch, §4.3 item 2 —
  logged even when the decorrelation aux is off).

---

## 5. Configuration design (fixes defect 8's root cause)

One frozen dataclass tree, no scattered constructor defaults:

```
NECConfig
├── data:    DataConfig     # d_seq, d_snap, seq_len T, feature name lists (schema, §6)
├── encoder: EncoderConfig  # hidden_dim=32, num_layers=1, dropout=0.0
├── gate:    GateConfig     # (input dim derived from encoder — not settable independently)
├── experts: ExpertConfig   # n_experts=2, hidden_dim=64, dropout=0.05,
│                           # input_mode: "snapshot" | "snapshot_plus_hidden"  (Decision A)
├── prior:   PriorConfig    # kind: "soft" (registry key); memoryless family params;
│                           # HMM params: transition_diag_bias, learn_pi0, tvtp: bool=False
└── train:   TrainConfig    # lr, weight_decay, gate_weight_decay, batch_size, steps,
                            # sigma_init, sigma_freeze_steps, aux_load_balance: bool=False,
                            # aux_alpha, seed, sequence_ordered: bool  (True required for HMM)
```

Rules:

- `NECConfig.validate()` runs at model construction and cross-checks every derived
  dimension (gate input == encoder hidden; expert input == `d_snap` or `d_snap + hidden`
  per `input_mode`; `dropout > 0` requires `num_layers ≥ 2`; `n_experts ≥ 2`). The
  prototype's 256-vs-64 mismatch becomes a `ValueError` with both numbers in the message,
  at build time.
- Modules take their sub-config, never loose ints — there is exactly one place a
  dimension can be written.
- A tiny `assert_shape(t, (B, self.cfg.hidden_dim), "h_T")` helper is called at the top
  of every `forward` — mismatches fail loudly with tensor name, expected, and actual.
- Config serializes to/from dict (JSON/YAML-able) so every run is reproducible and the
  Module 5 trial registry can log it verbatim.
- **HMM/memoryless coupling checks** in `validate()`: `prior.kind == "hmm"` requires
  `train.sequence_ordered == True` (the filter needs time-contiguous, chronological
  batches per entity — a data-loader constraint, §7); and `prior.tvtp == True` requires an
  encoder→transition path, deferred (§9bis) — setting it now raises `NotImplementedError`
  with a pointer, so the config surface exists but the half-built path can't be selected.

---

## 6. Data interface and feature-column contract (fixes defect 7)

The real data pipeline is out of scope; the *contract* is in scope and synthetic-testable.

```
FeatureSchema
    sequence_features: list[str]   # names, order = channel order of x_seq
    snapshot_features: list[str]   # names, order = channel order of x_snap
    target: str                    # e.g. "fwd_log_return_5d"

Batch  (what the model consumes)
    x_seq:  float32 (B, T, d_seq)   # trailing window, oldest → newest; row t uses info ≤ t
    x_snap: float32 (B, d_snap)     # prediction-date snapshot features
    y:      float32 (B,)            # forward return target
```

Contract clauses:

1. **The target, identifiers, and dates never live inside a feature tensor.** The
   prototype's `x[:, -1, :-1]` — a reserved trailing column silently dropped — is
   impossible by construction: there is no reserved column.
2. **The model never slices feature tensors.** `x_snap` is built by the (future) pipeline
   or the synthetic generator, not derived in `forward`. Snapshot features may overlap
   sequence features (typically: the last-step values of a named subset) but that mapping
   lives in the pipeline, documented by the schema.
3. `d_seq == len(sequence_features)` and `d_snap == len(snapshot_features)` are asserted
   where data meets model.
4. A `SyntheticRegimePanel` generator (in-package, also used by tests) emits batches of
   exactly this shape with a known latent regime: regime determined by the volatility of a
   designated sequence channel; `y = β_z · x_snap + ε` with per-regime coefficients
   (e.g., sign-flipped — momentum in calm, reversal in stress), so regime recovery and
   prediction quality are both checkable against ground truth. **Two regime-process modes**
   so both variations are testable against a known generator: `iid` (regime drawn per
   sample from a fixed marginal — the memoryless-favourable case) and `markov` (regime
   follows a first-order chain with a configurable, persistence-biased transition matrix —
   the HMM-favourable case). The `markov` mode is what lets a test show the `HMMRegimePrior`
   recover a sticky regime and beat the memoryless prior when persistence is real; both
   modes expose the ground-truth transition matrix and regime path for scoring.

---

## 7. The gate abstraction: shared likelihood + pluggable prior (serves both variations)

The gate is **not** a monolithic router. It is two components, and only one of them
varies between thesis variations:

**(A) `ExpertLikelihood` — shared, fixed across variations.** Pure function of expert
outputs and target: `log N_k = log N(y | μ_k, σ_k²)`, shape `(B, K)`. No parameters of its
own beyond what the experts own. Every prior below plugs into this same core. This is the
component that makes "same per-expert likelihood, different prior" true in code, not just
on the whiteboard.

**(B) `RegimePrior` — the pluggable component.** Forms the **log prior over experts**
`log π` *before* the Bayes combine. Two families:

> **A `RegimePrior` consumes gate logits and (optionally) recursive state, and returns a
> *log prior* on the simplex. The mixture NLL is `−LSE_k(log π_k + log N_k)` and the
> filtered posterior is `r_k = softmax_k(log π_k + log N_k)` — both well-defined for any
> prior.** Sparse (top-k) priors emit `−inf` at pruned entries, which drop out of the
> `logsumexp` naturally; the HMM prior consumes the previous `r` and returns the next.

```
PriorOutput (dataclass)
    log_prior:      Tensor (B, K)       # normalized log π: LSE_k = 0; −inf allowed
    aux_loss:       Tensor | None       # e.g. load balancing (memoryless family)
    info:           dict[str, Tensor]   # diagnostics (selected idx, temperature, A, …)

PriorContext (dataclass, all fields optional)
    prev_filtered:  Tensor | None       # r_{t-1}, log-domain; the HMM recursion's state
    step:           int | None          # temperature/annealing schedules

class RegimePrior(nn.Module, ABC):
    K: int
    stateful: bool                      # True ⇒ trainer must thread prev_filtered in time order
    @abstractmethod
    def forward(self, gate_logits: Tensor, ctx: PriorContext | None = None) -> PriorOutput
```

**Memoryless family (Variation 2)** — `ctx` ignored; prior is a function of the current
encoder output only:
- **Now:** `SoftRegimePrior` — `log_prior = log_softmax(gate_logits)`. The only prior
  fully wired in the baseline.
- **Later, no rewrite** (documented in the ABC, not implemented): `TopKRegimePrior`
  (mask outside top-k, renormalize in log domain); `GumbelSoftmaxRegimePrior` (Gumbel-
  perturbed logits, temperature from `ctx.step`); `HardRegimePrior` (one-hot `0/−inf`,
  straight-through gradient — note it *changes the effective objective* to the selected
  expert's NLL, which is exactly the routing-mechanism comparison).

**Recursive family (Variation 3)** — `stateful = True`:
- `HMMRegimePrior` — `log π_t = logsumexp_j(logA_{jk} + prev_filtered_j)` (the predict
  step), with `A` the row-softmax transition matrix (§4.3). It reads `ctx.prev_filtered`
  and its output combines with `log N_t` to produce both the timestep's NLL and the next
  `r_t` that the **trainer threads back** as the following step's `prev_filtered`. Requires
  chronological, entity-contiguous batches (`train.sequence_ordered`, enforced in §5).
  Filtering only — never smoothing (§3 correctness requirement).
  - *TVTP extension (deferred, config-guarded):* make `A` depend on a covariate (VIX /
    realized vol) or on `h_T` via `L = L_0 + W·covariate`. Flagged off with a strong
    identifiability caveat (arXiv:2605.14976 §9bis) — start static.

**Why this is a generalization, not a fork.** `SoftRegimePrior` is the special case of a
stateful prior that ignores its state. Swapping memoryless↔HMM changes *only* which
`RegimePrior` subclass the registry builds from `PriorConfig.kind`; the encoder, gate head,
experts, `ExpertLikelihood`, `MixtureNLL`, responsibilities, and all diagnostics are
byte-for-byte the same code. Committing to Variation 2 or 3 later is a config string, not a
rewrite — the requirement you set.

**Trainer's role in the recursion.** The training loop owns the time-threading so the
priors stay simple `nn.Module`s: for a stateful prior it iterates timesteps in order,
carrying `r_{t-1}`; for a memoryless prior it processes the batch in one shot. The loop
detects which via `prior.stateful`. `NECModel.forward` returns a structured `NECOutput`
(gate logits, prior output, expert means, log sigmas, filtered posterior, ŷ) so objectives
and diagnostics consume whatever they need without model changes.

**Design goal — one skeleton also yields the classical Hamilton baseline** (natural
comparison for Variation 3; not built this pass). Note the *emission* side is itself a
pluggable interface: `ExpertBank` produces `(μ_k, σ_k)`. Swap the neural experts for
**per-regime constant Gaussians** (learned `μ_k, σ_k` scalars, no network) or **per-regime
linear emissions**, keep the `HMMRegimePrior` + forward filter, and the same training loop
estimates a classical Hamilton / Gaussian-HMM model. So the full design has *two*
orthogonal plug axes — emission model (neural | classical) × prior/dynamics
(memoryless | Markov) — and the four corners are: our baseline (neural × memoryless),
HMM-NEC (neural × Markov), plain MoE-with-classical-experts (classical × memoryless), and
Hamilton (classical × Markov). Recording this now keeps the `ExpertBank` interface honest
(experts are an emission strategy, not hardcoded MLPs) even though only the neural experts
ship this pass.

---

## 8. Decision A (defect 6): what do the experts see? — **needs your sign-off**

**Question.** Experts consume the raw snapshot `x_snap` only, or `x_snap ⊕ h_T`
(snapshot concatenated with the encoder's final hidden state)?

**Recommendation: snapshot-only as the baseline default**, with
`experts.input_mode = "snapshot" | "snapshot_plus_hidden"` implemented from day one
(it is a config-derived input-dim change, ~zero marginal cost) and the comparison
pre-registered as a thesis ablation under the Module 5 protocol — exactly what the
ledger's keep/change/decide box prescribes ("run both, pre-registered").

**Reasoning for the default:**

1. *The original objection to snapshot-only dissolves once defect 1 is fixed.* The
   ledger's worry was that the encoder's entire value flowed through a routing decision
   that argmax prevented from training. Under the mixture NLL the gate receives the dense
   `π − r` gradient every step, so sequence information has a fully trainable path into
   the model through routing. Snapshot-only is now a *clean* division of labor —
   regime-from-history, prediction-from-snapshot — which the ledger itself calls
   defensible as a design.
2. *Interpretability of the thesis question.* If experts also read `h_T`, each expert
   becomes a sequence model and the MoE can quietly degenerate into an ensemble of
   sequence regressors whose gate does little — the secondary research question ("do
   learned experts correspond to regimes?") loses its cleanest probe, because expert
   specialization is no longer forced to happen *through* the regime channel.
3. *Variance budget.* Concatenation grows every expert's first layer by `n` inputs and
   routes K experts' gradients (plus the gate's) into one small encoder — more parameters
   and a shared-representation tug-of-war, against the effective-sample arithmetic that
   motivated shrinking the encoder in the first place.
4. *Falsifiability is preserved.* If snapshot-only underperforms, the flag flips and the
   ablation says so under purged walk-forward — a result either way, which is worth more
   to the thesis than a silently-made choice.

**Cost of the default:** if the true regime-conditional relationship needs sequence
context *inside* the conditional mean (not just for regime identification), snapshot-only
experts can't express it and the baseline understates the architecture. The ablation
covers this.

---

## 9. Decision B: gate supervision — **needs your sign-off**

**Question.** Does the gate learn regimes purely unsupervised (responsibilities from the
mixture likelihood alone), or does an auxiliary supervised signal (e.g., VIX- or
realized-vol-bucket labels) regularize it?

**Recommendation: purely unsupervised for the baseline.** Regime proxies (VIX, realized
vol, drawdowns) enter as *diagnostics and reporting devices only* — gate–VIX correlation,
expert alignment across folds sorted by expert σ_k or vol-correlation — never as a
training signal. Architect the trainer so a supervised auxiliary
(cross-entropy of gate logits toward provided labels — eq. (4.10.3) vs. (4.10.4):
"identical operator, different teacher") can be added later as one more config-gated aux
term, so nothing is foreclosed.

**Tradeoffs, stated:**

| | Unsupervised (recommended) | Auxiliary supervised |
|---|---|---|
| Thesis logic | Keeps the secondary research question ("do learned experts correspond to recognizable regimes?") *testable* — alignment with VIX is a finding | Circular: training toward VIX labels then "discovering" VIX alignment assumes the answer |
| Objective | One principled objective (mixture NLL = generalized EM); no aux-weight hyperparameter | Extra hyperparameter (aux weight, label definition, bucket thresholds) — each a researcher degree of freedom Module 5 makes you register |
| Regime definition | Learned; may find non-vol structure (or embarrassing splits — size, liquidity, artifacts — which the diagnostics are designed to expose) | Anchored to a hand-chosen definition that may be wrong (regimes ≠ vol buckets necessarily) |
| Training stability | Real collapse/label-switching risk — mitigated *without* touching the objective: uniform zero-init gate, gate weight decay, σ warm-start freeze, diversified experts, utilization/entropy alarms, optional load-balancing flag, multi-seed protocol | More stable, semantics pinned from step 0 |
| Data dependencies | None new (matters now: data pipeline is deliberately unbuilt) | Needs VIX/vol label plumbing before the baseline can even train |

The deciding argument is the first row: the syllabus's interpretability hypothesis is
explicitly "must be tested, not assumed." A supervised gate converts the thesis's most
interesting question into an assumption. If unsupervised gating proves unstable on real
data, the supervised auxiliary is the documented, config-gated fallback — and *that
finding itself* ("feature-conditioned gates need regime supervision to stay alive") is a
publishable ablation row. *(Note for Variation 3: the same unsupervised principle applies —
the transition matrix and emissions are learned by maximum likelihood through the forward
filter, no regime labels; "unsupervised" and "HMM" compose cleanly.)*

---

## 9bis. HMM-path decisions (Variation 3) — **flagged for sign-off if Variation 3 is chosen**

These bind only if the supervisor approves the HMM-gated variation. The baseline **ships
memoryless (`SoftRegimePrior`)**; the items below are architected-for, not built this pass,
but the decisions shape the interfaces so record them now.

### Decision C: how to train the HMM gate — **needs your sign-off (if Var. 3)**

**Question.** Backprop end-to-end through the differentiable forward-algorithm recursion
(treat the forward filter as a differentiable layer, one joint AdamW objective with the
experts), or alternate Baum-Welch-style EM (forward–backward E-step, gradient M-steps on
experts/transition)?

**Recommendation: end-to-end differentiable forward algorithm.** Reasoning:

1. *Baum-Welch buys nothing here.* Its appeal is closed-form M-steps, but the experts are
   neural nets with **no closed-form M-step** — the M-step is itself gradient descent
   either way. So "EM" would be gradient M-steps wrapped in an outer loop, versus one clean
   joint objective. The classical simplification does not apply once emissions are neural.
2. *Autograd already is the recursive score.* Backprop through the log-domain forward
   recursion computes the exact score — this is precisely arXiv:2205.01565's manual
   recursive score/Hessian algorithm, done automatically and exactly (§4.2). We get for
   free what that paper derives by hand.
3. *Baum-Welch's E-step is smoothed* (forward–backward), and smoothing is
   prediction-time-illegal (§3). Training-time smoothing is defensible *if* it never
   leaks into a prediction, but it is a footgun next to a strictly-causal filter that is
   already all we need. End-to-end filtering keeps one causal path.
4. *One objective, one optimizer, one monotonicity story* (generalized EM, §4.2) — simpler
   to reason about and to keep numerically stable via the same LSE machinery.

**Cost / when to revisit:** if joint training is unstable (the transition logits and
experts fighting early), a *filtering* EM variant — freeze `A`, take expert steps, then a
transition update from filtered (not smoothed) expected counts — is the documented
fallback, and worth one ablation line. Full forward–backward Baum-Welch stays off the table
unless a purely-offline diagnostic use is found. Recommend end-to-end; this is a real
decision, not a default.

### Decision D: regime-label permutation under rolling re-estimation — **needs your sign-off (if Var. 3)**

**The risk (arXiv:2603.04441).** Walk-forward evaluation re-fits models on a rolling
schedule. Each refit is a fresh optimization, and latent regime identities can **permute**
between refits ("regime 1" this window ↔ "regime 2" next window) — the label-switching of
Module 8 §failure-mode 2, now *across time*, which corrupts any per-regime table averaged
over folds and any narrative that a given expert "is" the stress regime.

**Does our protocol even trigger it?** Two sub-cases, and the plan should state which:
- *Fit-once-per-window (recommended default):* within a walk-forward window the model
  (including `A`) is trained once and frozen for that window's out-of-sample dates.
  **Within a window there is no permutation** — identity is fixed by that fit. Permutation
  can only occur *across* windows (independent refits).
- *Re-estimate-within-window:* not planned; would reintroduce intra-window permutation and
  is explicitly out of scope.

So the exposure is **across-refit** identity drift, and it is real for any multi-window
walk-forward.

**Recommendation for the baseline: canonical ordering by a declared statistic** — sort
experts by ascending `σ_k` (or by gate/VIX correlation) *after each fit, before any
cross-window aggregation*. This is the same alignment rule already mandated for the
memoryless case (§ the Module 8 reporting clause: "align expert labels across folds by a
declared statistic before any per-regime table") — Variation 3 does not need a new
mechanism, just the existing one applied at each refit. Cheap, deterministic, no extra
model.

**If sorting proves fragile** (near-ties in `σ_k`, regimes that genuinely swap variance
ordering), the documented upgrade is **arXiv:2603.04441's Wasserstein template tracking**:
map each refit's regime Gaussians to persistent templates via the closed-form 2-Wasserstein
distance between Gaussians, updated by exponential smoothing — geometric identity anchoring
that survives near-ties a scalar sort cannot. Recorded as the escalation path, not built
now. Recommend: scalar-sort for the baseline, template-tracking flagged as the upgrade.

### Two notes that touch existing requirements (not full decisions)

- **Number of regimes K.** The "don't hardcode K" requirement (§2) already covers Var. 3.
  arXiv:2603.04441 additionally does *predictive* model-order selection (pick `K` by
  one-step-ahead validation log-likelihood, strictly causal) rather than a fixed `K`. For
  the baseline `K` stays a fixed config value (default 2, reinforced for the HMM case by
  the arXiv:2605.14976 identifiability warning below); predictive/adaptive-`K` is a
  documented extension, and if used it is a selection event for the Module 5 registry.
- **Static vs. covariate-dependent (TVTP) transition matrix.** Baseline uses a **static
  learned `A`**. Making `A` depend on VIX/vol (or `h_T`) is the natural next step
  (arXiv:2605.14976's Model II), but that paper is a **strong caution**: the score-driven
  (GAS) TVTP coefficient is *statistically non-identifiable* (collapses to zero), and even
  covariate-driven TVTP coefficients need ~1000 observations to identify at `K=3`, while
  one-step point forecasts are nearly insensitive to transition specification anyway (the
  payoff of getting transitions right is *regime characterization*, i.e. our interpretability
  question, not rank-IC). Conclusion: static `A` for the baseline; TVTP is `prior.tvtp`,
  guarded off (§5), with this caveat attached so we don't over-reach.

---

## 10. Recent literature (MoE papers/): what the baseline adopts, defers, and rejects

Read after the plan's first draft; design deltas already folded into §4.3/§4.4 above.
None of these is required precedent — methodologically all three sit far from our
cross-sectional rank-IC setting — but each contributes one concrete thing.

**arXiv:2603.19136 (autoencoder-gated dual-expert + SAC-tuned threshold).**
Its one-component-at-a-time ablation attributes most of the gain to the regime
detection/routing module (+35.6% MAPE on removal) versus the dual experts (+6.8%) and
the RL controller (+15.3%). *Adopted:* two extra ablation baselines pre-registered for
the thesis grid, mirroring their discipline —
(a) **frozen uniform gate** (`w_k = 1/K` always; experts + mixture prediction, no
learned routing) and (b) **regime-as-feature** (single expert of matched total capacity,
gate signal appended to its input — their "No Dual Paths" variant), which asks whether
*architectural* separation of experts beats conditioning one model on the regime signal.
Both are cheap once the baseline exists; (a) is literally a `RegimePrior` variant of §7
(a fixed uniform log-prior).
*Noted, not adopted:* their gate is generative/one-sided (reconstruction-error anomaly
score → threshold), a different gate *head*, not a different router — our gate-head /
router separation already leaves that door open. Their RL threshold controller is out of
scope. *Caveats for the lit review:* price-level MAPE on 20 stocks, sentiment features,
no purged walk-forward — motivation-grade, not evidence-grade, for our claims.

**arXiv:2606.24062 (RAVEN — experts specialized by adaptive context length).**
A different specialization axis (temporal scale via nested windows) — related work for
the thesis, and a candidate future expert design, not the baseline. *Adopted:* the
expert-diversity penalty `L_div = ‖R − I‖_F²` as our config-gated decorrelation aux
(§4.3 item 2) plus always-on `R_jk` monitoring (§4.4) — their argument that routing-side
balance and output-side decorrelation address orthogonal failure modes maps cleanly onto
our collapse vs. homogenization pair. *Noted:* their K=3 > K=2 > K=4 result is specific
to nested-scale experts and does not overturn our K=2 default; it does reinforce keeping
`n_experts` a first-class ablation. Their router-entropy regularizer is the same family
as our load-balancing/entropy levers — no new mechanism needed.

**arXiv:2501.11873 (load-balancing implementation pitfalls).**
The load-bearing read for our gate objective. Micro-batch-scope balancing forces uniform
expert usage *within every batch*, destroying specialization exactly when batches are
internally homogeneous — and our date-sliced cross-sectional batches are the worst case,
since regime is a date-level variable. *Adopted wholesale into §4.3:* buffered/global
scope for `f̄_k`, stop-grad structure made explicit, from-scratch-runs rule for LBL
ablations, "weight cannot fix scope" caution, and the two-scope utilization dashboards
in §4.4. Their auxiliary-loss-free alternative (bias term nudged by selection frequency,
Wang et al. 2024) is recorded here as the fallback if even buffered LBL proves to fight
the NLL — one more config-gated strategy later, no interface change.

Also in the folder, unread in this pass but flagged as directly thesis-relevant:
arXiv:2604.09780 ("The Myth of Expert Specialization in MoEs: routing reflects geometry,
not domain expertise") speaks straight at the secondary research question — read before
writing the interpretability chapter, since it is the strongest available null
hypothesis for "learned experts = regimes"; and arXiv:2006.10119 (Markovian RNN,
HMM-based switching) is a concrete precedent to cite for the `HMMRegimePrior` variant.
MIGA (arXiv:2410.02241) and arXiv:2508.02686 are lit-review material for the proposal.

### 10bis. HMM-path literature (Variation 3): adopted, deferred, cautioned

Three papers specific to the HMM-gated variation; design deltas already folded into
§3 (filtering requirement), §4.3 (transition parameterization), and §9bis (decisions).

**arXiv:2603.04441 (Wasserstein HMM — strictly causal daily regime allocation).**
Directly on-point for Variation 3's operational risks. *Confirms our correctness spine:*
it is strictly causal with **filtered** probabilities `p(z_t | F_{t-1})` — independent
support for the filtering-not-smoothing invariant (§3). *Adopted as the escalation path
for Decision D:* its 2-Wasserstein template tracking (map each refit's regime Gaussians to
persistent templates via the closed-form Gaussian-Wasserstein distance, exponential-smooth
the templates) is our documented upgrade if scalar `σ_k`-sorting can't hold regime identity
across refits. *Bears on "don't hardcode K":* it selects regime count by predictive
(one-step-ahead) log-likelihood rather than fixed `K` or BIC — recorded as the adaptive-`K`
extension in §9bis. *Caveat for the lit review:* it is a portfolio-allocation/turnover
study on a 5-asset cross-asset universe with MVO, not cross-sectional equity rank-IC —
methodology transfer (causal rolling HMM, identity tracking), not a performance benchmark.

**arXiv:2605.14976 (multi-regime Markov switching with time-varying transition
probabilities).** Our **caution source** for the transition matrix. *Adopted as a
constraint, not a feature:* keep `A` **static** for the baseline. Their Monte Carlo shows
(i) the GAS/score-driven TVTP coefficient is *statistically non-identifiable* (ridge in the
`(σ², A)` likelihood; `Â → 0`, empirically fails to converge), and (ii) even covariate-
driven TVTP coefficients need ≈1000 observations to identify at `K=3`, while (iii) one-step
point forecasts are almost insensitive to transition specification — the value of correct
transitions is in **filtered regime probabilities** (regime characterization), i.e. exactly
our interpretability question, *not* rank-IC. Net: TVTP is `prior.tvtp`, guarded off (§5,
§9bis); if ever enabled, prefer the exogenous covariate form (their Model II) over GAS, and
budget for the identification difficulty. Also reinforces the `K=2` default for the HMM
variant (their `K=3` filtered-probability MSE is ~an order of magnitude worse than `K=2`).

**arXiv:2205.01565 (recursive score/Hessian in regime-switching models).** The
*technical underwriter of Decision C.* It gives a recursive algorithm computing the score
(and Hessian) of a regime-switching likelihood inside the prediction/update (filtering)
steps, with per-step re-scaling for numerical stability. *The takeaway for us:* backprop
through our log-domain differentiable forward filter **is** this recursion, computed
automatically — so "treat the forward algorithm as a differentiable layer" is not a
shortcut but the exact score, which is why Decision C recommends it over Baum-Welch. *Also
useful later:* for the **classical Hamilton baseline** (classical emissions × Markov prior,
§7 design goal), if we want econometric standard errors on `A`, this paper's outer-product-
of-score (OPG) estimator is what it recommends over the Hessian in finite samples — a
concrete recipe for that baseline's inference table, should the thesis want one.

---

## 11. Package layout

New top-level directory (keeps the prototype in `OP model/nec/` untouched as the artifact
the thesis critiques):

```
nec_baseline/
├── pyproject.toml              # minimal; deps: torch, numpy, pytest
├── README.md                   # points at this design doc; quickstart for the smoke test
├── nec_moe/
│   ├── __init__.py
│   ├── config.py               # NECConfig tree + validate() (§5)
│   ├── data.py                 # FeatureSchema, Batch, SyntheticRegimePanel (§6)
│   ├── encoder.py              # GRUEncoder
│   ├── gate.py                 # GateHead (LayerNorm → Linear, zero-init, logits out)
│   ├── experts.py              # Emission ABC; ExpertMLP/ExpertBank (neural, ships);
│   │                           # (ClassicalGaussianEmission stub for Hamilton baseline, §7)
│   ├── likelihood.py           # ExpertLikelihood core: log N(y|μ_k,σ_k²) — shared (§7A)
│   ├── priors.py               # RegimePrior ABC, PriorOutput/Context, SoftRegimePrior,
│   │                           # HMMRegimePrior (row-softmax A, log-domain predict step),
│   │                           # registry (§7); `Router` alias for familiarity
│   ├── model.py                # NECModel (assembly + shape assertions), NECOutput
│   ├── losses.py               # MixtureNLL (fused log-domain; returns NLL + filtered r),
│   │                           # load_balance_aux, expert_decorrelation_aux (config-gated)
│   ├── train.py                # Trainer: AdamW param groups, σ freeze schedule,
│   │                           # aux-loss composition, monitoring; memoryless one-shot path
│   │                           # AND stateful time-threaded path (selected by prior.stateful)
│   └── diagnostics.py          # entropy, utilization, sharpness, R_jk, regime-recovery,
│                               # (Wasserstein template-tracking stub for Decision D escalation)
└── tests/                      # §12
```

`likelihood.py` and `priors.py` split reifies the unifying principle in the file tree: the
shared core is one module, the pluggable priors another. The HMM prior's forward-filter
*predict step* ships this pass (it is what proves the abstraction is a generalization, not a
promise — §12.4); the full Variation-3 apparatus (TVTP, Wasserstein tracking, classical
emissions, walk-forward refit) is deferred (§13). Engineering standard throughout: type
hints on every public signature, docstrings with tensor shapes in the ledger's `(B, T, d)`
notation, dataclasses for structured values, no module-level state, single seed function.
Python ≥ 3.12 (system python3.14 per project convention), PyTorch as in the existing codebase.

---

## 12. Test plan

Principle: **every ledger defect gets a test that would have failed on the prototype**,
plus math-verification tests for the loss, plus one end-to-end smoke test. All tests run
on CPU with fixed seeds; the full suite targets < 90 s.

### 12.1 Defect-regression tests (would have caught the prototype)

| Test | Defect | Assertion |
|---|---|---|
| `test_gate_receives_gradient` | 1 | After one `backward()` of the mixture NLL, gate `Linear` weight grad is nonzero. Companion: a hard-argmax routing double produces exactly-zero gate grads — demonstrating the defect the design fixes. |
| `test_all_experts_enter_loss` | 2 | For each `k < K`: perturbing expert k's parameters changes the loss (no expert is collapsed out of the objective); parameterized over `K ∈ {2, 3, 5}`. |
| `test_model_emits_logits_not_probs` | 3 | Gate/model output rows do not lie on the simplex; no `nn.Softmax` module anywhere in the model tree; fused loss equals an explicit-softmax reference to tolerance on benign inputs. |
| `test_no_cross_batch_coupling` | 4 | In **train mode**, sample i's output is bit-identical when the rest of the batch is replaced/permuted (LayerNorm passes; BatchNorm provably fails this). |
| `test_default_capacity` | 5 | Default config builds a GRU (`num_layers ≤ 2`, `hidden ≤ 64`); total parameter count under a stated cap (~100k with defaults) vs. the prototype's ~4M. |
| `test_expert_input_modes` | 6 | Both `input_mode`s construct, validate, and forward with the correct derived input dim; wrong `x_snap` width raises. |
| `test_feature_contract` | 7 | Model raises (never silently slices) when `x_seq`/`x_snap` widths disagree with the schema; `Batch` construction rejects a target column smuggled into features (width check against schema). |
| `test_config_mismatch_fails_loudly` | 8 | Reconstructing the prototype's bug — gate expecting 64 while encoder emits 256 — is impossible via config (gate dim is derived); manually mis-wired modules raise `ValueError` naming expected/actual dims at construction, and `assert_shape` raises informatively on malformed runtime tensors. |

### 12.2 Loss/math verification tests

- `test_nll_matches_hand_reference` — fused loss equals a naive-but-safe float64
  reference on small tensors.
- `test_gate_gradient_is_pi_minus_r` — autograd gradient of the batch loss w.r.t. gate
  logits equals `(π − r)/B` analytically (Module 4 Ex. 4.5(b)).
- `test_expert_gradient_responsibility_weighted` — expert mean grads scale with
  responsibilities; a near-zero-responsibility expert receives near-zero grad (collapse
  twin, documented behavior).
- `test_log_domain_stability` — targets ~40σ from every component: loss finite and
  correct; demonstrably the regime where the non-log-domain pipeline underflows
  (Module 8 Ex. 8.1's cliff).
- `test_logit_gauge_invariance` — adding a constant to all gate logits leaves loss and ŷ
  unchanged.
- `test_sigma_gradient_stationarity` — at `σ_k²` = responsibility-weighted MSE, the
  `log σ_k` gradient vanishes.
- `test_prior_contract` — `SoftRegimePrior` log-prior normalizes (`LSE_k = 0`); registry
  round-trips from config; `PriorOutput` with `−inf` entries flows through the NLL
  finitely (future-proofing sparse priors).
- `test_load_balance_scope` — the arXiv:2501.11873 pitfall as a test: on a
  regime-homogeneous batch (all samples one regime) with a balanced running buffer,
  buffered `L_bal` stays near its floor while a per-batch-scope double penalizes the
  same (correct) sharp routing; and gradient flows only through `P̄_k` (`f̄_k` is
  stop-grad).
- `test_expert_decorrelation` — `L_div ≈ 0` for decorrelated expert outputs, maximal
  for duplicated experts; its gradient pushes duplicated experts apart.

### 12.3 Gate-abstraction & HMM-path tests (proves "config change, not rewrite")

These verify the two-variation abstraction actually holds — the shared likelihood core, the
stateful prior threading, and the filtering invariant — so choosing Variation 3 later is a
config swap. The HMM prior's forward-filter predict step ships this pass expressly so these
can run.

- `test_shared_likelihood_core` — the `ExpertLikelihood` output and the `MixtureNLL` /
  responsibility computation are **identical** whether the prior is `SoftRegimePrior` or
  `HMMRegimePrior`, given the same `log π` — i.e. only the prior differs (the abstraction's
  core claim, asserted in code).
- `test_hmm_forward_filter_matches_reference` — the log-domain forward recursion's
  per-timestep log-likelihood and filtered posterior equal a plain (small, float64) HMM
  forward-algorithm reference on a tiny fixed `(A, μ, σ, y)` sequence.
- `test_filtering_is_causal_no_lookahead` — **the leakage test, first-class (§3).**
  Perturbing `y_t` (or any input) at a future step `t' > t` leaves the filtered posterior
  and predictive density at `t` bit-identical; a smoothing double (forward–backward) is
  shown to *fail* this — encoding the filtering-not-smoothing invariant as an executable
  guard, the HMM-path analogue of `test_no_cross_batch_coupling`.
- `test_transition_is_row_stochastic` — `A = row_softmax(L)` sums to 1 per row for
  arbitrary unconstrained `L` (incl. after gradient steps); no projection needed.
- `test_transition_persistence_init` — with `transition_diag_bias > 0`, initial `A` has
  diagonal > off-diagonal (starts sticky).
- `test_stateful_threading_equivalence` — the trainer's time-threaded path, run on a
  batch reshaped to length-1 sequences with a memoryless prior, reproduces the one-shot
  path's loss (the two trainer paths agree where they must).
- `test_hmm_recovers_persistent_regime` — on `SyntheticRegimePanel(markov, high
  persistence)`, a trained `HMMRegimePrior` model recovers the latent regime path (state
  accuracy / AUC above threshold) **and** achieves lower held-out NLL than the same model
  with `SoftRegimePrior` — the payoff test showing persistence is exploited when real.
  (Companion: on `iid` regimes the two priors tie within tolerance — the HMM prior does no
  harm when there is no persistence to exploit.)

### 12.4 Smoke test (the "actually trains and predicts" gate)

`test_smoke_train_synthetic` — on `SyntheticRegimePanel` (K=2 regimes, sign-flipped
per-regime linear maps, regime inferable from a sequence channel's rolling vol):

1. builds the default-config model, trains a few hundred AdamW steps on CPU (< 60 s);
2. final train NLL improves on initial NLL by a stated margin;
3. **regime recovery:** AUC of gate probability vs. true latent regime > 0.8 on held-out
   synthetic data (the syllabus's "synthetic hidden-regime recovery test", proposal
   milestone 1);
4. **prediction sanity:** mixture-mean predictions correlate with the noiseless
   conditional mean above a stated threshold;
5. monitoring sanity: gate entropy starts at `log K` (zero-init check) and utilization
   never alarms.

Deterministic seed; thresholds chosen loose enough to be robust, tight enough to fail on
a broken gradient path (e.g., an accidental detach reproduces the argmax pathology and
fails item 3).

---

## 13. Explicitly out of scope for this pass

- ~~Real data pipeline (Stooq daily prices)~~ — **shipped in increment 4** (Stage B:
  `market_data.py`/`features.py`, offline-tested, yfinance fallback, live-verified).
  Still deferred from the data plan: French factors + VIX context data (Stage C), SEC
  fundamentals (Stage D), and the point-in-time universe/delistings work (Module 12) —
  without which every backtest on the default universe stays survivorship-biased and
  pipeline-verification-grade only.
- **Ships this pass** (to prove the abstraction): the shared `ExpertLikelihood` core, the
  `RegimePrior` ABC with the stateful hook, `SoftRegimePrior` (memoryless, the baseline's
  active prior), and the `HMMRegimePrior` **forward-filter predict step** with its
  causality/correctness tests (§12.3).
- ~~Deferred: memoryless routing variants (hard / top-k / Gumbel)~~ — **shipped in
  increment 2** (see build status), plus the `uniform` ablation prior.
- ~~Wiring the classical Gaussian emissions into the config registry (Hamilton
  baseline)~~ — **shipped in increment 3** with a parameter-recovery test; per-regime
  *linear* emissions remain a possible later addition.
- **Still deferred (interface/config-guarded only):** covariate-dependent/TVTP
  transitions (§9bis, strong identifiability caveat), Wasserstein template tracking for
  cross-refit identity (Decision D escalation), and any Baum-Welch/forward–backward path
  (Decision C).
- ~~Walk-forward evaluation, purging, rank-IC harness~~ — **shipped in increment 2**;
  ~~transaction costs/turnover~~ — **shipped in increment 3** (quantile long-short,
  entity-aligned turnover, cost drag); ~~deflated-Sharpe, multiple-testing corrections,
  and the trial registry~~ — **shipped in increment 6** (`registry.py`,
  `multiple_testing.py`; factor-zoo validation pinned as a test). Module 5's
  methodology machinery is now complete.
- Gate calibration (reliability diagrams, temperature scaling — Module 4 §4.9),
  explicit-EM training epochs (worth one ablation line later, per Module 8 capstone),
  load-balancing *on by default*, heteroscedastic input-dependent σ_k(x), and any
  Bayesian/VB machinery.

## 14. Open items for sign-off

*Bind now (Variation-agnostic — the baseline needs them regardless of which variation is approved):*

1. **Decision A (§8):** experts on snapshot-only (recommended default) vs.
   snapshot ⊕ h_T — flag implemented either way; which is the *default*?
2. **Decision B (§9):** purely unsupervised gate (recommended) vs. auxiliary supervised
   regime signal in the baseline objective?

*Bind only if Variation 3 (HMM-gated) is the approved thesis direction — recorded now because they shape the interfaces:*

3. **Decision C (§9bis):** train the HMM gate by end-to-end backprop through the
   differentiable forward algorithm (recommended) vs. Baum-Welch-style EM with gradient
   M-steps?
4. **Decision D (§9bis):** cross-refit regime-identity handling — scalar `σ_k`-sort
   canonical ordering (recommended baseline) vs. Wasserstein template tracking; and confirm
   the fit-once-per-walk-forward-window protocol (so intra-window permutation can't occur)?

*Rubber-stamps unless you object:*

5. `K = 2` default (§2, reinforced for the HMM case by the K=3 identifiability warning);
   package location/name `nec_baseline/nec_moe` (§11); design doc in `NEC_sources/`; both
   aux losses (buffered load balancing, expert decorrelation) implemented-with-tests but
   **default off** (§4.3, §10); static transition matrix with TVTP guarded-off (§9bis);
   the `HMMRegimePrior` forward-filter predict step + its causality tests **ship this pass**
   to prove the abstraction, while full Variation-3 apparatus stays deferred (§13).
