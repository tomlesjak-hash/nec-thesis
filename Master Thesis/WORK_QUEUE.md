# Work queue

Last updated 2026-09-26 (brief 06: CRSP replaces the free data; items 3 and 7 done). Everything still to be coded, in priority order. Each item names the brief
that specifies it. Start a session by reading this file, then the brief for the item being worked.

**Standing rules for every item.** Every numeric quantity is a config field with a documented
default and no hardcoded constant; nothing is chosen yet. Every claim gets a test that pins it.
Read the literature listed in the relevant brief's section 0 before writing code. Do not consume
out-of-sample data until the pre-registration is written and Q20 is answered.

**Reference documents**
- `Code_State_2026-09-21.md` — what the code was as of 21 September. Increasingly stale; trust the
  code over it.
- `Advisor_Questions.md` — OPEN and RESOLVED. Q6, Q7, Q11, Q19 settled. Q20 open.
- `Code_Change_Brief_2026-09-21.md` — brief 01, the original survey.
- `Code_Change_Brief_02_Residual_Frozen_Base.md` — brief 02, the residual design.
- `Code_Change_Brief_03_Gate_Interface_And_Hamilton.md` — brief 03, the gate interface.

---

## Done

- Brief 01 section 1.1, persistence diagnostics: `expected_durations`,
  `stationary_distribution`, `persistence_metrics`.
- Brief 01 section 1.4, the `wasserstein_template_tracking` stub made explicit.
- Brief 01 section 2, the dead-parameter audit and live parameter count.
- Brief 02, residual mode with a frozen base, zero-initialised correction heads, the correction
  penalty, and the objective registry seam.
- Item 1, the corrections to the residual implementation: between-expert statistics on `r` rather
  than `mu`, the `logsumexp(log_prior) == 0` guard where `f0`'s coefficient depends on it, the
  pairwise-correction-distance diagnostic, and the low-SNR sanity test.
- **Item 2, brief 03, the gate interface and the Hamilton gate** (2026-09-21). `RegimePrior.fit`
  (no-op default, called once per fold on the training block), `PrecomputedRegimePrior` (date-keyed
  log-prior table, no gradient, refuses to serve before `fit` or outside the fitted information
  set, and refuses to let the causal-apply path overwrite a fitted row), and
  `MarkovSwitchingRegimePrior` — Hamilton fitted by ML on the training block, canonically reordered
  by fitted variance, applied to the test block with frozen parameters through `filter(params)`,
  never the smoother. Multi-start is logged to the registry as a selection event.
  `HMMRegimePrior`'s docstring and the README now name it accurately as a baseline arm.

  Two things worth carrying forward. **The frozen-gate arm is now meaningful**: with a fitted
  Hamilton gate the residual mixture improves IC over the base by ~0.44 on sticky synthetic data,
  against ~-0.03 when the frozen gate was the zero-initialised (uniform) one — the gap that made
  the brief 02 arm vacuous is closed. **`start_jitter` is a real knob, not a formality**: measured
  on a simulated two-state series, 0.05-0.2 converges every start into the *same* basin
  (log-likelihood spread 0.000, so "best of N" selects from a population of one) while 0.5 reaches
  distinct optima (spread ~600 nats) at the cost of most starts failing. `gate_llf_spread` is
  logged per fit so this is visible rather than assumed; the value is not chosen.

---

## 1. Corrections to the residual implementation [DONE 2026-09-21]

Raised 2026-09-21 from the implementation report; confirmed against the code and fixed. Kept here
for the record of what was changed and why.

- `expert_output_correlation` and `expert_decorrelation_aux` must operate on the corrections `r`,
  not on `mu`. With `mu_k = f0 + r_k` the shared base dominates both: the correlation tends to 1
  regardless of expert behaviour, and the aux loss, if enabled, could only reduce `||R - I||` by
  inflating the corrections until they overcome the frozen base, in direct opposition to the
  correction penalty.
- Confirm `r` is materialised in the forward output rather than recovered by subtracting a
  recomputed `f0`.
- Assert `logsumexp(log_prior) == 0` wherever the `f0` coefficient depends on the prior summing to
  one, and test that at initialisation the posterior equals the prior exactly (true because all
  `mu_k` are identical when every `r_k` is zero).
- Add a diagnostic reporting pairwise distance between fitted corrections. Under zero-init the
  experts start identical and are separated only by the frozen gate handing them different weights,
  so "corrections never differentiated" is a named outcome to watch, not a surprise. It is the
  expected result when the gate is near-uniform.
- Add a low signal-to-noise variant of test 5, sized to an out-of-sample R-squared of about 0.005,
  asserting numerical sanity only: no NaN, sigma positive and bounded, correction magnitude finite,
  improvement over base small rather than spuriously large. Do not assert recovery.

## 2. The gate interface and the Hamilton gate [DONE 2026-09-21]

**Brief 03, sections 1 to 7.** Delivered; see the Done list above for what landed and the two
findings worth carrying forward. The critical path is clear: the remaining gates (item 6) now have
the `fit` / `PrecomputedRegimePrior` interface and the canonical ordering they each need.

## 3. End-to-end run on free data [DONE; superseded 2026-09-26]

Run on free data 2026-09-25 (`Smoke_Run_2026-09-25.md`), then superseded by the CRSP integration
run of brief 06 section F (`Smoke_Run_2026-09-26_CRSP.md`: same settings, only the data source
changed; data, gate, experts trained and wall clock only, no out-of-sample number). The original
item is kept below for the record.


Not a brief; a smoke run. Build a Stage B panel from the free source, fit the base, fit the Hamilton
gate, train the experts in residual mode, run the walk-forward evaluation. Small universe, few
folds, few seeds. The goal is integration, not a result.

Report base out-of-sample IC, mixture IC, improvement over base, correction magnitude distribution,
gate entropy, expert utilisation, expected regime durations, live parameter count, and wall clock
per fold, into a dated file in this folder.

**Nothing from this run may be quoted.** The universe is survivorship biased and the source is not
CRSP. The first plausible-looking number is exactly when a result quietly becomes a claim.

## 4. Documentation drift

`README.md` and `HANDBOOK.md` describe a jointly trained model the thesis is no longer making.
Update both to the decided design: gate fitted separately and frozen (Q19), base pre-trained and
frozen (Q7), residual correction form, experts the only trainable component, primary objective open
(Q20) with no argument made for one.

Add `DECISIONS.md` at the repo root: one entry per settled question (Q3 CRSP, Q6 residual form,
Q7 frozen base, Q11 depth grid, Q19 frozen gate) with decision, date and a pointer to
`Advisor_Questions.md`. Reference it from `AGENTS.md` so future sessions read the decisions rather
than re-deriving them.

## 5. Diagnostics

Independent of everything above; can run in parallel.

- **Brief 01 section 1.2**, the ICC variance decomposition of gate weights. Tests whether the gate
  behaves as a date-level regime variable at all, which is the premise of the thesis.
- **Brief 01 section 1.3**, the gate permutation test. Permute the gate series across dates,
  retrain the expert stage, build a null distribution. This is the strongest single piece of
  evidence the thesis can produce and it needs no additional data. Do not leave it to the end.

## 6. The remaining gates

In this order. Each becomes a `PrecomputedRegimePrior` on the section 1 interface.

- **Statistical jump model** — brief 01 section 4.2. The most structurally different from Hamilton
  and therefore the most informative second arm. Literature: Bemporad et al. (2018) *Automatica* 96;
  Nystrup, Lindström and Madsen (2020) *ESWA* 150; arXiv:2402.05272; arXiv:2410.14841.
- **Wasserstein** — brief 01 section 4.3. Literature: arXiv:2110.11848; arXiv:2310.01285 for
  hyperparameter sensitivity.
- **TVTP** — brief 01 section 4.1. Smallest change of the three, a localised edit to the existing
  filter, behind the config flag that currently raises. Literature: Diebold, Lee and Weinbach
  (1994); Filardo (1994) *JBES* 12(3). Keep the identifiability caution already cited in
  `config.py`.

Note that all three need the canonical regime ordering from brief 03 section 4, since all are fitted
per fold and all are invariant to relabelling.

## 7. CRSP seam [DONE 2026-09-26, by brief 06 section A]

CRSP access arrived, so brief 06 replaced the seam with the CRSP layer itself: `nec_moe/crsp.py`
(release `ciz202512`, S&P 500 membership INDNO 1000500, market INDNO 1000500, delisting returns,
the post-delisting fill), `scripts/extract_crsp.py` and `scripts/build_pit_panel.py`, everything
derived kept in `Data/derived/`; the free loaders are gone (section B). The provenance requirement
is met and enforced: every `TrialRegistry` row carries `data_source` (`"crsp_ciz202512"` or
`"synthetic"`), `post_delisting_return` and `hidden_init`, and the registry refuses a row without
them. The original item, for the record: define a loader interface against the cache CSV contract,
and add a `source` field to every registry entry.

## 8. Pre-registration

Fill in `PREREGISTRATION_TEMPLATE.md` with everything now settled: the architecture, the frozen gate
and base, the walk-forward protocol, the purge, the metric family, the multiple-testing accounting,
and how a null result is reported. Write it **before** Q20 is answered, so the evaluation protocol
is fixed independently of the objective.

This was the blank item 6 on the September meeting TO DO list.

---

## Blocked, not queued

- **Q20, the primary error function.** Open. Brief 02 added the registry seam and registered only
  the existing mixture NLL. Do not add MSE, Huber, rank or IC objectives, and do not argue for one,
  until this is answered. Note that the choice interacts with the frozen gate: under the likelihood
  the gradient to each expert is weighted by the posterior, which depends on the realised target,
  while under a point-prediction loss it is weighted by the prior alone.
- **Q18, the number of regimes K.** Open. `k_regimes` stays a config field.
- **Anything consuming out-of-sample data.** Blocked on item 8 and Q20.
