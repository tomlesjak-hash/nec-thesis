# Handoff: NEC Thesis Companion Textbook — Instructions for Fable 5

Written 2026-07-02 by Claude (Sonnet 5) for continuation in a new chat. This document plus four attached
reference files are everything needed to keep producing chapters in the exact established style. **Before
doing anything, re-list `Textbook/` and `Textbook/source/` in the project folder** — this snapshot may be
stale by the time you read it (more chapters may have been added or restructured).

## 1. Context

Tom (Physics BS → Fintech Master's, targeting quant) is writing a Master's thesis on a regime-conditional
Mixture-of-Experts model ("NEC") for stock return prediction. Rather than reading the assigned textbooks
(ESL, Bishop, Prince, Goodfellow, AFML, etc.) directly, he is having a **custom textbook built chapter by
chapter**, one chapter per module of his thesis syllabus, that merges the recommended sources into a single
non-redundant treatment aimed exactly at what the thesis needs — cutting irrelevant material, adding
worked-from-scratch derivations where sources are too terse, and tying everything back to the NEC model and
its known code defects. The end goal (his words): learn the full quant-model pipeline and the ML theory
behind it well enough to eventually design and improve his own model.

This handoff is scoped **only to the textbook-writing workflow**. It does not cover the broader project
goal of comparing/improving the three existing quant model codebases in the folder — that is a separate,
not-yet-started thread.

## 2. Files in this folder

| File | What it is |
|---|---|
| `00_INSTRUCTIONS_for_Fable5.md` | this document |
| `NEC_thesis_syllabus.md` | **source of truth.** Full thesis syllabus: every module, its reading list, and its assigned exercises. Chapters are written directly against this. |
| `NEC_math_syllabus.md` | companion roadmap mapping each syllabus module to the math it needs and why (bridges to Tom's physics background) |
| `Foundations_B_Probability_Statistics.tex` | **style exemplar A** — a Math Foundations chapter, fully up to date with the current house style (see §4) |
| `Module_4_Linear_Models_Regularization_Classification.tex` | **style exemplar B** — a Module chapter, fully up to date with the current house style |

Copy these two `.tex` files' preambles verbatim for any new chapter rather than re-deriving the macros —
they are the ground truth for every color, environment, and badge command described below. The originals
live in `Quant Model/Textbook/source/`, alongside six more finished chapters and all `.pdf` outputs.

## 3. Current state of the 8 chapters written so far

| Chapter | Pages | Exercises (core/worth/low/opt) | 4-tier priority badges? | Core exercises moved inline? |
|---|---|---|---|---|
| Foundations B — Probability & Statistics | 28 | 17 (7/3/6/1) | yes | **yes** |
| Foundations C — Optimization & Matrix Calculus | 13 | 7 (5/2/0/0) | yes | **yes** |
| Foundations D — Information & Variational | 11 | 8 (4/3/1/0) | yes | **yes** |
| Module 4 — Linear Models, Regularization, Classification | 23 | 15 (6/6/3/0) | yes | **yes** |
| Module 5 — Model Assessment & Financial Validation | 17 | 14 (9/4/1/0) | yes | **yes** |
| Module 6 — Neural Networks & Optimization | 12 | 11 (6/4/1/0) | yes | **yes** |
| Module 7 — Sequence Models | 11 | 9 (5/4/0/0) | yes | **yes** |
| Module 8 — Mixtures, EM, Variational | 11 | 10 (5/3/2/0) | yes | **yes** |
| Module 9 — HMMs & Classical Regime Switching | 19 | 10 (7/3/0/0) | yes | **yes** |

*(Tier splits regenerated 2026-07-02 by grep — an earlier version of this table had incorrect splits in every row, though totals and page counts were right. Don't trust this table blindly; regenerate with:
`for f in *.tex; do echo "$f: $(grep -c 'begin{exercise}' $f) ($(grep -c '^\\prioCore' $f)/$(grep -c '^\\prioWorth' $f)/$(grep -c '^\\prioLow' $f)/$(grep -c '^\\prioOpt' $f))"; done` in `Textbook/source/`.)*

**As of 2026-07-02, all eight chapters are uniform**: 4-tier badges everywhere (the old binary `[CORE]` tag
is gone; D/7/8 were migrated with per-exercise tier judgments logged in the changelog), all `\prioCore`
exercises inline as checkpoints (§4.4), all supplementary exercises in the end sections, all
content-audited (§4.5c), and all cross-chapter exercise references verified against the `.aux` files. New
chapters (Module 9 onward) should be born in this format — write them with inline core checkpoints from the
start rather than restructuring afterward.

## 4. House style — the exact conventions every chapter follows

Read this section together with the two attached `.tex` files; the files are the ground truth, this is the
explanation of *why* they look the way they do.

### 4.1 Chapter skeleton (in order)

1. Preamble (documentclass, packages, colors, environments, macros — copy from an exemplar, adjust only the
   chapter-numbering macro, e.g. `\renewcommand{\thesection}{5.\arabic{section}}` or `{B.\arabic{section}}`)
2. Title block naming the sources merged (e.g. "A single merged treatment of ESL §7.1–7.12, AFML Ch. 7, and
   Bailey–López de Prado (2014)")
3. `\begin{abstract}` — one paragraph: what the chapter builds and why, ending on a concrete "destination"
   sentence (what the reader can do by the end)
4. `\tableofcontents`
5. `\subsection*{How this chapter was assembled (and what was cut)}` — itemized list of every source section
   *excluded*, with a one-line reason each, plus a note on anything *added* because an exercise needed it
6. `\subsection*{Notation}` — defines every new symbol used in the chapter
7. Numbered `\section`s carrying the actual content (see §4.2)
8. A capstone section near the end, always tied explicitly to "the syllabus checkpoint" for that module —
   this is the section that ties everything together and states the closed-book self-test questions
9. `\section{Exercises}` — supplementary (non-core) exercises only, grouped under `\subsection*{On <topic>
   (Section~\ref{sec:x})}` headers (§4.4)
10. `\section*{Source map and what comes next}` — a `tabular` mapping each section to its primary source(s)
    and what it feeds into, plus a closing paragraph on where the *next* chapter picks up

### 4.2 Content sections

Within numbered sections: theorem/lemma/proof environments for anything load-bearing that the source books
assert without proof; two recurring `tcolorbox` call-outs used throughout —

- `necbox` (navy, titled "NEC connection") — ties the math just covered to the actual NEC model or its code
- `finbox` (maroon, titled "Finance reality check") — a concrete financial-data caveat or magnitude check

Depth rule (Tom's explicit, emphatic standing instruction): **do not assume prior background**, follow the
source books' full derivations rather than compressing to a "derivation route," build from the simplest
special case up to the general one (e.g. work the bivariate Gaussian in full before the general matrix
case), actually carry out each derivation rather than gesturing at it, and define every object the first
time it appears. Physics analogies (Boltzmann/softmax, free energy/log-sum-exp, variational principle,
normal modes, propagators) are welcomed and land well with Tom — use them where they genuinely illuminate,
not decoratively.

### 4.3 Exercise priority badges

Every exercise carries exactly one badge, placed on its own line right after `\begin{exercise}[...]\label{...}`:

```latex
\newcommand{\prioCore}{{\normalfont\small\textbf{\textcolor{necblue}{[\,\ensuremath{\bigstar\bigstar\bigstar}\ Core\,]}}}\hspace{0.5em}}
\newcommand{\prioWorth}{{\normalfont\small\textbf{\textcolor{finmaroon}{[\,\ensuremath{\bigstar\bigstar}\ Worth it\,]}}}\hspace{0.5em}}
\newcommand{\prioLow}{{\normalfont\small\textcolor{priogray}{[\,\ensuremath{\bigstar}\ Low priority\,]}}\hspace{0.5em}}
\newcommand{\prioOpt}{{\normalfont\small\textcolor{priogray}{[\,\ensuremath{\circ}\ Optional\,]}}\hspace{0.5em}}
```

Priority means **load-bearing-ness for the NEC/MoE thesis track, not difficulty.** A long, hard, classical
derivation the thesis never actually leans on (e.g. the QR/Frisch–Waugh pass in Module 4) is `\prioWorth` or
lower; a short exercise whose single result is directly used later (e.g. the gate gradient, the rank-IC null
variance) is `\prioCore` even if easy. State the "Importance labels" legend once, in prose, right at the top
of the Exercises section (see either exemplar file for the exact wording pattern).

This is the system that superseded an earlier crude binary `[CORE]`/no-tag scheme, which over-marked long
classical exercises as core just because they were labor-intensive. Foundations D, Module 7, and Module 8
have not been migrated to it yet (§3).

### 4.4 Core-exercise placement — the current structure (as of 2026-07-02)

This is the newest convention and the one most likely to be asked about. Structure:

- **`\prioCore` exercises live inline**, physically inserted at the end of the section or subsection whose
  content they test — not in the end-of-chapter exercises block. Insert them right before the next
  `\section{...}` (or `\subsection{...}`) begins, under a `\subsection*{Checkpoint exercise}` header. If two
  core exercises both belong at the same point (e.g. both test the same subsection), group them under **one**
  shared `\subsection*{Checkpoint exercise}` header rather than repeating the header.
- **Everything else (`\prioWorth`, `\prioLow`, `\prioOpt`) stays in the end-of-chapter `\section{Exercises}`**,
  grouped under `\subsection*{On <topic> (Section~\ref{sec:x})}` headers, in chapter order.
- The Exercises section's intro paragraph must say explicitly that core exercises are inline and this
  section holds supplementary material only (see either exemplar for exact phrasing) — don't leave stale
  language claiming "every exercise appears below."
- If moving many exercises around a large file by hand is error-prone, write a small Python script that (1)
  locates each `\begin{exercise}...\end{exercise}` block by its `\label{}`, (2) removes blocks in *reverse*
  document order so earlier indices stay valid, (3) re-inserts them grouped by target insertion point. This
  is exactly how Foundations B and Module 4 were restructured. (Exact-string editing works too if you have
  the verbatim text — that is how Module 5 was done.)

**The strict rule** (uniform across all eight chapters since 2026-07-02): *every* `\prioCore` exercise sits
inline as a checkpoint, including audit-added supplementary ones; *only* non-core exercises live in the end
section. Two core exercises anchored at the same point share one `\subsection*{Checkpoint exercise}` header
(examples: gaussmle+bootstrap in B, esl36+dfsupp in 4, icnull+icir in 5, xavier+he in 6, lstm+grulstm in 7).

**Renumbering hazard — this has bitten twice.** Later chapters cite earlier chapters' exercises by
hand-typed number (e.g. "Jensen (Foundations B, Exercise B.5)"), and both the June-22 restructuring and an
even earlier layout left ~20 stale numbers scattered across C, D, 4, 5, 6, and 8 (all fixed 2026-07-02).
Whenever anything renumbers exercises, sweep every chapter with
`grep -rnE "(Exercise|Exercises|Ex\.)[~ \\]+(B|C|D|[0-9])\.[0-9]+" *.tex` (ignore book refs like "ESL
Ex. 4.2") and fix hits against the new numbering. In new prose, cite by name plus number ("the Jensen
exercise, B.5") so a future stale number is at least self-diagnosing.

### 4.5 The three standing audits — do these for every chapter, always

**(a) Solvability audit.** Every exercise must be solvable using *only* that chapter plus earlier chapters —
never a bare citation to an un-derived result in one of the source books. If a tool is missing (an identity,
a lemma, a hint), **add it inline to the chapter text**, don't just cite the book it came from. Several
chapters have an added lemma or two purely because an assigned exercise was otherwise unsolvable (e.g.
Module 5's random-rank-moments lemma, max-of-Gaussians envelope, and Mills-ratio bound).

**(b) All-encompassing coverage audit — this is what Tom keeps asking for explicitly.** After drafting the
core exercises, list every major topic/section the chapter actually teaches in prose, and check that the
`\prioCore` set collectively touches all of them — the goal is that **doing only the core exercises is
sufficient to have practiced the entire chapter.** Where a topic is taught but has zero exercise anywhere
(core or not), either add one, or — if it's genuinely minor/contextual (e.g. Module 5's AIC/BIC aside, which
the thesis doesn't use since it standardizes on cross-validation) — explicitly note in the audit report that
it was left prose-only and why. Report this audit to Tom in plain language when delivering a chapter: which
sections have core coverage, and which prose-only gaps were judged acceptable and why.

**(c) Content audit — added 2026-07-02 at Tom's request: whenever a chapter is written *or edited*, also
verify its content.** Concretely: (i) recompute every numerical example touched or nearby (the audit that
added this rule caught a mixture-kurtosis example whose stated κ ≈ 7.7 required w = 0.95, not the w = 0.9
in the text); (ii) check each stated formula/theorem against the standard form (moments, distributions,
classical identities — most are memorizable or one-line derivable); (iii) verify claimed derivation steps
actually go through as described; (iv) run the cross-reference sweep from §4.4's renumbering-hazard note,
plus a scan for empty `\subsection*` headers left by moves (one was found in Module 4); (v) report findings
with fixes, distinguishing "wrong" from "stale" from "stylistic." Scope it to the chapters being edited
plus anything that cites them; a full-book pass is only needed if numbering changed.

### 4.6 Numbering, cross-references, and file conventions

- Sections numbered by the syllabus label: Foundations chapters use letters (`\renewcommand{\thesection}{B.\arabic{section}}`, equations `\numberwithin{equation}{section}` → B.3.1 etc.); Module chapters use the module number (`4.\arabic{section}` → equations 4.2.2 etc.)
- Exercises numbered per chapter too (`\theexercise` = `B.\arabic{exercise}` or `4.\arabic{exercise}`), but always **cited by `\label`/`\ref`, never by hand-typed number** — numbers shift when exercises move
- Later chapters cross-reference earlier ones by their real numbers, e.g. "(B.6.9)", "Module 4 §4.2" — so **never renumber or remove an existing label** without grepping every other chapter for references to it first
- **Folder convention:** only `.pdf` files live at `Textbook/` top level; all `.tex` sources and LaTeX build artifacts (`.aux`, `.log`, `.out`, `.toc`) live in `Textbook/source/`. Compile inside `source/`, then copy the resulting PDF up a level.
- **Compile procedure:** run `pdflatex -interaction=nonstopmode <file>.tex` **twice** (first pass writes labels/TOC, second pass resolves cross-references), then `cp` the PDF up to `Textbook/`.
- Exercises are **questions only** — no solutions, no worked answers, no Python labs — restated self-contained (so the reader never needs the original book open) with a source tag in the exercise heading: `[ESL Ex. 3.2]`, `[Syllabus]`, `[Supplementary]`, etc.

## 5. Step-by-step workflow for producing a new chapter

1. Read the relevant syllabus block in `NEC_thesis_syllabus.md` (and the companion entry in
   `NEC_math_syllabus.md`) end to end before writing anything; note the assigned reading and assigned
   exercises verbatim.
2. Outline the chapter against that reading list; decide what to cut and why (this becomes §4.1 item 5 of
   the finished chapter — keep notes as you go, don't reconstruct the rationale afterward).
3. Write the content sections in logical chunks (roughly half the chapter, then the other half, works well
   for context management), following the depth rule in §4.2.
4. Write exercises: pull the syllabus/source-book exercises verbatim first, adapt to be self-contained, add
   any supplementary exercises the chapter itself motivates. Assign priority badges (§4.3) as you go.
5. Run all three standing audits (§4.5): solvability, all-encompassing coverage, and content. Patch any
   gaps found — directly in the chapter text, not as a footnote promising to fix it later.
6. Place core exercises inline per §4.4, supplementary exercises in the end `\section{Exercises}`.
7. Compile (§4.6), fix any LaTeX errors, compile again.
8. Report to Tom: what was cut and why, the audit results (including any deliberately-accepted prose-only
   gaps), and present the PDF.

## 6. One open strategic question Tom has not yet resolved

Several written chapters (Module 4's Frisch–Waugh proof and Gaussian-prior derivation, Foundations C's
matrix-calculus prerequisites) defer a proof or two to a **Foundations A (linear algebra) chapter that does
not exist yet.** The standing workaround has been to patch the needed result inline in whichever chapter
needs it rather than block on writing Foundations A. Tom has not decided whether to eventually write
Foundations A properly (cleaning up these forward-references) or keep patching inline indefinitely — worth
surfacing if it comes up, not worth deciding unilaterally.

## 7. What's next, per the syllabus

At last check, the next chapters in sequence were (line numbers verified 2026-07-02; Module 9 was written
2026-07-02 and is done):

- **Math Foundations E — Special distributions and reparameterization** (incl. Gumbel-softmax / concrete
  distribution; Maddison et al. 2017, Kingma & Welling 2014) — syllabus line 776 of `NEC_thesis_syllabus.md`
- **Module 10 — Mixture-of-experts theory** (routing, load balancing) — syllabus line 807

Read the full relevant syllabus section before starting each — line numbers may have shifted if the
syllabus file has been edited since this was written.

## 8. Changelog

- **2026-07-02 (Sonnet 5):** document created.
- **2026-07-02 (Fable 5):** verified handoff `.tex` copies byte-identical to `Textbook/source/` originals;
  corrected every tier split in the §3 table (totals were right, splits were not) and added the regeneration
  one-liner; documented the exemplars' checkpoint deviations in §4.4. Restructured **Module 5** inline
  (strict rule). Coverage audit changes to Module 5: `ex:esl71`, `ex:esl710`, `ex:decile` upgraded
  Worth→Core (each was the sole exercise on a major section the core set otherwise skipped: optimism/df,
  CV-estimand hygiene, after-cost decile stack); new inline core `ex:walkforward` added (walk-forward was a
  syllabus topic with zero exercises anywhere, despite being the thesis's primary split scheme). Module 5 is
  now 17 pp, 14 exercises (9/4/1/0). Deliberate prose-only gaps, with reasons, are listed in the Exercises
  intro judgment notes and §4.5b still applies as written.
- **2026-07-02 (Fable 5, second pass):** applied the strict rule to Foundations B and Module 4 (moved
  `ex:mixkurt`, `ex:bootstrap`, `ex:dfsupp` inline; moved `ex:studentt`, `ex:klgauss`, `ex:esl34`,
  `ex:esl42` to the end sections; badges unchanged, so the §3 table splits still hold; both chapters
  renumbered as a result). Full content audit of B and 4: fixed the mixture-kurtosis example weight
  (w 0.9 → 0.95 to match κ ≈ 7.7), deleted an empty "On ridge and its posterior" header in Module 4,
  refiled ESL 2.8 from the mislabeled "MoE objective" group to the dimensionality group, and fixed ~20
  stale hand-typed cross-chapter exercise numbers across C, D, 4, 5, 6, and 8 (some dating from before the
  June-22 restructuring). Added standing audit §4.5(c) (content verification) and the §4.4 renumbering-
  hazard sweep. All eight chapters recompiled clean; handoff exemplar copies re-synced.
- **2026-07-02 (Fable 5, third pass):** Module 5 content-audited — clean, zero errors (flags only: three
  "Foundations A" citations, the §6 standing question; ESL's contested LOO-variance claim kept as sourced).
  **Module 6** restructured inline (strict rule): 6 core checkpoints (`prince5`, `backprop`,
  `xavier`+`he` shared, `bngrad`, `earlystop`), 5 supplementary at end. Coverage audit upgraded
  `ex:prince5` and `ex:earlystop` Worth→Core (the variance-head loss is the NEC experts' actual training
  objective; early-stopping patience is a registry hyperparameter — each was its section's only exercise).
  Accepted prose-only gaps: §6.1–6.2 expressiveness/depth-folding (qualitative intuition; collapse fact
  exercised in Worth/Low items), §6.6 practice recipes (derivations discharged in C.1–C.4), dropout-ridge
  fact (hedged prose). Content audit found one defect: prince5(b) cited "eq. 4.8.2" for the bounded-gradient
  fact — that is the log-likelihood; fixed to 4.8.3. Numerics verified programmatically (He half-Gaussian
  moments, BN gradient identities vs finite differences, early-stop↔ridge correspondence). No chapter cites
  Module 6 exercises by number, so the renumbering is safe. Now 11 exercises (6/4/1/0), 12 pp, compiles
  clean.
- **2026-07-02 (Fable 5, fourth pass — book-wide completion):** Foundations C, D and Modules 7, 8 brought
  to the uniform format. **C**: restructured (5 checkpoints; no badge changes); content audit fixed the
  momentum exercise's discretization hint (forward difference → backward difference for the friction term,
  which is what makes the μ = 1−γh dictionary exact); C renumbered (C.3 adam, C.4 chainrule, C.5 depthvar,
  C.6 momentum, C.7 vjp), and all 8 downstream C.x citations in Modules 6–7 updated. **D**: badges migrated
  (core: chain, fwdrev, elbo, meanfield; maxent set Worth to match its univariate parent B.15; bregman,
  gradest Worth; aep Low); restructured (4 checkpoints); two stale "(B.6.5)" refs fixed to (B.6.4); D
  renumbered (elbo D.5→D.3) and all 10 downstream D.x citations in Module 8 updated. **7**: badges migrated
  (core: bptt, vanishproof, gru, lstm, grulstm — grulstm Core because the capstone's shrink-the-encoder
  recommendation rests on it; tbptt, linrnn, birnn, milestone Worth); restructured (4 checkpoints,
  lstm+grulstm shared); fixed a confusing self-reference in the BPTT section's opening. **8**: badges
  migrated (core: lse, mstep, failures, vbpoint, moeem; moments, eslbridge, chooseK Worth; complete,
  bernoulli Low); restructured (5 checkpoints); content audit fixed the float64 underflow bound in ex:lse
  (2×745·log 2 ≈ 1033 was wrong given the e^−745 threshold the exercise itself states; now 2×745 ≈ 1490,
  ≈39σ) and two stale "Prop. D.3.2" refs (it is D.3.1). Also fixed Module 4's stale "(B.2.5)" → (B.2.1)
  for the base-rate example. Accepted prose-only-for-core gaps, with reasons, per chapter: C §C.1 landscape
  and §C.4 momentum (Adam subsumes operationally; documented in C's intro), D §D.2 coding and §D.7
  gradient-estimator preview (one-time insight / Module 10's job), 7 §7.1 interface and §7.5 overfitting
  checklist (synthesis sections; every line cross-referenced), 8 §8.1 moments-at-Worth (the leptokurtosis
  argument is Core inline in Foundations B; identifiability is exercised in ex:failures(b)). All eight
  chapters recompiled clean; final book-wide reference sweep validates every hand-typed exercise citation.
- **2026-07-02 (Fable 5, fifth pass — first new chapter):** **Module 9 — HMMs and Classical Regime
  Switching** written from scratch in the uniform format (born with inline core checkpoints; file
  `Module_9_HMMs_Regime_Switching.tex`, 19 pp, 10 exercises (7/3/0/0)). Sections: Markov
  chains/persistence (with the two-state spectral decomposition, sojourn times, and the
  indicator-autocovariance lemma added as tools), the HMM with its CI lemma proven directly from the
  factorization, forward/filtering (with scaling and the predict–update form), backward/smoothing (with
  the filtered-vs-smoothed **leakage boundary** as a protocol clause), Viterbi, Baum–Welch, Hamilton
  (1989) Markov-switching models (with the new **persistence-is-volatility-clustering proposition**:
  squared-return ACF decays at the chain's second eigenvalue — completes B.7.3's fat-tails argument), and
  the Markov-vs-feature gating comparison (degenerate-equivalence proposition + belief-filter-as-recurrent-
  gate containment). Audit findings: the syllabus's complexity claim "O(NT²) vs O(T^N)" is garbled
  (corrected to O(K²T) vs O(K^T·T), flagged in the preface); Bishop Ex. 13.1 requires d-separation, never
  built — restated via Bishop 13.2's direct method (verified against the local copy); 13.3 assigned
  alongside, 13.5–13.6 absorbed into the Baum–Welch exercise; Tsay is not in the local library and its two
  problems are unnumbered in the syllabus — two constructed in its stated spirit, flagged. Content audit:
  all four DP algorithms verified against brute-force enumeration; Baum–Welch monotonicity and parameter
  recovery verified on simulation; the clustering proposition verified against simulation at three lags;
  the Viterbi-vs-marginals counterexample in Ex. 9.4(d)'s hint verified realizable (pointwise path has
  posterior probability exactly zero) and the hint written from the verified construction. Coverage: every
  section core-touched except §9.2, whose factorization/CI content is exercised inside every core
  derivation (standalone bookkeeping is Worth-tier Ex. 9.8) — deliberate. Next chapter: Foundations E.
