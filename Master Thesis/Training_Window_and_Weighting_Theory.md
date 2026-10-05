# How much history, and how to weight it: theory notes for the training scheme

Created 2026-10-05 from the training-scheme discussion. These are study notes: the theory behind how
much past data each part of the model learns from, and how heavily. Decisions are marked
**[decided]**; open items are marked **[open]**. Decisions are also recorded in `Advisor_Questions.md`
(Q16) and in the decision register of `Progress_Tracker.md`.

---

## 0. Where this sits in the study

- **Sample:** 2000-2024 **[decided 2026-10-05]**. Compustat's GICS history starts in mid-1999, which
  makes 2000 the earliest start with sectors. Compared with 2015-2024, it roughly doubles the number of
  stress episodes: 2000-02, 2008-09, 2010, 2011, 2015-16, 2018, 2020 and 2022.
- **Window type:** expanding, never rolling **[decided]**. Older data is down-weighted rather than
  dropped (sections 4-6).
- **Machine:** Apple M4 Pro, 48 GB. The full panel fits in memory several times over.

---

## 1. The problem in one sentence

Old data may describe a market that no longer exists, which argues for forgetting it. But rare events
such as crises are exactly what a regime model must remember, which argues for keeping it.

---

## 2. Bias against variance

Let the true relationship at the forecast date be f_T, and the estimate be built from past data. The
expected squared forecast error splits into three parts:

$$
\mathbb E\big[(y-\hat f)^2\big]=\underbrace{\sigma^2_{\varepsilon}}_{\text{noise}}+\underbrace{\operatorname{Var}(\hat f)}_{\text{estimation noise}}+\underbrace{\big(\mathbb E[\hat f]-f_T\big)^2}_{\text{bias}^2}
$$

- **Variance** falls as more data is used. With equal weights on n observations it scales like 1/n.
- **Bias** appears when the relationship changed over the sample. Old data then pulls the estimate
  towards the old relationship.
- Memory is a single dial that trades one term against the other. The best setting depends on how
  large the changes are relative to the noise.

**The key fact for finance.** The signal-to-noise ratio of returns is tiny, so the variance term is
large. Discarding data is expensive, and the best memory is usually longer than intuition suggests.
That is the case for expanding windows in Gu, Kelly & Xiu (2020).

**What theory says about breaks.**

- Pesaran & Timmermann (2007) derive the window that minimises forecast error when there is a
  structural break: the trade-off above, made exact.
- Pesaran, Pick & Pranovich (2013) derive the best weights. After a break, pre-break observations get
  smaller but **not zero** weight, because they still reduce variance.
- When break dates are unknown, averaging forecasts over several windows is robust (Pesaran &
  Timmermann 2007; Clark & McCracken 2009), at a multiple of the compute.

---

## 3. Evidence that the cross-section of returns changes

- **Publication decay:** anomaly returns fall by about a quarter out of sample and by about half after
  publication (McLean & Pontiff 2016).
- **Fewer predictors since 2003:** the number of characteristics with independent predictive power
  fell sharply (Green, Hand & Zhang 2017).
- **Cheaper trading:** anomalies weakened after decimal quotes (2001) and in more liquid markets
  (Chordia, Subrahmanyam & Tong 2014).
- **Market structure:** electronic trading, Reg NMS (2007), and the rise of passive and ETF ownership.

So the bias term is real for the **cross-sectional** parts of the model (the base and the experts).

---

## 4. Exponential down-weighting (forgetting)

Give an observation that is a periods old the weight

$$
w_a=\rho^{\,a},\qquad 0<\rho<1,\qquad \rho=2^{-1/H}
$$

where H is the **half-life**: after H periods the weight is one half.

Two ways to summarise how much data the weights keep:

1. **Effective memory length** (as in Nystrup et al.), with forgetting factor lambda = rho:
   $$
   N_{\text{eff}}=\frac{1}{1-\lambda}
   $$
   This is the total weight, i.e. how many full-weight periods the weights add up to.
2. **Kish effective sample size**: how many equally weighted, independent observations the weights
   are worth for variance:
   $$
   \text{ESS}=\frac{\big(\sum_a w_a\big)^2}{\sum_a w_a^2}\ \xrightarrow{\ \text{long sample}\ }\ \frac{1+\rho}{1-\rho}
   $$

**The flaw of calendar-time forgetting for a regime model.** Weights fall with calendar age, whatever
happened. A stress expert has perhaps ten episodes in 25 years. With a half-life of a few years, 2008
is worth almost nothing by 2018, so the stress expert's data mostly disappears. The "important events"
are erased along with the stale ones.

---

## 5. Regime-clock decay (the construction adopted for the experts)

**Idea.** Measure an observation's age not in calendar days, but in **how much new experience of the
same regime has accumulated since**. Calm days are common, so calm experience ages fast. Stress days
are rare, so stress experience ages slowly.

**Definition.** For a training row from date s, used in a model fitted at date t:

$$
w_s(t)=\sum_{k=1}^{K}\xi_{s\mid s}(k)\,\rho^{\,n_k(s,t)},\qquad
n_k(s,t)=\sum_{u=s+1}^{t}\xi_{u\mid u}(k),\qquad \rho=2^{-1/H}
$$

- The first factor in each term is the gate's filtered probability that date s was in regime k.
- n_k(s,t) is the expected number of regime-k days between s and t, counted with filtered
  probabilities, which are known at each u. There is no look-ahead.
- H is one half-life, measured in **regime-days**, shared by all regimes.

**Why it resolves the dilemma.**

- In a long calm spell, calm data turns over quickly (recency, so drift is tracked), while the last
  crisis keeps its weight, because no new crisis experience has replaced it.
- After a cluster of recent crises, older crisis data fades, because there is now newer crisis
  experience.
- Nothing is hand-labelled as a "crisis": the gate's own probabilities do the work.

**Calendar-time equivalents.** A regime that occupies a share p of days has a calendar half-life of
about H/p days. For the 2-state gate fitted on the CRSP S&P 500 index 2015-2024 (about 70% calm, 30%
stress) and H = 500 regime-days:

| Regime | Share of days | Calendar half-life |
|---|---|---|
| Calm | 0.7 | about 714 trading days, about 2.8 years |
| Stress | 0.3 | about 1,667 trading days, about 6.6 years |
| A rare crisis state (K = 3) | 0.05 | about 10,000 trading days, about 40 years: effectively never forgotten |

**In the likelihood.** Each row's term in the per-row composite likelihood (Q24) is multiplied by its
weight:

$$
\ell_{\text{row}}^{w}=\sum_{t}\sum_{i}w_t\,\log\sum_k\pi_k(t)\,\mathcal N\big(r_{i,t+1};\mu_k(x_{i,t}),s_k^2\big)
$$

This is a weighted composite likelihood, so the arguments of Appendix D of the formulation PDF carry
over. Each term still has a zero-mean score; the weights change efficiency, not consistency.

**Variant to keep in mind.** The decay could be applied per expert, so that expert k's gradient on
row s is weighted by rho to the power n_k(s,t) only, instead of by the mixture above. The mixed form
above is simpler and is the default proposal; the per-expert form is closer to Tom's original wording
in Q16 (e). **[open, minor]**

**Origin and novelty.** The idea is Tom's (Advisor_Questions Q16 (e), 2026-09-26): "each expert's
decay clock ticks only while its own regime is active". Exponential forgetting is standard. The
closest finance precedent for weighting history by regime rather than by calendar time is Mulliner,
Harvey, Xia, Fang & Van Hemert (2025), who weight past dates by the similarity of their economic state
variables to today's. Measuring age in regime time with the gate's filtered probabilities is, as far
as I know, this study's own construction. In the thesis, present it
as a design choice motivated by sections 2-4, not as established practice.

---

## 6. One memory per component

The three parts of the model face different trade-offs, so they get different memories.

| Component | What it estimates | Bias risk | Variance risk | Memory |
|---|---|---|---|---|
| **Experts** | regime-specific cross-sectional mappings | moderate | very high for rare regimes | **regime-clock decay [decided 2026-10-05]** |
| **Base** | the average cross-sectional mapping | highest: structural drift shows here | low: all data informs it | **calendar exponential decay, long half-life [decided 2026-10-05]** |
| **Gate** | regime dynamics of the market series | real: parameters drift (section 7) | very high for transitions into rare regimes | **regime-clock forgetting via weighted Baum-Welch [decided 2026-10-05]** (section 7) |

---

## 7. The gate's memory **[decided 2026-10-05: option C]**

### 7.1 Tom's challenge

A full-history gate ignores the bias-variance trade-off, and the gate matters most of all: it decides
which expert speaks.

### 7.2 What the evidence says

Nystrup, Madsen & Lindström (2017) fit a 2-state Gaussian HMM to daily S&P 500 returns from 1928 to
2014, with exponential forgetting of the likelihood.

- They chose an effective memory of N_eff = 250 days, described as close to the shortest memory that
  still balances speed of adaptation against noise.
- The HMM's parameters vary over time far beyond their confidence intervals.
- Adaptive estimation gives better one-step density forecasts, **but that result holds after leaving
  out the 20 most negative contributions**. On the worst days, the crashes, fast forgetting does not
  win.

So the gate's parameters do drift, which supports Tom's challenge. But calendar forgetting fails
exactly on the rare extreme days the gate exists for, which supports the rare-event argument.

### 7.3 The failure mode to avoid

During a long calm spell, such as 2012-2019, a short-memory 2-state HMM can stop representing stress
at all. It splits "calm" into "very calm" and "slightly less calm", and when March 2020 arrives there
is no stress state to switch into. The labels drift, and the regimes lose their meaning.

### 7.4 Options

| | Gate memory | Adapts to drift | Keeps the stress definition | Cost |
|---|---|---|---|---|
| A | full history, equal weights | no | yes | none |
| B | calendar forgetting (Nystrup et al.) | yes | no: section 7.3 risk | weighted Baum-Welch |
| C | **regime-clock forgetting** (section 5 applied to the gate) | yes, for common regimes | yes | weighted Baum-Welch, two passes |

**How C works (weighted Baum-Welch).** Baum-Welch is the EM algorithm for HMMs (PDF Section 5).

1. An unweighted Baum-Welch pass on the training block gives regime probabilities.
2. Compute the regime-clock weights from them.
3. Re-run Baum-Welch with weighted M-steps:
   - each regime's mean and variance use its days weighted by that regime's own clock;
   - the transition probabilities out of regime j use transitions weighted by j's clock.

This is in-sample estimation on the training block, like ordinary Baum-Welch. Only the forward
filter is run after it.

### 7.5 How the gate's memory would be chosen: the optimisation Tom asked for

The gate **is** optimised: its parameters maximise the likelihood of the market series. The memory
(and K, Q18) are hyperparameters that likelihood cannot choose by itself, because more memory always
fits the training block differently. Two possible criteria:

1. **The gate's own job: one-step predictive log-likelihood of the market series on the validation
   block** (Nystrup et al.'s criterion). It is a proper scoring rule, it is cheap (no mixture
   training), it keeps the gate fitted separately and frozen (Q19), and it gives every gate in the
   comparison the same tuning budget. Report the worst-day contributions separately (section 7.2).
2. **Downstream:** the mixture's validation loss. It matches the end goal, but it needs a full mixture
   training per gate setting, it risks tuning the gate to the experts' noise, and it makes the
   cross-gate comparison unfair unless every gate gets the same budget.

**Decided 2026-10-05:** option C (regime-clock forgetting), with its half-life chosen by criterion 1;
criterion 2 is reported only as a diagnostic.

---

## 8. Tuning discipline (all memories)

- Each half-life is chosen from a **small preset grid**, for example: equal weights (no decay),
  10-year and 5-year calendar equivalents.
- The choice is made **once, in the pilot, on the early folds' validation blocks only**, then frozen
  and pre-registered (Q15).
- Equal weighting always stays in the grid as the reference, so "decay does not help" is a reportable
  outcome.
- Weights cost no extra compute per run. Only the pilot's selection runs cost compute.

---

## 9. Still open in the training scheme

- **Decided 2026-10-05:** test 2010-2024 with annual refits (15 folds); a pilot on validation years 2007-2009
  chooses every setting once with a regime-balanced validation loss; the main study trains on full blocks
  with frozen settings (see Q16).
- Warm-starting from the previous fold, training on every day or every fifth day, seeds, and how
  training settings are tuned.

---

## References

- Chordia, T., Subrahmanyam, A. & Tong, Q. (2014), *Journal of Accounting and Economics* 58(1).
- Clark, T. E. & McCracken, M. W. (2009), *International Economic Review* 50(2).
- Green, J., Hand, J. R. M. & Zhang, X. F. (2017), *Review of Financial Studies* 30(12).
- Gu, S., Kelly, B. & Xiu, D. (2020), *Review of Financial Studies* 33(5).
- Kish, L. (1965), *Survey Sampling*.
- McLean, R. D. & Pontiff, J. (2016), *Journal of Finance* 71(1).
- Mulliner, A., Harvey, C. R., Xia, C., Fang, E. & Van Hemert, O. (2025), "Regimes", *Journal of
  Portfolio Management* 52(4), 6-25.
- Nystrup, P., Madsen, H. & Lindström, E. (2017), "Long memory of financial time series and hidden
  Markov models with time-varying parameters", *Journal of Forecasting* 36(8).
- Pesaran, M. H., Pick, A. & Pranovich, M. (2013), *Journal of Econometrics* 177(2).
- Pesaran, M. H. & Timmermann, A. (2007), *Journal of Econometrics* 137(1).
