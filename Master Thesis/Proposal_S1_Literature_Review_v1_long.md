# Section 3 — Literature Review

**Frontier, current status, challenges and significance of the research topic**

*Draft 1 — prepared 2026-09-11. Target: ≥4,000 words (master's, English). In-text citations are numbered in order of first appearance; the reference list follows the section. A verification checklist that is NOT part of the submission appears at the very end.*

---

## 3.0 Framing

This thesis sits at the intersection of two research traditions that, without contact, converged on the same mathematical object within two years of each other. In 1989, Hamilton introduced the Markov-switching autoregression to econometrics, in which an unobserved discrete state evolves as a Markov chain and governs which of several parameterisations generates the data [1]. In 1991, Jacobs, Jordan, Nowlan and Hinton introduced the mixture of experts to machine learning, in which a gating network produces input-dependent weights over several specialised predictors whose outputs are combined [2]. Both are conditional mixtures. They differ in one respect that turns out to be decisive: the econometric object places a *stochastic process* over the mixing variable, endowing it with persistence, transition dynamics and an implied duration distribution, while the machine-learning object computes the mixing weights as a *deterministic function of the current input*, with no memory of the previous mixing decision at all.

Within four years the two traditions had touched. Weigend, Mangeas and Srivastava trained nonlinear gated experts on time series and framed the result explicitly as the discovery of regimes, comparing the architecture to hidden Markov models [3]. Bengio and Frasconi introduced the input–output hidden Markov model [4], in which transition probabilities are themselves a softmax function of exogenous inputs — simultaneously a mixture of experts with memory and a Markov-switching model with covariate-driven transitions. The contact did not persist. The modern mixture-of-experts literature, revived at scale after 2017, inherited the input-dependent router and discarded the temporal structure entirely.

The empirical asset pricing literature, meanwhile, adopted neural networks for the cross-section of expected returns without adopting either [5]. This section reviews the five bodies of work this thesis draws on — regime switching in financial econometrics, the mixture-of-experts architecture, machine learning for the cross-section of returns, the recent cluster of regime-aware neural models, and evaluation methodology under non-stationarity — and identifies the specific question that falls between them.

---

## 3.1 Regime switching in financial econometrics

### 3.1.1 The canonical model and its extensions

Hamilton's formulation supposes an unobserved state variable *s<sub>t</sub>* ∈ {1, …, K} following a first-order Markov chain with transition matrix *P*, where *P<sub>ij</sub>* = Pr(*s<sub>t</sub>* = *j* | *s<sub>t−1</sub>* = *i*), and observations drawn from a state-dependent density [1]. Because the state is latent, inference proceeds through a recursive filter that updates the state probability each period by combining the predictive density of the observation under each state with the prior implied by the transition matrix and the previous filtered probabilities. The filtered probability vector ξ<sub>t|t</sub> is the object of interest: it is the model's real-time belief about which regime currently prevails.

Two structural properties of this filter matter for the present thesis. First, the filtered probability is *persistent by construction*: because the prior is propagated through *P*, a single anomalous observation cannot move the belief far when the diagonal elements of *P* are large. Under a plausible calibration with a staying probability of 0.99 and two volatility states, a single daily return must be several times the calm-state standard deviation before the filter flips — the model deliberately ignores isolated shocks and responds only to accumulated evidence. Second, the implied duration of a regime is *geometric*: Pr(*D* = *d*) = *P<sub>jj</sub><sup>d−1</sup>*(1 − *P<sub>jj</sub>*), which is memoryless. The hazard of leaving a regime is the same on day one as on day two hundred. These two properties — genuine persistence, but a memoryless exit hazard — define both the strength and the principal limitation of the classical model.

Extensions have addressed each of these in turn. Hamilton and Susmel, and Gray, embedded conditional heteroskedasticity within a switching framework, addressing the observation that volatility clustering and regime persistence are easily confounded [6, 7]. Filardo, and Diebold, Lee and Weinbach, independently introduced time-varying transition probabilities (TVTP), in which *P<sub>ij</sub>*(*z<sub>t</sub>*) is a logistic or softmax function of observable covariates *z<sub>t</sub>*, so that the probability of switching responds to economic conditions rather than remaining constant [8, 9]. Durland and McCurdy relaxed the memoryless duration assumption by allowing transition probabilities to depend on the time already spent in the current regime, a duration-dependent model that anticipates the hidden semi-Markov literature [10]; fully explicit-duration hidden semi-Markov models have since been applied to financial return series and shown to reproduce stylised facts that the geometric-duration model cannot [11].

A separate and more recent line replaces likelihood-based estimation altogether. Bemporad, Breschi, Piga and Boyd formulate the fitting of piecewise models with persistent state sequences as an optimisation problem penalising state changes directly, solved by coordinate descent alternating between closed-form parameter updates and a dynamic-programming pass over state sequences [12]. Nystrup, Kolm and Lindström developed this into the *statistical jump model* for financial regime identification, where the objective minimises within-state squared deviation plus a penalty λ on each switch [13]. The construction has an attractive property for experimental work: at λ = 0 the model degenerates to k-means clustering with no persistence whatsoever, and as λ grows the fitted state sequence becomes arbitrarily persistent, so that persistence becomes a continuously tunable experimental variable rather than a modelling assumption. A continuous variant relaxes the hard state assignment to simplex-valued weights, and has been applied to asset-specific regime forecasting for dynamic allocation [14].

A third line abandons parametric state densities entirely. Horvath, Issa and Muguruza cluster market states by the Wasserstein distance between the empirical distributions of returns over rolling windows, a model-free approach that in the univariate case reduces to comparing sorted quantile functions and is robust to the heavy tails that undermine Gaussian state densities [15]. Finally, the smooth transition autoregressive family replaces the discrete latent state with a deterministic logistic function of an observable threshold variable, producing continuous regime weights without any latent process at all [16]; this provides a useful interpretable floor against which latent-state models can be compared.

### 3.1.2 What this literature has established

Several findings are robust enough to be treated as settled. Mixtures over regimes reproduce the fat tails, skewness and excess kurtosis of financial returns without recourse to exotic innovation distributions, because a mixture of two Gaussians with different variances is itself leptokurtic. Cross-asset and cross-country correlations rise sharply in market downturns, a phenomenon documented by Longin and Solnik using extreme value theory and shown to be inconsistent with a constant-correlation multivariate normal [17]; regime switching accommodates it naturally. Ang and Bekaert showed that an investor who ignores regimes in international allocation forgoes an economically meaningful amount of certainty-equivalent wealth, establishing that regimes are not merely a descriptive convenience but carry allocation value [18].

The single most important finding for the present thesis concerns predictability itself. Henkel, Martin and Nardari document that the short-horizon predictability of equity returns by standard valuation ratios is strongly state-dependent: it is essentially absent during expansions and substantial during recessions [19]. This is a direct statement that the mapping from predictor variables to expected returns is not stable across regimes — which is precisely the premise that a regime-conditional mixture of predictors is designed to exploit. If the relation between characteristics and subsequent returns were stable, there would be no reason to condition a predictor on market state at all.

### 3.1.3 What remains contested

Three difficulties recur. First, the number of regimes is not testable by standard means: under the null of a single regime, the transition probabilities are unidentified nuisance parameters, so the likelihood ratio statistic does not have its usual asymptotic chi-squared distribution [20]. Non-standard procedures exist but are computationally demanding, and applied work conventionally fixes the number of states at two or three and reports sensitivity. This is a methodological constraint the present thesis inherits and must address explicitly rather than ignore.

Second, regime-conditional means are poorly identified. A regime that occupies ten per cent of a thirty-year daily sample still supplies relatively few effective observations for estimating a conditional mean, and the estimation error in those means propagates into any downstream decision. Guidolin and Timmermann show that parameter uncertainty in multivariate regime-switching models is substantial, though they also find that the cost of ignoring regimes typically exceeds the cost of estimating them imprecisely [21].

Third — and this is the observation on which the thesis is built — real-time regime inference is imperfect in a specific and documented way. In their survey of the field, Ang and Timmermann note that filtered regime probabilities at times miss an important regime change and at other times issue false alarms, complicating the real-time tracking of market states [22]. This is not a criticism invented here; it is a limitation named in a leading survey of the literature. The thesis takes this statement as its premise and asks whether alternative regime mechanisms, which differ precisely in how they trade off responsiveness against stability, differ measurably when used as the conditioning device of a predictive model.

---

## 3.2 The mixture of experts

### 3.2.1 Origins and the classical period

The mixture of experts decomposes a prediction into ŷ = Σ<sub>k</sub> *g<sub>k</sub>*(**x**) *f<sub>k</sub>*(**x**), where the *f<sub>k</sub>* are expert networks and the gating function *g* produces weights on the simplex. Jacobs, Jordan, Nowlan and Hinton's central insight was not the architecture but the error function [2]. Under a naive squared-error objective on the combined output, experts learn cooperatively: each learns to correct the residual left by the others, producing entangled specialists that must be used together. Under an error function in which each expert is evaluated on its own output and weighted by its gate probability, experts compete: each is driven toward the region of input space where it already performs best, and localisation emerges without supervision. Expert specialisation is therefore a consequence of the objective, not of the architecture, which is a point of direct relevance to any claim that learned experts correspond to interpretable market states.

Jordan and Jacobs extended the model hierarchically and derived an expectation-maximisation algorithm for it [23]. The E-step computes, for each observation, the posterior probability that each expert generated it — a quantity they call the responsibility. This object is formally a regime posterior. The difference between it and Hamilton's filtered probability is exactly one term: the mixture-of-experts responsibility uses a prior that depends only on the current input, whereas the Markov-switching filtered probability uses a prior propagated through a transition matrix from the previous period. The two literatures were, at this point, one substitution apart.

Bengio and Frasconi made the substitution [4]. Their input–output hidden Markov model specifies Pr(*s<sub>t</sub>* = *j* | *s<sub>t−1</sub>* = *i*, **u**<sub>*t*</sub>) = softmax(φ<sub>*i*</sub>(**u**<sub>*t*</sub>; *W*)) — transition probabilities driven by exogenous inputs through a neural network. This is, term for term, the time-varying transition probability model of Filardo [8] and of Diebold, Lee and Weinbach [9], written in the same year by a different research community. Yuksel, Wilson and Gader's survey of the two decades that followed documents a rich body of sequential and hidden-Markov mixture-of-experts variants that were developed and then largely forgotten when the architecture was revived for a different purpose [24].

### 3.2.2 The modern revival and its different objective

Shazeer and co-authors reintroduced the mixture of experts at scale, using noisy top-*k* gating to activate only a few of thousands of experts per input, with an auxiliary load-balancing loss to prevent the router from collapsing onto a small subset [25]. Fedus, Zoph and Shazeer simplified this to top-1 routing with a capacity factor [26], and Zoph and co-authors catalogued the stability pathologies of sparse expert models and the remedies for them [27]. Puigcerver and co-authors subsequently showed that fully soft expert combinations avoid several of the optimisation difficulties of discrete routing [28].

It is essential to be precise about what this literature was optimising, because the objective does not transfer. Sparse routing exists to increase parameter count while holding per-token computation approximately constant: it is a device for scaling capacity under a compute budget. In a setting with billions of training tokens, the statistical cost of estimating the router is negligible relative to the capacity gained. The cross-section of equity returns is the opposite regime. He and co-authors provide the first rigorous statistical treatment of mixture-of-experts estimation, deriving an oracle risk bound in which the router-estimation cost scales as *O*((*M·d*/*n*) log(*M·d·n*)) for *M* experts, input dimension *d* and sample size *n* [29]. This term vanishes at language-model sample sizes; at equity-panel sample sizes, with a low signal-to-noise ratio, it is a first-order concern. The same analysis yields three further results used in this thesis: top-1 routing attains oracle risk when a dominant specialist exists and the partition boundary is linearly separable but incurs irreducible error when the best local predictor is itself a mixture, which argues for soft gating; linear gates incur constant error on curved decision boundaries whereas quadratic or kernel gates do not, which motivates covariate-driven transition functions; and always-active shared experts reduce the learning burden placed on the routed experts, which provides theoretical support for residual parameterisations.

This last point connects to a design that has since become common in large models. Shared-expert isolation, introduced in DeepSeekMoE, retains one or more always-on experts alongside the routed ones so that knowledge common to all inputs need not be replicated across specialists [30]. The shared expert there is trained jointly with the rest of the network. A distinct variant, borrowed from residual learning [31] and zero-initialised residual branches [32], pre-trains a base predictor, freezes it, and constrains the experts to produce corrections that are zero at initialisation. The two are frequently conflated but differ in what they identify: a frozen base makes the expert contribution a separately measurable quantity, whereas a jointly trained shared expert does not.

### 3.2.3 The discarded dimension

Ghahramani and Hinton derived switching state-space models as the fully dynamical generalisation of the mixture of experts, with both the experts and the gating network made recurrent and the switch variable given Markov dynamics [33]. The identity between switching models and dynamical mixtures of experts has therefore been in the literature for twenty-five years and cannot be claimed as a contribution. What can be observed is that the branch was not taken. The architectures that scaled after 2017 are memoryless routers: the gate sees the current input and nothing else. The temporal structure that the econometric literature spent thirty years refining — persistence, transition dynamics, duration — is absent from the modern architecture by inheritance rather than by demonstration that it is unnecessary.

---

## 3.3 Machine learning for the cross-section of expected returns

### 3.3.1 The factor zoo and three responses

Harvey, Liu and Zhu catalogued hundreds of published cross-sectional return predictors and argued that conventional significance thresholds are indefensible under the implied multiple testing, proposing substantially raised hurdles [34]. The literature responded along three lines that remain the main options today.

The first is shrinkage. Kozak, Nagel and Santosh argue that the correct response to a proliferation of weakly informative predictors is not selection but regularisation toward a small number of dominant principal components, and construct a stochastic discount factor from shrunk characteristic portfolios [35]. The second is conditional structure. Kelly, Pruitt and Su's instrumented principal components analysis allows factor loadings to be functions of observable firm characteristics while keeping the factors latent, which substantially improves fit relative to static loadings and separates the question of what the factors are from the question of which firms are exposed to them [36]. The third is nonlinearity. Gu, Kelly and Xiu conducted the definitive comparative study, evaluating linear models, penalised regressions, dimension reduction, regression trees, random forests, gradient boosting and neural networks of one to five hidden layers on a common characteristic panel under a fixed rolling train–validate–test protocol [5]. Chen, Pelger and Zhu subsequently embedded deep learning within a no-arbitrage restriction, using adversarial training to find the most mispriced portfolios [37]; Kelly and Xiu's survey synthesises the field [38].

### 3.3.2 What this literature has established, and the size of the prize

Three results recur across studies and set the expectations against which any new architecture must be judged. Machine learning methods outperform linear benchmarks out of sample, and neural networks are modestly the best of them. Performance peaks at shallow depth: in Gu, Kelly and Xiu's architecture ladder, a three-hidden-layer network of width 32–16–8 was the best configuration, with performance declining monotonically for deeper networks, a pattern the authors attribute to small effective sample size and low signal-to-noise ratio [5]. And the effect size is small in absolute terms — their best stock-level out-of-sample *R*² is approximately 0.40% at the monthly horizon.

That last number deserves emphasis in a proposal, because it calibrates everything that follows. The best available models explain well under one per cent of the variation in individual monthly stock returns. Any reported improvement is a small improvement on a small number, and any result substantially larger than this benchmark should be treated as evidence of a data leak or an evaluation error rather than as a discovery. It also implies that additional model flexibility is as likely to hurt as to help, which is the central tension in applying a capacity-increasing architecture such as a mixture of experts to this problem.

### 3.3.3 The unexamined assumption

Nearly every paper in this literature names non-stationarity in its motivation, and almost none models it. The universal treatment is rolling or expanding re-estimation: the model is refitted periodically on a moving window, and the resulting parameter drift is taken to accommodate structural change. This procedure is itself an implicit regime model — but a peculiarly impoverished one. It has a single hard-coded timescale, set by the window length rather than estimated. It has no recurrence: the model cannot recognise that current conditions resemble a period two decades ago more closely than they resemble last month, because the earlier period has fallen out of the window. And it provides no state variable that can be inspected, tested against external classifications, or used to condition anything.

The field's default answer to regime change is thus unstructured, unexamined and, to our knowledge, never compared against structured alternatives on this task. That gives the thesis a target which is simultaneously the standard practice and, on its face, the weakest available approach.

Two further questions remain contested and must be acknowledged. Whether the documented gains survive realistic transaction costs and the multiple-testing correction implied by the number of architectures tried is disputed. And Grinsztajn, Oyallon and Varoquaux show that tree-based ensembles continue to outperform deep learning on typical tabular data of this size and character, which firm characteristic panels resemble closely [39] — a finding that obliges any neural study on this data to include a gradient-boosting baseline rather than only linear and neural comparisons.

---

## 3.4 Regime-aware neural architectures: the current frontier

The period from 2024 to 2026 has produced a dense cluster of models that attempt to make neural forecasters regime-aware. This is the frontier against which the thesis must be positioned, and it must be described without self-generosity: the claim that mixtures of experts have not been applied to financial prediction, or that Markov-gated mixtures are new, is false and easily checked.

### 3.4.1 What has been built

The closest precedent on methodology is the regime-gated residual mixture of experts of Ye and Borde [40]. Working on cross-sectional five-day realised volatility for 1,027 US equities with replication on 1,552 Japanese stocks, they hold the regime variables, forecasting backbone, parameter budget, hyperparameter selection procedure and evaluation protocol fixed, and vary only the point at which regime information enters the network. Their finding is unusually clean. When the regime variables are appended to the forecasting input, training collapsed in 30 of 30 random seeds on the US panel and 28 of 30 on the Japanese panel; when the identical variables were supplied only to the gating network, no seed collapsed. Soft routing outperformed four distinct hard-routing schemes, including learned top-1 assignment, volatility-quantile partitions and GICS sector partitions. The gains were attributable principally to the residual parameterisation — a frozen base predictor plus zero-initialised correction experts — rather than to the number of experts, which was nearly irrelevant between two and six.

Three further studies target returns rather than volatility. RAVEN routes three-layer Transformer encoders operating on nested look-back windows, selecting among them by a learned patch-importance score, and the authors describe the result as adaptive regime awareness without explicit regime labels [41]. MIGA combines sequence experts with group aggregation and inner-group attention for stock ranking on Chinese indices, and documents that experts specialise by market capitalisation and by long versus short exposure without any supervision directing them to do so [42]. PRISM-VQ learns a discrete codebook of cross-sectional market archetypes by vector quantisation and uses the resulting code as the routing signal for a structure-conditioned mixture of experts producing time-varying factor loadings [43]. A fourth, an adaptive financial Transformer with regime-gated attention, uses a learned encoder over grouped technical indicators to bias attention weights on a single-asset return forecasting task [44].

A separate line places a filtering recursion in the gate. MoE-F formulates expert selection as a finite-state continuous-time hidden Markov model and derives closed-form filtering recursions for combining expert forecasts, with optimality guarantees, evaluated partly on financial market movement [45]. The distinction from the present thesis is precise and should be stated rather than elided: their latent state is over *which expert is currently performing best*, inferred from the experts' own running performance, not over a market regime inferred from market data. The state space describes the model, not the world. Related earlier work embeds hidden Markov transition structure inside recurrent networks [46], and a hidden Markov model has been used directly as the gating model of an expert mixture in a biomedical decoding application [47], establishing that the construction itself is not new in any field.

### 3.4.2 What this frontier has and has not settled

Three things are now reasonably well supported. Where regime information enters a network is decisive, and the gate is the right place for it [40]. Soft combination outperforms discrete assignment, consistently and across several partitioning schemes [40], in agreement with the theoretical result that top-1 routing is suboptimal when the best local predictor is itself a mixture [29]. And expert specialisation along economically interpretable lines can emerge without supervision [42], though these interpretability claims are asserted far more often than they are formally tested.

What has not been settled is the question this thesis asks. Every architecture surveyed above implements regime awareness as a *learned router over instantaneous features*. Ye and Borde's regime state variable, despite the name of the architecture, consists of two rolling volatility statistics — a twenty-day volatility of the equal-weighted market return and a twenty-day volatility of the residual return — with no latent state, no transition matrix, no persistence parameter and no filtering recursion [40]. PRISM-VQ's codes are assigned independently each day, and the paper measures their month-to-month persistence as an outcome rather than imposing it [43]. RAVEN states explicitly that it operates without regime labels [41]. In none of these is the gate the output of a regime-generating *process*.

The consequences of this are visible in the results. Ye and Borde decompose their forecasting advantage by market condition and find it concentrated in sustained high-volatility periods — roughly four times the full-sample advantage in the top volatility decile and nearly seven times during the 2020 crisis — but *below* the full-sample advantage in windows surrounding regime transitions themselves [40]. Their own interpretation is that the benefit of residual routing is greatest during sustained periods of elevated volatility rather than during transitions. That is the failure mode one would predict for a gate built from rolling statistics: such a gate reports where the market currently is, not whether it is about to leave. Their limitations section names alternative forms of market state as future work; an independent regime-gated architecture published in the same year lists the integration of Gaussian-mixture hidden Markov regime classification as future work [44]. The gap identified here is therefore not merely unoccupied but has been signposted by the papers occupying the adjacent cells.

A related diagnosis has been stated elsewhere: a recent drift-aware mixture-of-experts paper identifies memoryless routing as a limitation of existing architectures for adapting to abrupt regime shifts, though its proposed remedy — drift detection with dynamic instantiation of new experts — differs entirely from the approach taken here [48]. The observation that routing is memoryless is therefore already in print and cannot be claimed as an original diagnosis; what remains open is the comparison of remedies drawn from the regime-switching literature. Adjacent work reinforces the point from two directions: market-guided Transformers condition a single predictor on market-level information without any mixture structure [49], while volatility-sensitive expert mixtures partition the universe by asset-level volatility classification rather than by a market state process [50]. Neither introduces transition dynamics into the conditioning variable.

Finally, a methodological criticism applies to most of this cluster and is made most forcefully by Ye and Borde themselves: papers typically change the regime representation, the routing mechanism and the model capacity simultaneously, so that it cannot be determined which change produced the reported improvement [40]. Any contribution in this area must therefore be built as a controlled comparison or it will inherit the same objection.

---

## 3.5 Evaluation under non-stationarity

The methodological literature imposes requirements that shape the research design rather than merely the reporting. Harvey, Liu and Zhu's multiple-testing argument applies with equal force to architecture search: trying many model variants and reporting the best is the same statistical error as mining many predictors [34]. Bailey and López de Prado formalise this for performance statistics through the deflated Sharpe ratio, which adjusts for the number of trials and the non-normality of returns [51]. López de Prado's treatment of purged cross-validation and embargo periods addresses the leakage that arises when overlapping label windows allow information from a test period to influence training [52]; with multi-day forward return targets, this leakage is a certainty rather than a risk unless explicitly purged.

Formal tests of forecast accuracy require standard errors robust to the serial correlation induced by overlapping targets, which is the Diebold–Mariano test [53] with Newey–West covariance estimation [54]. And the recent literature adds one further requirement that the older methodological work does not: training stability must be reported as a first-class result, not as a footnote. In Ye and Borde's study the differences in mean accuracy between competing architectures were small and the differences in the proportion of seeds that failed to train were enormous [40]. A single-seed result in this domain is uninterpretable; distributions across many seeds are the minimum acceptable reporting standard.

One caution applies specifically to the present thesis. Ye and Borde define training collapse through a volatility-specific loss function, QLIKE, which is undefined for a signed return forecast and which diverges precisely when a volatility forecast approaches zero. That failure mode does not exist for cross-sectional returns. Whether the stability advantage of residual routing transfers from a volatility target to a signed return target is therefore an open empirical question, and this thesis must define its own stability criterion — dispersion of rank information coefficient across seeds, and the count of seeds falling below a stated threshold — rather than importing one that does not apply.

---

## 3.6 Challenges, gap and significance

**Challenges.** Three difficulties define the problem. The data are non-stationary in a way that is universally acknowledged and almost never modelled. The signal-to-noise ratio is so low that the best available models explain under one per cent of the variation in monthly individual stock returns [5], so that any additional model flexibility is as likely to degrade out-of-sample performance as to improve it. And the two most relevant literatures disagree about the object at the centre of the problem: econometrics treats the market state as a latent stochastic process to be filtered, while machine learning treats it as a deterministic function of the current input to be learned.

**Gap.** Regime switching and the mixture of experts are the same conditional-mixture object, developed in parallel and briefly joined in the mid-1990s [3, 4]. The econometric branch refined how the latent state evolves — persistence, covariate-driven transitions, explicit durations, penalty-based and distributional alternatives to likelihood estimation — but attached these mechanisms almost exclusively to linear or low-dimensional predictors. The machine-learning branch refined how specialists are combined but discarded the temporal structure, retaining a router that sees only the current input. The empirical asset pricing literature adopted neural predictors without adopting either, handling non-stationarity through rolling re-estimation. The 2024–2026 neural cluster sits at this junction and, in every instance surveyed, reaches for a learned router over instantaneous features. No existing study compares structurally distinct regime-switching processes — classical Markov switching, penalty-based jump models, covariate-driven transition probabilities, and model-free distributional clustering — as the gate of a neural mixture of experts, on the cross-section of equity returns, under matched expert capacity and a common walk-forward evaluation protocol.

**Significance.** The importance of this question follows from the statistical structure of the problem rather than from novelty alone. He and co-authors show that the cost of estimating a router is not negligible at finite sample size [29]; at equity-panel dimensions it is first-order. A regime-switching process is precisely a *restriction* on the class of admissible gates — it constrains the mixing weights to evolve with persistence and with a specified transition mechanism instead of varying freely with the current input. Under a low signal-to-noise ratio, such a restriction is a candidate form of regularisation with genuine economic content, since the predictability it conditions on is itself known to be state-dependent [19]. Whether that restriction pays for itself on this task has not been asked. The answer is informative in either direction: a negative result would be a substantive finding, given how much of the 2026 literature proceeds on the assumption that regime structure in the gate is beneficial, and given that the closest existing study reports its smallest advantage precisely in the transition windows that a regime process is designed to handle [40].

---

## References

[1] Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

[2] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79–87.

[3] Weigend, A. S., Mangeas, M., & Srivastava, A. N. (1995). Nonlinear gated experts for time series: discovering regimes and avoiding overfitting. *International Journal of Neural Systems*, 6(4), 373–399.

[4] Bengio, Y., & Frasconi, P. (1994). An input–output HMM architecture. *Advances in Neural Information Processing Systems*, 7, 427–434.

[5] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.

[6] Hamilton, J. D., & Susmel, R. (1994). Autoregressive conditional heteroskedasticity and changes in regime. *Journal of Econometrics*, 64(1–2), 307–333.

[7] Gray, S. F. (1996). Modeling the conditional distribution of interest rates as a regime-switching process. *Journal of Financial Economics*, 42(1), 27–62.

[8] Filardo, A. J. (1994). Business-cycle phases and their transitional dynamics. *Journal of Business & Economic Statistics*, 12(3), 299–308.

[9] Diebold, F. X., Lee, J.-H., & Weinbach, G. C. (1994). Regime switching with time-varying transition probabilities. In C. Hargreaves (Ed.), *Nonstationary Time Series Analysis and Cointegration* (pp. 283–302). Oxford University Press.

[10] Durland, J. M., & McCurdy, T. H. (1994). Duration-dependent transitions in a Markov model of U.S. GNP growth. *Journal of Business & Economic Statistics*, 12(3), 279–288.

[11] Bulla, J., & Bulla, I. (2006). Stylized facts of financial time series and hidden semi-Markov models. *Computational Statistics & Data Analysis*, 51(4), 2192–2209.

[12] Bemporad, A., Breschi, V., Piga, D., & Boyd, S. P. (2018). Fitting jump models. *Automatica*, 96, 11–21.

[13] Nystrup, P., Kolm, P. N., & Lindström, E. (2021). Feature selection in jump models. *Expert Systems with Applications*, 184, 115558.

[14] Shu, Y., Yu, C., & Mulvey, J. M. (2024). Dynamic asset allocation with asset-specific regime forecasts. arXiv:2406.09578.

[15] Horvath, B., Issa, Z., & Muguruza, A. (2021). Clustering market regimes using the Wasserstein distance. arXiv:2110.11848.

[16] van Dijk, D., Teräsvirta, T., & Franses, P. H. (2002). Smooth transition autoregressive models — a survey of recent developments. *Econometric Reviews*, 21(1), 1–47.

[17] Longin, F., & Solnik, B. (2001). Extreme correlation of international equity markets. *Journal of Finance*, 56(2), 649–676.

[18] Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

[19] Henkel, S. J., Martin, J. S., & Nardari, F. (2011). Time-varying short-horizon predictability. *Journal of Financial Economics*, 99(3), 560–580.

[20] Hansen, B. E. (1992). The likelihood ratio test under nonstandard conditions: testing the Markov switching model of GNP. *Journal of Applied Econometrics*, 7(S1), S61–S82.

[21] Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503–3544.

[22] Ang, A., & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313–337.

[23] Jordan, M. I., & Jacobs, R. A. (1994). Hierarchical mixtures of experts and the EM algorithm. *Neural Computation*, 6(2), 181–214.

[24] Yuksel, S. E., Wilson, J. N., & Gader, P. D. (2012). Twenty years of mixture of experts. *IEEE Transactions on Neural Networks and Learning Systems*, 23(8), 1177–1193.

[25] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. *International Conference on Learning Representations*. arXiv:1701.06538.

[26] Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120), 1–39.

[27] Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., & Fedus, W. (2022). ST-MoE: designing stable and transferable sparse expert models. arXiv:2202.08906.

[28] Puigcerver, J., Riquelme, C., Mustafa, B., & Houlsby, N. (2024). From sparse to soft mixtures of experts. *International Conference on Learning Representations*.

[29] He, S., Yang, B., Hu, J., Gao, Z., & Yang, Y. (2026). Towards a statistical understanding of mixture-of-experts. arXiv:2609.03501.

[30] Dai, D., Deng, C., Zhao, C., et al. (2024). DeepSeekMoE: towards ultimate expert specialization in mixture-of-experts language models. arXiv:2401.06066.

[31] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778.

[32] Zhang, H., Dauphin, Y. N., & Ma, T. (2019). Fixup initialization: residual learning without normalization. arXiv:1901.09321.

[33] Ghahramani, Z., & Hinton, G. E. (2000). Variational learning for switching state-space models. *Neural Computation*, 12(4), 831–864.

[34] Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

[35] Kozak, S., Nagel, S., & Santosh, S. (2020). Shrinking the cross-section. *Journal of Financial Economics*, 135(2), 271–292.

[36] Kelly, B. T., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: a unified model of risk and return. *Journal of Financial Economics*, 134(3), 501–524.

[37] Chen, L., Pelger, M., & Zhu, J. (2024). Deep learning in asset pricing. *Management Science*, 70(2), 714–750.

[38] Kelly, B. T., & Xiu, D. (2023). Financial machine learning. *Foundations and Trends in Finance*, 13(3–4), 205–363.

[39] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track*. arXiv:2207.08815.

[40] Ye, J., & Borde, G. V. (2026). Regime-gated residual mixture-of-experts for cross-sectional volatility forecasting. arXiv:2608.12251.

[41] He, C., Guan, Z., Liang, X., Lian, D., Li, J., Chen, E., Lee, P. P. C., Hu, G., & Chen, Z. (2026). RAVEN: a regime-aware variable-context expert network for financial time series forecasting. arXiv:2606.24062.

[42] Yu, Z., Wu, Y., Wang, G., & Weng, H. (2024). MIGA: mixture-of-experts with group aggregation for stock market prediction. arXiv:2410.02241.

[43] Kim, N., & Song, J. W. (2026). Vector-quantized discrete latent factors meet financial priors: dynamic cross-sectional stock ranking prediction for portfolio construction. arXiv:2605.13407.

[44] Sarkar, D. (2026). Adaptive financial Transformer with regime-gated attention for stock return prediction. arXiv:2606.29347.

[45] Saqur, R., Kratsios, A., Krach, F., Limmer, Y., Tian, Y., Willes, J., Horvath, B., & Rudzicz, F. (2025). Filtered not mixed: stochastic filtering-based online gating for mixture of large language models. *International Conference on Learning Representations*. arXiv:2406.02969.

[46] Ilhan, F., Karaahmetoglu, O., Balaban, I., & Kozat, S. S. (2021). Markovian RNN: an adaptive time series prediction network with HMM-based switching for nonstationary environments. *IEEE Transactions on Neural Networks and Learning Systems*. arXiv:2006.10119.

[47] Moly, A., Costecalde, T., Martel, F., et al. (2022). An adaptive closed-loop ECoG decoder for long-term and stable bimanual control of an exoskeleton by a tetraplegic. *Journal of Neural Engineering*, 19(2), 026021.

[48] Dynamic TMoE: drift-aware dynamic mixture-of-experts for non-stationary time series forecasting (2026). arXiv:2605.20678.

[49] Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H., & Huang, S. (2024). MASTER: market-guided stock Transformer for stock price forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38, 162–170.

[50] Adaptive market intelligence: a mixture of experts framework for volatility-sensitive stock forecasting (2025). arXiv:2508.02686.

[51] Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

[52] López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7. Wiley.

[53] Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.

[54] Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.

---
---

## NOT PART OF THE SUBMISSION — verification checklist

Delete this block before submitting. Every reference must be opened and checked against the source; the following need particular attention.

**Numbers quoted in the text that must be confirmed against the source PDF:**
- [5] Gu, Kelly & Xiu: the 0.40% monthly stock-level out-of-sample *R*² and the NN3 (32–16–8) result. Confirm which table.
- [40] Ye & Borde: 30/30 and 0/30 collapse counts; 28/30 Japanese; the 4.3×, 6.7× and 0.6× relative-gain figures in their Table 8. (These were read in full and are believed correct, but check once more when citing.)
- [19] Henkel, Martin & Nardari: confirm the exact characterisation of predictability in expansions versus recessions before paraphrasing it.
- [18] Ang & Bekaert: I deliberately avoided quoting a certainty-equivalent figure. If you want one, take it from the paper directly.
- [22] Ang & Timmermann: the "miss an important regime change / issue false alarms" wording is the premise of the whole thesis. Find the exact sentence and page and quote it rather than paraphrasing.

**Bibliographic details that are reconstructed and must be checked:**
- [11] Bulla & Bulla — volume, issue and page numbers.
- [13] Nystrup, Kolm & Lindström — there are several closely related jump-model papers by these authors; confirm which one you actually want to cite, or cite two.
- [14] Shu, Yu & Mulvey — author order and whether it has since been published in a journal.
- [15] Horvath, Issa & Muguruza — arXiv number and whether a journal version now exists.
- [30] Dai et al. — full author list.
- [47] the ECoG/REW-MSLM reference — I have the mechanism right (an HMM used as the gating model of an expert mixture) but the exact paper and author list need confirming; find it via PMID 35234665.
- [48] and [50] — these two are cited by arXiv number without confirmed author lists. Fill them in or drop them; an uncredited reference looks careless.

**Substantive gaps to close before submission:**
1. **Chinese-language literature.** Nothing in this draft is from CNKI or any Chinese-language source. Your committee is in Shanghai and may well know the A-share regime-switching and machine-learning literature. Its complete absence from a 54-item reference list will be noticed. Search CNKI for 机制转换 / 马尔可夫区制转换 + 股票收益 and add what you find.
2. **Word count.** The body of this section is approximately 4,600 words excluding the reference list and this block, against a 4,000-word floor. If you cut anything, cut from §3.2.2 rather than §3.4 — the frontier section is what the committee will read most closely.
3. **Sections 3.4.2 and 3.6 contain the novelty claim.** Read them against the wording agreed in the novelty audit and make sure you can defend every clause: "structurally distinct", "as the gate", "cross-section of equity returns", "matched capacity and common protocol". Each one is load-bearing.
