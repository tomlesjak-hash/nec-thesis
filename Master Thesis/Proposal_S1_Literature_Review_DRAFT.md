# Section 1. Literature Review

**Frontier, current status, challenges and significance of the research topic**

*Draft 4, 2026-09-11. Approximately 4,160 words (form requires ≥4,000 for a master's in English). 31 numbered references, cited in order of first appearance. A verification checklist that is NOT part of the submission appears at the end.*

---

## 1.0 Two literatures, one object

An unobserved market state that persists, shifts occasionally, and changes which predictors work is among the oldest ideas in financial econometrics. A gating network that assigns input-dependent weights to a set of specialised predictors is among the oldest ideas in neural computation. They are the same conditional-mixture object, introduced two years apart by Hamilton [1] and by Jacobs, Jordan, Nowlan and Hinton [2], and they differ in one respect. The econometric version places a stochastic process over the mixing variable, giving it persistence, transition dynamics and an implied duration distribution. The machine-learning version computes the mixing weights as a deterministic function of the current input, with no dependence on the previous mixing decision.

The two traditions made contact almost immediately. Weigend, Mangeas and Srivastava trained nonlinear gated experts on time series, framed the result explicitly as the discovery of regimes, and compared the architecture directly to hidden Markov models [3]. Bengio and Frasconi's input-output hidden Markov model specified transition probabilities as a learned softmax function of exogenous inputs [4], which is simultaneously a mixture of experts with memory and a Markov-switching model with covariate-driven transitions. The contact did not persist into the architectures that followed.

---

## 1.1 Regime-aware neural architectures, 2024-2026

The last twenty-four months have produced a dense cluster of neural forecasting models that condition predictions on market state.

### 1.1.1 The methodological benchmark

Ye and Borde propose a regime-gated residual mixture of experts for cross-sectional volatility forecasting [5]. Their stated motivation is methodological: existing approaches, they observe, "typically modify the regime representation, routing strategy, and model capacity simultaneously," with the result that "the contribution of the integration point remains unclear" [5]. Their design holds the state variables, the forecasting backbone, the parameter budget, the hyperparameter selection rule and the evaluation protocol fixed, and varies only the point at which regime information enters the network.

The study forecasts annualised five-day forward realised volatility for 1,027 US equities from December 2015 to November 2025, with replication on 1,552 Japanese TSE Prime stocks. Evaluation uses thirty non-overlapping walk-forward windows, each a 504-day development period split 85/15 into training and validation followed by a 63-day test period, producing approximately 1.9 million out-of-sample forecasts. Every neural configuration is run over thirty random seeds and reported as a distribution.

The principal result concerns integration pathway. When the two regime state variables are appended to the forecasting input, training collapsed in 30 of 30 seeds on the US panel and 28 of 30 on the Japanese panel; when the identical variables were supplied only to the gating network, no seed collapsed. The size of that second effect is easily overread. Against an identically specified model whose gate receives the stock features but not the regime variables, supplying the regime variables raises the information coefficient by 0.0004 and removes the single remaining collapsed run, an accuracy gain smaller than the across-seed standard deviation of 0.0012, and small beside the 0.0048 by which the residual mixture as a whole exceeds a capacity-matched single network. The demonstrated effect of routing on regime information in this architecture is therefore principally that it avoids the training instability caused by input concatenation, rather than that it substantially improves forecasts; the greater part of the architecture's advantage is attributable to the residual construction rather than to the regime information carried through it. Soft routing outperformed four hard-routing alternatives, a learned top-1 gate, volatility-quantile assignment, GICS sector assignment, and a market-by-idiosyncratic volatility split, each difference significant at *p* < 10⁻⁴.

A second set of ablations separates the contributions of the architecture. Expert count has little effect: reducing from four experts to two lowers the information coefficient by 0.0004, and increasing to six changes it by 0.0002 while raising the number of collapsed runs from zero to three. The residual parameterisation matters more. Replacing the zero-initialised final layer of each expert with ordinary random initialisation lowers the information coefficient from 0.5469 to 0.5430 and raises collapsed runs from 0 to 13 of 30; removing the frozen base entirely, which yields a standard mixture of experts, gives 0.5413 with 24 of 30 runs collapsed. The differences in accuracy are small in absolute terms, 0.0039 and 0.0056 against a level of roughly 0.547, while the differences in training reliability are large. The authors attribute the gain to the residual architecture rather than to the size of the expert ensemble.

Training collapse in this study is defined as a mean test QLIKE above 2. QLIKE is a volatility-specific loss, undefined for a signed return forecast, and it diverges as a volatility forecast approaches zero.

### 1.1.2 Expert mixtures applied to returns

Three recent studies apply expert mixtures to the cross-section of returns instead of volatility. RAVEN routes three-layer Transformer encoders operating over nested look-back prefixes, selecting among them by a learned patch-importance score, and reports correlations of approximately 0.036-0.039 with information ratios between 0.39 and 0.60 on HS300 and S&P 500 panels; the authors describe the mechanism as adaptive regime awareness without explicit regime labels [6]. MIGA combines sequence experts with group aggregation and inner-group attention for stock ranking on CSI 300, 500 and 1000, and reports that experts separate by market capitalisation and by long versus short exposure without supervision directing them to do so [7]. PRISM-VQ learns a codebook of 512 discrete cross-sectional archetypes by vector quantisation and uses the assigned code as the routing signal for a structure-conditioned mixture that generates time-varying factor loadings, evaluated on CSI 300 and S&P 500 over 2022-2024 with rank information coefficient, rank information ratio, and portfolio-level return, drawdown and Sharpe ratio under explicit transaction costs [8].

Both MIGA and PRISM-VQ measure expert specialisation instead of asserting it: MIGA documents the capitalisation and directional split, and PRISM-VQ tests whether expert activation distributions are statistically distinct and reports the overlap between their periods of high activation [7, 8]. Across the wider literature such claims are more often stated than tested.

The expert architectures in this group are sequence models, not tabular ones. RAVEN's experts are Transformer encoders applied to look-back windows of differing length, so that the routing decision selects an effective context length for each observation [6]; MIGA's experts operate on a stock's recent history and are combined by attention within expert groups, with an information-coefficient-based expert loss alongside a load-balancing term on the router [7]; PRISM-VQ's temporal stage prepends the discrete code as a structure token to a Transformer sequence encoder before the mixture produces loadings [8]. This follows from the input representation: where the input carries a time axis, the experts are sequence models, and where the input is a point-in-time vector of characteristics, as in Ye and Borde [5], the experts are compact multilayer perceptrons.

A fourth study, an adaptive financial Transformer with regime-gated attention, applies a GRU-based encoder over eleven groups of technical indicators to produce softmax weights that bias self-attention logits [9]. It operates on a single asset, not a cross-section, its gate weights feature groups rather than experts, and the differences between its optimised and baseline configurations are not statistically significant on any reported metric. Its own future work proposes integrating Gaussian-mixture hidden Markov models for regime classification, so a paper with "regime-gated" in its title treats the use of an actual regime model in the gate as still to be done.

### 1.1.3 The source of the gate signal

In each of the architectures above the gate is a learned function of instantaneous features. Ye and Borde's regime state variable consists of two rolling statistics: the twenty-day volatility of the equal-weighted market return, shared across all stocks on a given date, and the twenty-day volatility of the residual return after removing a beta estimated over the preceding 120 days [5]. There is no latent state, no transition matrix, no persistence parameter and no filtering recursion. PRISM-VQ assigns its codes independently on each date and measures their month-to-month persistence as an outcome, reporting rates of 4.9 per cent on CSI 300 and 7.4 per cent on the S&P 500 against a uniform baseline of 1 per cent [8]. RAVEN states that it operates without regime labels [6].

Ye and Borde decompose their advantage over a capacity-matched baseline by market condition. Against a full-sample gain of 0.0048 in information coefficient, the gain is 0.0207 in the top decile of market volatility and 0.0322 during the COVID crisis period of February to June 2020, or approximately 4.3 and 6.7 times the full-sample figure, and 0.0079 in the 2022 bear market. In windows of ±10 days around regime flips it is 0.0031, or 0.6 times the full-sample gain. The authors conclude that the benefit of residual routing "is greatest during sustained periods of elevated market volatility rather than during regime transitions themselves" [5]. Their limitations section names alternative forms of market state as future work.

One line of work places a filtering recursion in the gate. MoE-F formulates expert selection as a finite-state continuous-time hidden Markov model and derives closed-form Wonham-Shiryaev filtering recursions with optimality guarantees, evaluated partly on financial market movement prediction [10]. The latent state there is over which expert is currently performing best, inferred from the experts' own running performance, not over a market state inferred from market data.

The use of a Markov process as the gating mechanism of an expert mixture is not confined to the recent literature. An adaptive decoder for exoskeleton control mixes and switches independent expert decoders using a hidden Markov model as the gating model, adopted for robustness against non-stationarity in the input signal [11]. Markovian RNN embeds hidden Markov transition structure inside a recurrent network for prediction in non-stationary environments, with the switching variable governing which set of recurrent parameters applies [12]. Earlier still, Ghahramani and Hinton derived switching state-space models explicitly as the fully dynamical generalisation of the mixture of experts, with recurrent expert and gating networks and Markov dynamics on the switch variable [13]. The identity between switching models and dynamical expert mixtures has therefore been established in the literature for twenty-five years.

The limitation of memoryless routing has likewise been stated within the machine-learning literature. A recent drift-aware mixture of experts identifies fixed expert pools and memoryless routing as obstacles to adapting to abrupt regime shifts, and proposes maximum-mean-discrepancy drift detection with dynamic instantiation of new experts as a remedy [14].

### 1.1.4 Two variants of the base-plus-experts construction

The base-plus-experts decomposition used by Ye and Borde has a parallel in large language models. Shared-expert isolation, introduced in DeepSeekMoE and carried into subsequent large mixture models, retains one or more always-active experts alongside the routed ones so that knowledge common to all inputs need not be replicated across specialists [15]. The rationale corresponds to a theoretical result of He and co-authors, who show that always-active shared experts reduce the learning burden placed on the routed experts [16].

The two constructions differ in two respects. In the language-model version the shared expert is trained jointly with the routed experts; Ye and Borde pre-train a base network, freeze it, and train only the gate and the correction experts in a second stage [5]. And in the language-model version the routed experts produce full-magnitude contributions, whereas Ye and Borde's produce corrections that are exactly zero at initialisation and are penalised through a shrinkage term on the aggregate correction. Their ablations separate zero initialisation from the presence of a base, but their non-residual comparison arm removes the base and switches to joint training at the same time, and no published study appears to pair a jointly trained base with zero-initialised residual experts.

---

## 1.2 The two source literatures

### 1.2.1 Regime switching in financial econometrics

Hamilton's model supposes a latent state following a first-order Markov chain and observations drawn from a state-dependent density, with inference through a recursive filter that combines the predictive density under each state with a prior propagated through the transition matrix [1]. Two structural properties follow. The filtered probability is persistent by construction: when the diagonal elements of the transition matrix are large, a single anomalous observation moves the belief only slightly, so the model responds to accumulated evidence, not to isolated shocks. And the implied regime duration is geometric, hence memoryless: the hazard of leaving a regime is the same on the first day as on the two-hundredth.

Subsequent work developed these properties in several directions. Time-varying transition probability models make the switching probability a logistic function of observable covariates, so that transitions respond to economic conditions instead of occurring at a constant rate [17]; the construction coincides with the input-output hidden Markov model of the same year [4]. Statistical jump models replace likelihood estimation with an optimisation problem minimising within-state squared deviation plus an explicit penalty λ on each state change, solved by coordinate descent alternating closed-form parameter updates with a dynamic-programming pass over state sequences [18]. At λ = 0 the model reduces to k-means clustering with no persistence, and as λ increases the fitted state sequence becomes arbitrarily persistent. A continuous variant relaxes hard state assignment to simplex-valued weights and has been applied to asset-specific regime forecasting for dynamic allocation, explicitly in contrast to broad regimes affecting an entire asset universe [19]. A further family abandons parametric state densities, clustering market states by the Wasserstein distance between empirical return distributions over rolling windows, which in the univariate case reduces to comparing sorted quantile functions [20].

These families differ in what they take a regime to be. For classical Markov switching a regime is a parameter set generating observations, and the state is inferred by likelihood [1]. For a jump model it is a centroid in feature space, and persistence enters as an explicit penalty, not as a transition probability [18]. For a covariate-driven transition model it is a state whose exit hazard is a function of observable conditions [17]. For distributional clustering it is a shape of the return distribution, defined without reference to generating parameters [20]. Each therefore produces a different sequence of mixing weights from the same data, and each trades responsiveness against stability by a different mechanism. This is a distinction of kind, not of degree, and differs from the comparisons of configuration already present in the mixture-of-experts literature, which hold the routing paradigm fixed while varying engineering choices, the number and size of experts [21], sigmoid against softmax gating [22], and dispatch against aggregation weighting [23].

A limitation of real-time regime inference is documented within this tradition. In their survey of regime changes and financial markets, Ang and Timmermann note that filtered regime probabilities at times miss an important regime change and at other times issue false alarms, complicating the real-time tracking of market states [24]. Two estimation limitations are common to the whole family. The number of regimes is not testable by standard means, because under the null of a single regime the transition probabilities are unidentified nuisance parameters and the likelihood ratio statistic loses its usual asymptotic distribution; applied work conventionally fixes the number of states and reports sensitivity instead [24]. And regime-conditional means are poorly identified, since a regime occupying a small share of the sample supplies few effective observations for estimating its parameters.

The empirical case for regime structure is nonetheless well established. Ang and Timmermann's survey documents that mixtures over regimes reproduce the fat tails and skewness of financial returns without recourse to non-Gaussian innovations, that cross-asset correlations rise in market downturns in a manner inconsistent with a constant-correlation multivariate normal, and that investors who ignore regimes in allocation decisions forgo measurable certainty-equivalent wealth [24].

### 1.2.2 The mixture of experts

What does the work in Jacobs and co-authors' formulation is the error function, not the architecture [2]. Under a squared-error objective on the combined output, experts learn cooperatively, each correcting the residual left by the others, and the resulting specialists are entangled. Under an objective in which each expert is evaluated on its own output weighted by its gate probability, experts compete, and localisation emerges without supervision. Specialisation is therefore a consequence of the training objective.

Jordan and Jacobs derived an expectation-maximisation algorithm for the hierarchical form, whose E-step computes for each observation the posterior probability that each expert generated it [25]. That quantity differs from Hamilton's filtered probability in the prior alone: the expert responsibility uses a prior depending only on the current input, while the Markov-switching filter uses a prior propagated through a transition matrix from the previous period. Bengio and Frasconi's model is the version in which that prior is propagated [4].

It was revived at scale for a different purpose. Sparse top-*k* routing activates a small number of many experts per input, with auxiliary load-balancing losses to prevent the router from concentrating on a subset, and exists to increase parameter count while holding per-input computation approximately constant [26]. At language-model sample sizes the statistical cost of estimating the router is small relative to the capacity gained. He and co-authors provide a statistical treatment of mixture-of-experts estimation, deriving an oracle risk bound in which the router-estimation cost scales as *O*((*M·d*/*n*) log(*M·d·n*)) for *M* experts, input dimension *d* and sample size *n* [16]. The same analysis shows that top-1 routing attains oracle risk when a dominant specialist exists and the partition boundary is linearly separable, but incurs irreducible error when the best local predictor is itself a mixture, and that linear gates incur constant error on curved decision boundaries whereas quadratic or kernel gates do not.

Routing collapse, the concentration of the gate on a small subset of experts, leaving the remainder untrained, is a recognised failure mode of this architecture, and the standard remedy is an auxiliary penalty encouraging the average routing weights toward a uniform allocation [26]. Ye and Borde carry such a penalty alongside the shrinkage term on the aggregate correction, and report that the resulting gate remains broadly distributed: the mean maximum routing weight across observations is 0.36, against 0.25 for uniform routing over four experts, with the two most strongly loaded experts typically showing opposite-signed correlations with the regime variables [5].

---

## 1.3 Machine learning for the cross-section of expected returns

Gu, Kelly and Xiu evaluate linear models, penalised regressions, dimension reduction, regression trees, random forests, gradient boosting and neural networks of one to five hidden layers on a common panel of firm characteristics under a fixed rolling train-validate-test protocol [27]. Machine learning methods outperform linear benchmarks out of sample, and neural networks are modestly the best of them. Performance peaks at shallow depth instead of increasing with it, which they attribute to small effective sample size and a weak signal. Their stated conclusion is that performance peaks at three hidden layers and declines thereafter. The location of that peak, however, depends on which criterion is applied, and their own results divide along a clear line. On forecast accuracy the three-layer network leads: it attains the highest monthly stock-level *R*² at 0.40 per cent, the highest among large capitalisations at 0.70 per cent, the highest monthly *R*² for the aggregate index at 1.80 per cent, and the largest market-timing Sharpe gain. On realised portfolio performance the four-layer network leads: it produces the best decile-spread strategy at 2.3 per cent per month, the smallest maximum drawdown, the mildest worst month, and the authors describe it as dominating the alternatives by a large margin in cumulative performance on both the long and the short side. The two families of criterion therefore select different depths, a divergence the paper reports without reconciling.

Statistical testing separates none of them in any case. Their Diebold-Mariano tests find **no significant difference between any pair of the five depths**, the largest statistic among them being 0.59; the authors further state that neural networks improve over tree-based models but that "the difference is not statistically significant," and that under a multiple-comparison correction their advantage over penalised linear models becomes only marginal. The finding that survives is therefore the shape of the curve rather than any particular depth: shallow is sufficient, and depth beyond a few layers does not help. A later study varying depth on characteristic-sorted portfolios of large-capitalisation US stocks reaches the same qualitative conclusion while placing its optimum at two hidden layers [31]. Neither study separates depth from capacity, since each added layer also adds parameters under the geometric pyramid convention both adopt, and neither contains a mixture architecture, so none of this speaks to how deep the components of a mixture should be. Two practices from this literature do transfer regardless of depth: forecasts are averaged across networks trained from several random initialisations, because the stochastic nature of the optimisation makes a single run unreliable, and regularisation is applied through early stopping, batch normalisation and learning-rate shrinkage, not through architectural restraint alone [27]. The best stock-level out-of-sample *R*² is approximately 0.40 per cent at the monthly horizon. Characteristics are cross-sectionally rank-transformed period by period and mapped into [-1, 1].

The magnitude of that figure characterises the problem: the best available models explain well under one per cent of the variation in individual monthly stock returns, so that added model flexibility is as likely to degrade out-of-sample performance as to improve it. Kelly and Xiu survey the surrounding literature, including the multiple-testing problem created by a proliferation of weakly informative predictors and the shrinkage and conditional-structure responses to it [28].

Non-stationarity is named in the motivation of most of this literature and modelled in almost none of it. The standard treatment is rolling or expanding re-estimation, in which the model is refitted periodically on a moving window. That procedure carries an implicit model of structural change with a single timescale set by the window length and never estimated; it has no recurrence, so a period resembling conditions from two decades earlier cannot be recognised as such once that period has left the window; and it produces no state variable that can be inspected, tested against external classifications, or used as a conditioning input.

There is direct evidence that state matters for this problem. Henkel, Martin and Nardari document that the short-horizon predictability of equity returns by standard valuation ratios is essentially absent during expansions and substantial during recessions [29]. The mapping from predictors to expected returns is therefore not stable across states.

A parallel line within this literature conditions model structure on firm characteristics, not on market state. Instrumented principal components allow factor loadings to be functions of observable characteristics while the factors themselves remain latent, and deep learning has been embedded within a no-arbitrage restriction so that the pricing kernel is estimated adversarially against the most mispriced portfolios [28]. These approaches condition on the cross-sectional position of a firm at a point in time; the time-series question of which conditioning relation prevails in the current period is treated separately, through the estimation window.

---

## 1.4 Challenges in the research area

**Signal strength.** At an out-of-sample *R*² below one per cent [27], the room for improvement is narrow relative to the variance of any estimate, and the router-estimation cost identified by He and co-authors [16] is of first-order magnitude at equity-panel sample sizes, where at language-model scale it is negligible.

**Evaluation.** Multi-day forward targets overlap, requiring purging and embargo periods to prevent test-period information from influencing training [30], and trying many architectures and reporting the best carries the same statistical liability as testing many predictors [28]. The recent literature adds a further requirement: in Ye and Borde's results the differences in mean accuracy between architectures were small while the differences in the proportion of seeds that failed to train were large [5], so single-seed results are uninformative. Stability criteria in current use are target-specific: the QLIKE threshold serving the volatility literature has no established counterpart for signed return forecasts.

**Pace.** Six or more directly relevant architectures appeared within twelve months. Claims of novelty in this area that rest on the absence of a single experiment date quickly.

**Comparability of regime mechanisms.** The families in Section 1.2.1 differ not only in what they take a regime to be but in how they can be embedded in a neural model. Covariate-driven transition models [17] and the continuous jump model [19] are differentiable, so their parameters can in principle be learned jointly with a predictor; classical filtering with a fitted transition matrix and distributional clustering [20] are not, and can enter only as a pre-computed state sequence. Placing these mechanisms on a common footing therefore confronts a trade-off between using each in its natural form and holding the training procedure fixed, a difficulty that does not arise in studies implementing a single mechanism.

---

## 1.5 Gap and significance

Regime switching and the mixture of experts are the same conditional-mixture object, developed in parallel and joined briefly in the mid-1990s [3, 4, 13]. The econometric branch refined how the latent state evolves, persistence, covariate-driven transitions, and penalty-based and distributional alternatives to likelihood estimation, and attached these mechanisms almost exclusively to linear or low-dimensional predictors. The machine-learning branch refined how specialists are combined and retained a router depending only on the current input, under an objective concerned with computational scaling [26]. The empirical asset pricing literature adopted neural predictors without adopting either, treating non-stationarity through rerolled estimation windows [27, 28].

Gating an expert mixture with a Markov or filtering process is not itself new: it appears in Weigend, Mangeas and Srivastava [3], in Bengio and Frasconi [4], in switching state-space models [13], in Markovian recurrent networks [12], in hidden-Markov-gated decoders outside finance [11], and in MoE-F [10]. Nor is the observation that modern routing is memoryless [14]. What is absent is a comparison: no existing study evaluates structurally distinct regime-switching processes, classical Markov switching, penalty-based jump models, covariate-driven transition probabilities, and model-free distributional clustering, as the gate of a neural mixture of experts, on the cross-section of equity returns, under matched expert capacity and a common walk-forward evaluation protocol. The existing comparisons within the mixture-of-experts literature vary configuration inside a fixed routing paradigm [21, 22, 23]; the existing regime-gated financial models each implement one mechanism.

The significance of that absence follows from the statistical structure of the problem. A regime-switching process is formally a restriction on the class of admissible gates: it constrains the mixing weights to evolve with persistence and through a specified transition mechanism instead of varying freely with the current input. Under a signal-to-noise ratio as low as this one, and with a router-estimation cost that is material at this sample size [16], such a restriction is a candidate form of regularisation with economic content, since the predictability being conditioned on is itself state-dependent [29]. Evidence on whether the restriction is beneficial would be informative in either direction, and is presently unavailable: the closest existing study reports its smallest advantage precisely in the transition windows that a regime process is constructed to handle [5].

The literature also lacks the instrument as well as the result. There is no common, capacity-matched, protocol-matched setting in which regime mechanisms can be compared against one another on the cross-section of returns, and in its absence evidence accumulated on any one mechanism cannot be placed beside evidence on another. Ye and Borde's study illustrates the value of such a setting on a narrower question: it introduces no new state variable, fixing everything else and isolating a single design choice [5], and its finding can accordingly be read without ambiguity, which is not true of studies varying several elements simultaneously.

---

## References

[1] Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

[2] Jacobs, R. A. Jordan, M. I. Nowlan, S. J. & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79-87.

[3] Weigend, A. S. Mangeas, M. & Srivastava, A. N. (1995). Nonlinear gated experts for time series: discovering regimes and avoiding overfitting. *International Journal of Neural Systems*, 6(4), 373-399.

[4] Bengio, Y. & Frasconi, P. (1994). An input-output HMM architecture. *Advances in Neural Information Processing Systems*, 7, 427-434.

[5] Ye, J. & Borde, G. V. (2026). Regime-gated residual mixture-of-experts for cross-sectional volatility forecasting. arXiv:2608.12251.

[6] He, C. Guan, Z. Liang, X. Lian, D. Li, J. Chen, E. Lee, P. P. C. Hu, G. & Chen, Z. (2026). RAVEN: a regime-aware variable-context expert network for financial time series forecasting. arXiv:2606.24062.

[7] Yu, Z. Wu, Y. Wang, G. & Weng, H. (2024). MIGA: mixture-of-experts with group aggregation for stock market prediction. arXiv:2410.02241.

[8] Kim, N. & Song, J. W. (2026). Vector-quantized discrete latent factors meet financial priors: dynamic cross-sectional stock ranking prediction for portfolio construction. arXiv:2605.13407.

[9] Sarkar, D. (2026). Adaptive financial Transformer with regime-gated attention for stock return prediction. arXiv:2606.29347.

[10] Saqur, R. Kratsios, A. Krach, F. Limmer, Y. Tian, Y. Willes, J. Horvath, B. & Rudzicz, F. (2025). Filtered not mixed: stochastic filtering-based online gating for mixture of large language models. *International Conference on Learning Representations*. arXiv:2406.02969.

[11] Moly, A. Costecalde, T. Martel, F. et al. (2022). An adaptive closed-loop ECoG decoder for long-term and stable bimanual control of an exoskeleton by a tetraplegic. *Journal of Neural Engineering*, 19(2), 026021.

[12] Ilhan, F. Karaahmetoglu, O. Balaban, I. & Kozat, S. S. (2021). Markovian RNN: an adaptive time series prediction network with HMM-based switching for nonstationary environments. *IEEE Transactions on Neural Networks and Learning Systems*. arXiv:2006.10119.

[13] Ghahramani, Z. & Hinton, G. E. (2000). Variational learning for switching state-space models. *Neural Computation*, 12(4), 831-864.

[14] Dynamic TMoE: drift-aware dynamic mixture-of-experts for non-stationary time series forecasting (2026). arXiv:2605.20678.

[15] Dai, D. Deng, C. Zhao, C. et al. (2024). DeepSeekMoE: towards ultimate expert specialization in mixture-of-experts language models. arXiv:2401.06066.

[16] He, S. Yang, B. Hu, J. Gao, Z. & Yang, Y. (2026). Towards a statistical understanding of mixture-of-experts. arXiv:2609.03501.

[17] Filardo, A. J. (1994). Business-cycle phases and their transitional dynamics. *Journal of Business & Economic Statistics*, 12(3), 299-308.

[18] Bemporad, A. Breschi, V. Piga, D. & Boyd, S. P. (2018). Fitting jump models. *Automatica*, 96, 11-21.

[19] Shu, Y. Yu, C. & Mulvey, J. M. (2024). Dynamic asset allocation with asset-specific regime forecasts. arXiv:2406.09578.

[20] Horvath, B. Issa, Z. & Muguruza, A. (2021). Clustering market regimes using the Wasserstein distance. arXiv:2110.11848.

[21] Slicing and dicing: configuring optimal mixtures of experts (2026). arXiv:2605.11689.

[22] On the role of sigmoid versus softmax gating in mixture-of-experts models (2026). arXiv:2602.01466.

[23] Dispatch versus aggregation weighting in sparse mixture-of-experts layers (2026). arXiv:2608.08853.

[24] Ang, A. & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of Financial Economics*, 4, 313-337.

[25] Jordan, M. I. & Jacobs, R. A. (1994). Hierarchical mixtures of experts and the EM algorithm. *Neural Computation*, 6(2), 181-214.

[26] Shazeer, N. Mirhoseini, A. Maziarz, K. Davis, A. Le, Q. Hinton, G. & Dean, J. (2017). Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. *International Conference on Learning Representations*. arXiv:1701.06538.

[27] Gu, S. Kelly, B. & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.

[28] Kelly, B. T. & Xiu, D. (2023). Financial machine learning. *Foundations and Trends in Finance*, 13(3-4), 205-363.

[29] Henkel, S. J. Martin, J. S. & Nardari, F. (2011). Time-varying short-horizon predictability. *Journal of Financial Economics*, 99(3), 560-580.

[30] López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7. Wiley.

[31] Lai, S. (2025). Multilayer perceptron neural network models in asset pricing: an empirical study on large-cap US stocks. arXiv:2505.01921.

---
---

## NOT PART OF THE SUBMISSION, verification checklist

Delete this block before submitting.

**References [21], [22] and [23] are the weakest entries and must be checked first.** They were added to support the sentence distinguishing structural comparisons from configuration comparisons. I have the arXiv numbers but not confirmed titles or author lists, and the titles as written are reconstructions. Open each one, correct the entry, and confirm it actually compares what the sentence says it compares. If any does not, drop it, the sentence works with two supporting citations, or with none if rephrased.

**Reference [11], confirm the paper.** The mechanism is right: an expert mixture gated by a hidden Markov model, outside finance. The exact paper and author list need confirming; the relevant record is PMID 35234665. This citation matters because it is evidence that HMM-gated mixtures are an established construction, which is the honest framing.

**Other bibliographic details reconstructed from notes:** [14] and [15] author lists; whether [18] Bemporad et al. or the Nystrup-Kolm-Lindström financial application is the citation wanted for jump models, or both; [19] author order and journal status; [20] arXiv number and journal status.

**Figures quoted in the text, confirm against the source PDFs:**
- [5] Ye & Borde: all of it, 30/30, 28/30, 0/30, the *p* < 10⁻⁴ routing comparisons, IC values 0.5469 / 0.5430 / 0.5413, collapse counts 0 / 13 / 24, the K2 and K6 deltas, and the Table 8 ratios 4.3× / 6.7× / 0.6×. These were read in full and are believed correct.
- [27] Gu, Kelly & Xiu: **RESOLVED 2026-09-12 against the published paper (RFS 33(5), 2223-2274), not only the tables.** Monthly, all stocks: NN1 0.33, NN2 0.39, NN3 0.40, NN4 0.39, NN5 0.36. Annual, all stocks: NN1 2.64, NN2 2.70, NN3 3.40, NN4 3.60, NN5 2.79. Diebold-Mariano statistics among NN1-NN5 range from -1.04 to 0.59, none significant. Lai's "3-4 layers" reading is the fair one; the flat "NN3 is best" claim is not supported. Confirm the table numbers once against the published paper.
- [31] Lai: the depth ordering (two layers ahead of three) is the citable part. Its *R*² values (3.66 and 2.86 per cent) are portfolio-level on 420 large caps and are not comparable to [27]'s stock-level 0.40 per cent, do not place the two figures side by side.
- [6] RAVEN: correlation 0.036-0.039 and ICIR 0.39-0.60 come from notes, not from a full read. Verify or remove the numbers.
- [8] PRISM-VQ: codebook size 512 and the 4.9 / 7.4 per cent persistence figures.
- [24] Ang & Timmermann: the "miss an important regime change / issue false alarms" wording. Find the exact sentence and page and quote it. Note that [24] is also carrying the unidentified-nuisance-parameter point; the primary citation for that is Hansen (1992), *Journal of Applied Econometrics* 7(S1), S61-S82, add it as a 31st reference if a reviewer is likely to press.
- [29] Henkel, Martin & Nardari: confirm the expansion/recession characterisation before paraphrasing.

**Substantive gap: no Chinese-language or CNKI sources.** Search 机制转换 / 马尔可夫区制转换 + 股票收益 and add two or three.

**Deliberately omitted, in case a reviewer asks.** Harvey, Liu & Zhu (2016), now carried by [28]; Grinsztajn et al. (2022) on trees versus deep learning on tabular data, which belongs in Section 3 as the justification for a gradient-boosting baseline; Diebold & Mariano (1995) and Newey & West (1987), also Section 3; Hansen (1992) as above; the duration-dependence and hidden semi-Markov line; and MASTER, FactorVAE, Switch Transformer, ST-MoE and Soft-MoE. All are in `Proposal_S1_Literature_Review_v1_long.md`.

**Content that belongs in later sections and is deliberately absent here.** Section 1.1.4 observes that no study pairs a jointly trained base with zero-initialised residual experts; the corresponding decision (a frozen base) and the pilot that would test it belong in Sections 3 and 5. Section 1.4 notes that stability criteria are target-specific; defining one for signed returns belongs in Section 3. Section 1.2.1 notes that the number of regimes is not testable; fixing K and sweeping it belongs in Section 3.
