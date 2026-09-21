# Section 6. Prior Experience and Accomplished Achievements Related to the Proposed Topic

*Draft 2, 2026-09-12. Approximately 280 words.*

---

The closest preparation is the Quantitative Investment Programme at SAIF. It covered the practice of quantitative investment and, more usefully, the three things this study spends most of its time on: building a panel of firm-level data, choosing the metrics a signal is judged by, and evaluating a model against them. The empirical work here follows that process, and reports results in that vocabulary.

An econometrics course supplied the statistical background the regime-switching literature assumes, though not to the depth that literature eventually demands. A financial mathematics course introduced stochastic processes, which is how a latent market state with transition probabilities is expressed, but stopped short of stochastic calculus. That limit shaped the design rather than being worked around. The nearest continuous-time precedent for filtering-based gating is written in stochastic calculus, and Section 1 cites it as prior work instead of rebuilding it; the mechanisms selected here are its discrete-time counterparts, which suit daily and monthly data anyway.

The physics degree supplies the mathematics the machine learning needs. Feed-forward networks and their training are built from linear algebra and multivariable calculus, and being comfortable with both is what makes the September schedule realistic. There has been no formal machine learning coursework. That is worth stating because it explains the shape of the study: the architectures used are small and well documented, and the contribution is the comparison, not a new network design.

Outside coursework, an existing codebase already contains the data pipeline, baseline models including gradient-boosted trees, and a backtesting component, which is why Section 3 treats the engineering as moderate. A quantitative alpha competition provided practice at building cross-sectional signals and evaluating them walk-forward, which is the design this thesis uses.
