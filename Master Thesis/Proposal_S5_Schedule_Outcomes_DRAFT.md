# Section 5. Research Schedule and Expected Outcomes

*Draft 2, 2026-09-12. Approximately 390 words. Assumes a first draft due 3 January 2027.*

---

## Research schedule

Writing and experimentation run in parallel. Only the results and discussion depend on experiments finishing; the background, methodology and data chapters do not, and are drafted while the runs are in progress.

**September.** Finalise and defend this proposal. Resolve database access, adopting the fallback specification at the end of the month if it is not confirmed, since the data source fixes the characteristic set and everything downstream. Build the data pipeline and the walk-forward evaluation harness. Cover the theoretical ground the later work assumes: the filtering recursions behind Markov switching, expectation-maximisation for mixtures and its relation to regime inference, the optimisation used to fit penalty-based state sequences, distributional distances, and the gradient machinery a differentiable gate needs.

**October.** Implement each regime mechanism behind the common interface and check its fitted state sequence against known episodes before attaching it to a mixture. Sweep base depth in parallel, since the base trains standalone. Begin the literature review chapter, which draws on Section 1.

**November.** Complete the mechanism by depth grid in the first half of the month and the construction arms in the second, then run the full comparison once the configuration is locked. Write the methodology and data chapters alongside.

**December.** Run the diagnostics in the first week and freeze results on 5 December; after that date nothing is re-run except to correct an error. Write the results and discussion, assemble tables and figures, and check every citation and reported figure against its source. The complete draft is finished by mid-December, leaving the final two weeks for revision and supervisor comments.

## Expected outcomes

Four results, three of them informative whatever the comparison shows.

The grid of regime mechanisms against expert capacity, which does not exist in the published literature for any target, and which establishes whether the ordering of mechanisms holds across architectures.

The comparison itself. If a mechanism wins, the result identifies which structural property carries the benefit. If none does, that is also substantive, given how much recent work assumes regime structure in the gate helps.

The diagnostics: how much of the expert correction the gate is actually responsible for, and whether fitted regime weights line up with market states anyone would recognise.

The comparison framework, released so further mechanisms can be added without rebuilding the baseline.
