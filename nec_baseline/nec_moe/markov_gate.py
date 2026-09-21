"""The Hamilton gate: Markov switching, fitted by ML, applied causally (brief 03).

Hamilton (1989) models a series as driven by a latent two-state Markov chain,
estimated by maximum likelihood through the forward filter (Hamilton 1994,
ch. 22; the EM view is Hamilton 1990). This module wraps
``statsmodels.tsa.regime_switching`` into a :class:`
~nec_moe.priors.PrecomputedRegimePrior` so the fitted regime probabilities
become the gate the experts are handed and cannot influence (Q19).

Four things here are easy to get wrong, and each is enforced rather than
documented:

**Filtering, never smoothing.** Kim & Nelson (1999, ch. 4) treat this as a
*modelling decision*, not an implementation detail, and it is the reason this
file exists in the shape it does. The filtered probability at ``t`` conditions
on the series through ``t``; the smoothed probability conditions on the whole
sample, including the future. In Bishop's notation (PRML 13.2.2/13.2.4) the
scaled forward variable ``alpha-hat(z_n)`` is the filtered quantity and
``gamma(z_n)`` is the smoothed one; only the first may reach a gate.
``smoothed_marginal_probabilities`` is the more natural-looking name and the
one most people reach for first, which is exactly why a test asserts this
module never mentions it.

**Fit on train, apply frozen.** Parameters are estimated on the training block
alone. Probabilities for later dates come from re-running the *filter* over
the extended series with those exact parameters — legal, because the filter at
``t`` uses only the series through ``t``. Re-fitting would not be, because
maximum likelihood over the extended series uses every observation including
the future. The applied parameters are asserted element-wise identical to the
fitted ones, so a silent re-estimation cannot pass.

**Label switching.** The likelihood is invariant to permuting the states, so
fold 3's "regime 1" need not be fold 4's. States are therefore sorted by a
*declared* statistic (fitted conditional variance, ascending, by default), so
regime 0 is always the calmest, and the applied permutation is recorded.
Stephens (2000) is the reference for why an identifiability constraint of this
kind is a convention rather than a solution: it works when the regimes are
well separated in the sort statistic and can fail when they are not, which is
what the template-tracking escalation in :mod:`nec_moe.diagnostics` is for.

**Multi-start is a selection event.** The likelihood is multimodal. Every
random start's converged log-likelihood is logged as a trial, so the
multiplicity of "best of N" reaches the deflated Sharpe like any other
selection in this project, and convergence failures are reported rather than
silently absorbed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .config import MarkovGateConfig, NECConfig
from .data import Panel
from .priors import PrecomputedRegimePrior, PriorOutput

__all__ = [
    "SERIES_REGISTRY",
    "ORDERING_REGISTRY",
    "MarkovFit",
    "MarkovSwitchingRegimePrior",
    "date_level_series",
]

# --------------------------------------------------------------------------- #
# What the gate is fitted on (brief 03 section 2)
# --------------------------------------------------------------------------- #


def _named_sequence_feature(panel: Panel, cfg: MarkovGateConfig) -> np.ndarray:
    """A market-level sequence feature, by name, at each date's last timestep.

    The panel's feature contract guarantees every sequence feature at date
    ``t`` is computed from data at or before ``t``, so this is causal by
    construction. Market features are cross-sectionally constant on a date, so
    the per-date mean recovers the series exactly.
    """
    names = panel.schema.sequence_features if panel.schema is not None else ()
    if cfg.series_feature not in names:
        raise ValueError(
            f"markov_gate.series_feature={cfg.series_feature!r} is not a "
            f"sequence feature of this panel; available: {list(names)}. "
            "Synthetic panels carry no named features — use "
            "series='sequence_channel' there"
        )
    return _per_date_mean(panel, panel.x_seq[:, -1, names.index(cfg.series_feature)])


def _raw_sequence_channel(panel: Panel, cfg: MarkovGateConfig) -> np.ndarray:
    """A sequence channel by integer index — the unnamed/synthetic-panel path."""
    ch = cfg.series_channel
    if not 0 <= ch < panel.x_seq.shape[2]:
        raise ValueError(
            f"markov_gate.series_channel={ch} out of range for d_seq="
            f"{panel.x_seq.shape[2]}"
        )
    return _per_date_mean(panel, panel.x_seq[:, -1, ch])


def _per_date_mean(panel: Panel, values: Tensor) -> np.ndarray:
    """Cross-sectional mean per date, in ascending date order."""
    dates = torch.unique(panel.date, sorted=True)
    out = torch.stack([values[panel.date == d].mean() for d in dates])
    return out.detach().double().numpy()


#: Which date-level series the Hamilton gate is fitted on. A registry, not a
#: hardcoded choice, because the series is a modelling decision (§2). Every
#: entry must be knowable at date ``t``: the target ``y`` is excluded by
#: construction, being a forward return.
SERIES_REGISTRY: dict[str, Callable[[Panel, MarkovGateConfig], np.ndarray]] = {
    "market_excess_return": _named_sequence_feature,
    "sequence_channel": _raw_sequence_channel,
}


def date_level_series(panel: Panel, cfg: MarkovGateConfig) -> tuple[Tensor, np.ndarray]:
    """``(dates (D,), series (D,))`` — the gate's univariate input."""
    try:
        builder = SERIES_REGISTRY[cfg.series]
    except KeyError:
        raise ValueError(
            f"unknown markov_gate.series {cfg.series!r}; registered: "
            f"{sorted(SERIES_REGISTRY)}"
        ) from None
    series = builder(panel, cfg)
    dates = torch.unique(panel.date, sorted=True)
    if len(series) != len(dates):
        raise ValueError(
            f"series length {len(series)} != number of dates {len(dates)}"
        )
    if not np.isfinite(series).all():
        raise ValueError(
            f"series {cfg.series!r} contains non-finite values; a Markov "
            "switching fit on NaNs converges to nonsense rather than failing"
        )
    return dates, series


# --------------------------------------------------------------------------- #
# Canonical ordering of the fitted regimes (brief 03 section 4)
# --------------------------------------------------------------------------- #

#: How to relabel the fitted states so a regime index means the same thing in
#: every fold. Ascending in the chosen statistic, so regime 0 is the calmest
#: (variance) or the lowest-mean (mean) state.
ORDERING_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "variance": lambda variances, means: np.argsort(variances, kind="stable"),
    "mean": lambda variances, means: np.argsort(means, kind="stable"),
}


@dataclass
class MarkovFit:
    """A fitted, frozen Markov switching model and the trace behind it."""

    params: np.ndarray  # frozen; the applied filter must reuse these exactly
    llf: float
    transition: np.ndarray  # (K, K) ROW-stochastic, canonically ordered
    expected_durations: np.ndarray  # (K,) canonically ordered
    variances: np.ndarray  # (K,) canonically ordered
    means: np.ndarray  # (K,) canonically ordered
    permutation: tuple[int, ...]  # applied canonical order (§4), auditable
    n_starts: int  # candidates the winner was selected from (§5)
    n_converged: int
    chosen_start: int  # which start won
    start_llfs: tuple[float, ...] = field(default_factory=tuple)

    def metrics(self) -> dict[str, float]:
        """Registry-safe summary; per-regime values in canonical order."""
        finite = [x for x in self.start_llfs if math.isfinite(x)]
        m: dict[str, float] = {
            "gate_llf": self.llf,
            "gate_n_starts": float(self.n_starts),
            "gate_n_converged": float(self.n_converged),
            "gate_chosen_start": float(self.chosen_start),
            # How multimodal the likelihood actually looked from these starts.
            # ~0 means every start found the same optimum: the multi-start
            # search explored nothing, and "best of N" was a selection from a
            # population of one. Large means the local optima are real and the
            # multiplicity genuinely matters.
            "gate_llf_spread": (max(finite) - min(finite)) if len(finite) > 1 else 0.0,
        }
        for k in range(len(self.variances)):
            m[f"gate_variance_{k}"] = float(self.variances[k])
            m[f"gate_stay_prob_{k}"] = float(self.transition[k, k])
            d = float(self.expected_durations[k])
            if math.isfinite(d):
                m[f"gate_expected_duration_{k}"] = d
        return m


# --------------------------------------------------------------------------- #
# The prior
# --------------------------------------------------------------------------- #


class MarkovSwitchingRegimePrior(PrecomputedRegimePrior):
    """Hamilton's Markov switching model as a frozen, date-keyed gate.

    This is the Hamilton *arm* of the comparison. It is not
    :class:`~nec_moe.priors.HMMRegimePrior`, which is a jointly fitted latent
    Markov mixture estimated by gradient descent through the forward
    recursion; that one is a baseline (see its docstring).
    """

    def __init__(self, n_experts: int, cfg: MarkovGateConfig) -> None:
        super().__init__(n_experts)
        if cfg.k_regimes != n_experts:
            raise ValueError(
                f"markov_gate.k_regimes={cfg.k_regimes} must equal "
                f"experts.n_experts={n_experts}: the gate's regimes are the "
                "mixture's components"
            )
        if cfg.order_by not in ORDERING_REGISTRY:
            raise ValueError(
                f"unknown markov_gate.order_by {cfg.order_by!r}; registered: "
                f"{sorted(ORDERING_REGISTRY)}"
            )
        self.cfg = cfg
        self.fit_result: MarkovFit | None = None
        self._model_kwargs: dict | None = None

    # ------------------------------------------------------------- fitting
    def _build_model(self, series: np.ndarray):
        """Construct the statsmodels model over ``series`` (no fitting)."""
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        c = self.cfg
        common = dict(
            k_regimes=c.k_regimes,
            trend=c.trend,
            switching_trend=c.switching_trend,
            switching_variance=c.switching_variance,
        )
        if c.order > 0:
            return MarkovAutoregression(series, order=c.order, **common)
        return MarkovRegression(series, **common)

    def fit(self, train_panel: Panel) -> None:
        """Fit by maximum likelihood on the **training block** only."""
        dates, series = date_level_series(train_panel, self.cfg)
        model = self._build_model(series)
        result, trace = self._multi_start_fit(model)
        perm = self._canonical_order(result)
        self.fit_result = self._summarize(result, perm, trace)
        # the fitted table: filtered probabilities over the training dates
        log_prior = self._filtered_log_prior(result, perm)
        self.set_fitted_table(dates, log_prior)

    def _multi_start_fit(self, model):
        """Best of ``search_reps`` random starts; every start is a trial (§5).

        statsmodels has its own ``search_reps``, but it does not expose the
        per-start likelihoods, and §5 needs them: "best of N" is a selection
        event whose multiplicity has to reach the corrections. So the loop is
        ours, each start fits with ``search_reps=0``, and every converged
        log-likelihood is recorded.
        """
        import warnings

        c = self.cfg
        rng = np.random.default_rng(c.start_seed)
        best, best_llf, best_i = None, -np.inf, -1
        llfs: list[float] = []
        n_converged = 0
        for i in range(max(c.search_reps, 1)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    start = None if i == 0 else _jitter(model, rng, c.start_jitter)
                    res = model.fit(
                        start_params=start, maxiter=c.maxiter,
                        search_reps=0, disp=0,
                    )
                except Exception:  # a start that diverges is a datum, not a crash
                    llfs.append(float("nan"))
                    continue
            converged = bool(res.mle_retvals.get("converged", False))
            n_converged += int(converged)
            llf = float(res.llf)
            llfs.append(llf)
            if converged and llf > best_llf:
                best, best_llf, best_i = res, llf, i
        if best is None:
            raise RuntimeError(
                f"Markov switching gate: none of {max(c.search_reps, 1)} random "
                "starts converged. Reported rather than silently falling back "
                "to a non-converged fit — inspect the series, k_regimes and "
                "maxiter before continuing"
            )
        return best, {
            "llfs": tuple(llfs),
            "n_starts": len(llfs),
            "n_converged": n_converged,
            "chosen": best_i,
        }

    def _canonical_order(self, result) -> np.ndarray:
        """The declared relabelling (§4), applied before anything is stored."""
        variances, means = _regime_moments(result, self.cfg.k_regimes)
        return ORDERING_REGISTRY[self.cfg.order_by](variances, means)

    def _summarize(self, result, perm: np.ndarray, trace: dict) -> MarkovFit:
        variances, means = _regime_moments(result, self.cfg.k_regimes)
        transition = _row_stochastic_transition(result)[np.ix_(perm, perm)]
        return MarkovFit(
            params=np.asarray(result.params, dtype=float).copy(),
            llf=float(result.llf),
            transition=transition,
            expected_durations=np.asarray(result.expected_durations, dtype=float)[perm],
            variances=variances[perm],
            means=means[perm],
            permutation=tuple(int(i) for i in perm),
            n_starts=trace["n_starts"],
            n_converged=trace["n_converged"],
            chosen_start=trace["chosen"],
            start_llfs=trace["llfs"],
        )

    def _filtered_log_prior(self, result, perm: np.ndarray) -> Tensor:
        """Log of the **filtered** marginal probabilities, canonically ordered.

        `filtered_marginal_probabilities`, never the smoothed attribute: the
        filtered quantity at date t conditions on the series through t, which
        is what a gate applied at t is allowed to know (Kim & Nelson ch. 4).
        """
        probs = np.asarray(result.filtered_marginal_probabilities, dtype=float)
        if probs.ndim != 2:  # pragma: no cover - shape contract of statsmodels
            raise ValueError(f"unexpected filtered probability shape {probs.shape}")
        if probs.shape[0] < probs.shape[1]:  # (K, T) in some versions
            probs = probs.T
        probs = probs[:, perm]
        probs = np.clip(probs, self.cfg.prob_floor, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return torch.log(torch.from_numpy(probs).to(torch.float32))

    # -------------------------------------------------- causal application
    def apply_causal(self, panel: Panel) -> None:
        """Extend the table to ``panel``'s later dates with frozen parameters.

        Constructs the model over the **extended** series and runs
        ``filter(params)`` — not ``fit`` — with the parameters estimated on the
        training block, then asserts they came back element-wise identical.
        """
        if self.fit_result is None:
            raise ValueError("apply_causal before fit: nothing is frozen yet")
        dates, series = date_level_series(panel, self.cfg)
        model = self._build_model(series)
        frozen = self.fit_result.params
        applied = model.filter(frozen)  # NOT .fit(): no re-estimation
        used = np.asarray(applied.params, dtype=float)
        if not np.array_equal(used, frozen):
            raise ValueError(
                "the applied parameters differ from the training fit's: the "
                "model was re-estimated on data it must not see. "
                f"max |difference| = {np.abs(used - frozen).max():.3e}"
            )
        perm = np.asarray(self.fit_result.permutation, dtype=int)
        self.extend_causal_table(dates, self._filtered_log_prior(applied, perm))

    # ------------------------------------------------------------ forward
    def forward(self, gate_logits: Tensor, ctx=None) -> PriorOutput:
        out = super().forward(gate_logits, ctx)
        if self.fit_result is not None:
            out.info["transition"] = torch.from_numpy(
                self.fit_result.transition.astype("float32")
            )
        return out


def _jitter(model, rng: np.random.Generator, scale: float) -> np.ndarray:
    """A randomly perturbed start, in the model's unconstrained parameterization.

    ``scale`` is a config field (``markov_gate.start_jitter``): too small and
    every start lands in the same basin, making the multi-start search
    decorative; too large and most starts fail to converge. The convergence
    count is reported per fit so the setting can be judged rather than assumed.
    """
    start = np.asarray(model.start_params, dtype=float)
    return start + rng.normal(0.0, scale, size=start.shape)


def _regime_moments(result, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(variances (K,), means (K,))`` of the fitted regimes.

    Read off ``param_names`` rather than by index arithmetic on ``params``:
    the layout shifts with ``trend``, ``exog`` and each switching flag
    (``sigma2[0], sigma2[1]`` when the variance switches, a single shared
    ``sigma2`` when it does not), so positional indexing is a bug waiting for
    the first config change.
    """
    names = list(result.model.param_names)
    params = np.asarray(result.params, dtype=float)

    def by_prefix(prefix: str, default: float) -> np.ndarray:
        vals = np.full(k, np.nan)
        for name, value in zip(names, params, strict=False):
            if name == prefix:  # shared across regimes (not switching)
                vals[:] = value
            elif name.startswith(f"{prefix}["):
                idx = int(name[len(prefix) + 1 : -1])
                if 0 <= idx < k:
                    vals[idx] = value
        return np.where(np.isnan(vals), default, vals)

    return by_prefix("sigma2", 1.0), by_prefix("const", 0.0)


def _row_stochastic_transition(result) -> np.ndarray:
    """statsmodels' ``regime_transition``, transposed into our convention.

    statsmodels returns ``P[i, j] = P(S_t = i | S_{t-1} = j)`` — **column**
    stochastic — while every transition matrix in this package (and
    :func:`nec_moe.diagnostics.expected_durations`) is row-stochastic
    ``A[j, k] = P(z_t = k | z_{t-1} = j)``. Verified empirically against an
    asymmetric chain; getting it wrong transposes the persistence of every
    regime, which is silent for a symmetric chain and wrong for a real one.
    """
    p = np.asarray(result.regime_transition, dtype=float)
    if p.ndim == 3:  # (K, K, n_tvtp) for time-invariant transitions
        p = p[:, :, 0]
    return p.T


def build_markov_gate(cfg: NECConfig) -> MarkovSwitchingRegimePrior:
    return MarkovSwitchingRegimePrior(cfg.experts.n_experts, cfg.markov_gate)
