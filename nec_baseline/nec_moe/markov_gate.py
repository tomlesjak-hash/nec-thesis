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

**The gate weight (Q27).** The served row at date t is built from the
filtered probability ``xi_t`` and the frozen, canonically ordered transition
matrix ``A`` by :func:`gate_weight_probs`: by default the average over the
target window of the ``j``-step-ahead probabilities,
``(1/h) sum_{j=1..h} xi_t A^j``, which makes the gate-weighted forecast the
conditional expected ``h``-day return. ``h`` comes from the panel's target.
Training and test dates get the same kind of weight, and every one is
``F_t``-measurable: it uses the series through t and the frozen ``A`` only.
"""

from __future__ import annotations

import itertools
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
    "START_SCHEME_REGISTRY",
    "causal_vol_assignment",
    "informed_centre",
    "validate_start",
    "distinct_optima",
    "MarkovFit",
    "MarkovSwitchingRegimePrior",
    "date_level_series",
    "gate_weight_probs",
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


def _french_market_excess(panel: Panel, cfg: MarkovGateConfig) -> np.ndarray:
    """Kenneth French daily Mkt-RF, aligned to the panel's dates.

    The series brief 03 §2 specified: the market excess return, from the
    French daily factors already cached for Stage C, so no new data source is
    involved. Mkt-RF for date ``t`` is the market's return over day ``t``,
    known at the close of ``t`` — the same timing as the panel's own features
    at ``t``, and strictly before the forward-return target over
    ``(t, t + h]``.

    Aligned by calendar date through the panel's ``date_labels``. A panel date
    absent from the factor file raises: forward-filling a return would invent
    a day that did not happen.
    """
    import pandas as pd

    from .context_data import load_french_factors

    if panel.date_labels is None:
        raise ValueError(
            "series='market_excess_return' aligns French factors by calendar "
            "date, but this panel carries no date_labels (synthetic panels do "
            "not) — use series='sequence_channel' there"
        )
    factors = load_french_factors(cfg.context_dir, momentum=False)
    codes = torch.unique(panel.date, sorted=True)
    labels = pd.to_datetime([panel.date_labels[int(c)] for c in codes])
    missing = labels.difference(factors.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} panel date(s) have no French factor row, first "
            f"{[d.date().isoformat() for d in missing[:5]]}"
        )
    return factors["mkt_rf"].reindex(labels).to_numpy(dtype=float)


#: Which date-level series the Hamilton gate is fitted on. A registry, not a
#: hardcoded choice, because the series is a modelling decision (§2). Every
#: entry must be knowable at date ``t``: the target ``y`` is excluded by
#: construction, being a forward return.
#:
#: ``market_excess_return`` is the French Mkt-RF series brief 03 §2
#: specified. (It previously read a sequence feature named ``mkt_ret_1d``,
#: which is neither an excess return nor present in the point-in-time panel,
#: whose sequence features are ret_1d / rel_ret_1d / vol_20d / volume_z_20d;
#: that reader lives on under the honest name ``sequence_feature``.)
SERIES_REGISTRY: dict[str, Callable[[Panel, MarkovGateConfig], np.ndarray]] = {
    "market_excess_return": _french_market_excess,
    "sequence_feature": _named_sequence_feature,
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


# --------------------------------------------------------------------------- #
# Starting values for the ML fit (brief 04 section B)
# --------------------------------------------------------------------------- #


def causal_vol_assignment(
    series: np.ndarray, window: int, calm_share: float, k: int, min_history: int
) -> np.ndarray:
    """Per-date regime assignment by trailing volatility, using no future data.

    Two ingredients, both strictly causal, so the assignment at date ``t``
    depends on the series through ``t`` and nothing later:

    - realised volatility over a **trailing** window ``[t-w+1, t]`` — never a
      centred one, which would put ``w/2`` future observations into every
      date's value even at initialisation;
    - cut points from an **expanding** quantile of those volatilities, again
      over dates ``<= t`` only. This is the specification (brief 04 B.2 as
      amended; accepted 2026-09-26, audit G-5 (a)): the cut at date ``t``
      uses volatilities through ``t``, not a quantile over the whole training
      block, so no later date can move the split at an earlier one and "no
      future observation enters" holds date by date, checkable by
      perturbation.

    ``calm_share`` is the share of dates placed in the calmest regime; for
    ``k > 2`` the remaining mass is split evenly among the others. Returns an
    int array, ``-1`` where no assignment is possible yet (the window has not
    filled, or fewer than ``min_history`` volatilities exist to cut on).
    """
    import pandas as pd

    vol = pd.Series(series).rolling(window, min_periods=window, center=False).std()
    cuts = [calm_share + j * (1.0 - calm_share) / (k - 1) for j in range(k - 1)]
    thresholds = [
        vol.expanding(min_periods=min_history).quantile(c).to_numpy() for c in cuts
    ]
    v = vol.to_numpy()
    assigned = np.zeros(len(series), dtype=int)
    for th in thresholds:
        assigned += (v > th).astype(int)
    undefined = np.isnan(v) | np.isnan(thresholds[0])
    assigned[undefined] = -1
    return assigned


def _param_index(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(n.split("[")[0], []).append(i)
    return groups


def informed_centre(
    model, series: np.ndarray, calm_share: float, window: int, persistence: float,
    cfg: MarkovGateConfig,
) -> np.ndarray:
    """One data-driven starting vector, in statsmodels' CONSTRAINED space.

    Split the training block's dates into ``K`` groups by causal trailing
    volatility (:func:`causal_vol_assignment`), set each regime's mean and
    variance to its group's sample moments, and set the transition matrix to
    ``persistence`` on the diagonal with ``(1-p)/(K-1)`` off it — informed
    about both what a regime of daily returns looks like and how long it
    lasts, unlike statsmodels' default start (``p = 0.5``: no persistence).

    Written **by parameter name**, never by position: the layout of
    ``param_names`` shifts with every trend/switching/order flag. Anything the
    centre has no information about (AR terms, exog) keeps statsmodels' own
    default for that name.
    """
    k = cfg.k_regimes
    assign = causal_vol_assignment(
        series, window, calm_share, k, cfg.start_min_history
    )
    means, variances = np.empty(k), np.empty(k)
    for r in range(k):
        members = series[assign == r]
        if len(members) < cfg.start_min_group_size:
            raise ValueError(
                f"informed start (q={calm_share}, w={window}): regime {r} has "
                f"{len(members)} dates, below start_min_group_size="
                f"{cfg.start_min_group_size}; the training block is too short "
                "for this grid point"
            )
        means[r], variances[r] = members.mean(), members.var(ddof=1)

    names = list(model.param_names)
    vec = np.asarray(model.start_params, dtype=float).copy()
    for i, n in enumerate(names):
        if n.startswith("p["):
            src, dst = (int(x) for x in n[2:-1].split("->"))
            vec[i] = persistence if src == dst else (1.0 - persistence) / (k - 1)
        elif n.startswith("const["):
            vec[i] = means[int(n[6:-1])]
        elif n == "const":
            vec[i] = series.mean()
        elif n.startswith("sigma2["):
            vec[i] = variances[int(n[7:-1])]
        elif n == "sigma2":
            vec[i] = series.var(ddof=1)
    validate_start(names, vec, k)
    return vec


def validate_start(names: list[str], vec: np.ndarray, k: int) -> None:
    """Raise unless ``vec`` is a valid constrained start.

    Variances positive, every transition probability in ``[0, 1]``, and each
    origin's free probabilities summing to at most 1 (statsmodels leaves the
    last destination implicit, so its row sums to 1 exactly when that holds).
    """
    by = _param_index(names)
    for i in by.get("sigma2", []):
        if not vec[i] > 0:
            raise ValueError(f"invalid start: {names[i]} = {vec[i]} is not > 0")
    row_mass: dict[int, float] = {}
    for i in by.get("p", []):
        if not 0.0 <= vec[i] <= 1.0:
            raise ValueError(f"invalid start: {names[i]} = {vec[i]} not in [0, 1]")
        src = int(names[i][2:-1].split("->")[0])
        row_mass[src] = row_mass.get(src, 0.0) + float(vec[i])
    for src, mass in row_mass.items():
        if mass > 1.0 + 1e-12:
            raise ValueError(
                f"invalid start: transitions out of regime {src} sum to {mass} > 1"
            )


def _default_jitter_starts(model, series, cfg: MarkovGateConfig):
    """The pre-brief-04 scheme, reproduced exactly.

    Start 0 is statsmodels' own default (``None``); start ``i > 0`` adds
    absolute ``N(0, start_jitter)`` noise to ``model.start_params`` — in the
    **constrained** space, which is the diagnosed defect. Kept so the B.1
    comparison is reproducible, not because it is recommended.
    """
    rng = np.random.default_rng(cfg.start_seed)
    base = np.asarray(model.start_params, dtype=float)
    starts: list[np.ndarray | None] = [None]
    for _ in range(1, max(cfg.search_reps, 1)):
        starts.append(base + rng.normal(0.0, cfg.start_jitter, size=base.shape))
    return starts


def _centres(model, series, cfg: MarkovGateConfig) -> list[np.ndarray]:
    return [
        informed_centre(model, series, q, w, p, cfg)
        for q, w, p in itertools.product(
            cfg.start_vol_quantiles, cfg.start_vol_windows, cfg.start_persistences
        )
    ]


def _informed_grid_starts(model, series, cfg: MarkovGateConfig):
    """Deterministic: one start per (q, w, p) centre, unperturbed."""
    return list(_centres(model, series, cfg))


def _informed_jitter_starts(model, series, cfg: MarkovGateConfig):
    """Each centre perturbed ``start_draws_per_centre`` times, validly.

    ``untransform_params`` into the unconstrained space the optimizer works
    in, add noise **relative to each coordinate's magnitude**, and
    ``transform_params`` back — so a start is valid by construction rather
    than by luck (measured: 0 invalid starts in 2,000 draws even at noise
    sd 3.0 in unconstrained units). Noise sd per coordinate is
    ``start_jitter_rel * max(|u_i|, floor_i)``, with the floor at the series
    std for location/scale coordinates and ``start_jitter_unit`` for
    dimensionless ones (transition logits, AR terms), because several
    unconstrained coordinates sit exactly at 0 and purely relative noise
    would never move them.
    """
    rng = np.random.default_rng(cfg.start_seed)
    names = list(model.param_names)
    scale_floor = np.array(
        [
            float(np.std(series))
            if n.split("[")[0] in ("const", "sigma2")
            else cfg.start_jitter_unit
            for n in names
        ]
    )
    starts: list[np.ndarray | None] = []
    for centre in _centres(model, series, cfg):
        u = np.asarray(model.untransform_params(centre), dtype=float)
        sd = cfg.start_jitter_rel * np.maximum(np.abs(u), scale_floor)
        for _ in range(cfg.start_draws_per_centre):
            vec = np.asarray(
                model.transform_params(u + rng.normal(0.0, sd)), dtype=float
            )
            validate_start(names, vec, cfg.k_regimes)
            starts.append(vec)
    return starts


#: How the multi-start ML fit generates its starting values (brief 04 B).
#:
#: Default ``"informed_jitter"``. Measured on simulated daily-scale two-state
#: series of 3,000 days, 20 starts each (brief 04 B.1, 2026-09-25):
#:
#: ====================  ===========================  =========  ========  =========
#: series                scheme                       converged  distinct  best llf
#: ====================  ===========================  =========  ========  =========
#: well separated        default_jitter, 0.05         9 / 20     1         9248.95
#: well separated        default_jitter, 0.50         4 / 20     2         9248.95
#: well separated        informed + relative jitter   20 / 20    1         9248.95
#: weak separation       default_jitter, 0.05         7 / 20     1         9508.75
#: weak separation       informed + relative jitter   20 / 20    4         9508.75
#: mean switch only      default_jitter, 0.05         7 / 20     1         9409.16
#: mean switch only      informed + relative jitter   20 / 20    5         9412.44
#: ====================  ===========================  =========  ========  =========
#:
#: Reproduced independently on differently-parameterised 3,000-day series
#: when this was implemented (24 informed_jitter starts = 8 centres x 3):
#:
#: ====================  ======================  =====  ==========  =========
#: series                scheme                  conv   failed      best llf
#: ====================  ======================  =====  ==========  =========
#: well separated        default_jitter, 0.05    6/20   14 raised   9716.98
#: well separated        informed_jitter         24/24  0           9716.98
#: weak separation       default_jitter, 0.05    6/20   14 raised   9425.46
#: weak separation       informed_jitter         24/24  0           9433.98
#: mean switch only      default_jitter, 0.50    4/20   16 raised   9554.51
#: mean switch only      informed_jitter         24/24  0           9556.92
#: ====================  ======================  =====  ==========  =========
#:
#: Every default_jitter failure was a *raised* start (invalid), none a
#: non-converged one — the diagnosed mechanism exactly. What this shows:
#: (1) the old scheme loses most starts even at small jitter, because its
#: noise lands in the constrained space at the wrong scale;
#: (2) every start finding one optimum is *not* a defect when the likelihood
#: has one dominant optimum — the second "optimum" the old scheme reached at
#: jitter 0.5 was hundreds of nats worse, not a discovery; (3) where the
#: likelihood genuinely is multimodal, the old scheme could not see it, while
#: informed starts found a better optimum (+3.3 nats, mean-switch series).
START_SCHEME_REGISTRY: dict[str, Callable[..., list[np.ndarray | None]]] = {
    "default_jitter": _default_jitter_starts,
    "informed_grid": _informed_grid_starts,
    "informed_jitter": _informed_jitter_starts,
}


def distinct_optima(llfs: list[float], tol: float) -> list[float]:
    """Distinct converged optima, best first: values within ``tol`` nats merge.

    The specification (brief 04 B.3 as amended; accepted 2026-09-26, audit
    G-5 (b)): optima are clustered by a **gap** of at most ``tol`` (1 nat by
    default) between neighbouring sorted values, not by rounding to a 1-nat
    grid, so two values a hair apart on either side of a rounding boundary
    count as one optimum.
    """
    out: list[float] = []
    for x in sorted(llfs, reverse=True):
        if not out or out[-1] - x > tol:
            out.append(x)
    return out


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
    # per start: "failed" (raised) | "nonconverged" | "converged" (brief 04 B.3)
    start_status: tuple[str, ...] = field(default_factory=tuple)
    distinct_optima_tol: float = 1.0

    @property
    def distinct_optima(self) -> list[float]:
        """Distinct CONVERGED optima, best first (within ``distinct_optima_tol``)."""
        converged = [
            llf for llf, st in zip(self.start_llfs, self.start_status, strict=True)
            if st == "converged"
        ]
        return distinct_optima(converged, self.distinct_optima_tol)

    def metrics(self) -> dict[str, float]:
        """Registry-safe summary; per-regime values in canonical order."""
        finite = [x for x in self.start_llfs if math.isfinite(x)]
        optima = self.distinct_optima
        m: dict[str, float] = {
            "gate_llf": self.llf,
            "gate_n_starts": float(self.n_starts),
            "gate_n_converged": float(self.n_converged),
            "gate_n_failed": float(self.start_status.count("failed")),
            "gate_n_nonconverged": float(self.start_status.count("nonconverged")),
            "gate_chosen_start": float(self.chosen_start),
            # These two are what answer "is the likelihood multimodal on this
            # data". One distinct optimum is NOT a defect: where the
            # likelihood has one dominant maximum, every valid start should
            # find it (brief 04 B.1, point 2).
            "gate_n_distinct_optima": float(len(optima)),
            # Range over every start that produced a log-likelihood, converged
            # or not. Kept for continuity, but it does NOT measure
            # multimodality: a start that ran without converging can sit
            # hundreds of nats below any optimum and dominate it. (An earlier
            # reading of a large spread as "the search found distinct optima"
            # was wrong for exactly that reason.)
            "gate_llf_spread": (max(finite) - min(finite)) if len(finite) > 1 else 0.0,
        }
        # the gap to the runner-up optimum is undefined with a single one;
        # omitted rather than reported as 0, which would read as a tie
        if len(optima) > 1:
            m["gate_best_minus_second"] = optima[0] - optima[1]
        for k in range(len(self.variances)):
            m[f"gate_mean_{k}"] = float(self.means[k])
            m[f"gate_variance_{k}"] = float(self.variances[k])
            m[f"gate_stay_prob_{k}"] = float(self.transition[k, k])
            # the full row-stochastic matrix, canonically ordered (brief 04 C.4)
            for j in range(len(self.variances)):
                m[f"gate_transition_{k}_{j}"] = float(self.transition[k, j])
            d = float(self.expected_durations[k])
            if math.isfinite(d):
                m[f"gate_expected_duration_{k}"] = d
        return m


# --------------------------------------------------------------------------- #
# The gate weight (Q27)
# --------------------------------------------------------------------------- #


def gate_weight_probs(
    filtered: np.ndarray, transition: np.ndarray, kind: str, horizon: int | None
) -> np.ndarray:
    """The gate's probability rows from filtered probabilities (Q27).

    ``filtered`` is ``(T, K)``, row t the filtered probability ``xi_t``;
    ``transition`` is the ``(K, K)`` row-stochastic ``A`` in the same state
    order. ``kind``:

    - ``"filtered"``: ``xi_t`` (today's regime);
    - ``"predicted"``: ``xi_t A`` (tomorrow's regime);
    - ``"window"``: ``(1/h) sum_{j=1..h} xi_t A^j``, the expected share of
      the target window ``(t, t+h]`` spent in each regime. With ``h = 1`` it
      is ``"predicted"``; with ``A = I`` it is ``"filtered"``.

    ``horizon`` is the target's ``h``; the window weight raises without it
    rather than assume one. Rows stay on the simplex: ``A`` is row-stochastic.
    """
    xi = np.asarray(filtered, dtype=float)
    a = np.asarray(transition, dtype=float)
    if xi.ndim != 2 or a.shape != (xi.shape[1], xi.shape[1]):
        raise ValueError(
            f"filtered must be (T, K) and transition (K, K); got {xi.shape} and {a.shape}"
        )
    if kind == "filtered":
        return xi.copy()
    if kind == "predicted":
        return xi @ a
    if kind == "window":
        if horizon is None or horizon < 1:
            raise ValueError(
                f"gate_weight='window' needs the target's horizon h >= 1, got {horizon}: "
                "the gate reads it from the panel it is fitted on"
            )
        step, acc = xi, np.zeros_like(xi)
        for _ in range(horizon):
            step = step @ a
            acc += step
        return acc / horizon
    raise ValueError(f"unknown gate_weight {kind!r}; use 'filtered', 'predicted' or 'window'")


# --------------------------------------------------------------------------- #
# The prior
# --------------------------------------------------------------------------- #


class MarkovSwitchingRegimePrior(PrecomputedRegimePrior):
    """Hamilton's Markov switching model as a frozen, date-keyed gate.

    This is the Hamilton *arm* of the comparison. It is not
    :class:`~nec_moe.priors.HMMRegimePrior`, which is a jointly fitted latent
    Markov mixture estimated by gradient descent through the forward
    recursion; that one is a baseline (see its docstring).

    The table holds ``cfg.gate_weight`` rows (:func:`gate_weight_probs`,
    Q27). ``horizon`` is the target horizon the model was configured for
    (``DataConfig.horizon_periods``, passed by :func:`build_markov_gate`);
    :meth:`fit` reads ``h`` from the training panel's target and raises if
    the two disagree, so there is one horizon, the target's.
    """

    def __init__(
        self, n_experts: int, cfg: MarkovGateConfig, horizon: int | None = None
    ) -> None:
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
        #: the target horizon the gate weight is built for; set from the
        #: training panel by fit()
        self.horizon: int | None = horizon
        #: date code -> the filtered probability xi_{t|t} (canonical order) of
        #: every date the fit saw or the frozen filter reached: what the
        #: experts' regime-clock decay counts with (brief 09 C.2)
        self._filtered: dict[int, np.ndarray] = {}

    def _panel_horizon(self, panel: Panel) -> int:
        """The panel target's horizon, checked against the configured one."""
        h = panel.horizon
        if self.horizon is not None and self.horizon != h:
            raise ValueError(
                f"the gate was built for a {self.horizon}-period target, but this "
                f"panel's target {panel.schema.target!r} has horizon {h}"
            )
        return h

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
        """Fit by maximum likelihood on the **training block** only.

        With ``order = p > 0`` (``MarkovAutoregression``) the first ``p`` dates
        are the autoregression's initial conditions: statsmodels returns
        filtered probabilities for the other ``nobs - p`` dates only, so the
        table starts at ``dates[p]`` (audit finding G-2). The first ``p``
        training dates have no gate row; the harness leaves them out of expert
        training and reports how many (``FoldResult.gate_excluded_dates``).
        """
        self.horizon = self._panel_horizon(train_panel)
        dates, series = date_level_series(train_panel, self.cfg)
        model = self._build_model(series)
        result, trace = self._multi_start_fit(model, series)
        perm = self._canonical_order(result)
        self.fit_result = self._summarize(result, perm, trace)
        # the fitted table: gate weights from the filtered probabilities over
        # the training dates (Q27), the same rule apply_causal uses
        filtered = self._filtered_canonical(result, perm)
        self._remember_filtered(dates[self.cfg.order :], filtered)
        self.set_fitted_table(dates[self.cfg.order :], self._gate_log_prior(filtered))

    def _multi_start_fit(self, model, series: np.ndarray):
        """Best converged fit over the scheme's starts; every start is a trial.

        statsmodels has its own ``search_reps``, but it does not expose the
        per-start likelihoods, and brief 03 §5 needs them: "best of N" is a
        selection event whose multiplicity has to reach the corrections. So
        the loop is ours: starts come from ``START_SCHEME_REGISTRY``, each fits
        with ``search_reps=0``, and every start's outcome is recorded as one of
        three things, kept apart because they mean different things (B.3):

        - ``failed``: the fit raised — for the old scheme, usually an invalid
          start (a negative variance, a probability outside [0, 1]);
        - ``nonconverged``: it ran but the optimizer did not converge;
        - ``converged``: a candidate for the maximum.
        """
        import warnings

        c = self.cfg
        starts = START_SCHEME_REGISTRY[c.start_scheme](model, series, c)
        best, best_llf, best_i = None, -np.inf, -1
        llfs: list[float] = []
        status: list[str] = []
        for i, start in enumerate(starts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = model.fit(
                        start_params=start, maxiter=c.maxiter,
                        search_reps=0, disp=0,
                    )
                except Exception:  # a start that diverges is a datum, not a crash
                    llfs.append(float("nan"))
                    status.append("failed")
                    continue
            converged = bool(res.mle_retvals.get("converged", False))
            llf = float(res.llf)
            llfs.append(llf)
            status.append("converged" if converged else "nonconverged")
            if converged and llf > best_llf:
                best, best_llf, best_i = res, llf, i
        if best is None:
            raise RuntimeError(
                f"Markov switching gate: none of {len(starts)} starts converged "
                f"({status.count('failed')} raised, "
                f"{status.count('nonconverged')} did not converge). Reported "
                "rather than silently falling back to a non-converged fit — "
                "inspect the series, k_regimes, maxiter and start_scheme"
            )
        return best, {
            "llfs": tuple(llfs),
            "status": tuple(status),
            "n_starts": len(llfs),
            "n_converged": status.count("converged"),
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
            start_status=trace["status"],
            distinct_optima_tol=self.cfg.distinct_optima_tol,
        )

    @staticmethod
    def _filtered_canonical(result, perm: np.ndarray) -> np.ndarray:
        """``(T, K)`` filtered probabilities, canonically ordered.

        From `filtered_marginal_probabilities`, never the smoothed attribute:
        the filtered quantity at date t conditions on the series through t,
        which is what a gate applied at t is allowed to know (Kim & Nelson
        ch. 4).
        """
        probs = np.asarray(result.filtered_marginal_probabilities, dtype=float)
        if probs.ndim != 2:  # pragma: no cover - shape contract of statsmodels
            raise ValueError(f"unexpected filtered probability shape {probs.shape}")
        if probs.shape[0] < probs.shape[1]:  # (K, T) in some versions
            probs = probs.T
        return probs[:, perm]

    def _gate_log_prior(self, filtered: np.ndarray) -> Tensor:
        """Log of the gate weights (Q27) from canonically ordered filtered
        probabilities: :func:`gate_weight_probs` moves them forward with the
        frozen canonical ``A`` (``fit_result.transition``) for
        ``cfg.gate_weight``; ``prob_floor`` and renormalisation come after.
        """
        assert self.fit_result is not None
        probs = gate_weight_probs(
            filtered, self.fit_result.transition, self.cfg.gate_weight, self.horizon
        )
        probs = np.clip(probs, self.cfg.prob_floor, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return torch.log(torch.from_numpy(probs).to(torch.float32))

    def _remember_filtered(self, dates: Tensor, filtered: np.ndarray, *,
                           only_after: int | None = None) -> None:
        """Keep each date's filtered probability; a fitted date is never
        overwritten by the causal application (as in the log-prior table)."""
        for d, row in zip(dates.tolist(), filtered, strict=True):
            if only_after is not None and d <= only_after:
                continue
            self._filtered.setdefault(int(d), np.asarray(row, dtype=float).copy())

    def filtered_probabilities(self, dates: Tensor) -> np.ndarray:
        """``(D, K)`` filtered probabilities ``xi_{t|t}`` of ``dates``,
        canonically ordered: the gate's input to the experts' regime-clock
        decay (brief 09 C.2). Raises for a date the gate never reached."""
        missing = [int(d) for d in dates.tolist() if int(d) not in self._filtered]
        if missing:
            raise ValueError(
                f"{len(missing)} date(s) have no filtered probability (first "
                f"{missing[:5]}): the gate was not fitted or applied there"
            )
        return np.stack([self._filtered[int(d)] for d in dates.tolist()])

    def _extra_state(self) -> dict:
        """The frozen fit (parameters, canonical permutation, multi-start
        trace), the horizon its gate weight was built for and the filtered
        probabilities, so a resumed fold reuses exactly this gate (brief 07
        C.1) and the same decay weights (brief 09 C)."""
        return {"fit_result": self.fit_result, "horizon": self.horizon,
                "filtered": dict(self._filtered)}

    def _load_extra_state(self, extra: dict) -> None:
        self.fit_result = extra.get("fit_result")
        self.horizon = extra.get("horizon", self.horizon)
        self._filtered = dict(extra.get("filtered", {}))

    # -------------------------------------------------- causal application
    def apply_causal(self, panel: Panel) -> None:
        """Extend the table to ``panel``'s later dates with frozen parameters.

        Constructs the model over the **extended** series and runs
        ``filter(params)`` — not ``fit`` — with the parameters estimated on the
        training block, then asserts they came back element-wise identical.
        The series begins at the training block's first date, far more than
        ``order`` dates before the first test date, so with an autoregressive
        gate every test date still gets a filtered probability; the output is
        aligned to ``dates[order:]`` as in :meth:`fit`.
        """
        if self.fit_result is None:
            raise ValueError("apply_causal before fit: nothing is frozen yet")
        self._panel_horizon(panel)  # the same target horizon as the fit's
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
        filtered = self._filtered_canonical(applied, perm)
        self._remember_filtered(dates[self.cfg.order :], filtered,
                                only_after=self.fit_max_date)
        self.extend_causal_table(dates[self.cfg.order :], self._gate_log_prior(filtered))

    # ------------------------------------------------------------ forward
    def forward(self, gate_logits: Tensor, ctx=None) -> PriorOutput:
        out = super().forward(gate_logits, ctx)
        if self.fit_result is not None:
            out.info["transition"] = torch.from_numpy(
                self.fit_result.transition.astype("float32")
            )
        return out


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
    return MarkovSwitchingRegimePrior(
        cfg.experts.n_experts, cfg.markov_gate, horizon=cfg.data.horizon_periods
    )
