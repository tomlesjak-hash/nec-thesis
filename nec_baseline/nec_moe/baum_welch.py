"""Native Baum-Welch for the Hamilton gate's order-0 Gaussian model (brief 09 D).

The model is the one ``statsmodels``' ``MarkovRegression(trend="c",
switching_trend=True, switching_variance=True)`` fits: a ``K``-state Markov
chain with row-stochastic transition matrix ``A[j, k] = P(z_t = k | z_{t-1} =
j)``, and an emission ``m_t | z_t = k ~ N(nu_k, sigma_k^2)``. statsmodels
maximises its likelihood but takes no observation weights, and the gate's
regime-clock memory (Q16 (d)(e)) needs weighted M-steps, hence this module.

Notation follows ``MoE_HMM_Gate_Formulation.pdf`` Section 5 (Bishop PRML
13.2; Rabiner 1989 III-C, V-A):

- ``eta_t(k) = N(m_t; nu_k, sigma_k^2)``, the emission density;
- the **scaled forward pass** is the Hamilton filter:
  ``alpha_hat_t(k) = xi_{t|t}(k)`` (the filtered probability) and
  ``c_t = p(m_t | F_{t-1})``, so ``log p(m_{1:T}) = sum_t log c_t``
  (Bishop's ``c_n``; Rabiner's scale factor is ``1/c_t``);
- the scaled backward pass ``beta_hat_T = 1``,
  ``beta_hat_t(j) = sum_k A_jk eta_{t+1}(k) beta_hat_{t+1}(k) / c_{t+1}``;
- ``gamma_t(k) = alpha_hat_t(k) beta_hat_t(k)`` (smoothed) and
  ``xi_t(j, k) = alpha_hat_t(j) A_jk eta_{t+1}(k) beta_hat_{t+1}(k) / c_{t+1}``.

Nothing underflows: every stored quantity is a probability or a ratio of two
densities (``eta_{t+1}(k) / c_{t+1}``), and the log-likelihood is a sum of logs.

**The M-steps** (Bishop 13.18-13.21; PDF Section 5.5), with optional
per-regime weights ``omega_t(k)`` (brief 09 D.2; all ones for the ordinary
fit):

- ``nu_k = sum_t omega_t(k) gamma_t(k) m_t / sum_t omega_t(k) gamma_t(k)``;
- ``sigma_k^2`` likewise with ``(m_t - nu_k)^2``, then floored at
  ``variance_floor`` (a config field; a collapsing variance is the classic
  degenerate EM solution);
- ``A_jk = sum_t omega_t(j) xi_t(j, k) / sum_t omega_t(j) gamma_t(j)``: the
  transitions out of ``j`` use ``j``'s weights, and the denominator equals
  the row sum of the numerator exactly;
- the initial distribution ``rho``, by one of two rules
  (:data:`INITIAL_RULES`):

  - ``"stationary"``: ``rho`` is the stationary distribution of ``A`` (with
    statsmodels' floor), which is what ``MarkovRegression`` does by default
    (``initialize_steady_state``), so the native fit maximises **the same
    likelihood** as the statsmodels backend. ``rho`` then depends on ``A``,
    so ``A``'s M-step maximises ``sum_jk N_jk log A_jk + sum_k g_k log
    rho_k(A)`` numerically (``N`` the weighted expected transition counts,
    ``g`` the weighted first-date posterior), started from the closed form;
    it is still an exact M-step, so the unweighted iteration is a
    (generalised) EM and never lowers the likelihood;
  - ``"estimated"``: the textbook free ``rho_k = gamma_1(k)`` (Rabiner's
    ``pi_bar``, Bishop 13.18).

The E-step is never weighted: the posteriors are those of the current
parameters under the full, unweighted model. Only how they are averaged into
new parameters changes (brief 09 D.2), which makes the weighted iteration a
weighted generalised-EM heuristic rather than the maximiser of a known
objective; :func:`baum_welch` reports whether it settled and never stops
silently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "INITIAL_RULES",
    "INITIAL_FLOOR",
    "HMMParams",
    "EStep",
    "BaumWelchResult",
    "stationary_distribution",
    "gaussian_logpdf",
    "forward",
    "backward",
    "e_step",
    "m_step",
    "expected_complete_loglik",
    "baum_welch",
    "params_from_statsmodels",
    "params_to_statsmodels",
    "filter_series",
]

#: How the initial regime distribution is set (see the module docstring).
INITIAL_RULES: tuple[str, ...] = ("stationary", "estimated")

#: statsmodels bounds the steady-state initial probabilities away from zero
#: by this much (``MarkovSwitching.initial_probabilities``), without
#: renormalising; the native ``"stationary"`` rule copies it exactly.
INITIAL_FLOOR = 1e-20

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class HMMParams:
    """Parameters of the ``K``-state Gaussian Markov switching model.

    ``transition`` is row-stochastic, ``A[j, k] = P(z_t = k | z_{t-1} = j)``;
    ``initial`` is the distribution of the first date's regime.
    """

    means: np.ndarray  # (K,)
    variances: np.ndarray  # (K,)
    transition: np.ndarray  # (K, K)
    initial: np.ndarray  # (K,)

    @property
    def k(self) -> int:
        return int(len(self.means))

    def permuted(self, perm: Sequence[int] | np.ndarray) -> HMMParams:
        """Relabel the regimes: new regime ``i`` is old regime ``perm[i]``."""
        p = np.asarray(perm, dtype=int)
        return HMMParams(
            means=self.means[p].copy(),
            variances=self.variances[p].copy(),
            transition=self.transition[np.ix_(p, p)].copy(),
            initial=self.initial[p].copy(),
        )


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """The stationary distribution of a row-stochastic ``A``, computed exactly
    as statsmodels computes its steady-state initial probabilities (pseudo-
    inverse of the stacked balance and normalisation equations, then floored
    at :data:`INITIAL_FLOOR`), so the two backends start the filter from the
    same numbers."""
    p = np.asarray(transition, dtype=float).T  # statsmodels' column-stochastic P
    m = p.shape[0]
    stacked = np.c_[(np.eye(m) - p).T, np.ones(m)].T
    probabilities = np.linalg.pinv(stacked)[:, -1]
    return np.maximum(probabilities, INITIAL_FLOOR)


def gaussian_logpdf(series: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """``(T, K)``: ``log N(m_t; nu_k, sigma_k^2)``."""
    x = np.asarray(series, dtype=float)[:, None]
    v = np.asarray(variances, dtype=float)[None, :]
    return -0.5 * (_LOG_2PI + np.log(v) + (x - np.asarray(means, dtype=float)[None, :]) ** 2 / v)


def forward(
    log_eta: np.ndarray, transition: np.ndarray, initial: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The scaled forward pass: ``(filtered (T, K), log_c (T,))``.

    ``filtered[t] = xi_{t|t}``; ``log_c[t] = log p(m_t | F_{t-1})``, the
    one-step predictive log density, so ``log_c.sum()`` is the
    log-likelihood. The recursion is sequential, so it runs on Python floats
    (about three times faster than per-step numpy calls at ``K <= 4``)."""
    t_len, k = log_eta.shape
    peak = log_eta.max(axis=1)
    eta = np.exp(log_eta - peak[:, None]).tolist()
    a = np.asarray(transition, dtype=float).tolist()
    pred = [float(v) for v in initial]
    filtered: list[list[float]] = [[]] * t_len
    scale = [0.0] * t_len
    rk = range(k)
    for t in range(t_len):
        e = eta[t]
        joint = [pred[i] * e[i] for i in rk]
        s = sum(joint)
        f = [v / s for v in joint]
        filtered[t] = f
        scale[t] = s
        pred = [sum(f[j] * a[j][i] for j in rk) for i in rk]
    log_c = np.log(np.asarray(scale)) + peak
    return np.asarray(filtered), log_c


def backward(log_eta: np.ndarray, log_c: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """The scaled backward pass ``beta_hat`` ``(T, K)``, from the forward
    pass's ``log_c``: ``beta_hat_t(j) = sum_k A_jk e_{t+1}(k)
    beta_hat_{t+1}(k)`` with ``e_t(k) = eta_t(k) / c_t``."""
    t_len, k = log_eta.shape
    e = np.exp(log_eta - log_c[:, None]).tolist()
    a = np.asarray(transition, dtype=float).tolist()
    beta: list[list[float]] = [[]] * t_len
    nxt = [1.0] * k
    beta[t_len - 1] = nxt
    rk = range(k)
    for t in range(t_len - 2, -1, -1):
        en = e[t + 1]
        w = [en[i] * nxt[i] for i in rk]
        nxt = [sum(a[j][i] * w[i] for i in rk) for j in rk]
        beta[t] = nxt
    return np.asarray(beta)


@dataclass(frozen=True)
class EStep:
    """One E-step: everything the M-step and the diagnostics read."""

    loglik: float
    filtered: np.ndarray  # (T, K) xi_{t|t}
    log_c: np.ndarray  # (T,) log p(m_t | F_{t-1})
    gamma: np.ndarray  # (T, K) smoothed marginals
    ratio: np.ndarray  # (T, K) eta_t(k) / c_t
    beta: np.ndarray  # (T, K) scaled backward variables

    def transition_counts(
        self, transition: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """``N_jk = sum_{t<T} omega_t(j) xi_t(j, k)``, vectorised: the
        expected number of ``j -> k`` transitions, weighted by ``j``'s
        weights (unweighted when ``weights`` is None)."""
        left = self.filtered[:-1] if weights is None else self.filtered[:-1] * weights[:-1]
        right = self.ratio[1:] * self.beta[1:]
        return np.asarray(transition, dtype=float) * (left.T @ right)


def e_step(series: np.ndarray, params: HMMParams) -> EStep:
    """Forward-backward at ``params`` (unweighted, always)."""
    log_eta = gaussian_logpdf(series, params.means, params.variances)
    filtered, log_c = forward(log_eta, params.transition, params.initial)
    beta = backward(log_eta, log_c, params.transition)
    gamma = filtered * beta
    return EStep(
        loglik=float(log_c.sum()), filtered=filtered, log_c=log_c,
        gamma=gamma, ratio=np.exp(log_eta - log_c[:, None]), beta=beta,
    )


def _stationary_transition_step(counts: np.ndarray, first: np.ndarray) -> np.ndarray:
    """The M-step for ``A`` when ``rho = stationary(A)``: maximise
    ``sum_jk N_jk log A_jk + sum_k g_k log rho_k(A)`` over row-stochastic
    ``A`` (rows as softmax of logits with the last entry fixed at 0), started
    from the closed form ``N_jk / sum_l N_jl``.

    The gradient is analytic, so the step is exact to the optimiser's
    precision and the EM iteration settles instead of creeping on
    finite-difference noise. For the stationary distribution ``rho = rho A``,
    ``d rho = rho dA Z`` with the fundamental matrix ``Z = (I - A + 1
    rho')^{-1}``, so ``d/dA_jl sum_k g_k log rho_k = rho_j (Z v)_l`` with
    ``v_k = g_k / rho_k``; the softmax chain rule gives the logit gradient
    ``A_jm (G_jm - sum_l G_jl A_jl)`` with ``G_jl = N_jl / A_jl + rho_j
    (Z v)_l``."""
    from scipy.optimize import minimize

    k = counts.shape[0]
    closed = counts / counts.sum(axis=1, keepdims=True)
    if float(first.sum()) == 0.0:
        return closed
    tiny = 1e-300

    def unpack(z: np.ndarray) -> np.ndarray:
        logits = np.concatenate([z.reshape(k, k - 1), np.zeros((k, 1))], axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        rows = np.exp(logits)
        return rows / rows.sum(axis=1, keepdims=True)

    def value_and_grad(z: np.ndarray) -> tuple[float, np.ndarray]:
        a = unpack(z)
        rho = stationary_distribution(a)
        value = float(np.sum(counts * np.log(a + tiny)) + np.sum(first * np.log(rho)))
        fundamental = np.linalg.inv(np.eye(k) - a + np.outer(np.ones(k), rho))
        g = counts / np.maximum(a, tiny) + np.outer(rho, fundamental @ (first / rho))
        grad = a * (g - (g * a).sum(axis=1, keepdims=True))
        return -value, -grad[:, : k - 1].ravel()

    start = np.log(np.clip(closed, 1e-12, None))
    start = (start[:, :-1] - start[:, -1:]).ravel()
    res = minimize(value_and_grad, start, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 0.0, "gtol": 1e-12})
    best = res.x if res.fun <= value_and_grad(start)[0] else start
    return unpack(best)


def m_step(
    series: np.ndarray,
    estep: EStep,
    transition: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    initial: str = "stationary",
    variance_floor: float = 0.0,
) -> HMMParams:
    """New parameters from the posteriors of ``estep`` (computed at the
    current ``transition``), with per-regime weights ``omega_t(k)``
    (``weights``, ``(T, K)``; None means all ones)."""
    if initial not in INITIAL_RULES:
        raise ValueError(f"unknown initial rule {initial!r}; use one of {INITIAL_RULES}")
    m = np.asarray(series, dtype=float)
    g = estep.gamma if weights is None else estep.gamma * weights
    mass = g.sum(axis=0)
    if np.any(mass <= 0):
        raise ValueError(
            f"a regime has no weighted posterior mass ({mass.tolist()}): its "
            "emission cannot be estimated; the weights or the starting values "
            "leave it empty"
        )
    means = (g * m[:, None]).sum(axis=0) / mass
    variances = (g * (m[:, None] - means[None, :]) ** 2).sum(axis=0) / mass
    variances = np.maximum(variances, variance_floor)
    counts = estep.transition_counts(transition, weights)
    first = estep.gamma[0] if weights is None else estep.gamma[0] * weights[0]
    if initial == "estimated":
        a = counts / counts.sum(axis=1, keepdims=True)
        rho = np.maximum(first / first.sum(), INITIAL_FLOOR)
    else:
        a = _stationary_transition_step(counts, first)
        rho = stationary_distribution(a)
    return HMMParams(means=means, variances=variances, transition=a, initial=rho)


def expected_complete_loglik(
    series: np.ndarray,
    estep: EStep,
    params: HMMParams,
    transition_old: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """The (weighted) expected complete-data log-likelihood ``Q`` of
    ``params`` under the posteriors of ``estep`` (Bishop 13.17), each term
    multiplied by its regime's weight: the objective the weighted M-step
    maximises, logged every iteration (brief 09 D.2)."""
    g = estep.gamma if weights is None else estep.gamma * weights
    log_eta = gaussian_logpdf(series, params.means, params.variances)
    counts = estep.transition_counts(transition_old, weights)
    tiny = 1e-300
    return float(
        np.sum(g * log_eta)
        + np.sum(counts * np.log(params.transition + tiny))
        + np.sum(g[0] * np.log(params.initial))
    )


def _param_change(old: HMMParams, new: HMMParams, scale: float) -> float:
    """Largest change across the parameters, each on its own natural scale:
    means in units of the series' standard deviation, variances in logs,
    probabilities as they are."""
    return float(max(
        np.max(np.abs(new.means - old.means)) / scale,
        np.max(np.abs(np.log(new.variances) - np.log(old.variances))),
        np.max(np.abs(new.transition - old.transition)),
        np.max(np.abs(new.initial - old.initial)),
    ))


@dataclass(frozen=True)
class BaumWelchResult:
    """The outcome of one Baum-Welch run (one start, or the weighted pass)."""

    params: HMMParams
    loglik: float  # unweighted log-likelihood at ``params``
    filtered: np.ndarray  # (T, K) xi_{t|t} at ``params``
    log_c: np.ndarray  # (T,) one-step predictive log densities at ``params``
    n_iter: int
    converged: bool
    # per iteration: (log-likelihood before the step, the weighted objective
    # Q after the M-step, the largest parameter change of the step)
    trace: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)


def baum_welch(
    series: np.ndarray,
    start: HMMParams,
    *,
    weights: np.ndarray | None = None,
    initial: str = "stationary",
    variance_floor: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 5000,
) -> BaumWelchResult:
    """Iterate E- and M-steps from ``start`` until the largest parameter
    change is below ``tol`` (see :func:`_param_change`), or ``maxiter``
    iterations. ``converged`` says which; the caller decides what a run that
    did not settle means (it is never silently treated as converged).

    With ``initial="stationary"`` the start's ``initial`` is replaced by the
    stationary distribution of its ``A``, so the first E-step already uses
    the model's own rule."""
    m = np.asarray(series, dtype=float)
    if m.ndim != 1 or len(m) < 2 or not np.isfinite(m).all():
        raise ValueError("the series must be a finite 1-D array of at least 2 dates")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(m), start.k) or not np.isfinite(weights).all() or (
            weights < 0
        ).any():
            raise ValueError(
                f"weights must be finite, nonnegative and (T, K) = ({len(m)}, {start.k}), "
                f"got shape {weights.shape}"
            )
    if initial not in INITIAL_RULES:
        raise ValueError(f"unknown initial rule {initial!r}; use one of {INITIAL_RULES}")
    params = start
    if initial == "stationary":
        params = HMMParams(start.means, start.variances, start.transition,
                           stationary_distribution(start.transition))
    scale = float(np.std(m)) or 1.0
    trace: list[tuple[float, float, float]] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, maxiter + 1):  # noqa: B007 - read after the loop
        est = e_step(m, params)
        new = m_step(m, est, params.transition, weights=weights, initial=initial,
                     variance_floor=variance_floor)
        q = expected_complete_loglik(m, est, new, params.transition, weights)
        change = _param_change(params, new, scale)
        trace.append((est.loglik, q, change))
        params = new
        if change < tol:
            converged = True
            break
    final = e_step(m, params)
    return BaumWelchResult(
        params=params, loglik=final.loglik, filtered=final.filtered, log_c=final.log_c,
        n_iter=n_iter, converged=converged, trace=tuple(trace),
    )


def filter_series(series: np.ndarray, params: HMMParams) -> tuple[np.ndarray, np.ndarray]:
    """The Hamilton filter at frozen ``params``: ``(filtered (T, K), log_c
    (T,))``. What the gate runs forward over training and test dates; it
    never re-estimates anything."""
    log_eta = gaussian_logpdf(np.asarray(series, dtype=float), params.means, params.variances)
    return forward(log_eta, params.transition, params.initial)


# --------------------------------------------------------------------------- #
# Conversion to and from statsmodels' constrained parameter vector
# --------------------------------------------------------------------------- #


def params_from_statsmodels(
    names: Sequence[str], vector: np.ndarray, k: int, initial: str = "stationary"
) -> HMMParams:
    """Read ``MarkovRegression``'s constrained parameter vector **by name**
    (``p[i->j]`` = P(z_t = j | z_{t-1} = i) for ``j < K - 1``, the last
    destination implicit; ``const[k]``; ``sigma2[k]``), as the start-value
    machinery writes it. The initial distribution is the stationary one
    under either rule (a start has no posterior yet)."""
    vec = np.asarray(vector, dtype=float)
    a = np.zeros((k, k))
    means = np.full(k, np.nan)
    variances = np.full(k, np.nan)
    for name, value in zip(names, vec, strict=True):
        if name.startswith("p["):
            src, dst = (int(x) for x in name[2:-1].split("->"))
            a[src, dst] = value
        elif name.startswith("const["):
            means[int(name[6:-1])] = value
        elif name.startswith("sigma2["):
            variances[int(name[7:-1])] = value
        else:
            raise ValueError(
                f"parameter {name!r} is outside the native model (order-0, switching "
                "mean and variance)"
            )
    a[:, k - 1] = 1.0 - a[:, : k - 1].sum(axis=1)
    if np.isnan(means).any() or np.isnan(variances).any():
        raise ValueError(f"the vector does not name every regime's mean and variance: {names}")
    del initial  # a start carries no posterior: the stationary rule under both
    return HMMParams(means=means, variances=variances, transition=a,
                     initial=stationary_distribution(a))


def params_to_statsmodels(params: HMMParams, names: Sequence[str]) -> np.ndarray:
    """The constrained statsmodels vector of ``params``, in ``names``' order."""
    out = np.empty(len(names))
    for i, name in enumerate(names):
        if name.startswith("p["):
            src, dst = (int(x) for x in name[2:-1].split("->"))
            out[i] = params.transition[src, dst]
        elif name.startswith("const["):
            out[i] = params.means[int(name[6:-1])]
        elif name.startswith("sigma2["):
            out[i] = params.variances[int(name[7:-1])]
        else:
            raise ValueError(f"parameter {name!r} is outside the native model")
    return out
