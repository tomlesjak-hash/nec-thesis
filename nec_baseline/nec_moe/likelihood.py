"""The shared expert-likelihood core (design doc §7A).

Both thesis variations — memoryless routing and HMM gating — consume the same
per-expert, per-timestep predictive log-likelihood computed here:

    log N_k = log N(y | mu_k, sigma_k^2)
            = -1/2 log(2 pi) - log sigma_k - (y - mu_k)^2 / (2 sigma_k^2)

computed **directly in the log domain** (never exponentiate a density and take
the log afterwards — Module 8 Ex. 8.1's float64 cliff: the naive pipeline
underflows whenever every component's squared standardized residual exceeds
~1490, which fat-tailed returns hit routinely).

This module is deliberately a pure function: it has no parameters and no
state, which is what makes "same likelihood, different prior" literally true
in code.
"""

from __future__ import annotations

import math

from torch import Tensor

__all__ = ["expert_log_likelihood"]

_LOG_2PI = math.log(2.0 * math.pi)


def expert_log_likelihood(mu: Tensor, log_sigma: Tensor, y: Tensor) -> Tensor:
    """Per-expert Gaussian log-density of the observed target.

    Parameters
    ----------
    mu : ``(B, K)`` expert means.
    log_sigma : ``(K,)`` or ``(B, K)`` log noise scales (broadcast).
    y : ``(B,)`` observed targets.

    Returns
    -------
    ``(B, K)`` log-likelihoods ``log N(y_b | mu_bk, sigma_k^2)``.
    """
    if mu.ndim != 2:
        raise ValueError(f"mu must be (B, K), got {tuple(mu.shape)}")
    if y.ndim != 1 or y.shape[0] != mu.shape[0]:
        raise ValueError(
            f"y must be (B,) matching mu's batch, got {tuple(y.shape)} vs "
            f"{tuple(mu.shape)}"
        )
    resid = y.unsqueeze(-1) - mu  # (B, K)
    inv_var = (-2.0 * log_sigma).exp()  # 1 / sigma^2, broadcast
    return -0.5 * _LOG_2PI - log_sigma - 0.5 * resid.pow(2) * inv_var
