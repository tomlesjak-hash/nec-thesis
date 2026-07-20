"""Deterministic figures for Part II (Chapters 4-8). Run from NEC_textbook/:

    python3.14 scripts/make_figures_part2.py

numpy + matplotlib only; every figure is seeded and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

BLUE, RED, GOLD, GREY = "#2B4C7E", "#A84C32", "#8A7B4D", "#888888"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


# ------------------------------------------------------------------ EM bound
def em_bound() -> None:
    rng = np.random.default_rng(4)
    y = np.concatenate([rng.normal(-2.0, 0.7, 120), rng.normal(2.0, 0.7, 80)])
    w, s = np.array([0.5, 0.5]), 0.7
    mu2 = 2.0                                  # fixed; theta = mu1 varies

    def comp(yv, m):
        return np.exp(-0.5 * ((yv - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    def loglik(m1):
        return np.log(w[0] * comp(y, m1) + w[1] * comp(y, mu2)).mean()

    def elbo(m1, m1_anchor):
        p1 = w[0] * comp(y, m1_anchor)
        p2 = w[1] * comp(y, mu2)
        r1 = p1 / (p1 + p2)
        r2 = 1.0 - r1
        t1 = r1 * (np.log(w[0] * comp(y, m1)) - np.log(np.maximum(r1, 1e-300)))
        t2 = r2 * (np.log(w[1] * comp(y, mu2)) - np.log(np.maximum(r2, 1e-300)))
        return (t1 + t2).mean()

    def em_step(m1_anchor):
        p1 = w[0] * comp(y, m1_anchor)
        r1 = p1 / (p1 + w[1] * comp(y, mu2))
        return (r1 * y).sum() / r1.sum()

    grid = np.linspace(-4.5, 1.5, 400)
    m_a = 0.9                                  # deliberately bad start
    m_b = em_step(m_a)
    m_c = em_step(m_b)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(grid, [loglik(m) for m in grid], color="k", lw=1.6,
            label=r"log-likelihood $\ell(\mu_1)$")
    ax.plot(grid, [elbo(m, m_a) for m in grid], color=BLUE, lw=1.2,
            label=r"ELBO at $\mu_1^{(0)}$")
    ax.plot(grid, [elbo(m, m_b) for m in grid], color=RED, lw=1.2,
            label=r"ELBO at $\mu_1^{(1)}$")
    for m, c in [(m_a, BLUE), (m_b, RED), (m_c, "k")]:
        ax.axvline(m, color=c, lw=0.7, ls=":", alpha=0.8)
        ax.plot([m], [loglik(m)], "o", color=c, ms=4)
    ax.annotate(r"$\mu_1^{(0)}$", (m_a, loglik(m_a)), textcoords="offset points",
                xytext=(6, -14), color=BLUE)
    ax.annotate(r"$\mu_1^{(1)}$", (m_b, loglik(m_b)), textcoords="offset points",
                xytext=(6, -14), color=RED)
    ax.annotate(r"$\mu_1^{(2)}$", (m_c, loglik(m_c)), textcoords="offset points",
                xytext=(6, 6), color="k")
    ax.set_xlabel(r"$\mu_1$ (all other parameters fixed)")
    ax.set_ylabel("mean log-likelihood / ELBO")
    ax.set_ylim(-3.4, -1.9)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "em_bound.pdf")
    plt.close(fig)


# ------------------------------------------------------- Gumbel on the simplex
def gumbel_simplex() -> None:
    rng = np.random.default_rng(7)
    z = np.array([0.8, 0.2, -0.5])
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

    def project(p):                            # barycentric -> 2-D triangle
        return p @ corners

    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
    for ax, tau in zip(axes, [5.0, 1.0, 0.3]):
        g = -np.log(-np.log(rng.uniform(1e-9, 1 - 1e-9, (400, 3))))
        a = (z + g) / tau
        p = np.exp(a - a.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        xy = project(p)
        tri = plt.Polygon(corners, closed=True, fill=False, color=GREY, lw=0.8)
        ax.add_patch(tri)
        ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.35, color=BLUE, lw=0)
        m = project(np.exp(z) / np.exp(z).sum())
        ax.plot([m[0]], [m[1]], "o", color=RED, ms=5, zorder=5)
        ax.set_title(rf"$\tau = {tau}$", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        for c, lab in zip(corners, ["$k{=}1$", "$k{=}2$", "$k{=}3$"]):
            ax.annotate(lab, c, textcoords="offset points",
                        xytext=(0, -9 if c[1] == 0 else 4),
                        ha="center", fontsize=8, color=GREY)
    fig.suptitle(r"samples of $\mathrm{softmax}((z+g)/\tau)$;"
                 r"  red dot $=\mathrm{softmax}(z)$", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "gumbel_simplex.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- forward filter
def hmm_filter() -> None:
    rng = np.random.default_rng(11)
    T, stay = 500, 0.97
    sig = np.array([0.6, 2.0])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.uniform() < stay else 1 - s[t - 1]
    r = rng.normal(0.0, sig[s])

    A = np.array([[stay, 1 - stay], [1 - stay, stay]])
    log_a = np.log(A)

    def loglik(x):                             # (2,) per-state log density
        return -0.5 * np.log(2 * np.pi) - np.log(sig) - 0.5 * (x / sig) ** 2

    log_r = np.log(np.array([0.5, 0.5]))
    filt = np.zeros(T)
    for t in range(T):
        log_prior = np.logaddexp(log_r[0] + log_a[0], log_r[1] + log_a[1])
        a = log_prior + loglik(r[t])
        log_r = a - np.logaddexp(a[0], a[1])
        filt[t] = np.exp(log_r[1])

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1.0]})
    ax = axes[0]
    ax.plot(r, color=BLUE, lw=0.5)
    for t0, t1 in _spans(s == 1):
        ax.axvspan(t0, t1, color=RED, alpha=0.12, lw=0)
    ax.set_ylabel("return")
    ax = axes[1]
    ax.plot(filt, color=RED, lw=1.0,
            label=r"filtered $\mathbb{P}(S_t = \mathrm{turb} \mid y_{1:t})$")
    for t0, t1 in _spans(s == 1):
        ax.axvspan(t0, t1, color=RED, alpha=0.12, lw=0)
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("date $t$")
    ax.set_ylabel("probability")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "hmm_filter.pdf")
    plt.close(fig)


def _spans(mask):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return zip(idx[::2], idx[1::2])


# ------------------------------------------------------- symmetric landscape
def symmetry_landscape() -> None:
    rng = np.random.default_rng(3)
    n, beta, s_eps = 4000, 1.5, 0.4
    x = rng.standard_normal(n)
    flip = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0)
    y = flip * beta * x + rng.normal(0.0, s_eps, n)
    # model sigma = std(y): the package's sigma_init regime -- responsibilities
    # start gentle, which is exactly when the saddle's plateau is binding
    s_mod = float(y.std())

    def nll(b1, b2):
        d1 = np.exp(-0.5 * ((y - b1 * x) / s_mod) ** 2)
        d2 = np.exp(-0.5 * ((y - b2 * x) / s_mod) ** 2)
        return -np.log(0.5 * (d1 + d2) / (s_mod * np.sqrt(2 * np.pi))
                       + 1e-300).mean()

    def grad(b1, b2):
        d1 = np.exp(-0.5 * ((y - b1 * x) / s_mod) ** 2)
        d2 = np.exp(-0.5 * ((y - b2 * x) / s_mod) ** 2)
        r1 = d1 / (d1 + d2)
        r2 = 1.0 - r1
        g1 = -(r1 * (y - b1 * x) * x).mean() / s_mod**2
        g2 = -(r2 * (y - b2 * x) * x).mean() / s_mod**2
        return np.array([g1, g2])

    def descend(b0, steps, lr=0.25):
        path = [np.array(b0, dtype=float)]
        for _ in range(steps):
            path.append(path[-1] - lr * grad(*path[-1]))
        return np.array(path)

    grid = np.linspace(-2.4, 2.4, 121)
    zz = np.array([[nll(b1, b2) for b1 in grid] for b2 in grid])

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cs = ax.contourf(grid, grid, zz, levels=24, cmap="Blues_r", alpha=0.85)
    fig.colorbar(cs, ax=ax, label="mixture NLL", shrink=0.85)
    ax.plot(grid, grid, color=GREY, lw=0.8, ls=":",
            label=r"symmetric subspace $b_1 = b_2$")

    # equal step budgets (60): the warm start converges in ~51 steps; the
    # near-symmetric start (delta = 1e-5) needs ~197 -- still pinned at the
    # saddle here, escaping only after ~4x the budget (Ex. 8.3's law:
    # ~25 extra steps per decade of initial symmetry, measured)
    p1 = descend([0.10, 0.10001], 60)
    p2 = descend([0.90, -0.60], 60)
    ax.plot(p1[:, 0], p1[:, 1], color=RED, lw=1.4,
            label="GD from near-symmetric init")
    ax.plot(p2[:, 0], p2[:, 1], color=GOLD, lw=1.4,
            label="GD from warm-started init")
    for p in (p1, p2):
        ax.plot([p[0, 0]], [p[0, 1]], "o", color="k", ms=4, zorder=6)
        ax.plot([p[-1, 0]], [p[-1, 1]], "*", color="k", ms=9, zorder=6)
    ax.plot([0], [0], "s", color="k", ms=5)
    ax.annotate("pooled point\n(saddle)", (0, 0), textcoords="offset points",
                xytext=(8, 8), fontsize=8)
    ax.set_xlabel(r"expert-1 slope $b_1$")
    ax.set_ylabel(r"expert-2 slope $b_2$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "symmetry_landscape.pdf")
    plt.close(fig)


if __name__ == "__main__":
    em_bound()
    gumbel_simplex()
    hmm_filter()
    symmetry_landscape()
    print("figures written to", FIG)
