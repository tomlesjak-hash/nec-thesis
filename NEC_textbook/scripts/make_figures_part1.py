"""Figures for Part I. Deterministic (fixed seeds), numpy + matplotlib only.

Run from NEC_textbook/:  python3.14 scripts/make_figures_part1.py
Outputs land in figures/ as vector PDFs.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures"
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def spearman_null(n_names: int, n_dates: int, rng: np.random.Generator) -> np.ndarray:
    """Per-date Spearman IC of a zero-skill signal against iid returns."""
    ics = np.empty(n_dates)
    for t in range(n_dates):
        pred = rng.standard_normal(n_names)
        y = rng.standard_normal(n_names)
        rp = np.argsort(np.argsort(pred)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rp -= rp.mean()
        ry -= ry.mean()
        ics[t] = (rp @ ry) / (np.linalg.norm(rp) * np.linalg.norm(ry))
    return ics


def fig_null_ic() -> None:
    rng = np.random.default_rng(0)
    n = 100
    ics = spearman_null(n, 20_000, rng)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    ax1.hist(ics, bins=80, density=True, color="#7a9cc6", alpha=0.85)
    grid = np.linspace(-0.45, 0.45, 400)
    sd = 1.0 / np.sqrt(n - 1)
    ax1.plot(grid, np.exp(-grid**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)),
             color="#333333", lw=1.2, label=rf"$\mathcal{{N}}(0, 1/(N-1))$")
    ax1.set_xlabel("daily rank IC (zero skill, $N=100$)")
    ax1.set_ylabel("density")
    ax1.legend(frameon=False, fontsize=8)

    # one zero-skill strategy's running mean IC with 2SE bands
    run = ics[:750]
    t = np.arange(1, len(run) + 1)
    cum = np.cumsum(run) / t
    ax2.plot(t, cum, color="#b3543e", lw=1.0, label="running mean IC")
    ax2.plot(t, 2 * sd / np.sqrt(t), color="#666666", lw=0.8, ls="--",
             label=r"$\pm 2\,\mathrm{SE}$")
    ax2.plot(t, -2 * sd / np.sqrt(t), color="#666666", lw=0.8, ls="--")
    ax2.axhline(0.0, color="#bbbbbb", lw=0.6)
    ax2.set_xlabel("number of test dates $T$")
    ax2.set_ylabel(r"$\overline{\mathrm{IC}}$ after $T$ dates")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{FIG}/null_ic.pdf")
    plt.close(fig)


def fig_regimes() -> None:
    rng = np.random.default_rng(7)
    # 2-state Markov scale mixture: sticky chain, calm/turbulent vols
    n = 1500
    stay = 0.98
    sig = np.array([0.6, 2.2])
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    r = sig[s] * rng.standard_normal(n)

    fig = plt.figure(figsize=(6.6, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.5, wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(r, lw=0.4, color="#33507a")
    for t0 in np.flatnonzero(np.diff(np.concatenate([[0], s, [0]])) != 0).reshape(-1, 2):
        ax.axvspan(t0[0], t0[1], color="#d9a066", alpha=0.18, lw=0)
    ax.set_xlim(0, n)
    ax.set_xlabel("date")
    ax.set_ylabel("return")
    ax.set_title("two-state Markov scale mixture (shaded = turbulent state)", fontsize=9)

    # left: unconditional density is fat-tailed vs the moment-matched normal
    ax1 = fig.add_subplot(gs[1, 0])
    grid = np.linspace(-7, 7, 400)
    p1 = float(np.mean(s == 1))
    mix = ((1 - p1) * np.exp(-grid**2 / (2 * sig[0] ** 2)) / (sig[0] * np.sqrt(2 * np.pi))
           + p1 * np.exp(-grid**2 / (2 * sig[1] ** 2)) / (sig[1] * np.sqrt(2 * np.pi)))
    sd = np.sqrt((1 - p1) * sig[0] ** 2 + p1 * sig[1] ** 2)
    ax1.semilogy(grid, mix, color="#b3543e", lw=1.2, label="mixture")
    ax1.semilogy(grid, np.exp(-grid**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)),
                 color="#333333", lw=1.0, ls="--", label="normal, same variance")
    ax1.set_ylim(1e-6, 1)
    ax1.set_xlabel("return")
    ax1.set_ylabel("density (log scale)")
    ax1.legend(frameon=False, fontsize=8)

    # right: sign-flip regression — pooled fit is blind
    ax2 = fig.add_subplot(gs[1, 1])
    m = 400
    x = rng.standard_normal(m)
    reg = rng.integers(0, 2, m)
    beta = np.where(reg == 0, 1.5, -1.5)
    y = beta * x + 0.5 * rng.standard_normal(m)
    ax2.scatter(x[reg == 0], y[reg == 0], s=4, color="#33507a", alpha=0.6,
                label=r"regime 1: $\beta=+1.5$")
    ax2.scatter(x[reg == 1], y[reg == 1], s=4, color="#d9a066", alpha=0.7,
                label=r"regime 2: $\beta=-1.5$")
    b = np.polyfit(x, y, 1)
    gx = np.linspace(-3, 3, 2)
    ax2.plot(gx, np.polyval(b, gx), color="#b3543e", lw=1.4,
             label=rf"pooled fit: $\hat\beta={b[0]:+.2f}$")
    ax2.set_xlabel("feature $x$")
    ax2.set_ylabel("target $y$")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")

    fig.savefig(f"{FIG}/regimes.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_null_ic()
    fig_regimes()
    print("figures written to", FIG)
