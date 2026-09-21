"""GP data layer + fitness — the bridge from the panel to genetic_factor.py.

Replaces three things in your QIP `genetic_factor.py`:

  1. `read_pickle()` / `load_data()`  -> `load_gp_data()` here
  2. the five raw-level terminals      -> `build_terminals()` here
  3. `calc_ir()` (quintile L/S, lag 2) -> `fitness()` here

Import from your GP instead of the CN loaders:

    from tq_gp import load_gp_data, build_terminals, fitness, TRAIN, VALID, TEST

    d  = load_gp_data(window=TRAIN)          # excludes BAD_TICKERS
    T  = build_terminals(d)                  # dimensionless only
    ir = fitness(expr_output, d)             # industry-neutral IR/sqrt(TVR)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tq_data import BAD_TICKERS, load_panel  # noqa: E402

PANEL = Path(__file__).resolve().parents[1] / "panel"

#: REVISED 2026-08-09. The first run trained on 2011-2017 only; its alphas hit
#: local daily IR 0.10-0.14 and then failed Trexsim's 0.07 gate three times.
#: Seven years is too narrow — those alphas had never seen 2008 at all, while
#: the platform evaluates across 2006-2022.
#:
#: Widened to 13 training years INCLUDING the financial crisis, while keeping a
#: real holdout. Training on the full 2006-2022 would leave nothing to
#: distinguish a genuine alpha from a fitted one before spending a submission,
#: and `valid_ir` is what caught the degenerate top-8 last time.
#:
#: Cost of the widening: 2006-2010 is where the 212 missing delisted names
#: cluster and where Wikipedia membership is unreliable, so training data is
#: dirtier. Accepted deliberately — the alphas are failing on regime
#: generalisation, not on noise.
TRAIN = ("2006-01-01", "2018-12-31")   # 13y, includes 2008
VALID = ("2019-01-01", "2020-12-31")   # COVID crash
TEST = ("2021-01-01", "2022-12-31")    # untouched until the very end
EARLY = ("2006-01-01", "2010-12-31")   # the biased sub-block, for reference

TRADING_DAYS = 252

#: Minimum median cross-sectional dispersion (std / median|x|). Below this the
#: alpha is effectively a constant and carries no cross-sectional information.
MIN_DISPERSION = 1e-4

#: Minimum mean daily turnover. The tutorial's own examples run 0.21-1.58.
MIN_TURNOVER = 0.01

#: Minimum average positions per day. Platform threshold is 160; keep local
#: search a little looser so near-misses are still visible in the output.
MIN_NUMSTK = 100

#: Minimum EFFECTIVE positions, 1/sum(w^2). Catches what MIN_NUMSTK cannot:
#: a book with 400 non-NaN positions where one name holds most of the weight.
MIN_EFFECTIVE_N = 50

#: Denominator floor for IR/sqrt(TVR) — stops the metric diverging as TVR -> 0.
TVR_FLOOR = 0.05

#: Round-trip trading cost in basis points, for the net-IR robustness column.
#: Your QIP code used 12 bps (China A-shares); US large-cap is nearer 5.
COST_BPS = 5.0

#: Trexsim's PRIMARY qualification gate: daily IR > 0.07. IR/sqrt(TVR) is only
#: the tie-break used by the correlation-override clause — an alpha below the
#: IR gate does not qualify at all, so its IR/sqrt(TVR) is irrelevant.
#: Without this, fitness prefers a low-turnover alpha at IR 0.053 over a real
#: one at IR 0.107, purely because dividing by a smaller sqrt(TVR) wins.
PLATFORM_IR_MIN = 0.07


# ------------------------------------------------------------------- data --

def load_gp_data(window: tuple[str, str] | None = TRAIN,
                 *, panel_dir: Path = PANEL,
                 universe_only: bool = True) -> dict[str, pd.DataFrame]:
    """Load the panel for one window, excluding corrupt tickers.

    `exclude=BAD_TICKERS` is applied here — this is THE answer to "how do we
    exclude bad data when training". It happens once, at load, so every
    terminal and the fitness function all see the same clean frame. The parquet
    files on disk are never modified.

    With `universe_only`, every 2D field is masked to the point-in-time
    universe so the GP can never see a stock it could not have traded.
    """
    p = load_panel(panel_dir, exclude=BAD_TICKERS)
    if not p:
        raise FileNotFoundError(f"no parquet files in {panel_dir}")

    if window is not None:
        lo, hi = window
        p = {k: v.loc[lo:hi] for k, v in p.items()}

    uni = p["universe"].astype(bool)
    if universe_only:
        wide = [k for k, v in p.items()
                if v.shape == uni.shape and k not in ("universe",)]
        for k in wide:
            p[k] = p[k].where(uni)

    p["universe"] = uni
    return p


# -------------------------------------------------------------- terminals --

def build_terminals(d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Dimensionless GP terminals.

    THIS IS THE FIX FOR THE CLONE PROBLEM. Your DEAP run was handed raw
    `closes/highs/lows/opens/amounts`. Price *levels* differ across stocks by
    orders of magnitude and are extremely stable, so the first thing any GP
    finds is the size factor — `neg(closes)` — and every hall-of-fame slot
    then fills with cosmetic variations of it (all 10 correlated 0.99+).

    Every terminal below is a ratio, a return, or a bounded rank. None carries
    a price level, so there is no size attractor to collapse onto. This also
    matches Trexsim's "make expressions dimensionless" rule, which protects
    against corporate-action retro-adjustment.
    """
    c, o, h, l = d["close"], d["open"], d["high"], d["low"]
    v, adv = d["volume"], d["adv60"]

    def safe(x):
        return x.replace(0, np.nan)

    t: dict[str, pd.DataFrame] = {}

    # price shape — all ratios
    t["c_ma20"] = c / safe(c.rolling(20, min_periods=10).mean())
    t["c_ma60"] = c / safe(c.rolling(60, min_periods=30).mean())
    t["hl"] = h / safe(l)
    t["co"] = (c - o) / safe(o)                       # intraday return
    t["gap"] = (o - c.shift(1)) / safe(c.shift(1))    # overnight return

    # stochastic position in the 20d range — bounded [0, 1]
    rng_hi = h.rolling(20, min_periods=10).max()
    rng_lo = l.rolling(20, min_periods=10).min()
    t["stoch"] = (c - rng_lo) / safe(rng_hi - rng_lo)

    # returns
    for n in (1, 5, 20):
        t[f"ret{n}"] = d[f"ret{n}"]

    # volume / liquidity — ratios against each stock's own history
    t["v_ma20"] = v / safe(v.rolling(20, min_periods=10).mean())
    t["dv_adv"] = d["dollar_volume"] / safe(adv)

    # realised vol, and size as a bounded cross-sectional rank
    t["vol20"] = d["ret1"].rolling(20, min_periods=10).std()
    t["size"] = adv.rank(axis=1, pct=True)

    # market context — genuinely cross-sectional despite the index being 1D
    if "ret1_spx" in d:
        spx = d["ret1_spx"].iloc[:, 0]
        t["beta60"] = d["ret1"].rolling(60, min_periods=30).corr(spx)

    # --- three terminals carrying information the others do not ------------
    # Amihud illiquidity: price impact per dollar traded. A long-documented
    # priced factor, and the only liquidity measure here that is not just a
    # volume ratio.
    t["amihud"] = (d["ret1"].abs() / safe(d["dollar_volume"])) * 1e9

    # Distance below the 52-week high — the anchor behind momentum and the
    # disposition effect. Bounded (0, 1].
    t["hi52"] = c / safe(c.rolling(252, min_periods=120).max())

    # Idiosyncratic return: today's move with the industry move removed. The
    # sim neutralises by industry, so the residual is what actually gets traded.
    if "industry" in d:
        g = d["industry"].iloc[0]
        ok = g.notna()
        if ok.any():
            grp = d["ret1"].loc[:, ok.values].T.groupby(g[ok]).transform("mean").T
            idio = d["ret1"].copy()
            idio.loc[:, ok.values] = d["ret1"].loc[:, ok.values] - grp
            t["idio"] = idio

    # WINSORISE EVERY TERMINAL cross-sectionally at 1%/99%.
    # Ratios like v_ma20 and dv_adv spike 50x on a news day. Unwinsorised, a
    # GP expression can earn most of its in-sample IR from a handful of extreme
    # observations — which is precisely the kind of edge that does not survive
    # a change of universe. Clipping costs almost nothing and removes that
    # degree of freedom from the search.
    out = {}
    for k, w in t.items():
        w = sanitize(w)
        lo = w.quantile(0.01, axis=1)
        hi = w.quantile(0.99, axis=1)
        out[k] = w.clip(lo, hi, axis=0).astype("float32")
    return out


# ---------------------------------------------------------------- fitness --

#: Values beyond this are treated as missing rather than kept.
MAX_MAGNITUDE = 1e15


def sanitize(a: pd.DataFrame) -> pd.DataFrame:
    """Drop non-finite AND absurd-magnitude cells.

    Replacing inf alone is not enough. `div` guards only against an EXACT zero
    denominator (matching Trexsim's `at_zero2nan`), so a denominator of, say,
    1e-300 yields ~1e300 — finite, but squaring it inside `.std()` overflows to
    inf and pandas raises
        RuntimeWarning: invalid value encountered in subtract
    from `(avg - values) ** 2`. Masking instead of clipping is deliberate: a
    stock whose value is meaningless should carry no position rather than an
    enormous one.
    """
    return (a.replace([np.inf, -np.inf], np.nan)
             .where(a.abs() < MAX_MAGNITUDE))


def _industry_neutral(a: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    """Subtract the industry mean each day, DROPPING unclassified names.

    Trexsim runs with `Neut=industry`, so an alpha that is really an industry
    bet scores well locally and badly on the platform unless you neutralize
    the same way. `industry` is constant over time, so group the COLUMNS once
    rather than looping days.

    The dropping matters. An earlier version left tickers with a NaN industry
    code untouched instead of excluding them. Since every classified name gets
    demeaned to ~0 under a constant alpha, the handful of UNCLASSIFIED names
    became the only non-zero positions — so the GP could score well by
    producing a constant and letting the book fall entirely on names whose
    industry code happens to be missing. Trexsim has no unclassified names
    (empty value 0), so excluding them here also matches the platform.
    """
    g = industry.iloc[0]
    valid = g.notna()
    if not valid.any():
        return a
    out = a.copy()
    out.loc[:, ~valid.values] = np.nan            # cannot neutralize -> do not trade
    sub = out.loc[:, valid.values].T.groupby(g[valid]).transform("mean").T
    out.loc[:, valid.values] = out.loc[:, valid.values] - sub
    return out


def fitness(alpha: pd.DataFrame, d: dict[str, pd.DataFrame],
            *, delay: int = 1, neutralize: bool = True,
            metric: str = "ir_over_sqrt_tvr") -> float:
    """Mirror Trexsim's scoring: industry-neutral, rank-weighted, dollar-neutral.

    Differs from your QIP `calc_ir` in four ways that all matter:
      - industry-neutralized (the sim's Neut=industry)
      - full cross-section, weighted, not top/bottom quintile buckets
      - dollar-neutral book with the median removed, as the sim does
      - returns IR/sqrt(TVR), the metric the correlation-override clause uses

    Returns -1.0 for a degenerate alpha so the GP discards it.
    """
    if not isinstance(alpha, pd.DataFrame) or alpha.shape != d["close"].shape:
        return -1.0

    a = sanitize(alpha).where(d["universe"])
    if a.notna().sum().sum() < 100:
        return -1.0

    # DEGENERACY GUARD 1 — an alpha with no cross-sectional spread carries no
    # information. `div(ret20, ret20)` is 1 everywhere; deeply nested
    # ts_max/ts_decay chains flatten to near-constants. Both scored highly
    # before this check existed.
    scale = a.abs().median(axis=1).replace(0, np.nan)
    disp = (a.std(axis=1) / scale).replace([np.inf, -np.inf], np.nan)
    if float(disp.fillna(0).median()) < MIN_DISPERSION:
        return -1.0

    if neutralize and "industry" in d:
        a = _industry_neutral(a, d["industry"])

    # WINSORISE THE ALPHA before sizing. Without it a single extreme value
    # takes most of the book: positions are value-weighted, so one cell at 100x
    # the others dominates the PnL. Trexsim has cs_winsor for exactly this.
    lo = a.quantile(0.01, axis=1)
    hi = a.quantile(0.99, axis=1)
    a = a.clip(lo, hi, axis=0)

    a = a.sub(a.median(axis=1), axis=0)              # sim subtracts the median
    gross = a.abs().sum(axis=1).replace(0, np.nan)
    pos = a.div(gross, axis=0)                       # unit gross, dollar-neutral

    # CONCENTRATION GUARD. `numstk` counts non-NaN positions, which misses the
    # case where one name holds 90% of the book and 400 others split the rest —
    # numstk reports 400, the alpha is really trading one stock. Effective N =
    # 1 / sum(w^2) (inverse Herfindahl) measures what is actually held.
    w2 = (pos ** 2).sum(axis=1)
    eff_n = (1.0 / w2.replace(0, np.nan)).mean()
    if not np.isfinite(eff_n) or eff_n < MIN_EFFECTIVE_N:
        return -1.0

    pnl = (pos.shift(delay) * d["ret1"]).sum(axis=1, min_count=1)
    tvr = (pos - pos.shift(1)).abs().sum(axis=1, min_count=1)

    pnl = pnl.dropna()
    if len(pnl) < 100 or pnl.std() == 0 or not np.isfinite(pnl.std()):
        return -1.0

    # DEGENERACY GUARD 3 — position count. The platform requires numstk > 160.
    # The 30-seed run produced four alphas holding 0.4-0.6 stocks per day: an
    # expression that is NaN almost everywhere concentrates the entire book in
    # one or two names, posts a spectacular in-sample IR (3.28) and then blows
    # up out of sample (valid_ir -2.40). Dispersion and turnover both look fine
    # for such an alpha, so neither earlier guard caught it.
    if float(pos.notna().sum(axis=1).mean()) < MIN_NUMSTK:
        return -1.0

    # DEGENERACY GUARD 2 — an alpha that never trades is not an alpha. The
    # tutorial's own worked examples run tvr 0.21-1.58; anything two orders
    # below that is a static holding, not a signal.
    t = float(tvr.mean())
    if not np.isfinite(t) or t < MIN_TURNOVER:
        return -1.0

    ir = np.sqrt(TRADING_DAYS) * pnl.mean() / pnl.std()
    if metric == "ir":
        return float(ir)
    # Floor the denominator: IR/sqrt(TVR) diverges as TVR -> 0, so without a
    # floor the GP maximises fitness by driving turnover to zero rather than
    # by finding signal. That is exactly what the first smoke test produced.
    score = ir / np.sqrt(max(t, TVR_FLOOR))

    # Soft gate on the platform's PRIMARY threshold. Squared so the pull is
    # firm, multiplicative so ordering is preserved among sub-threshold
    # candidates and the GP keeps a usable gradient early on.
    ir_daily = ir / np.sqrt(TRADING_DAYS)
    if ir_daily < PLATFORM_IR_MIN:
        score *= (max(ir_daily, 0.0) / PLATFORM_IR_MIN) ** 2
    return float(score)


def report(alpha: pd.DataFrame, d: dict[str, pd.DataFrame],
           **kw) -> dict[str, float]:
    """Full metric set for a candidate — IR, turnover, drawdown, stock count."""
    a = sanitize(alpha).where(d["universe"])
    if kw.get("neutralize", True) and "industry" in d:
        a = _industry_neutral(a, d["industry"])
    a = a.sub(a.median(axis=1), axis=0)
    pos = a.div(a.abs().sum(axis=1).replace(0, np.nan), axis=0)
    pnl = (pos.shift(kw.get("delay", 1)) * d["ret1"]).sum(axis=1, min_count=1).dropna()
    tvr = (pos - pos.shift(1)).abs().sum(axis=1, min_count=1)
    nav = (1 + pnl).cumprod()
    ir = np.sqrt(TRADING_DAYS) * pnl.mean() / pnl.std() if pnl.std() else np.nan
    t = float(tvr.mean())

    # NET of trading costs — borrowed from your QIP calc_net_ir, but repriced.
    # QIP used 12 bps two-way (China A-shares); US large-cap is nearer 5 bps.
    # This is the honest check on a high-turnover alpha: gross IR can look fine
    # while net IR is halved.
    net = pnl - 0.5 * COST_BPS * 1e-4 * tvr.reindex(pnl.index).fillna(0)
    net_ir = (np.sqrt(TRADING_DAYS) * net.mean() / net.std()
              if net.std() else np.nan)

    return {
        "ir": float(ir),
        "ret": float(TRADING_DAYS * pnl.mean()),
        "tvr": t,
        # floored to match `fitness` — otherwise the CSV and the ranking metric
        # disagree for any low-turnover alpha
        "ir_over_sqrt_tvr": float(ir / np.sqrt(max(t, TVR_FLOOR))),
        "net_ir": float(net_ir),
        "max_dd": float(-(nav / nav.cummax() - 1).min()),
        "numstk": float(pos.notna().sum(axis=1).mean()),
        # platform IR is per-day, not annualised: threshold is 0.07
        "ir_daily": float(ir / np.sqrt(TRADING_DAYS)),
        "days": int(len(pnl)),
    }
