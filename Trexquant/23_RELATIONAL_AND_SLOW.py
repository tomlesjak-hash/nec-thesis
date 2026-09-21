# =============================================================================
# RELATIONAL & SLOW-HORIZON SWEEP
# =============================================================================
# WHAT IS ACTUALLY MISSING FROM THE POOL
# --------------------------------------
# Every alpha built so far ranks a stock on ITS OWN characteristics -- its own
# price path, its own trading hour, its own accounting. Two things are absent:
#
#   1. RELATIONAL signals. How the stock moves WITH something else: the index,
#      volatility, gold. The data dictionary states this explicitly --
#      "rolling beta of each stock to SPX/VIX/GLD is a genuine CROSS-SECTIONAL
#      signal even though the index itself is 1D." Never used.
#
#   2. SLOW signals. A1/A2/A3 are all 1-day-horizon. Nothing in the pool looks
#      back further than ~60 days, and the whole momentum family -- the most
#      replicated anomaly in the literature -- is missing entirely.
#      A2 and A3 are also reversal-flavoured, so a momentum alpha is closer to
#      structurally OPPOSITE than merely different.
#
# Both are near-mechanically decorrelated from what you hold: a 250-day signal
# and a 1-day signal cannot share much variance, because they turn the book
# over on incompatible timescales.
#
# ALL UNGATED. If the shared event gate has been the correlating factor, these
# remove it entirely. Add `* w` only if a signal misses IR on its own.
#
# ALL SLOW -> expect TVR 0.02-0.15, i.e. break-even in the tens of bp.
#
# WARM-UP WARNING: the 250-day windows consume a year of sample. Expect the
# start date to jump to ~2008 and slow mode to start later still. That is
# warm-up being consumed correctly, NOT a bug -- but it means these alphas are
# measured across the crisis from a standing start, so check the drawdown.
#
# SIGN: if IR comes back negative, flip it. Noted per-block where the
# literature has a strong prior on direction.
# =============================================================================


# =============================================================================
# TIER 1 — RELATIONAL. The genuinely absent dimension. START HERE.
# =============================================================================

# --- R1. Market beta — the low-beta anomaly ----------------------------------
# Frazzini & Pedersen "Betting Against Beta". PUBLISHED, robust, and one of the
# most replicated results in finance -- low-beta stocks earn higher
# risk-adjusted returns because leverage-constrained investors bid up high-beta
# names instead of borrowing.
# STRONG PRIOR ON SIGN: short high beta, i.e. NEGATIVE. Written that way.
beta  = ts_corr_binary(ret1, ret1_spx, 250)
score = -(cs_rank(beta) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R2. Gold beta -----------------------------------------------------------
# GENUINELY UNMINED. Equity factor models use market, size, value, momentum,
# quality, vol -- almost never commodity exposure. But a stock's covariance
# with gold proxies its sensitivity to real rates, inflation expectations and
# safe-haven demand, none of which the standard factors span.
# No strong prior on sign. Try as written, then flipped.
gbeta = ts_corr_binary(ret1, ret1_gld, 120)
score = cs_rank(gbeta) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R3. VIX beta ------------------------------------------------------------
# How the stock responds to VOLATILITY shocks, not to market direction. Related
# to Ang-Hodrick-Xing-Zhang's volatility-innovation factor (published), but the
# 60-day rolling per-stock construction is not their portfolio-sort method.
# Stocks that rally when VIX spikes are natural hedges and should be expensive
# -> lower expected return -> expect the NEGATIVE sign to win.
vbeta = ts_corr_binary(ret1, ret1_vix, 60)
score = -(cs_rank(vbeta) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R4. Beta INSTABILITY — original -----------------------------------------
# Not the level of beta, its VOLATILITY. How much does this stock's market
# relationship wander?
#
# Thesis: unstable beta is unhedgeable risk. A stock whose market sensitivity
# jumps around cannot be hedged with a fixed index position, so anyone holding
# it carries residual exposure they did not choose. That should command a
# premium -- and it is invisible to every standard factor model, all of which
# assume beta is a constant to be estimated rather than a process.
#
# I have not seen this as a named factor. It is the most original entry here.
bser  = ts_corr_binary(ret1, ret1_spx, 60)
inst  = ts_std(bser, 120)
score = cs_rank(inst) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R5. Beta term structure — original --------------------------------------
# Short-window beta minus long-window beta: is this stock's market sensitivity
# RISING or FALLING relative to its own norm?
#
# Rising beta means the name is being repriced as a macro asset rather than on
# its fundamentals -- typically what happens as a story stock becomes a
# crowded, correlated position. Falling beta means it is decoupling.
# Differencing two betas removes the stock's static beta level, so this is
# orthogonal to R1 by construction rather than by hope.
b_fast = ts_corr_binary(ret1, ret1_spx, 60)
b_slow = ts_corr_binary(ret1, ret1_spx, 250)
score  = cs_rank(b_fast - b_slow) - 1.0
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R6. Idiosyncratic share — original construction -------------------------
# What fraction of this stock's movement is NOT explained by the market.
# Correlation squared is R-squared for a single regressor, so 1 - corr^2 is
# the idiosyncratic share of variance.
#
# Low R-squared means the stock trades on its own information. Related to the
# published idiosyncratic-volatility literature, but that measures the LEVEL of
# residual vol (which is mostly just total vol); this measures the PROPORTION,
# which is scale-free and a different quantity.
b2    = ts_corr_binary(ret1, ret1_spx, 120)
idio  = 1.0 - (b2 * b2)
score = cs_rank(idio) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — SLOW HORIZON. The momentum family, absent from the pool entirely.
# =============================================================================

# --- M1. 12-1 momentum -------------------------------------------------------
# Jegadeesh & Titman. PUBLISHED and the single most replicated cross-sectional
# anomaly there is. The "skip the last month" convention is not cosmetic: the
# most recent month REVERSES, so including it fights the signal. Hence
# ts_delay(close, 20) in the numerator rather than close.
# Note this makes it near-orthogonal to your reversal alphas by construction --
# it deliberately excludes exactly the window they trade.
mom   = ts_delay(close, 20) / at_zero2nan(ts_delay(close, 250))
score = cs_rank(mom) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- M2. 52-week high proximity ----------------------------------------------
# George & Hwang. Published, and empirically it DOMINATES plain momentum in
# their tests. How close is the stock to its own 12-month high -- an anchoring
# effect: traders treat the 52-week high as a reference point and under-react
# to news that should push price through it.
# Bounded [0,1] by construction, so it needs no normalisation.
prox  = close / at_zero2nan(ts_max(high, 250))
score = cs_rank(prox) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- M3. Information discreteness --------------------------------------------
# Da, Gurun & Warachka, "Frog in the Pan". Momentum delivered in many small
# steps is under-reacted to and persists; momentum delivered in a few large
# jumps is noticed and does not.
#
# ts_mean of a boolean gives the FRACTION of up-days over the window -- a
# continuous signal that is high for a steady grinder and near 0.5 for a stock
# that went nowhere in a straight line punctuated by jumps.
sgn   = ret1 / at_zero2nan(at_signsqrt(ret1) * at_signsqrt(ret1))   # = sign(ret1), float
disc  = ts_mean(sgn, 250)
score = cs_rank(disc) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- M4. Momentum quality — original combination -----------------------------
# 12-1 momentum scaled by HOW SMOOTHLY it was delivered. This crosses M1 with
# M3: rank both, multiply.
#
# The premise is that neither leg is the signal -- the interaction is. A stock
# up 40% in steady increments and a stock up 40% on two gap-ups are the same
# under M1 and different under this. Frog-in-the-Pan tests the two legs in a
# double sort; this is the continuous version.
m = cs_rank(ts_delay(close, 20) / at_zero2nan(ts_delay(close, 250))) - 1.0
sg = ret1 / at_zero2nan(at_signsqrt(ret1) * at_signsqrt(ret1))
d = cs_rank(ts_mean(sg, 250)) - 1.0
score = m * d
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — SEASONALITY AS SIGNAL. You have used the calendar fields only as
# gates. This uses the calendar dimension as the signal itself.
# =============================================================================

# --- S1. Same-calendar-period return — Heston & Sadka ------------------------
# Stocks that performed well in a given month HISTORICALLY tend to perform well
# in that month again -- a genuine and persistent annual seasonality that
# survives controls for the standard factors.
#
# ts_delay(ret20, 250) is roughly this stock's return over the same calendar
# window one year ago; smoothing over 5 days blurs the trading-day misalignment
# between years (250 is an approximation of a calendar year).
#
# CAUTION: this is one of the more fragile published effects and the 250-day
# alignment is imprecise. Low prior, but it is cheap and completely orthogonal
# to everything you hold.
seas  = ts_mean(ts_delay(ret20, 250), 5)
score = cs_rank(seas) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER AND WHAT TO DO NEXT
# =============================================================================
# 1. R1, M1, M2 first -- the three with the strongest published priors. If the
#    platform cannot reproduce low-beta, 12-1 momentum, or 52-week high, that
#    is a fact about the universe or sample worth knowing BEFORE spending runs
#    on the original constructions.
# 2. Then R4, R5, R6 -- the original relational ideas, and the best interview
#    material in this file.
# 3. Then M3, M4, R2, R3, S1.
#
# IF A SIGNAL CLEARS:
#   * Industry-neutralise it. Beta and momentum are both heavily sector-driven
#     -- utilities are low-beta as a sector, tech was momentum for a decade --
#     so the raw version is partly a sector bet. The neutralised version is a
#     separate alpha, not a tweak, and frequently the better one:
#         alpha = at_zero2nan(cs_winsor(cs_indneut(score, industry), 0.01,
#                                       remove_extreme=False))
#   * THEN try the event gate. Slow signal x event weighting is a combination
#     you have not tested -- every gated alpha so far has had a fast core.
#
# IF EVERYTHING HERE FAILS TOO: the constraint is not the signal, it is that
# three alphas sharing one gate have crowded the correlation space. At that
# point the highest-value move is re-tuning A1/A2/A3 for IR rather than hunting
# alpha #4 -- combined OS performance is the other half of the score, and you
# have never re-walked fomc_pre on A3 (note 18, section 5, finding 3).
# =============================================================================
