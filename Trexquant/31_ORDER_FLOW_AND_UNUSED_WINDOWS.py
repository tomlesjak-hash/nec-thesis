# =============================================================================
# ORDER FLOW, PRICE IMPACT, AND THE TIME WINDOWS YOU HAVE NEVER USED
# =============================================================================
# WHAT IS STILL MISSING
# ---------------------
# 1. SIGNED ORDER FLOW. Every intraday alpha you have built INFERS imbalance
#    indirectly -- from return skew (A2), from close-vs-VWAP (A4), from volume
#    concentration. You can construct it DIRECTLY. The Lee-Ready tick rule
#    signs volume by the concurrent price move, and you have both hourly
#    returns and hourly volume. This is the actual variable the whole
#    microstructure literature is built on.
#
# 2. PRICE IMPACT. Kyle's lambda / Amihud illiquidity = how far price moves per
#    unit of volume. You have never measured it, and the intraday version is
#    much cleaner than the daily one because the window is short enough that
#    the move and the volume are genuinely paired.
#
# 3. TWO UNUSED TIME WINDOWS. You have first-hour and last-hour OHLC plus daily
#    OHLC -- which means MIDDAY is derivable and you have never touched it, and
#    so is the share of the day's range made in each window.
#
# All fast, all microstructure, all in the Sharpe band that can clear 1.11.
#
# SIGN CONVENTION HELPER used throughout:
#   at_signsqrt(x) * at_signsqrt(x) = |x|,  so  x / |x| = sign(x).
# =============================================================================


# =============================================================================
# TIER 1 — SIGNED ORDER FLOW.  The variable everything else has been proxying.
# =============================================================================

# --- O1. Closing-hour order imbalance ----------------------------------------
# Lee & Ready (1991): sign volume by the concurrent price move. Applied to the
# closing hour, then accumulated over 5 days and normalised by total volume, so
# it is bounded in [-1, +1] and dimensionless.
#
# This is the direct measurement of what A2 infers from skew. Same economic
# object, completely different statistic -- so it may either decorrelate
# cleanly or collide hard. If it collides, that is itself a useful finding: it
# means A2's skew really is measuring imbalance, which is currently only an
# assumption in note 17.
#
# SIGN: positive = continuation (persistent buy pressure keeps pushing).
# Flip if IR is negative -- that reading is that closing imbalance is
# uninformed and reverts, which is the note-15 mechanism.
sgn   = return_last_hour / at_zero2nan(at_signsqrt(return_last_hour) * at_signsqrt(return_last_hour))
oflow = sgn * volume_last_hour
sig   = ts_sum(oflow, 5) / at_zero2nan(ts_sum(volume_last_hour, 5))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- O2. Opening-hour order imbalance ----------------------------------------
# The open is retail and overnight-news driven; the close is institutional.
# Different clientele, so this should decorrelate from O1 even though the
# construction is identical.
sgn   = return_first_hour / at_zero2nan(at_signsqrt(return_first_hour) * at_signsqrt(return_first_hour))
oflow = sgn * volume_first_hour
sig   = ts_sum(oflow, 5) / at_zero2nan(ts_sum(volume_first_hour, 5))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- O3. Imbalance divergence — open versus close ----------------------------
# Retail buying at the open while institutions sell into the close is a
# specific, meaningful configuration: the two clienteles disagree, and the
# literature says the institutional side is the informed one.
#
# Differencing also removes any market-wide flow common to both windows,
# isolating the disagreement rather than the direction.
sl    = return_last_hour  / at_zero2nan(at_signsqrt(return_last_hour) * at_signsqrt(return_last_hour))
sf    = return_first_hour / at_zero2nan(at_signsqrt(return_first_hour) * at_signsqrt(return_first_hour))
il    = ts_sum(sl * volume_last_hour, 5)  / at_zero2nan(ts_sum(volume_last_hour, 5))
iff   = ts_sum(sf * volume_first_hour, 5) / at_zero2nan(ts_sum(volume_first_hour, 5))
score = cs_rank(il - iff) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- O4. Imbalance persistence -----------------------------------------------
# Not the level of imbalance -- how CONSISTENTLY it has pointed the same way.
# A single day's imbalance is noise; the same sign for twenty days is an order
# being worked. ts_mean of the sign alone discards magnitude entirely, which is
# the point: it measures direction agreement, not size.
sgn   = return_last_hour / at_zero2nan(at_signsqrt(return_last_hour) * at_signsqrt(return_last_hour))
score = cs_rank(ts_mean(sgn, 20)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — PRICE IMPACT.  Kyle's lambda, measured intraday.
# =============================================================================
# Amihud (2002) illiquidity = |return| / dollar volume. As a LEVEL this is
# mostly a size proxy and ranks the same names forever -- that is the clone
# trap that killed the first GP run. The tradable version is the SHOCK:
# how much did it cost to move this stock TODAY versus its own recent norm.
#
# A liquidity-supply shock is the Nagel mechanism stated directly, at the
# single-stock level rather than through VIX. Gate B tested it in calendar
# time; this tests it cross-sectionally, which gate B could not.

# --- I1. Closing-hour impact shock -------------------------------------------
absr  = at_signsqrt(return_last_hour) * at_signsqrt(return_last_hour)
imp   = absr / at_zero2nan(volume_last_hour)
shock = imp / at_zero2nan(ts_mean(imp, 20))
score = cs_rank(shock) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- I2. Impact asymmetry, close versus open ---------------------------------
# Is this name harder to move at the close than at the open? A high ratio says
# liquidity evaporates into the auction for this stock specifically -- which is
# exactly where closing-auction alphas should be strongest, and it is a
# stock-level property rather than a market-wide one.
al    = at_signsqrt(return_last_hour)  * at_signsqrt(return_last_hour)
af    = at_signsqrt(return_first_hour) * at_signsqrt(return_first_hour)
il    = al / at_zero2nan(volume_last_hour)
iff   = af / at_zero2nan(volume_first_hour)
score = cs_rank(il / at_zero2nan(iff)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — THE TIME WINDOWS YOU HAVE NEVER USED.
# =============================================================================

# --- M1. Midday return -------------------------------------------------------
# You have first-hour close and last-hour open, so the MIDDLE of the session is
# derivable and you have never looked at it.
#
# Midday is the quietest part of the day: overnight news is absorbed, the
# closing auction has not started, retail participation is lowest. It is the
# window most dominated by patient institutional execution and least polluted
# by noise traders -- arguably the cleanest read on real demand in the whole
# session, and certainly the least mined.
mid   = (open_price_last_hour - close_price_first_hour) / at_zero2nan(close_price_first_hour)
score = cs_rank(ts_mean(mid, 5)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- M2. Midday versus bookends ----------------------------------------------
# Long names that drift up quietly midday while giving it back at the bookends.
# Patient accumulation versus noisy round-tripping -- and differencing removes
# whatever moved the stock on the day as a whole.
mid   = (open_price_last_hour - close_price_first_hour) / at_zero2nan(close_price_first_hour)
ends  = return_first_hour + return_last_hour
score = cs_rank(ts_mean(mid, 5) - ts_mean(ends, 5)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- M3. Range timing --------------------------------------------------------
# What SHARE of the day's total range was made in the closing hour. Bounded
# [0,1] by construction, no normalisation needed.
#
# A day whose entire range was made at the close is a day where information or
# flow arrived late. Distinct from volume concentration (already tested,
# failed): this measures where PRICE moved, not where shares traded, and the
# two diverge exactly when the closing hour is thin.
rng_l = high_price_last_hour - low_price_last_hour
rng_d = high - low
score = cs_rank(rng_l / at_zero2nan(rng_d)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- M4. Range efficiency ----------------------------------------------------
# Net move divided by total range travelled in the closing hour. Near 1 means
# the hour trended cleanly in one direction; near 0 means it churned and went
# nowhere. A directional order produces a trend; two-sided liquidity provision
# produces churn.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
score = cs_rank(net / at_zero2nan(high_price_last_hour - low_price_last_hour)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 4 — INDUSTRY-RELATIVE MICROSTRUCTURE.  Cheap, and never tried.
# =============================================================================
# Every microstructure signal you have built is ranked against the WHOLE
# universe. But closing-hour behaviour is strongly sector-driven -- utilities
# and REITs carry far heavier index-fund closing flow than small tech names --
# so a universe-wide rank is partly a sector bet.
#
# cs_indneut asks instead: is this stock's closing-hour behaviour unusual FOR
# ITS SECTOR. That is a different question and a different alpha, not a tweak.
# It is also the cheapest thing in this file to test.

# --- N1. Industry-relative closing-hour skew ---------------------------------
# A2's exact variable, sector-neutralised.
score = cs_rank(cs_indneut(skew_return_last_hour, industry)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- N2. Industry-relative order imbalance -----------------------------------
sgn   = return_last_hour / at_zero2nan(at_signsqrt(return_last_hour) * at_signsqrt(return_last_hour))
sig   = ts_sum(sgn * volume_last_hour, 5) / at_zero2nan(ts_sum(volume_last_hour, 5))
score = cs_rank(cs_indneut(sig, industry)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. O1, O3  -- signed order flow. The genuinely absent variable, and the
#               strongest theoretical footing in the file.
# 2. M1, M3  -- untouched time windows. Nobody mines midday.
# 3. I1      -- impact shock: Nagel's mechanism cross-sectionally, which is
#               what gate B could not test.
# 4. N1, N2  -- cheapest runs here, one operator each.
# 5. O2, O4, I2, M2, M4.
#
# ADD THE EVENT GATE to anything clearing 0.05 ungated -- that has been worth
# +0.014 to +0.036 every time, and all of these are fast cores so the
# mechanism applies.
#
# IF O1 COLLIDES WITH A2: that is a genuinely useful result, not a wasted run.
# It would mean closing-hour skew really is measuring order imbalance, which
# note 17 currently ASSUMES without evidence. Record it either way -- it is a
# better answer to "how do you know your mechanism is right" than anything
# currently in that note.
# =============================================================================
