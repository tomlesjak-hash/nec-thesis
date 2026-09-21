# =============================================================================
# LIQUIDITY RESILIENCE & PASSIVE-OWNERSHIP DYNAMICS
# =============================================================================
# WHAT JUST WORKED, AND WHY IT POINTS HERE
# ----------------------------------------
#   P2  mandated-flow-WEIGHTED core   IR 0.106, Sharpe 1.68   <- best you own
#   P8  event-day liquidity DEPTH     IR 0.096, TVR 0.042, break-even 73bp
#
# P8's `cheap` = event-day volume / event-day |return| is INVERSE PRICE IMPACT
# measured on mandated-flow days. You are LONG the names that absorb forced
# flow without moving -- deep, heavily-indexed mega-caps.
#
# Both hits come from the same idea: measure a stock's RELATIONSHIP TO
# MANDATED FLOW as a slow characteristic. That idea has three dimensions and
# you have used one:
#
#   HOW MUCH flow arrives        -> P1/P2 (mflow share)          used
#   HOW DEEP the book is for it  -> P8 (cheap)                    used
#   HOW LIQUIDITY CHANGES        -> untouched
#   HOW IT IS TRENDING           -> untouched
#
# The last two are below. Both inherit P8's profile: 250-day windows, tiny
# turnover, break-evens in the tens of bp.
#
# NOTE THE ALGEBRA: in P8 the ts_sum(stress) terms cancel between numerator and
# denominator, so `cheap` reduces to
#     ts_sum(volume*stress, N) / ts_sum(absr*stress, N)
# which is what is used directly below.
#
# STRESS MUST BE FLOAT. `+` on numpy booleans is logical OR, so the `* 1.0` on
# each condition is required for an arithmetic 0-8 count and for any ts_
# operator to accept it.
# =============================================================================

fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)

stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0)
quiet  = (stress < 1.0) * 1.0                 # 1 on ordinary days, 0 on event days
w      = (stress > 0) * (1.0 + stress)

absr   = at_signsqrt(ret1) * at_signsqrt(ret1)          # |ret1|

sz    = cs_rank(ts_mean(close * volume, 60))
large = (sz >  1.5)
small = (sz <= 1.5)


# =============================================================================
# TIER 1 — LIQUIDITY RESILIENCE.  The sharpest extension of P8.
# =============================================================================

# --- L1. Resilience: event-day depth relative to ordinary-day depth -----------
# P8 measures the LEVEL of depth on event days -- which is mostly a size and
# liquidity ranking, and is why the `large` half won. This measures whether a
# name's depth HOLDS UP when forced flow arrives, relative to its own ordinary
# state. That strips out the static size component entirely.
#
# High = the book deepens when mandated flow hits (market makers and index
# arbitrageurs step in because the flow is predictable and uninformed).
# Low = the book thins exactly when it is needed -- these are the names where
# forced flow moves price, so a liquidity provider should be paid MORE there.
#
# The two readings predict OPPOSITE signs, which makes this a genuine test
# rather than a fishing expedition. Written long-high; flip and record which.
d_ev  = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(absr * stress, 250))
d_qu  = ts_sum(volume * quiet, 250)  / at_zero2nan(ts_sum(absr * quiet, 250))

score = cs_rank(d_ev / at_zero2nan(d_qu)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- L2. Same, on the large half ---------------------------------------------
# P8 only qualified once restricted to `large`. If L1 behaves the same way, the
# effect lives in mega-caps for both -- which is the mandated-flow story, since
# that is where index and ETF ownership concentrates.
d_ev  = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(absr * stress, 250))
d_qu  = ts_sum(volume * quiet, 250)  / at_zero2nan(ts_sum(absr * quiet, 250))

score = cs_rank(d_ev / at_zero2nan(d_qu)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * large)

# --- L3. The CONTROL — ordinary-day depth alone ------------------------------
# P8's `cheap` on NON-event days. If this works as well as P8, then P8 is just
# ranking liquid stocks and the event conditioning contributes nothing -- the
# whole mandated-flow interpretation would be decoration on a liquidity factor.
#
# THIS IS THE MOST IMPORTANT RUN IN THE FILE and it is not primarily a
# submission. It is the question an interviewer asks about P8, and right now
# you cannot answer it.
score = -(cs_rank(ts_sum(volume * quiet, 250)
                  / at_zero2nan(ts_sum(absr * quiet, 250))) - 1.0)
alpha = -at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * large)


# =============================================================================
# TIER 2 — PASSIVE-OWNERSHIP DYNAMICS.  Index-addition proxy.
# =============================================================================

# --- L4. Rising mandated-flow share ------------------------------------------
# mflow over 60 days minus mflow over 250 days: is this stock's volume becoming
# MORE concentrated on forced-flow dates than it used to be?
#
# A stock added to an index, or gaining ETF membership, sees its mandated-flow
# share rise. Index-addition effects are documented (Shleifer 1986; Harris &
# Gurel 1986) -- a permanent demand shift with no information content, which is
# the purest possible mandated-flow event.
#
# Unlike L1 this is a CHANGE, so it should decorrelate from both P2 and P8,
# which rank levels.
mf_fast = ts_sum(volume * stress, 60)  / at_zero2nan(ts_sum(volume, 60))
mf_slow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(volume, 250))

score = cs_rank(mf_fast - mf_slow) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- L5. Rising mandated-flow share, gated -----------------------------------
mf_fast = ts_sum(volume * stress, 60)  / at_zero2nan(ts_sum(volume, 60))
mf_slow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(volume, 250))

score = cs_rank(mf_fast - mf_slow) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- L6. Depth trend ---------------------------------------------------------
# Is this name's event-day absorption capacity improving or deteriorating?
# Deteriorating depth in a heavily-indexed name means a crowding problem
# building -- the same flow arriving into a thinner book.
d_fast = ts_sum(volume * stress, 60)  / at_zero2nan(ts_sum(absr * stress, 60))
d_slow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(absr * stress, 250))

score = cs_rank(d_fast / at_zero2nan(d_slow)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — EVENT-TYPE DECOMPOSITION.  Which forced flow actually pays?
# =============================================================================
# Your gate bundles eight events. Nothing has ever tested which of them carries
# the signal. Splitting the depth measure by event TYPE gives three alphas that
# trade different day-sets AND tells you which mechanism is real.
#
# Each uses a single-condition mask, so the day-sets are near-disjoint and the
# resulting alphas should decorrelate from each other and from P8.

# --- L7. Expiration-day depth ------------------------------------------------
# Dealer delta-hedging of expiring options. The most purely MECHANICAL flow in
# the set -- hedges must be adjusted regardless of view (Stoll & Whaley).
ox    = (opex * 1.0) + (qopex * 1.0)
score = cs_rank(ts_sum(volume * ox, 250)
                / at_zero2nan(ts_sum(absr * ox, 250))) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * large)

# --- L8. Rebalancing-window depth --------------------------------------------
# Quarter-end and month-end: index reconstitution and fund rebalancing to
# target weights. The flow most directly tied to passive ownership.
rb    = (qend * 1.0) + (mend * 1.0) + (mstart * 1.0)
score = cs_rank(ts_sum(volume * rb, 250)
                / at_zero2nan(ts_sum(absr * rb, 250))) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * large)

# --- L9. Macro-event depth ---------------------------------------------------
# FOMC windows. This flow is INFORMATION-driven, not mandated -- so under the
# revised mechanism it should be the WEAKEST of the three. If L9 beats L7 and
# L8, the mandated-flow account is wrong and note 38 needs revising again.
mc    = (fomc_pre * 1.0) + (fomc_post * 1.0)
score = cs_rank(ts_sum(volume * mc, 250)
                / at_zero2nan(ts_sum(absr * mc, 250))) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * large)


# =============================================================================
# TIER 4 — MANDATED-FLOW WEIGHT ON OTHER CORES.
# =============================================================================
# P2 applied the mflow weight to A1's core and gained +0.012 (0.094 -> 0.106).
# It has never been applied to A2's or A3's core. If the mechanism is general
# rather than specific to one expression, it should lift those too -- and each
# is a different alpha.

# --- L10. mflow weight on A2's closing-hour skew core ------------------------
mflow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(volume, 250))
mw    = cs_rank(mflow)

sk    = ts_mean(skew_return_last_hour, 1)
core2 = cs_rank(sk) - 1.0

alpha = at_zero2nan(cs_winsor(core2 * mw, 0.01, remove_extreme=False) * w)

# --- L11. mflow weight on A3's GP core ---------------------------------------
mflow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(volume, 250))
mw    = cs_rank(mflow)

core3 = (ts_max(ts_mean(ts_max(ts_corr_binary(
             (ts_mean(ts_std(ret1, 20), 20) - ts_zscore(ts_std(ret1, 20), 10)),
             ret20, 60), 10), 20), 5)
         - (close / at_zero2nan(ts_mean(close, 20))))

alpha = at_zero2nan(cs_winsor(core3 * mw, 0.01, remove_extreme=False) * w)


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. L3  -- the CONTROL. Run it first even though it is the least likely
#           submission. If ordinary-day depth works as well as event-day depth,
#           P8 is a liquidity factor wearing a mandated-flow costume, and you
#           need to know that BEFORE describing it in an interview.
# 2. L1, L2 -- resilience. The sharpest new idea, and its two readings predict
#           opposite signs so either outcome is informative.
# 3. L10, L11 -- mflow weight on the other two cores. Cheapest wins available:
#           proven mechanism, proven cores, one multiplication.
# 4. L4, L6 -- ownership dynamics.
# 5. L7, L8, L9 -- event-type decomposition. Run all three together; the
#           COMPARISON is the point, not any single IR.
#
# EXPECT P8's PROFILE: 250-day windows, TVR ~0.03-0.10, break-evens in the tens
# of bp. Judge on IR/sqrt(TVR).
#
# CHECK DRAWDOWN ON EVERY ONE. At this turnover these are near-static
# characteristic tilts, and a static tilt is exactly what carried M4 to a 19.5
# drawdown and a slow-mode failure. Low turnover is not the same as low risk.
#
# CHECK CORRELATION AGAINST P8 for everything in tiers 1 and 3 -- they share
# the depth construction. L4 and L6 rank CHANGES rather than levels and should
# be the safest.
# =============================================================================
