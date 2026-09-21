# =============================================================================
# MANDATED-FLOW FACTORS — built from your own falsification
# =============================================================================
# WHAT NOTE 38 ESTABLISHED
# ------------------------
# Two independent tests rejected the volatility channel:
#   * gate B (macro vol regime, calendar time) -- never added to any core
#   * hivol vs lovol split (cross-sectional)   -- LOW vol won, 0.089 vs 0.076
#
# And one lever has worked every single time: the 8-event calendar gate,
# +0.014 to +0.036 on every core it touched.
#
# The mechanism is therefore MANDATED FLOW -- index funds, ETF create/redeem,
# MOC algorithms, pension rebalancing, tax-driven selling. Flow that must
# transact regardless of price. Low-volatility mega-caps win because mechanical
# flow is the LARGEST SHARE of their volume, not because they are risky.
#
# THE GAP THIS FILE FILLS
# -----------------------
# You have measured WHEN mandated flow arrives (the calendar gate). You have
# never measured WHICH STOCKS RECEIVE IT. That is a per-stock characteristic,
# it is constructible from data you have, and it is the direct version of what
# the volatility split was proxying badly.
#
# Everything here is a CHARACTERISTIC, which is the construction type that has
# actually cleared for you (M4 0.103, A2 0.102, A3 0.073) as opposed to
# directional forecasts, which have failed uniformly.
#
# NOTE ON `stress`: it is a sum of booleans used in arithmetic, exactly as in
# every alpha you have submitted, so `volume * stress` is a valid float array
# and may enter ts_ operators. Do not wrap it in a comparison first.
# =============================================================================

# ---- the event intensity series (time-series, identical for every stock) -----
fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)

# CRITICAL: each condition is multiplied by 1.0 BEFORE summing.
# In numpy, `+` on boolean arrays is LOGICAL OR, not arithmetic addition. Every
# alpha submitted so far used `fomc_pre + fomc_post + ...` on raw booleans, so
# `stress` was a bool -- True if ANY event was live -- and `w` evaluated to
# {0.0, 2.0}. The event gate has been BINARY in every submitted alpha, and the
# "(1 + stress) intensity weighting" described in note 16 never executed.
#
# The `* 1.0` below forces arithmetic addition, so stress is a genuine 0-8
# count and w is a genuine graded weight 2.0-9.0. This is also why bare
# `ts_sum(stress, N)` threw "Unsupported data type: bool".
stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0)

w = (stress > 0) * (1.0 + stress)


# =============================================================================
# TIER 0 — TEST THE GRADED WEIGHT ON ALPHAS THAT ALREADY QUALIFY.
# =============================================================================
# Run this BEFORE anything else in the file. The `w` defined above is a
# genuinely graded weight for the first time. Dropping it into a core that
# already qualifies tests the mechanism note 16 CLAIMS to have tested but did
# not -- that days where several mandated-flow events coincide pay more than
# ordinary event days.
#
# It is also a DIFFERENT alpha from the binary version: different day-weights,
# so it may clear the 50% correlation gate and count separately rather than
# replace. Check correlation against the binary parent.
#
# Shown on A1's tuned core (currently IR 0.094 on the lovol half).
stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 80))
         - (((ts_median(cs_zscore(relvol), 10) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w)


# =============================================================================
# TIER 1 — PASSIVE OWNERSHIP PROXY.  The central idea in this file.
# =============================================================================

# --- P1. Mandated-flow share of volume ---------------------------------------
# What fraction of this stock's annual volume arrives on high-event days,
# weighted by event intensity.
#
# THIS IS A PROXY FOR PASSIVE OWNERSHIP. A stock held largely by index funds
# and ETFs sees a disproportionate share of its volume on rebalancing dates,
# expiration dates and month-ends, because that is when its holders are
# COMPELLED to trade. A stock held by discretionary investors trades on news
# and earnings instead, spread evenly across the calendar.
#
# The dataset has no ownership or index-membership field. This constructs the
# closest available substitute from volume timing alone, and it is the direct
# measurement that note 38 section 6.3 identifies as missing.
#
# Sign as written: long high mandated-flow share. That is the direct prediction
# of the revised mechanism -- the premium should be largest where the flow is.
mflow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(volume, 250))
score = cs_rank(mflow) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- P2. Same, as a WEIGHT on the proven reversal core -----------------------
# If P1 identifies where mandated flow concentrates, then the reversal core
# should be paid most in exactly those names. This is the low-volatility split
# done properly: a continuous tilt toward the right names instead of a crude
# volatility proxy, and with NO loss of universe coverage.
#
# If this beats the lovol split (IR 0.094), the mechanism revision in note 38
# is confirmed AND you have a better alpha.
mflow = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(volume, 250))
mw    = cs_rank(mflow)                                   # [1,2]

stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 80))
         - (((ts_median(cs_zscore(relvol), 10) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core1 * mw, 0.01, remove_extreme=False) * w)

# --- P3. Event-day volume AMPLIFICATION --------------------------------------
# A ratio version: how much heavier is this stock's volume on event days than
# on ordinary days. P1 measures the SHARE of volume; this measures the
# MULTIPLE, which is scale-free and does not inherit the stock's overall
# turnover level -- a cleaner separation of the flow signal from a size proxy.
ev_v  = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(stress, 250))
al_v  = ts_mean(volume, 250)
score = cs_rank(ev_v / at_zero2nan(al_v)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- P4. Index co-movement ---------------------------------------------------
# Passively-held stocks are TRADED as part of a basket, so their returns should
# track the index more tightly on mandated-flow days than on ordinary days.
#
# This is deliberately NOT plain market beta (which failed as a signal). It is
# the DIFFERENCE between event-window beta and unconditional beta -- how much
# more index-like a name becomes when mechanical flow dominates. Differencing
# removes the beta level, which is the part that did not work.
b_all = ts_corr_binary(ret1, ret1_spx, 250)
b_ev  = ts_corr_binary(ret1 * stress, ret1_spx * stress, 250)
score = cs_rank(b_ev - b_all) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — TAX-LOSS SELLING.  Mandated flow the calendar gate does not capture.
# =============================================================================
# Year-end tax-loss harvesting is forced, price-insensitive selling: holders of
# losing positions sell in December for the tax deduction regardless of value,
# then the selling pressure lifts in January. Documented since Rozeff & Kinney
# (1976) and Reinganum (1983), and it is a MANDATED-FLOW effect in exactly the
# sense note 38 identifies -- which is why it belongs in this file rather than
# with the momentum ideas that failed.
#
# Your 8-event gate contains no year-end term at all. This is a distinct
# forced-flow window it has never covered.
#
# NUMSTK NOTE: a hard December gate would be active ~20 days a year, and with
# numstk averaged over 120 days that lands near 60-125 -- BELOW the 160 floor.
# So both versions below use a CONTINUOUS proximity weight instead, keeping
# every day live and the full universe intact.

# --- P5. Tax-loss candidates, year-end weighted ------------------------------
# Long the year's losers, weighted by how close year-end is. `prox` rises
# smoothly as December approaches instead of switching on.
yr_ret = close / at_zero2nan(ts_delay(close, 250))
prox   = 1.0 / (1.0 + days_until_last_trading_day_of_year)

score = -(cs_rank(yr_ret) - 1.0) * prox
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- P6. January rebound -----------------------------------------------------
# The other side: the same losers should outperform once the selling pressure
# lifts. Weighted by proximity to the START of the year.
#
# If P5 and P6 BOTH clear, they trade near-disjoint parts of the calendar on
# opposite sides of the same names -- the note-14 / note-15 relationship again,
# and two submissions from one mechanism.
yr_ret = close / at_zero2nan(ts_delay(close, 250))
prox   = 1.0 / (1.0 + days_since_first_trading_day_of_year)

score = -(cs_rank(yr_ret) - 1.0) * prox
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — PER-STOCK EVENT SENSITIVITY.
# =============================================================================

# --- P7. Event-day return sensitivity ----------------------------------------
# Does this stock actually MOVE on mandated-flow days? P1 measures whether the
# flow arrives; this measures whether it moves the price -- which is the part
# that matters, since flow absorbed without price impact pays nothing.
#
# The two together are a price-impact-of-mandated-flow measure, and the
# difference between them is informative: heavy event volume with no event
# volatility means deep liquidity, which is where the fee is SMALLEST.
absr  = at_signsqrt(ret1) * at_signsqrt(ret1)
ev_r  = ts_sum(absr * stress, 250) / at_zero2nan(ts_sum(stress, 250))
al_r  = ts_mean(absr, 250)
score = cs_rank(ev_r / at_zero2nan(al_r)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- P8. Flow absorbed without impact ----------------------------------------
# High event volume, LOW event price response = this name absorbs mandated flow
# cheaply. The revised mechanism says that is where the liquidity-provision fee
# is smallest, so the sign should be NEGATIVE -- short the cheap-to-trade names,
# long the ones where flow moves price.
#
# This is the sharpest test in the file: it is the one construction whose sign
# the revised mechanism predicts and the volatility mechanism does not.
absr  = at_signsqrt(ret1) * at_signsqrt(ret1)
ev_v  = ts_sum(volume * stress, 250) / at_zero2nan(ts_sum(stress, 250))
ev_r  = ts_sum(absr * stress, 250)   / at_zero2nan(ts_sum(stress, 250))
cheap = ev_v / at_zero2nan(ev_r)

score = -(cs_rank(cheap) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. P2  -- mandated-flow-weighted reversal core. Highest expected value: it
#           takes an alpha ALREADY at IR 0.094 and replaces a crude volatility
#           proxy with a direct measurement of the thing that proxy stood for,
#           at NO cost in universe coverage.
# 2. P1  -- the passive-ownership proxy standalone. Also the missing direct
#           measurement flagged in note 38 section 6.3, so it is worth running
#           for the theory note even if it does not qualify.
# 3. P5, P6 -- tax-loss selling. A forced-flow window your gate has never
#           covered, and the only genuinely new CALENDAR territory left.
# 4. P8  -- sharpest mechanism test; its sign discriminates between the two
#           competing accounts.
# 5. P3, P4, P7.
#
# ALL UNGATED except P2 -- ungated is a free structural difference from a pool
# that mostly wears the gate. Add `* w` only if a signal misses IR alone.
#
# EXPECT LOW TURNOVER. The 250-day windows mean these rankings move slowly, so
# TVR should land at 0.05-0.30 and break-evens in the tens of bp. Judge on
# IR/sqrt(TVR).
#
# WARM-UP: 250-day windows push the start date out by a year. Check the start
# date and numstk before reading anything into the IR.
#
# WHATEVER HAPPENS, RECORD P1's RESULT. Note 38 currently says the direct
# measurement of mandated flow is missing and lists it as what you would build
# with better data. P1 is that measurement built from data you already have --
# it belongs in the note either way.
# =============================================================================
