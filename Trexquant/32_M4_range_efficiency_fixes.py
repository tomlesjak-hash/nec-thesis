# =============================================================================
# M4 RANGE EFFICIENCY — drawdown repair
# =============================================================================
# BASELINE (fast mode, 2006-03-31 -> 2021-12-30)
#   IR 0.103 | TVR 0.582 | Ret 0.088 | DD 19.5 | IR/sqrt(TVR) 0.135
#   Booksize 1.00 | NumStks 763x770 | Liq 776 / LiqN 315
#   -> Sharpe 1.64 (joint best in pool), break-even 6.0bp, Calmar 0.45 (worst)
#
# WHAT THE ALPHA ACTUALLY IS
# --------------------------
#   at_signsqrt(x) * at_signsqrt(x) = |x|
# so `net` is the ABSOLUTE net move and the signal is UNSIGNED. It ranks how
# cleanly the closing hour trended in EITHER direction -- long the trenders,
# short the churners. This is not a directional forecast, it is a ranking of a
# STOCK CHARACTERISTIC (Kaufman's efficiency ratio, computed intraday).
#
# Reading: a cleanly-trending closing hour means one-sided directional pressure
# working through the book; a churning hour means two-sided flow with liquidity
# being supplied on both sides. You are long names with concentrated demand.
#
# WHY THE DRAWDOWN IS 19.5
# ------------------------
# TVR 0.582 = positions barely turn over. Efficiency is a PERSISTENT stock
# property -- some names always trend into the close, some always churn -- so
# you hold roughly the same book for months. Persistent positions on a
# persistent characteristic = sustained factor exposure = a 3.6x-annual-vol
# drawdown when that exposure goes against you. It is a long losing REGIME,
# not a spike.
#
# THE FIX: convert the signal from a LEVEL to a SHOCK. Ranking the level asks
# "which stocks are trendy" -- a static characteristic. Ranking against the
# stock's own recent norm asks "which stocks are trendy TODAY, unusually for
# them", which is a trading signal rather than a portfolio tilt.
#
# This is the same normalisation that mattered in A2 (raw trade size ranks
# stocks by typical trade size forever; the ratio to ts_mean makes it a
# surprise) and the same failure mode as the fundamental LEVELS in file 22.
#
# EXPECT: DD falls sharply, TVR rises (probably 1.0-1.5), IR may fall somewhat.
# Watch Calmar and IR/sqrt(TVR) rather than IR alone -- IR 0.09 with DD 8 is a
# far better alpha than IR 0.103 with DD 19.5.
#
# ALSO: your run was FAST mode (2006-03-31 start; slow starts ~2007-03).
# Run slow before submitting.
# =============================================================================

# ---- baseline, for reference -------------------------------------------------
net  = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff  = net / at_zero2nan(high_price_last_hour - low_price_last_hour)


# =============================================================================
# F1 — EFFICIENCY SHOCK.  The main fix. Run this first.
# =============================================================================
# Today's efficiency relative to this stock's own 20-day norm. Removes the
# persistent "this name always trends" component and leaves the innovation.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff   = net / at_zero2nan(high_price_last_hour - low_price_last_hour)
shock = eff / at_zero2nan(ts_mean(eff, 20))

score = cs_rank(shock) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# F2 — EFFICIENCY Z-SCORE.  Same idea, dispersion-aware.
# =============================================================================
# ts_zscore divides by the stock's own STANDARD DEVIATION of efficiency rather
# than its mean, so it accounts for how variable the measure is for that name.
# A name whose efficiency swings wildly needs a bigger move to count as
# unusual. Usually the better of the two normalisations; try both.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff   = net / at_zero2nan(high_price_last_hour - low_price_last_hour)

score = cs_rank(ts_zscore(eff, 20)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# F3 — INDUSTRY-NEUTRAL LEVEL.  Cheapest possible test of the DD source.
# =============================================================================
# If the drawdown comes from a SECTOR tilt rather than a stock-level one, this
# fixes it while keeping the level formulation and most of the IR. Liquid mega
# caps and thin names have structurally different closing-hour efficiency, and
# so do sectors with heavy index-fund membership.
#
# DIAGNOSTIC VALUE: if F3 cuts DD substantially, the exposure was sectoral; if
# it does nothing, it is a stock-level characteristic and F1/F2 are the answer.
# One operator, and it tells you which repair is the right one.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff   = net / at_zero2nan(high_price_last_hour - low_price_last_hour)

score = cs_rank(cs_indneut(eff, industry)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# F4 — SHOCK + EVENT GATE.  Stack the fix with the proven lever.
# =============================================================================
# The gate has added +0.014 to +0.036 every time it has been applied to a fast
# core. Apply it only AFTER F1/F2 establishes which normalisation wins --
# tuning two things at once is what hid the day-0 earnings finding in note 14.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff   = net / at_zero2nan(high_price_last_hour - low_price_last_hour)
shock = eff / at_zero2nan(ts_mean(eff, 20))
score = cs_rank(shock) - 1.0

fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)

stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)


# =============================================================================
# F5 — VOLUME-CONFIRMED EFFICIENCY.
# =============================================================================
# A clean trend on heavy volume is a real order being worked. A clean trend on
# LIGHT volume is a thin book drifting on nothing -- the same efficiency
# reading with the opposite meaning, and very plausibly where the losses live,
# since thin-book drift reverses.
#
# Multiplying by the volume rank keeps both but tilts the book toward the
# reading you actually want.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff   = net / at_zero2nan(high_price_last_hour - low_price_last_hour)
shock = eff / at_zero2nan(ts_mean(eff, 20))
vconf = cs_rank(volume_last_hour / at_zero2nan(ts_mean(volume_last_hour, 20)))

score = (cs_rank(shock) - 1.0) * vconf
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# F6 — SIGNED EFFICIENCY.  A DIFFERENT ALPHA, not a repair.
# =============================================================================
# Drop the absolute value. (close - open) / (high - low) is bounded [-1, +1]
# and measures DIRECTIONAL efficiency: +1 means the hour opened at its low and
# closed at its high.
#
# This is a genuinely different signal -- directional rather than characteristic
# -- and should be submitted separately if it clears. It is also the intraday
# twin of the stochastic oscillator at the core of A1, so check correlation
# against A1 specifically.
sgnnet = close_price_last_hour - open_price_last_hour
score  = cs_rank(sgnnet / at_zero2nan(high_price_last_hour - low_price_last_hour)) - 1.0
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# F7 — EFFICIENCY PERSISTENCE.
# =============================================================================
# How consistently has this name trended into the close over 10 days? One
# efficient hour is noise; ten in a row is an order being worked across
# sessions -- the same multi-day-order logic that motivated the periodicity
# tests, but on efficiency rather than on returns (which failed).
#
# Note this INCREASES persistence rather than reducing it, so expect DD to stay
# high or worsen. Run it for the mechanism read, not as a drawdown fix.
net   = at_signsqrt(close_price_last_hour - open_price_last_hour) * at_signsqrt(close_price_last_hour - open_price_last_hour)
eff   = net / at_zero2nan(high_price_last_hour - low_price_last_hour)

score = cs_rank(ts_mean(eff, 10)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER AND WHAT TO WATCH
# =============================================================================
# 1. F3  -- one operator, and it DIAGNOSES whether the drawdown is sectoral or
#           stock-level. Do this first: it tells you which repair to pursue.
# 2. F1, F2 -- the main fix. Take whichever gives the better CALMAR, not the
#           better IR.
# 3. F4  -- add the gate to the winner.
# 4. F6  -- separate submission if it clears; check correlation vs A1.
# 5. F5, F7.
#
# JUDGE ON CALMAR AND IR/sqrt(TVR), NOT IR. The baseline already has a
# top-of-pool IR; what it lacks is drawdown control. IR 0.085 with DD 8 beats
# IR 0.103 with DD 19.5 for anything you would actually run, and the
# competition's combined-OS score is computed on aggregated PnL where a 19.5
# drawdown drags the whole book.
#
# THEN TUNE, one knob at a time:
#   ts_mean(eff, 20) / ts_zscore(eff, 20)  -> 5, 10, 20, 60
#   the volume window in F5               -> 10, 20, 60
#
# AND RUN SLOW MODE before submitting. Your figures are fast (2006-03-31 start;
# intra1 alphas start ~2007-03 in slow).
# =============================================================================
