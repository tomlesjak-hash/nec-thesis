# =============================================================================
# GP CORES REVISITED — with the improved gate
# =============================================================================
# SELECTION METHOD: not local IR (which had rank correlation -0.70 with platform
# IR and would mislead you). Instead, each of the 28 GP cores was decomposed
# into its BUILDING BLOCKS and scored by how many it shares with the cores
# already submitted:
#
#   A1's core contains  stoch = (close - ts_min(low,20)) / (ts_max(high,20) - ts_min(low,20))
#   A3's core contains  close / ts_mean(close, 20)      [MA reversal]
#
# Shared blocks are what drive collision -- that is why residual reversal and
# industry-neutral skew both failed the 50%% check. Every core below except the
# last shares ZERO blocks with your pool.
#
# ALL have low local turnover (0.023-0.073), which on the A3 calibration
# (local 0.057 -> platform 0.251) projects to platform TVR of roughly 0.10-0.32
# and break-evens in the 8-28bp range. These would be the most COST-VIABLE
# alphas you own.
#
# RUN UNGATED FIRST, then gated. Ungated is a free structural difference from a
# pool that mostly wears the gate.
# =============================================================================

# ---- CURRENT BEST GATE (the D2/M4 windows -- looser than A1's or A3's) ------
# WHY RETESTING IS JUSTIFIED: the gate these cores were originally tried with
# was A1's TIGHT version (booksize 0.48, ~40% of days). The current windows are
# far looser -- fomc_post went 0 -> 12, opex 1 -> 10, mend 2 -> 15 -- so booksize
# now sits at ~0.96 and numstk near the full universe. It has stopped being a
# FILTER and become a graded WEIGHT. Cores that previously died on coverage or
# on being over-filtered are genuinely worth another run.
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


==============================================================================
# CANDIDATE 1  [r1_14]
# local_ir 0.1015 | local_tvr 0.023 | max_corr 0.07 | proj platform TVR ~0.10 | proj break-even ~27bp
#
# ZERO overlap with pool. LOWEST max_corr (0.07) AND LOWEST tvr (0.023) of all 28.
#  Built on the OVERNIGHT return -- a data channel NOTHING in your pool uses.
#  Projected platform TVR ~0.10 -> break-even ~28bp, 2.5x better than A3.
#  THE STANDOUT. Run this first.
==============================================================================
core = ts_sum(ts_std(ts_max(ts_max(ts_delay(((close / at_zero2nan(ts_mean(close, 60))) * ts_skew(((close / at_zero2nan(ts_mean(close, 60))) * ((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1)))), 20)), 60), 10), 20), 60), 60)

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 2  [r1_20]
# local_ir 0.0830 | local_tvr 0.073 | max_corr 0.13 | proj platform TVR ~0.32 | proj break-even ~9bp
#
# ZERO overlap. Pure VOLATILITY STRUCTURE -- no price-level or reversal term at all.
#  Isolates A3's vol-regime component without A3's MA-reversal leg, so it is the
#  clean test of which half of A3 carries the signal (open question in note 18).
==============================================================================
core = ts_corr_binary(ts_max((ts_mean(ts_std(ret1, 20), 60) / at_zero2nan(ts_std(ret1, 20))), 60), ts_mean_exp(ts_mean_exp(ts_std(ret1, 20), 60, 0.5), 60, 0.5), 60)

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 3  [r1_17]
# local_ir 0.0995 | local_tvr 0.033 | max_corr 0.27 | proj platform TVR ~0.15 | proj break-even ~19bp
#
# ZERO overlap. Overnight x intraday x vol interaction, tvr 0.033.
#  A three-channel product -- the only core here combining all three.
==============================================================================
core = ts_sum(ts_std(ts_max(ts_delay((((ts_std(ret1, 20) - ts_delay(ts_std(ret1, 20), 10)) * ts_zscore(((close - open) / at_zero2nan(open)), 60)) * (ts_zscore((close / at_zero2nan(ts_mean(close, 60))), 5) * ((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1))))), 60), 60), 5), 60)

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 4  [r2_6]
# local_ir 0.0750 | local_tvr 0.042 | max_corr 0.38 | proj platform TVR ~0.18 | proj break-even ~15bp
#
# ZERO overlap. Beta INNOVATION (corr minus its own lag) minus dollar-volume ratio.
#  Trades the CHANGE in market sensitivity, not its level.
==============================================================================
core = ts_mean(ts_max(ts_mean_exp(((ts_corr_binary(ret1, ret1_spx, 60) - ts_delay(ts_corr_binary(ret1, ret1_spx, 60), 5)) - ((close * volume) / at_zero2nan(ts_mean(close * volume, 60)))), 10, 0.5), 60), 10)

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 5  [r2_4]
# local_ir 0.0874 | local_tvr 0.032 | max_corr 0.40 | proj platform TVR ~0.14 | proj break-even ~20bp
#
# ZERO overlap. Size-scaled price ratio plus a beta-skew/overnight correlation term.
==============================================================================
core = (((close / at_zero2nan(ts_mean(close, 60))) / at_zero2nan((cs_rank(ts_mean(close * volume, 60)) - 1))) + ts_sum(ts_corr_binary(ts_skew(ts_corr_binary(ret1, ret1_spx, 60), 60), ts_mean(((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1))), 10), 60), 60))

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 6  [r2_3]
# local_ir 0.0720 | local_tvr 0.052 | max_corr 0.43 | proj platform TVR ~0.23 | proj break-even ~12bp
#
# ZERO overlap. Correlation between the intraday move and the overnight gap,
#  smoothed. A pure open-vs-close decomposition -- closest GP core to the
#  Lou-Polk-Skouras clientele mechanism.
==============================================================================
core = ts_median(ts_delay(ts_delay(ts_std(ts_corr_binary(((close - open) / at_zero2nan(open)), ((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1))), 10), 20), 60), 60), 60)

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 7  [r2_7]
# local_ir 0.0836 | local_tvr 0.040 | max_corr 0.50 | proj platform TVR ~0.18 | proj break-even ~16bp
#
# ZERO overlap. Same family as r2_4, different tail. max_corr 0.50 -- check first.
==============================================================================
core = (((close / at_zero2nan(ts_mean(close, 60))) / at_zero2nan((cs_rank(ts_mean(close * volume, 60)) - 1))) + ts_mean(ts_min(ts_skew(ts_max(ts_corr_binary(ret1, ret1_spx, 60), 60), 60), 60), 5))

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 8  [r2_2]
# local_ir 0.0952 | local_tvr 0.053 | max_corr 0.00 | proj platform TVR ~0.23 | proj break-even ~12bp
#
# LOWEST max_corr of all 28 (0.00) but shares the MA-reversal block with A3.
#  Reversal squared, scaled by log size. Run last of this set for that reason.
==============================================================================
core = -((((close / at_zero2nan(ts_mean(close, 20))) * ts_sum(at_signlog((cs_rank(ts_mean(close * volume, 60)) - 1)), 20)) * (close / at_zero2nan(ts_mean(close, 20)))))

# ungated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False))

# gated
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


# =============================================================================
# RUN ORDER AND NOTES
# =============================================================================
# 1. r1_14  -- lowest max_corr, lowest turnover, and the ONLY core built on the
#              overnight return. Best expected value by a clear margin.
# 2. r1_20  -- pure volatility structure. Also answers the open question in
#              note 18 section 4: is A3's signal the vol-regime term or the
#              MA-reversal term? r1_20 is the vol term ALONE.
# 3. r1_17, r2_3 -- overnight/intraday decomposition cores.
# 4. r2_6, r2_4, r2_7, r2_2.
#
# FLIP THE SIGN on anything with negative IR. Four of your working alphas
# needed it.
#
# DO NOT tune these cores. The GP already optimised their internal windows and
# note 14 measured that re-tuning them was worth only +0.006. Tune the GATE if
# anything clears -- that has been worth +0.014 to +0.036 every time.
#
# EXCLUDED and why:
#   r1_11  = A3's core, already submitted
#   r1_16  = A1's core, already submitted
#   r1_4   = same seed as A1, near-duplicate expression
# =============================================================================
