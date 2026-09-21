# GP core batch — event-weighted, ready to paste
# Gate block is IDENTICAL to the submitted A3 (18_THEORY_vol_regime_coupling.md).
# Paste ONE core at a time. If IR comes out NEGATIVE, flip the sign: core = -(...)
# Run Compute Correlation against A1/A2/A3 before submitting any of these.

GATE = '''
fomc_pre  = (days_until_next_fomc_meeting <= 0)
fomc_post = (days_since_last_fomc_meeting <= 0)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 15)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 1)
stress = fomc_pre + fomc_post + opex + qopex + qend + mend + mstart + holiday
w      = (stress > 0) * (1.0 + stress)
'''


==============================================================================
# CANDIDATE 1  [r1_18]   local_ir 0.0821  tvr 0.055  max_corr 0.15
# TIER 1 - closest structural kin to A3: [coupling term] - [MA20 reversal]. Same skeleton.
==============================================================================
core = (ts_mean(ts_max(at_signlog(ts_delay(ts_corr_binary(ts_corr_binary(ret1, ret1_spx, 60), ((close - ts_min(low, 20)) / at_zero2nan(ts_max(high, 20) - ts_min(low, 20))), 20), 10)), 20), 60) - at_signlog((close / at_zero2nan(ts_mean(close, 20)))))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 2  [r2_1]   local_ir 0.0840  tvr 0.088  max_corr 0.42
# TIER 1 - explicit -(close/MA20) reversal + intraday skew + overnight gap.
==============================================================================
core = (ts_max(ts_zscore(ts_skew(((close - open) / at_zero2nan(open)), 5), 60), 60) + (((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1))) + -((close / at_zero2nan(ts_mean(close, 20))))))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 3  [r1_9]   local_ir 0.0750  tvr 0.066  max_corr 0.49
# TIER 1 - reversal via -ts_sum(close/MA20,5), momentum-size coupling. Compact. maxcorr 0.49 = risk.
==============================================================================
core = (((-(ts_corr_binary(ret20, (cs_rank(ts_mean(close * volume, 60)) - 1), 60)) - (cs_rank(ts_mean(close * volume, 60)) - 1)) - ts_sum((close / at_zero2nan(ts_mean(close, 20))), 5)) - ts_max((cs_rank(ts_mean(close * volume, 60)) - 1), 5))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 4  [r1_20]   local_ir 0.0830  tvr 0.073  max_corr 0.13
# TIER 2 - PURE vol-regime coupling, no reversal term. Isolates A3's term A. Best diagnostic.
==============================================================================
core = ts_corr_binary(ts_max((ts_mean(ts_std(ret1, 20), 60) / at_zero2nan(ts_std(ret1, 20))), 60), ts_mean_exp(ts_mean_exp(ts_std(ret1, 20), 60, 0.5), 60, 0.5), 60)

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 5  [r2_2]   local_ir 0.0952  tvr 0.053  max_corr 0.00
# TIER 3 - maxcorr 0.00, the most decorrelated core in either run. Pure reversal x size.
==============================================================================
core = -((((close / at_zero2nan(ts_mean(close, 20))) * ts_sum(at_signlog((cs_rank(ts_mean(close * volume, 60)) - 1)), 20)) * (close / at_zero2nan(ts_mean(close, 20)))))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 6  [r1_14]   local_ir 0.1015  tvr 0.023  max_corr 0.07
# TIER 3 - maxcorr 0.07, tvr 0.023 -> lowest platform turnover -> best break-even cost.
==============================================================================
core = ts_sum(ts_std(ts_max(ts_max(ts_delay(((close / at_zero2nan(ts_mean(close, 60))) * ts_skew(((close / at_zero2nan(ts_mean(close, 60))) * ((open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1)))), 20)), 60), 10), 20), 60), 60)

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 7  [r1_6]   local_ir 0.0881  tvr 0.045  max_corr 0.12
# TIER 4 - maxcorr 0.12, compact. corr(momentum-size, stochastic).
==============================================================================
core = -(ts_sum(ts_std(ts_min(ts_corr_binary((ret20 - (cs_rank(ts_mean(close * volume, 60)) - 1)), ((close - ts_min(low, 20)) / at_zero2nan(ts_max(high, 20) - ts_min(low, 20))), 5), 10), 60), 20))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


==============================================================================
# CANDIDATE 8  [r1_21]   local_ir 0.1077  tvr 0.077  max_corr 0.10
# TIER 4 - maxcorr 0.10, explicit -stochastic reversal, ts_mean_exp smoothed.
==============================================================================
core = ts_mean_exp(((ret5 + ts_corr_binary(ts_median(-(ret20), 60), (close / at_zero2nan(ts_mean(close, 20))), 60)) * -(((close - ts_min(low, 20)) / at_zero2nan(ts_max(high, 20) - ts_min(low, 20))))), 20, 0.5)

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)
