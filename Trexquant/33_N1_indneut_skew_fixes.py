# =============================================================================
# N1 INDUSTRY-NEUTRAL CLOSING-HOUR SKEW — turnover repair
# =============================================================================
# BASELINE (fast, 2006-03-31 -> 2021-12-30)
#   IR 0.088 | TVR 1.306 | Ret 0.040 | DD 4.1 | IR/sqrt(TVR) 0.077
#   Booksize 1.00 | NumStks 774x775 | Liq 362 / LiqN 156
#   -> Sharpe 1.40, Calmar 0.98, break-even 1.2bp
#
# DIAGNOSIS: this is NOT an IR problem. DD 4.1 and Calmar 0.98 are the BEST in
# your pool, and coverage at 774 names is the widest. The single weakness is
# turnover: TVR 1.306 gives a 1.2bp break-even, the worst alpha you own.
#
# You have 0.018 of IR headroom above the 0.07 gate. Spend it on turnover:
#     TVR 1.31 -> IR/sqrt(TVR) 0.077   (now)
#     TVR 0.70 -> IR/sqrt(TVR) 0.096   even if IR falls to 0.080
#     TVR 0.50 -> IR/sqrt(TVR) 0.106   even if IR falls to 0.075
#
# =============================================================================
# THE KEY DISTINCTION — SMOOTH THE RANK, NOT THE RAW SKEW
# -------------------------------------------------------
# On A2 you already tested smoothing the RAW skew and it destroyed the signal:
#     ts_mean(skew, 1) -> IR 0.088     ts_mean(skew, 5) -> IR 0.061
#
# Smoothing the RANK is a different operation and should behave differently.
# Raw skew is unbounded with fat tails, so averaging five days lets one extreme
# day dominate the window and washes out the information in the other four.
# A rank is bounded in [0,1] and stationary by construction, so averaging it
# cannot be dominated by an outlier -- it just damps how fast positions move.
#
# The first reduces INFORMATION. The second reduces CHURN. Same word, different
# operation, and the A2 result does not carry over.
#
# ts_mean_exp is named in the operator docs as "the turnover-control tool" and
# has not been tested on any alpha in this project.
#
# CORRELATION WARNING -- READ FIRST
# ---------------------------------
# This uses the SAME VARIABLE as A2 (skew_return_last_hour) at the SAME HORIZON
# (no smoothing on either). The only differences are industry neutralisation
# and the absent gate. This is the highest-risk pair in your pool -- check
# correlation against A2 before spending a submission, and note that at
# IR/sqrt(TVR) 0.077 vs A2's 0.087 the override clause is unavailable to you.
#
# Fixing the turnover also RAISES IR/sqrt(TVR) toward the override threshold,
# so the repair below helps on both fronts at once.
# =============================================================================


# =============================================================================
# T1 — EXPONENTIAL RANK SMOOTHING.  The main fix. Run first.
# =============================================================================
base  = cs_rank(cs_indneut(skew_return_last_hour, industry)) - 1.0
score = ts_mean_exp(base, 5, 0.5)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# T2 — HEAVIER SMOOTHING.  If T1 holds IR, push further.
# =============================================================================
base  = cs_rank(cs_indneut(skew_return_last_hour, industry)) - 1.0
score = ts_mean_exp(base, 10, 0.5)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# T3 — SLOWER DECAY.  Same window, more weight on older days.
# =============================================================================
# exp_factor controls how fast old observations decay. 0.5 halves the weight
# each step; 0.8 holds positions much longer for the same nominal window.
# This is the knob that trades turnover against staleness most directly.
base  = cs_rank(cs_indneut(skew_return_last_hour, industry)) - 1.0
score = ts_mean_exp(base, 5, 0.8)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# T4 — SIMPLE RANK SMOOTHING.  The control for T1.
# =============================================================================
# Flat average of the last 3 ranks. If T4 matches T1, the exponential weighting
# is doing nothing and the simpler expression is the better one to submit --
# fewer parameters, easier to defend. Worth the run for that reason alone.
base  = cs_rank(cs_indneut(skew_return_last_hour, industry)) - 1.0
score = ts_mean(base, 3)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# T5 — SECTOR NEUTRALISATION INSTEAD OF INDUSTRY.
# =============================================================================
# A coarser grouping removes less variation, so more signal survives -- but
# less of the sector tilt is stripped, which is what gave you DD 4.1. Expect
# higher IR and higher drawdown.
#
# Also a genuinely DIFFERENT alpha from T1 rather than a tuning of it: the two
# hold different books because they neutralise against different means. If both
# clear and they pass the 50% check against each other, that is two
# submissions.
#
# (Verify `sector` exists as a variable in the sim before relying on this --
# it is in your local panel, but confirm on the platform.)
base  = cs_rank(cs_indneut(skew_return_last_hour, sector)) - 1.0
score = ts_mean_exp(base, 5, 0.5)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# T6 — NEUTRALISED SKEW ASYMMETRY.  Furthest from A2.
# =============================================================================
# Differencing the two hours removes whatever is common to both, isolating what
# is specific to the CLOSE. Combined with industry neutralisation this is two
# independent steps away from A2's raw variable, which is the best correlation
# profile available while keeping the same underlying idea.
#
# If the A2 correlation check fails on T1, come here rather than abandoning.
sk    = skew_return_last_hour - skew_return_first_hour
base  = cs_rank(cs_indneut(sk, industry)) - 1.0
score = ts_mean_exp(base, 5, 0.5)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# T7 — SMOOTHED + EVENT GATE.
# =============================================================================
# Only if T1 clears comfortably. The gate has added +0.014 to +0.036 to every
# fast core, but it also moves this alpha TOWARD A2 (which is gated) on the one
# axis where they currently differ most. Run it, but check correlation again
# afterwards -- a gain in IR that costs you the submission is not a gain.
base  = cs_rank(cs_indneut(skew_return_last_hour, industry)) - 1.0
score = ts_mean_exp(base, 5, 0.5)

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
# RUN ORDER AND JUDGEMENT
# =============================================================================
# 1. T1  -- the fix. If TVR falls below ~0.8 with IR still above 0.075, done.
# 2. T2 / T3 -- push further in whichever direction T1 indicates.
# 3. T4  -- the control. Prefer it if it matches; simpler is more defensible.
# 4. T6  -- the fallback if the A2 correlation check fails.
# 5. T5, T7.
#
# JUDGE ON IR/sqrt(TVR), NOT IR. IR only has to clear 0.07; every point above
# that is worth less than the turnover it costs. This alpha's drawdown and
# coverage are already best-in-pool -- turnover is the only thing standing
# between it and being the best risk-adjusted alpha you own.
#
# STOP CONDITION: if IR falls below 0.075 the headroom is gone and further
# smoothing risks the gate on an out-of-sample run. Take the best IR/sqrt(TVR)
# among variants that keep IR >= 0.075.
# =============================================================================
