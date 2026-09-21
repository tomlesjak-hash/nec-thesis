# =============================================================================
# RANGE GEOMETRY, EARNINGS CYCLE, AND DELAYED ENTRY
# =============================================================================
# HONEST FRAMING
# --------------
# The mandated-flow thesis is exhausted. These are lower-conviction than that
# set was -- they are not derived from a live mechanism, they are unmined
# corners of data you already hold. Ranked below by prior, which is genuinely
# lower than in files 39-40.
#
# THE ONE REAL GAP: the daily high/low bars. You have used them ONLY inside
# A1's stochastic oscillator. Everything else in your pool is built from close,
# volume, intraday hours, or calendar variables. Bar GEOMETRY -- how today's
# range sits against yesterday's, where the close sits inside it, how often the
# stock gaps -- is a whole channel with nothing in it.
#
# All are CHARACTERISTICS, which is the construction type that clears for you
# (M4 0.103, A2 0.102, P8 0.096) as against directional forecasts, which have
# failed uniformly.
#
# STRESS AS FLOAT throughout (numpy `+` on booleans is logical OR).
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
w      = (stress > 0) * (1.0 + stress)

sz    = cs_rank(ts_mean(close * volume, 60))
large = (sz >  1.5)
vl    = cs_rank(ts_std(ret1, 40))
lovol = (vl <= 1.5)


# =============================================================================
# TIER 1 — BAR GEOMETRY.  The untouched channel.
# =============================================================================

# --- R1. Range containment ---------------------------------------------------
# How far today's range sits INSIDE yesterday's, normalised by yesterday's
# range. Positive means an inside day -- the stock traded within the prior
# session's boundaries on both sides.
#
# Sustained containment is volatility compression: disagreement is narrowing
# and positions are being held rather than turned over. The classic reading is
# that compression precedes expansion, but as a CHARACTERISTIC the more likely
# story is that persistently-contained names are ones where flow is orderly and
# two-sided -- the same property P8's depth measure was picking up, reached
# from price geometry instead of from volume.
prev  = at_zero2nan(ts_delay(high, 1) - ts_delay(low, 1))
cont  = ((ts_delay(high, 1) - high) + (low - ts_delay(low, 1))) / prev

score = cs_rank(ts_mean(cont, 20)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R2. Range containment, gated + large ------------------------------------
prev  = at_zero2nan(ts_delay(high, 1) - ts_delay(low, 1))
cont  = ((ts_delay(high, 1) - high) + (low - ts_delay(low, 1))) / prev

score = cs_rank(ts_mean(cont, 20)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * large)

# --- R3. Close versus typical price ------------------------------------------
# (high+low+close)/3 is the session's centre of gravity. Where the close sits
# relative to it says whether the day finished above or below where it actually
# traded -- a one-bar version of the close-vs-VWAP displacement that reached
# IR 0.083, but computable from daily bars with no intraday data.
#
# Distinct from A1's stochastic: that measures position in a 10-day range, this
# measures position within a SINGLE day against its own mean.
tp    = (high + low + close) / 3.0
score = cs_rank(ts_mean((close - tp) / at_zero2nan(tp), 5)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R4. Gap propensity ------------------------------------------------------
# Average absolute overnight gap over 60 days, as a fraction of price.
#
# A stock that gaps frequently prices its news discontinuously -- information
# arrives outside the session and the open jumps. A stock that grinds
# continuously is traded through the session. This is a characteristic of the
# INFORMATION ENVIRONMENT rather than of the price path, and nothing in your
# pool measures it.
gap   = open - ts_delay(close, 1)
agap  = (at_signsqrt(gap) * at_signsqrt(gap)) / at_zero2nan(ts_delay(close, 1))

score = cs_rank(ts_mean(agap, 60)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R5. Gap share of total movement -----------------------------------------
# What fraction of this stock's total price movement happens OVERNIGHT rather
# than during trading. Scale-free, and a cleaner statement of R4's idea because
# it divides out how volatile the name is overall.
#
# Lou-Polk-Skouras's clientele argument in characteristic form: names whose
# returns arrive overnight have a different holder base from names whose
# returns arrive intraday.
gap   = open - ts_delay(close, 1)
agap  = at_signsqrt(gap) * at_signsqrt(gap)
aday  = at_signsqrt(close - open) * at_signsqrt(close - open)

score = cs_rank(ts_sum(agap, 60) / at_zero2nan(ts_sum(agap + aday, 60))) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — EARNINGS CYCLE POSITION.  One line, never run.
# =============================================================================

# --- R6. Days-to-earnings as a pure signal -----------------------------------
# Long stocks about to report, short stocks that just did. This is the
# earnings-announcement premium (Savor & Wilson) as a bare cross-sectional
# alpha -- elevated returns around scheduled announcements as compensation for
# bearing event risk.
#
# You have used this variable as a GATE since note 14, where it produced the
# largest single improvement in the project (+0.025 from adding day 0). You
# have never ranked it. It is one line.
#
# RISK: it is obvious, so `max corr (others) < 0.90` is a real concern -- other
# competitors will have tried it. Cheap enough to be worth finding out.
du    = trading_days_until_next_earnings_announcement
score = -(cs_rank(du) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R7. Earnings cycle, event-gated -----------------------------------------
du    = trading_days_until_next_earnings_announcement
score = -(cs_rank(du) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- R8. Non-monotone earnings cycle -----------------------------------------
# The premium is unlikely to be linear in days-to-announcement. Note 14 found
# the gate worked best at 0-12 days and that day 0 alone carried a quarter of
# the gain -- which means the relationship is concentrated near zero, not
# smooth across the cycle. A rank cannot express that; this can.
#
# 1/(1+du) decays sharply: it is ~1.0 at announcement and near zero within a
# fortnight, which matches the shape note 14 measured.
du    = trading_days_until_next_earnings_announcement
score = cs_rank(1.0 / (1.0 + du)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)


# =============================================================================
# TIER 3 — DELAYED ENTRY.  Trading a different part of the response curve.
# =============================================================================
# Every alpha you own takes its position the day the signal fires. But a
# liquidity-provision premium is collected over the days that follow, and if
# any of it accrues on days 2-5 rather than day 1, a delayed version captures a
# DIFFERENT part of the same effect.
#
# Because the two books hold the same names at different times, correlation
# depends on how fast the signal decays: a fast-decaying signal gives a
# near-uncorrelated delayed version. A2's core decays in about a day
# (ts_mean(sk,5) lost 0.027 versus ts_mean(sk,1)), which makes it the best
# candidate here -- and also means the delayed version may have no signal left.
# Cheap to find out either way.

# --- R9. A2's core, entered 2 days late --------------------------------------
sk    = ts_mean(skew_return_last_hour, 1)
score = ts_delay(cs_rank(sk) - 1.0, 2)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- R10. A1's core, entered 3 days late -------------------------------------
# A1's core is slower (ts_zscore window 80), so more of it should survive a
# delay -- but for the same reason it will correlate more with the parent.
# The pair R9/R10 tests decay speed against correlation directly.
stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 80))
         - (((ts_median(cs_zscore(relvol), 10) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(ts_delay(core1, 3), 0.01, remove_extreme=False) * w * lovol)


# =============================================================================
# TIER 4 — LONGER-LAG SERIAL DEPENDENCE.
# =============================================================================

# --- R11. Lag-5 autocorrelation ----------------------------------------------
# Lag-1 autocorrelation is bid-ask bounce and inventory effects -- microstructure
# noise. Lag-5 is a weekly rhythm, which is where institutional order-splitting
# and rebalancing cycles would show up rather than market-making.
#
# Same operator as the lag-1 version, different economic content.
score = cs_rank(ts_corr_binary(ret1, ts_delay(ret1, 5), 120)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- R12. Volume irregularity ------------------------------------------------
# Coefficient of variation of daily volume. LOW means volume arrives evenly --
# programmatic, scheduled participation. HIGH means volume is event-driven and
# lumpy.
#
# This is the daily-bar version of the execution-regularity idea from the
# iceberg literature, and it needs no intraday data. Also a direct alternative
# proxy for passive ownership: index funds trade steadily, discretionary
# holders trade in bursts.
score = -(cs_rank(ts_std(volume, 20) / at_zero2nan(ts_mean(volume, 20))) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. R6, R8 -- one line each, and R8's shape is what note 14 actually measured.
#              Highest prior in the file despite being the simplest.
# 2. R12    -- alternative passive-ownership proxy from daily data only. If the
#              mandated-flow thesis has anything left, this is where.
# 3. R1, R5 -- bar geometry. R5 is the cleaner construction of the two.
# 4. R3, R4, R11.
# 5. R9, R10, R2, R7.
#
# JUDGE ON IR/sqrt(TVR) AND CHECK DRAWDOWN. The 60-120 day windows make most of
# these slow, near-static tilts -- the profile that produced M4's 19.5 drawdown
# and slow-mode failure. Low turnover is not low risk.
#
# IF THIS FILE ALSO COMES BACK EMPTY: stop hunting. You have seven qualified
# alphas and the other half of the score is combined out-of-sample performance.
# The unspent levers there are (a) the graded-weight test from file 39 tier 0,
# which has never run on any core, and (b) re-walking fomc_pre on A3, still
# sitting at a value chosen when total IR was 0.057.
# =============================================================================
