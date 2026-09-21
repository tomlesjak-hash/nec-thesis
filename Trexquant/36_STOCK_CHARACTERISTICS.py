# =============================================================================
# STOCK CHARACTERISTICS — the construction type that has actually worked
# =============================================================================
# THE PATTERN IN YOUR OWN RESULTS
# -------------------------------
#   WORKED   M4 range efficiency   IR 0.103  -- unsigned CHARACTERISTIC
#   WORKED   A2 closing-hour skew  IR 0.102  -- shape statistic, not direction
#   WORKED   A3 vol-regime coupling IR 0.073 -- a per-stock CORRELATION
#   FAILED   momentum, value, quality, 52w high, periodicity, agreement
#            products, tent transforms -- all DIRECTIONAL forecasts
#
# The three that qualified all rank stocks by a STRUCTURAL PROPERTY of how the
# name trades, not by a prediction of where it is going. That is a real pattern
# and it is worth spending the remaining runs on rather than more forecasts.
#
# Six characteristics below, none of them ever computed in this project, all
# from data you already have. Each is a well-defined quantity from the
# microstructure or econometrics literature -- so each has a mechanism to
# defend in an interview, unlike a GP expression.
#
# HELPERS
#   at_signsqrt(x) * at_signsqrt(x) = |x|
#   sign(x) = x / |x|
#   Booleans may sit in plain arithmetic but NEVER enter a ts_ operator.
# =============================================================================


# =============================================================================
# C1 — RETURN AUTOCORRELATION.  "Does reversal even work on this name?"
# =============================================================================
# The stock's own first-order return autocorrelation over 60 days. Strongly
# NEGATIVE means this name systematically reverses -- bid-ask bounce and
# inventory effects dominate its tape. Near zero means it random-walks.
#
# This is the single most direct measurement of the property your entire pool
# is trying to monetise, and you have never measured it. Every reversal alpha
# you own implicitly ASSUMES reversal works; this ranks stocks by whether it
# actually does.
#
# Sign as written: long high autocorrelation (trending names). Flip if
# negative -- the alternative reading is that names where reversal works best
# are where liquidity provision pays most, which is the note-14 thesis.
score = cs_rank(ts_corr_binary(ret1, ts_delay(ret1, 1), 60)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# C2 — VARIANCE RATIO (Lo & MacKinlay).  Mean-reverting vs trending.
# =============================================================================
# If returns were i.i.d., 5-day volatility would be sqrt(5) times 1-day
# volatility. The ratio of realised to implied-by-scaling measures departure
# from a random walk: below 1 means returns offset each other (mean reversion),
# above 1 means they compound (trending).
#
# Related to C1 but not the same statistic -- autocorrelation captures the
# one-lag relationship, the variance ratio aggregates across all lags inside
# the horizon. Where they disagree is where the reversal is at lags 2-5 rather
# than at lag 1, which is a genuinely different name set.
#
# sqrt(5) = 2.2360679
vr    = ts_std(ret5, 60) / at_zero2nan(ts_std(ret1, 60) * 2.2360679)
score = cs_rank(vr) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# C3 — VOLATILITY TERM STRUCTURE.  Short vol against long vol.
# =============================================================================
# 5-day vol over 60-day vol. Above 1 means this stock's volatility is currently
# elevated relative to its own baseline -- a vol term structure in backwardation.
#
# Distinct from the vol LEVEL (which just ranks volatile stocks forever, the
# clone trap) and from A3's vol-regime term (which correlates vol against
# returns). This is purely the SLOPE, and it is scale-free so it does not
# inherit the level's persistence.
#
# Nagel's mechanism says liquidity provision is paid most when EXPECTED
# volatility is high. Gate B tested that in calendar time via VIX and failed;
# this tests it per-stock, which gate B structurally could not.
vts   = ts_std(ret1, 5) / at_zero2nan(ts_std(ret1, 60))
score = cs_rank(vts) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# C4 — RETURN-VOLUME CORRELATION.  Who is trading this name.
# =============================================================================
# Per-stock correlation between daily return and daily volume over 60 days.
#
# POSITIVE means volume arrives on up-days -- buying pressure needs volume to
# push price, the classic signature of demand-driven, retail-heavy or
# attention-driven flow. NEGATIVE means volume arrives on down-days: selling
# pressure, or informed sellers working out of positions.
#
# This is a foundational microstructure quantity (the volume-volatility and
# volume-return relations run back to Karpoff 1987) and it is a CHARACTERISTIC
# of the shareholder base rather than a forecast. Nothing in your pool measures
# who the marginal trader is.
score = cs_rank(ts_corr_binary(ret1, volume, 60)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# C5 — SEMIVOLATILITY ASYMMETRY.  Upside vol against downside vol.
# =============================================================================
# Separate the return series into its positive and negative parts, take the
# volatility of each, and rank the ratio.
#
# A stock whose downside volatility greatly exceeds its upside volatility grinds
# up and gaps down -- crash-prone, and typically crowded. The reverse profile
# grinds down and spikes up. Downside beta is published (Ang, Chen & Xing); the
# per-stock SEMIVOLATILITY RATIO is much less traded, and unlike downside beta
# it needs no market series, so it isolates a property of the stock itself.
#
# CONSTRUCTION: (sgn+1)/2 is 1 on up days and 0 on down days, so ret1 times it
# is the positive part. Built with the sign trick because a boolean cannot
# enter ts_std.
sgn = ret1 / at_zero2nan(at_signsqrt(ret1) * at_signsqrt(ret1))
up  = ret1 * (sgn + 1.0) * 0.5
dn  = ret1 * (1.0 - sgn) * 0.5

asym  = ts_std(up, 60) / at_zero2nan(ts_std(dn, 60))
score = cs_rank(asym) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# C6 — ANNOUNCEMENT RETURN PERSISTENCE.  Frazzini & Lamont.
# =============================================================================
# Firms whose past earnings-announcement-day returns were high continue to earn
# high returns on FUTURE announcement dates. Documented, cross-sectional, and a
# firm characteristic rather than a price forecast.
#
# CONSTRUCTION TRICK: trading_days_until_next_earnings counts DOWN, so the daily
# change is -1 on ordinary days and jumps sharply POSITIVE the day an
# announcement resets the counter. sign(jump) therefore isolates announcement
# days without needing announcement dates.
#
# Averaging ret1 over just those days across 250 trading days gives this stock's
# typical announcement-day return -- roughly four observations, which is thin,
# so treat this as the highest-variance idea in the file.
#
# This is also the first time you use trading_days_until_next_earnings as a
# SIGNAL rather than a gate. Your own notes call it the best single conditioner
# in the dataset.
du   = trading_days_until_next_earnings_announcement
jump = du - ts_delay(du, 1)
s    = jump / at_zero2nan(at_signsqrt(jump) * at_signsqrt(jump))   # +1 on reset
ann  = (s + 1.0) * 0.5                                             # 1 on announce day

sig   = ts_sum(ret1 * ann, 250) / at_zero2nan(ts_sum(ann, 250))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# C7 — C6 CONDITIONED ON AN UPCOMING ANNOUNCEMENT.
# =============================================================================
# Frazzini-Lamont's premium is earned ON announcement dates, so holding the
# characteristic permanently dilutes it. This concentrates the book into names
# about to report.
#
# COST: cross-sectional gate. `du <= 5` keeps maybe 8% of names on any day --
# roughly 60 of 750, which is BELOW the 160 floor. `du <= 20` keeps ~30%,
# around 230, which is legal. Start at 20 and only tighten if numstk allows.
du   = trading_days_until_next_earnings_announcement
jump = du - ts_delay(du, 1)
s    = jump / at_zero2nan(at_signsqrt(jump) * at_signsqrt(jump))
ann  = (s + 1.0) * 0.5

sig   = ts_sum(ret1 * ann, 250) / at_zero2nan(ts_sum(ann, 250))
near  = (du <= 20)
score = (cs_rank(sig) - 1.0) * near
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. C1  -- return autocorrelation. Most direct measurement of the property
#           your whole pool monetises, never measured, one operator.
# 2. C4  -- return-volume correlation. Foundational microstructure quantity,
#           and the only thing here that describes the SHAREHOLDER BASE rather
#           than the price series -- best decorrelation prospects.
# 3. C3  -- vol term structure. Tests Nagel per-stock, which gate B could not.
# 4. C2, C5.
# 5. C6, C7 -- highest variance, best story if they land.
#
# ALL UNGATED, deliberately. Most of your pool wears the event gate and that is
# where collisions have come from. Add `* w` only if a signal misses IR alone.
#
# THESE ARE SLOW SIGNALS -- 60-day windows mean rankings barely move day to day.
# Expect TVR 0.05-0.30 and break-evens in the tens of bp, which would make them
# the most tradeable things in your pool alongside A3. Judge on IR/sqrt(TVR).
#
# WARM-UP: the 250-day window in C6/C7 pushes the start date out by a year.
# Check numstk and the start date before reading anything into the IR.
# =============================================================================
