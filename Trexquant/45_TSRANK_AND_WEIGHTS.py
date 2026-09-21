# =============================================================================
# TIME-SERIES RANK, FUNDAMENTAL WEIGHTS, AND CROSS-SPLITS
# =============================================================================
# THREE ANGLES NOT YET USED
# -------------------------
# 1. ts_rank. Every alpha in the pool ranks stocks against EACH OTHER (cs_rank).
#    ts_rank ranks today's value against THAT STOCK'S OWN past. It is a
#    fundamentally different transform: it strips out the cross-sectional level
#    and keeps only "is this unusual FOR THIS NAME". Two alphas on the same
#    variable through cs_rank and ts_rank hold very different books, because a
#    permanently-high-skew stock ranks top cross-sectionally forever but sits
#    mid-pack against its own history.
#
#    This is the cheapest decorrelation available on variables already proven.
#
# 2. FAILED SIGNALS AS WEIGHTS. Fundamentals failed as ranked signals -- their
#    standalone Sharpes (0.25-0.8) sit below the 1.11 bar. But P2 showed that a
#    weight which cannot clear on its own can still ADD to a working core
#    (+0.012). A weight only needs to correlate with where the core works; it
#    does not need its own IR.
#
# 3. CROSS-SPLITS. You split one dimension at a time. lovol x large is ~190
#    names -- above the 160 floor -- and is a DIFFERENT book from either half
#    alone, so it can be submitted alongside both.
#
# DOLLAR VOLUME THROUGHOUT (close * volume). Per note 43, share volume is
# MUL-adjusted and mixing it with an invariant quantity creates a look-ahead
# through the split factor. close * volume is DIV x MUL and cancels exactly.
# =============================================================================

fomc_pre  = (days_until_next_fomc_meeting <= 2)
fomc_post = (days_since_last_fomc_meeting <= 15)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)
earn_pre  = (trading_days_until_next_earnings_announcement <= 15)
earn_post = (trading_days_until_next_earnings_announcement >= 12)

stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0) \
       + (earn_pre * 1.0) + (earn_post * 1.0)
w      = (stress > 0) * (1.0 + stress)

sz    = cs_rank(ts_mean(close * volume, 60))
vl    = cs_rank(ts_std(ret1, 60))
bt    = cs_rank(ts_corr_binary(ret1, ret1_spx, 60))
px    = cs_rank(close)

small = (sz <= 1.5)
large = (sz >  1.5)
lovol = (vl <= 1.5)
hivol = (vl >  1.5)
lobet = (bt <= 1.5)
hibet = (bt >  1.5)
lopx  = (px <= 1.5)
hipx  = (px >  1.5)

dv = close * volume                     # adjustment-invariant


# =============================================================================
# TIER 1 — ts_rank.  Same variables, different lens. Best decorrelation odds.
# =============================================================================

# --- T1. Closing-hour skew, ranked against its OWN history -------------------
# A2 ranks skew cross-sectionally: a stock with structurally high closing-hour
# skew sits at the top every day, so A2 carries a persistent tilt toward those
# names. This asks instead whether TODAY is unusual for THIS stock, which
# removes that tilt entirely and holds a different set of names.
#
# Same variable, same horizon, near-orthogonal book.
score = cs_rank(ts_rank(skew_return_last_hour, 60)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)

# --- T2. Same, shorter own-history window ------------------------------------
# 20 days makes "unusual for this stock" a much fresher judgement. Faster, and
# further from A2's persistent ranking.
score = cs_rank(ts_rank(skew_return_last_hour, 20)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)

# --- T3. Dollar-volume surprise, ts_rank -------------------------------------
# Where today's dollar volume sits in this stock's own 60-day distribution.
# Ranking volume cross-sectionally just ranks big companies -- a static size
# proxy and the clone trap that killed the first GP run. ts_rank makes it a
# genuine attention/participation surprise.
score = cs_rank(ts_rank(dv, 60)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)

# --- T4. Range position, ts_rank ---------------------------------------------
# A1's stochastic measures position in a 10-day high/low range and then ranks it
# cross-sectionally. This measures the same quantity against the stock's own
# distribution of range positions -- so a name that habitually closes near its
# high is not permanently long.
stoch = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
score = -(cs_rank(ts_rank(stoch, 60)) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)

# --- T5. Intraday range, ts_rank ---------------------------------------------
# Is today's range unusually wide for this stock? A volatility surprise that is
# scale-free by construction and does not inherit the vol level, which is what
# the lovol split already conditions on.
rng   = (high - low) / at_zero2nan(close)
score = cs_rank(ts_rank(rng, 60)) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)


# =============================================================================
# TIER 2 — FUNDAMENTALS AS WEIGHTS.  The P2 pattern, applied to failed signals.
# =============================================================================
# A weight does not need its own IR. It only needs to correlate with WHERE the
# core works. P2 proved this: mflow could not clear alone but added +0.012 to a
# core that already qualified.
#
# NUMSTK WARNING: fundamental coverage runs ~2000-2800 of ~3250 names, so a
# NaN weight drops that stock from the book. Expect numstk to fall to roughly
# 75-85% of the gated core's count. Check it -- with lovol already halving the
# universe, the product could approach the 160 floor.

stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 80))
         - (((ts_median(cs_zscore(relvol), 10) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

# --- T6. Reversal core weighted by profitability -----------------------------
# Hypothesis: reversal is safer in profitable companies, because a price drop in
# a quality name is more likely to be flow and less likely to be information.
# Fading flow in a deteriorating business is how a reversal strategy gets run
# over -- this tilts away from that.
fw    = cs_rank(return_on_assets)
alpha = at_zero2nan(cs_winsor(core1 * fw, 0.01, remove_extreme=False) * w * lovol)

# --- T7. Reversal core weighted by balance-sheet cash -------------------------
# Same logic through solvency rather than profitability. A cash-rich company's
# drawdown is less likely to be a solvency signal.
fw    = cs_rank(cash_to_total_assets_ratio)
alpha = at_zero2nan(cs_winsor(core1 * fw, 0.01, remove_extreme=False) * w * lovol)

# --- T8. Reversal core weighted by value --------------------------------------
# earnings_yield has the best coverage of the valuation fields (2766). Fading
# selling in an already-cheap name has more support underneath it.
fw    = cs_rank(earnings_yield)
alpha = at_zero2nan(cs_winsor(core1 * fw, 0.01, remove_extreme=False) * w * lovol)


# =============================================================================
# TIER 3 — SPLIT-VARIABLE DYNAMICS.  Changes, not levels.
# =============================================================================
# sz, vl, bt and px are used only to CUT the universe. Their rates of change
# have never been signals -- and a change is orthogonal to a level by
# construction, so these should decorrelate from the splits themselves.

# --- T9. Beta drift ----------------------------------------------------------
# Short-window beta minus long-window beta: is this name being repriced as a
# macro asset rather than on its own fundamentals? Rising beta typically marks a
# name becoming a crowded, correlated position. Differencing removes the beta
# level, which failed as a signal on its own.
b_fast = ts_corr_binary(ret1, ret1_spx, 60)
b_slow = ts_corr_binary(ret1, ret1_spx, 250)
score  = cs_rank(b_fast - b_slow) - 1.0
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)

# --- T10. Liquidity growth ---------------------------------------------------
# 20-day dollar volume over 250-day dollar volume. A name whose turnover is
# expanding is gaining attention, coverage or index membership -- the same
# demand-shift idea as the mandated-flow work, measured without needing the
# event calendar at all.
g     = ts_mean(dv, 20) / at_zero2nan(ts_mean(dv, 250))
score = cs_rank(g) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)

# --- T11. Volatility regime shift --------------------------------------------
# 20-day vol over 250-day vol. Distinct from the lovol SPLIT, which conditions
# on the level: this ranks the change within the low-vol half.
vs    = ts_std(ret1, 20) / at_zero2nan(ts_std(ret1, 250))
score = cs_rank(vs) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w * lovol)


# =============================================================================
# TIER 4 — CROSS-SPLITS.  Free alphas from books you already have.
# =============================================================================
# ~190 names each. Above the 160 floor but with little margin -- CHECK numstk,
# and remember it averages over the trailing 120 days so a thin patch can pull
# it under even when the average looks acceptable.
#
# lovol∩large and lovol∩small hold DISJOINT names, so they decorrelate from
# each other AND both differ from the plain lovol book.

# --- T12. A1 core, low vol AND large -----------------------------------------
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lovol * large)

# --- T13. A1 core, low vol AND small -----------------------------------------
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lovol * small)

# --- T14. A1 core, low vol AND low price -------------------------------------
# Price is close to independent of size, beta and volatility -- a $40 and a $400
# stock can be the same market cap -- so this cuts on an axis the others do not.
# Tick size is fixed at a cent, so the relative tick is ~10x larger on a $40
# name, which changes spread economics and how much of reversal is bid-ask
# bounce rather than premium.
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lovol * lopx)


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. T1, T2  -- ts_rank on A2's variable. Cheapest genuine decorrelation
#               available: proven variable, transform you have never used.
# 2. T12, T13 -- cross-splits on a core already at IR 0.094. Nearly guaranteed
#               to decorrelate; the only question is whether IR survives 190
#               names.
# 3. T6, T8  -- fundamental weights. The P2 pattern, and the first use of the
#               fundamental group in anything that qualifies.
# 4. T10, T9 -- split-variable dynamics.
# 5. T3, T4, T5, T7, T11, T14.
#
# CHECK numstk ON TIERS 2 AND 4. Both compound restrictions on top of lovol.
#
# IF ts_rank CLEARS ANYWHERE, sweep the window (20 / 60 / 120) before doing
# anything else -- window tuning has beaten gate tuning three times now
# (A2 +0.027 vs +0.014; S8's ts_zscore 10 -> 80 was the largest single lever in
# that alpha).
# =============================================================================
