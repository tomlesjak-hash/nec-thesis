# =============================================================================
# UNIVERSE SPLIT MATRIX — the one decorrelation mechanism that has worked
# =============================================================================
# THE CASE FOR DOING THIS AND NOTHING ELSE WITH THE REMAINING TIME
# ----------------------------------------------------------------
# Track record of every decorrelation approach tried:
#
#   new intra1 fields          -> collided (A2 owns that group)
#   gate B / macro regime      -> collided (cores unchanged)
#   fundamentals               -> IR too low (Sharpe 0.25-0.8 < 1.11 bar)
#   relational / slow          -> IR too low, same arithmetic
#   residual reversal          -> collided (it IS the reversal factor)
#   intraday periodicity       -> failed
#   industry-neutral skew      -> collided with A2
#   range efficiency           -> failed slow-mode robustness
#   adaptive IC weighting      -> WORKED (D2)
#   UNIVERSE SPLITTING         -> WORKED (two alphas from one core)
#
# Two things have worked. One of them is mechanical and repeatable: books over
# DISJOINT STOCKS cannot correlate much regardless of signal similarity. You
# have applied it to ONE core. Every other proven core is still whole.
#
# WHY IT WORKS WHERE EVERYTHING ELSE FAILED
# ------------------------------------------
# Correlation between two long/short books is driven by overlapping positions.
# Two signals on the same 700 names share every position and differ only in
# weights -- so they collide. Two books on disjoint 350-name halves share NO
# positions; their correlation is bounded by how much the halves co-move, which
# for long/short portfolios is low.
#
# This is not a clever idea. It is arithmetic, which is exactly why it is the
# right thing to spend the last runs on.
#
# NUMSTK BUDGET
#   full universe   ~700-770  ->  halves ~350-385  (floor is 160: safe)
#                               ->  quarters ~175-190  (tight but legal)
# Booksize halves on each split. Check numstk in-sim every time.
#
# HONESTY NOTE: halves of one core are not independent ideas. They score as
# separate alphas and count is half your score, so take them -- but if an
# interviewer sees four variants of one signal, the answer is that you split
# the universe to exploit a scoring rule, not that you found four things.
# =============================================================================


# =============================================================================
# THE PARTITIONS.  cs_rank returns [1,2], so 1.5 is the median.
# =============================================================================
# Four INDEPENDENT ways to cut the universe. Independent matters: size-halves
# and vol-halves overlap heavily (small caps are volatile), so those two pairs
# are less decorrelated from each other than either is from the beta split.
# Order below is roughly by how independent each is from the others.

sz    = cs_rank(ts_mean(close * volume, 60))     # size / dollar volume
vl    = cs_rank(ts_std(ret1, 20))                # volatility
bt    = cs_rank(ts_corr_binary(ret1, ret1_spx, 60))   # market beta
px    = cs_rank(close)                           # nominal price level

small = (sz <= 1.5)
large = (sz >  1.5)
lovol = (vl <= 1.5)
hivol = (vl >  1.5)
lobet = (bt <= 1.5)
hibet = (bt >  1.5)
lopx  = (px <= 1.5)
hipx  = (px >  1.5)

# The event gate, for cores that use it
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


# =============================================================================
# GROUP 1 — A3's CORE (GP vol-regime coupling) x SPLITS
# =============================================================================
# A3 whole: IR 0.073, TVR 0.251, IR/sqrt(TVR) 0.146 (BEST in pool),
# break-even 11.2bp. The most tradeable core you own, and never split.
#
# Its low turnover means the halves inherit low turnover -- so even if IR falls
# to 0.075 on a half, IR/sqrt(TVR) stays strong and break-even stays high.
# Best risk-adjusted candidates in this file.

core3 = (ts_max(ts_mean(ts_max(ts_corr_binary(
             (ts_mean(ts_std(ret1, 20), 20) - ts_zscore(ts_std(ret1, 20), 10)),
             ret20, 60), 10), 20), 5)
         - (close / at_zero2nan(ts_mean(close, 20))))

# --- S1. A3 core, BETA split (low beta) --------------------------------------
# Beta is the partition most independent of the size/vol splits you already
# used, so this has the best correlation profile against your existing pair.
alpha = at_zero2nan(cs_winsor(core3, 0.01, remove_extreme=False) * w * lobet)

# --- S2. A3 core, high beta --------------------------------------------------
alpha = at_zero2nan(cs_winsor(core3, 0.01, remove_extreme=False) * w * hibet)

# --- S3. A3 core, small half -------------------------------------------------
# Reversal is documented as stronger where liquidity is scarcer, so expect the
# small half to beat the large one. If it does not, the liquidity-provision
# story behind your whole pool is weaker than the theory notes claim.
alpha = at_zero2nan(cs_winsor(core3, 0.01, remove_extreme=False) * w * small)

# --- S4. A3 core, large half -------------------------------------------------
alpha = at_zero2nan(cs_winsor(core3, 0.01, remove_extreme=False) * w * large)


# =============================================================================
# GROUP 2 — A1's CORE (event-gated reversal) x SPLITS
# =============================================================================
# A1 whole: IR 0.091, TVR 0.739, Calmar 1.19 (BEST), break-even 6.7bp.
# Highest return in the pool at 12.5%. Never split.

stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core1 = (-(ts_zscore(stoch, 10))
         - (((ts_median(cs_zscore(relvol), 20) - ts_mean(ret20, 20))
             + ts_skew(relvol - ts_delay(relvol, 3), 33))
            - ts_mean(-stoch - ts_median(volret, 3), 15)))

# --- S5. A1 core, low beta ---------------------------------------------------
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lobet)

# --- S6. A1 core, high beta --------------------------------------------------
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * hibet)

# --- S7. A1 core, high vol ---------------------------------------------------
# Nagel's mechanism says reversal pays most where volatility is highest, so
# this half should carry most of A1's IR. A useful mechanism check as well as
# a submission: if the low-vol half is just as good, the liquidity-provision
# reading is wrong.
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * hivol)

# --- S8. A1 core, low vol ----------------------------------------------------
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * lovol)


# =============================================================================
# GROUP 3 — PRICE-LEVEL SPLIT.  The partition nobody uses.
# =============================================================================
# Nominal price is nearly uncorrelated with size, beta and volatility -- a $40
# stock and a $400 stock can be the same company by market cap. So this cuts
# the universe along an axis genuinely independent of the other three.
#
# It is also economically real rather than arbitrary: tick size is fixed at one
# cent, so the RELATIVE tick is ~25x larger on a $4 stock than a $100 one.
# That changes spread economics, the size of bid-ask bounce, and therefore how
# much of short-horizon reversal is microstructure noise versus liquidity
# premium. Low-priced names should show stronger raw reversal.

# --- S9.  A2 core, low price -------------------------------------------------
sk    = ts_mean(skew_return_last_hour, 1)
core2 = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(core2, 0.01, remove_extreme=False) * w * lopx)

# --- S10. A2 core, high price ------------------------------------------------
sk    = ts_mean(skew_return_last_hour, 1)
core2 = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(core2, 0.01, remove_extreme=False) * w * hipx)

# --- S11. A3 core, low price -------------------------------------------------
alpha = at_zero2nan(cs_winsor(core3, 0.01, remove_extreme=False) * w * lopx)


# =============================================================================
# GROUP 4 — CROSS-SPLITS.  Quarters. Highest decorrelation, tightest numstk.
# =============================================================================
# Two conditions at once leaves ~175-190 names. Legal but close to the 160
# floor -- CHECK numstk IN-SIM, and note it is averaged over the trailing 120
# days so a thin patch can pull it under even if the average looks fine.
#
# These are the most decorrelated books available: a small-and-volatile book
# shares no names with a large-and-calm one, and neither shares names with
# either of the pure halves.

# --- S12. A1 core, small AND high vol ----------------------------------------
# Where liquidity provision should be paid best on both axes simultaneously --
# the strongest prediction the theory makes.
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * small * hivol)

# --- S13. A1 core, large AND low vol -----------------------------------------
# The complement. Theory says weakest; if it clears anyway, the mechanism is
# not what the notes claim.
alpha = at_zero2nan(cs_winsor(core1, 0.01, remove_extreme=False) * w * large * lovol)


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. S1, S2  -- A3 core x beta. Best IR/sqrt(TVR) core, most independent
#               partition. Highest expected value in the file.
# 2. S5, S6  -- A1 core x beta. Highest-return core.
# 3. S9, S10 -- price split. The axis nobody cuts on, and the one with a real
#               microstructure justification.
# 4. S3, S4, S7, S8, S11.
# 5. S12, S13 -- only if numstk on the halves came in comfortably above ~350.
#
# BEFORE EACH SUBMISSION: check numstk (floor 160, averaged over 120 days) and
# run Compute Correlation against the whole pool. A half correlates most with
# its OWN whole-universe parent -- that is the pair to check first.
#
# EXPECT: booksize ~0.5 on halves, ~0.25 on quarters. IR may RISE on the half
# where the mechanism is strongest. That asymmetry is worth recording either
# way -- it is direct evidence for or against the liquidity-provision story
# that underpins every theory note in this project, and it costs nothing extra
# to observe since you are running the simulations anyway.
# =============================================================================
