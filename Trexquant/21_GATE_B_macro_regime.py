# =============================================================================
# GATE FAMILY B — macro volatility regime  (calendar time, not event time)
# =============================================================================
# WHY THIS EXISTS
# ---------------
# Gate A (the 8-event stress union) conditions on MANDATED FLOW in EVENT time.
# Gate B conditions on LIQUIDITY SUPPLY in CALENDAR time. Same underlying
# thesis -- reversal pays more when liquidity providers withdraw -- expressed
# through a completely different variable.
#
# This is Nagel (2012) "Evaporating Liquidity" applied exactly as the paper
# states it: reversal returns are compensation for supplying liquidity, and
# that compensation is PREDICTABLE BY EXPECTED VOLATILITY. Gate A was the same
# claim in event time. Gate B is the direct test.
#
# Note 14 section 7 listed "High-VIX conditioning -- Nagel's direct result" as
# prediction #4 and marked it UNRUN. This closes that loop.
#
# WHY IT MULTIPLIES YOUR ALPHA COUNT
# ----------------------------------
# Gate A fires on periodic calendar dates, evenly spread across the sample.
# Gate B fires on persistent macro stress -- clustered in 2008, 2011, 2015,
# 2020. The two day-sets overlap only by coincidence, so
#     (same core) x (gate A)  and  (same core) x (gate B)
# should be far below the 50% self-correlation limit. Notes 14 and 15 already
# demonstrated this with opposite earnings gates on near-disjoint day sets.
#
# 3 proven cores x 2 gates = 6 alphas from work already done.
#
# DATA TRAP -- READ THIS
# ----------------------
# DO NOT build any condition on `close_spx`. The data dictionary records its
# maximum as 2396; the S&P last traded there in March 2017 and closed 2021 at
# 4766. The series is truncated or stale, and a gate using SPX LEVELS would
# silently stop working partway through the sample without erroring.
# Everything below uses ret1_spx / ret1_vix (return series, unaffected) and
# close_vix (guarded with at_zero2nan -- its stated range starts at 0, and VIX
# has never printed 0, so zeros are missing data in disguise).
# =============================================================================

# ---- GATE B ------------------------------------------------------------------
vix    = at_zero2nan(close_vix)
mktvol = ts_std(ret1_spx, 20)

r1 = (vix > ts_mean(vix, 60))                    # VIX above its own 3m trend
r2 = (ts_zscore(vix, 20) > 1.0)                  # VIX shock vs recent history
r3 = (mktvol > ts_mean(mktvol, 120))             # realised mkt vol elevated
r4 = (ts_sum(ret1_spx, 20) < 0)                  # market down over 20 days
r5 = (ts_sum(ret1_vix, 5) > 0)                   # vol rising this week

regime = r1 + r2 + r3 + r4 + r5
wm     = (regime > 0) * (1.0 + regime)

# =============================================================================
# APPLY TO EACH PROVEN CORE. Run these six first -- the cores are already
# validated on the platform, so only the conditioner is being tested.
# =============================================================================

# --- B1. A2's core (closing-hour skew) x gate B ------------------------------
# A2 with gate A: IR 0.102, TVR 1.380, DD 4.7
sk    = ts_mean(skew_return_last_hour, 1)
score = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wm)

# --- B2. A3's core (vol-regime coupling) x gate B ----------------------------
# A3 with gate A: IR 0.073, TVR 0.251, DD 11.8, IR/sqrt(TVR) 0.145
# NOTE: this core already contains a volatility term, so gate B may be partly
# redundant with it. If B2 underperforms B1 and B3, that is informative rather
# than a failure -- it says the core had already captured the vol conditioning.
core  = (ts_max(ts_mean(ts_max(ts_corr_binary(
             (ts_mean(ts_std(ret1, 20), 20) - ts_zscore(ts_std(ret1, 20), 10)),
             ret20, 60), 10), 20), 5)
         - (close / at_zero2nan(ts_mean(close, 20))))
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * wm)

# --- B3. A1's core (event-gated reversal) x gate B ---------------------------
# A1 with gate A: IR 0.091, TVR 0.739, DD 10.5
# This is the PUREST test of Nagel: A1 is explicitly a liquidity-provision
# alpha, and Nagel's result is specifically about liquidity provision and
# expected volatility. If gate B works anywhere, it should work here.
stoch  = (close - ts_min(low, 10)) / at_zero2nan(ts_max(high, 10) - ts_min(low, 10))
relvol = volume / at_zero2nan(ts_mean(volume, 20))
volret = cs_winsor(ts_std(ret1, 20) / at_zero2nan(ret20), 0.02, remove_extreme=False)

core = (-(ts_zscore(stoch, 10))
        - (((ts_median(cs_zscore(relvol), 20) - ts_mean(ret20, 20))
            + ts_skew(relvol - ts_delay(relvol, 3), 33))
           - ts_mean(-stoch - ts_median(volret, 3), 15)))

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * wm)

# =============================================================================
# GATE C — the INVERSE regime. Run only if gate B works.
# =============================================================================
# If reversal is paid MORE in stressed markets, the mirror alpha is a signal
# that works in CALM markets. Calm days are the complement of gate B's day set,
# so a core conditioned on calm is close to mechanically decorrelated from the
# same core conditioned on stress -- two submissions from one core.
#
# This is exactly the note-14 / note-15 relationship: same thesis, opposite
# conditioner, near-disjoint days, both qualified.
calm = (regime == 0)
wc   = calm * 1.0

# --- C1. A2's core x calm regime ---------------------------------------------
sk    = ts_mean(skew_return_last_hour, 1)
score = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wc)

# =============================================================================
# GATE D — stack A and B. Run after both are measured separately.
# =============================================================================
# Multiplying the two weights says: size up when mandated flow is heavy AND
# liquidity supply is thin. If the two mechanisms are independent, the product
# should beat either alone. If it does not, they were measuring the same thing.
#
# WATCH numstk AND booksize. Gate A alone leaves booksize at 0.91; the product
# will cut it further. Below ~160 names the alpha is disqualified regardless
# of IR -- that is what killed the earnings-gated version in note 16.
wd = w * wm

sk    = ts_mean(skew_return_last_hour, 1)
score = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wd)

# =============================================================================
# TUNING ORDER once a variant clears
# =============================================================================
# Walk ONE window at a time, recording IR / ret / TVR, exactly as you did for
# A3. Suggested ranges:
#     r1  ts_mean(vix, 60)      -> 20, 40, 60, 120
#     r2  ts_zscore threshold   -> 0.5, 1.0, 1.5
#     r3  ts_mean(mktvol, 120)  -> 60, 120, 250
#     r4  ts_sum(ret1_spx, 20)  -> 5, 10, 20, 60
#     r5  ts_sum(ret1_vix, 5)   -> 3, 5, 10
#
# AND RE-WALK THE FIRST KNOB AT THE END. Coordinate descent gives no guarantee
# that an early decision survives later moves -- fomc_pre in A3 was decided at
# IR 0.057 and never re-checked at 0.073, and the same mistake hid the day-0
# earnings finding in note 14 for two days.
# =============================================================================
