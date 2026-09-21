# =============================================================================
# H4 BETA-RESIDUAL REVERSAL — gated variants
# =============================================================================
# UNGATED BASELINE: IR 0.051
#
# CONTEXT: A1's core was 0.055 ungated and reached 0.091 with gate A (+0.036).
# A2's was 0.088 -> 0.102 (+0.014). You need +0.019 from a base almost
# identical to A1's, so the prior here is good.
#
# WHY BOTH GATE FAMILIES APPLY: residual reversal is the PUREST
# liquidity-provision strategy in the pool -- it is paid for absorbing purely
# idiosyncratic shocks, with market and factor exposure already stripped out.
# Gate A (mandated flow) and gate B (Nagel's expected-volatility result) are
# both direct statements about when that service is best compensated.
#
# RUN G1 AND G2 BOTH. If each clears and they clear the 50% check against each
# other, that is TWO alphas from one core -- their day-sets barely overlap
# (periodic calendar vs clustered macro stress).
# =============================================================================

# ---- the core (unchanged, IR 0.051) -----------------------------------------
beta = ts_corr_binary(ret1, ret1_spx, 60)
res  = ret5 - (beta * ts_sum(ret1_spx, 5))
sig  = -res / at_zero2nan(ts_std(ret1, 20))
core = cs_rank(sig) - 1.0


# =============================================================================
# G1 — GATE A (event stress).  Highest prior of clearing IR.
# =============================================================================
# The exact windows that took A3 to 0.073. Proven +0.036 on A1's reversal core,
# which is the closest analogue to this one.
#
# RISK: all three existing alphas wear this gate, so it is also the highest
# correlation risk. Check before submitting.
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

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)


# =============================================================================
# G2 — GATE B (macro vol regime).  Nagel's direct result. Best decorrelation.
# =============================================================================
# Nagel (2012): reversal returns are compensation for liquidity provision and
# are PREDICTABLE BY EXPECTED VOLATILITY. This core is residual reversal with
# market exposure stripped -- the cleanest possible instance of the strategy
# Nagel is describing. If gate B works anywhere in this project, here.
#
# Gate B failed against your EXISTING cores because those cores were already in
# the pool wearing gate A; the collision was the core, not the gate.
vix    = at_zero2nan(close_vix)
mktvol = ts_std(ret1_spx, 20)

r1 = (vix > ts_mean(vix, 60))
r2 = (ts_zscore(vix, 20) > 1.0)
r3 = (mktvol > ts_mean(mktvol, 120))
r4 = (ts_sum(ret1_spx, 20) < 0)
r5 = (ts_sum(ret1_vix, 5) > 0)

regime = r1 + r2 + r3 + r4 + r5
wm     = (regime > 0) * (1.0 + regime)

alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * wm)


# =============================================================================
# G3 — VOLATILITY WEIGHTING.  Nagel applied CROSS-SECTIONALLY. No numstk cost.
# =============================================================================
# Nagel's claim is about expected volatility, and gate B tests it in CALENDAR
# time (is the MARKET volatile today). But the same claim has a cross-sectional
# reading: reversal should pay more in the NAMES whose own volatility is
# elevated, because those are the names where absorbing a shock is riskiest.
#
# Implemented as a WEIGHT, not a gate -- so every stock stays in the book and
# numstk is untouched. That is the advantage over any cross-sectional gate,
# which is what destroyed the earnings-gated variant in note 16 (numstk 156).
#
# cs_rank returns [1,2], so vw runs [1,2]: the highest-vol names get twice the
# weight of the lowest, nothing gets zeroed.
vw = cs_rank(ts_std(ret1, 20))

alpha = at_zero2nan(cs_winsor(core * vw, 0.01, remove_extreme=False) * w)


# =============================================================================
# G4 — GATE A x GATE B stacked.
# =============================================================================
# Size up when mandated flow is heavy AND liquidity supply is thin. If the two
# mechanisms are independent the product should beat either alone; if not, they
# were measuring the same thing and this tells you so.
#
# WATCH numstk AND booksize -- the product cuts active days hard. Below 160
# names it is disqualified regardless of IR.
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w * wm)


# =============================================================================
# TUNING ORDER — only after one variant clears
# =============================================================================
# The core has THREE untuned windows and they were my arbitrary choices, not
# fitted values. On A2 the window sweep was worth +0.027 while the gate was
# worth +0.014, so tune the core BEFORE tuning the gate.
#
#   1. beta window        ts_corr_binary(..., 60)   -> 20, 60, 120, 250
#      Short beta is noisy, long beta is stale. This is the knob most likely
#      to matter: it sets how much market exposure actually gets removed.
#
#   2. reversal horizon   ret5 / ts_sum(ret1_spx, 5)
#      -> try ret1 with ts_sum(ret1_spx, 1), and ret10 with ts_sum(..., 10).
#      MUST MOVE TOGETHER -- the subtracted market term has to span the same
#      window as the stock return, or you are subtracting a mismatched hedge.
#
#   3. vol-scaling window ts_std(ret1, 20)          -> 10, 20, 60
#
# Then re-walk knob 1. Coordinate descent gives no guarantee an early choice
# survives later moves -- that is what hid the day-0 earnings finding for two
# days (note 14 section 4) and it is why A3's fomc_pre is still unverified.
# =============================================================================
