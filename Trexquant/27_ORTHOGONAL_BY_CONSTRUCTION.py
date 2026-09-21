# =============================================================================
# ORTHOGONAL BY CONSTRUCTION — for a pool of 5 correlated reversal alphas
# =============================================================================
# WHY EVERYTHING KEEPS COLLIDING
# ------------------------------
# The correlation is not coming from your INPUTS, it is coming from the
# SHORT-TERM REVERSAL FACTOR. Every fast, cross-sectional, mean-reversion
# signal on US equities loads on the same underlying thing. Residual reversal
# collided for exactly this reason: stripping market and industry exposure does
# not remove the reversal factor -- the residual IS the reversal factor, in its
# purest form. That is why it had the highest Sharpe AND the worst collision.
#
# So no further reversal construction will help, however clever. Only two
# things reliably break correlation:
#
#   (A) HOLD DIFFERENT STOCKS. Two books over disjoint universes cannot
#       correlate much no matter how similar the signals are. This is the
#       most RELIABLE lever available and it works on cores you have ALREADY
#       PROVEN -- no new signal research required.
#
#   (B) TRADE A CONTINUATION EFFECT. Opposite payoff structure to reversal
#       rather than a variation on it.
#
# Both are below. (A) is the higher-probability path; (B) is the better story.
# =============================================================================


# =============================================================================
# PART A — UNIVERSE SPLITTING.  Highest probability of working.
# =============================================================================
# Take a core you have ALREADY QUALIFIED and run it on the top half and bottom
# half of the universe by size. The two books hold DISJOINT stocks, so their
# correlation is bounded by how much the two halves co-move -- which is far
# below the correlation between two signals on the same names.
#
# NUMSTK MATH: your alphas carry ~700 names, so each half is ~350 -- more than
# double the 160 floor. This is the one cross-sectional split you can afford.
# Do NOT split into thirds; ~230 each is survivable but leaves no margin, and
# booksize falls to a third.
#
# EXPECT: booksize roughly halves on each. IR may RISE on the small half --
# short-term reversal is well documented as stronger in smaller, less liquid
# names, where liquidity provision is scarcer and better paid. That is a real
# prediction, not a hope: if the small half does NOT beat the large half, the
# liquidity-provision story behind your whole pool is weaker than claimed.
#
# cs_rank returns [1,2], so 1.5 is the median.

sz    = cs_rank(ts_mean(close * volume, 60))
small = (sz <= 1.5)
large = (sz > 1.5)

# --- A1. Your best core, SMALL half ------------------------------------------
# Substitute whichever core you like. Shown with A2's (closing-hour skew,
# IR 0.102) because it has the most headroom to lose to the split.
sk    = ts_mean(skew_return_last_hour, 1)
core  = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w * small)

# --- A2. Same core, LARGE half -----------------------------------------------
sk    = ts_mean(skew_return_last_hour, 1)
core  = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w * large)

# --- A3. Split by VOLATILITY instead of size ---------------------------------
# A second, independent way to cut the universe -- and the halves are different
# from the size halves, so this yields another disjoint pair rather than a
# relabelling of the same one. Nagel's mechanism says the high-vol half should
# win.
vl    = cs_rank(ts_std(ret1, 20))
hivol = (vl > 1.5)
lovol = (vl <= 1.5)

sk    = ts_mean(skew_return_last_hour, 1)
core  = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w * hivol)

# --- A4. Low-vol half --------------------------------------------------------
sk    = ts_mean(skew_return_last_hour, 1)
core  = cs_rank(sk) - 1.0
alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w * lovol)


# =============================================================================
# PART B — CONTINUATION EFFECTS.  Opposite payoff structure to your pool.
# =============================================================================

# --- B1. High-volume return premium ------------------------------------------
# Gervais, Kaniel & Mingelgrin (JoF 2001). TREXQUANT'S OWN TUTORIAL USES THIS
# AS ITS WORKED CASE STUDY -- so it is known to function on this platform,
# which is a stronger prior than anything in the last three files.
#
# The effect: stocks with unusually HIGH volume over a period earn higher
# subsequent returns. The mechanism is visibility, not value -- a volume shock
# puts the stock in front of more investors, and attention drives buying
# pressure from constrained buyers who cannot short.
#
# CONTINUATION, not reversal. Note the sign is POSITIVE: long high volume.
# This runs opposite to the reversal instinct behind your whole pool, which is
# exactly why it is worth a run.
vshock = volume / at_zero2nan(ts_mean(volume, 50))
sig    = ts_mean(vshock, 5)
score  = cs_rank(sig) - 1.0
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- B2. Same, event-gated ---------------------------------------------------
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- B3. Post-earnings announcement drift ------------------------------------
# Bernard & Thomas (1989). Prices UNDER-react to earnings and drift in the
# direction of the surprise for weeks -- one of the most durable anomalies in
# finance, and a pure continuation effect.
#
# CONSTRUCTION TRICK: trading_days_until_next_earnings_announcement counts DOWN
# to the next report, then RESETS to ~62 the day after one lands. So a high
# value means "this name reported recently" -- which is the PEAD window, and it
# is available without any earnings-date or surprise data.
#
# ret20 for a recently-reported name is dominated by the announcement reaction,
# so ranking ret20 WITHIN that group is a surprise proxy. Long the names that
# jumped, short the ones that dropped, and hold the drift.
#
# NUMSTK: `du > 50` keeps roughly the third of names that reported in the last
# ~2 weeks -> ~230 of 700. Above the floor but with less margin than Part A.
# CHECK IT IN-SIM before submitting.
du     = trading_days_until_next_earnings_announcement
recent = (du > 50)
score  = (cs_rank(ret20) - 1.0) * recent
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- B4. PEAD, shorter reaction window ---------------------------------------
# ret5 instead of ret20 -- captures the announcement jump more cleanly with
# less pre-announcement drift mixed in. Faster, so higher turnover.
du     = trading_days_until_next_earnings_announcement
recent = (du > 55)
score  = (cs_rank(ret5) - 1.0) * recent
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- B5. PEAD x volume confirmation ------------------------------------------
# Bernard-Thomas drift is stronger when the announcement was accompanied by
# heavy volume -- high volume means the surprise was genuinely news rather than
# noise. Combines B1's mechanism with B3's, which is a real interaction rather
# than a bolt-together: both are attention effects.
du     = trading_days_until_next_earnings_announcement
recent = (du > 50)
vconf  = cs_rank(volume / at_zero2nan(ts_mean(volume, 50)))
score  = (cs_rank(ret20) - 1.0) * vconf * recent
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. B1  -- Trexquant's own tutorial case study. If anything works, this does,
#           and it is continuation so it should not collide.
# 2. B3  -- PEAD. Different mechanism, different holding period, uses the one
#           variable your notes call "the best single conditioner in the
#           dataset" as a SIGNAL rather than a gate for the first time.
# 3. A1/A2 -- the size split on your best core. Nearly guaranteed to
#           decorrelate; the only question is whether IR survives halving the
#           book. Run BOTH halves: they are separately submittable.
# 4. A3/A4, B5, B2, B4.
#
# ON CORRELATION: for Part A, correlation between the two halves is bounded by
# how much small and large caps co-move -- typically well under 50% for
# long/short books on disjoint names. Check it, but this is the closest thing
# to a guarantee available to you.
#
# ON PART A AND HONESTY: two halves of one core are not two ideas. They will
# score as two alphas and that is worth having with count as half the score,
# but do not present them in an interview as independent discoveries -- an
# interviewer who sees both will ask, and the answer is that you split the
# universe to exploit a scoring rule.
# =============================================================================
