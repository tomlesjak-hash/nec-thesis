# =============================================================================
# BOOK-STRUCTURE ALPHAS — decorrelating by changing WHICH NAMES YOU HOLD
# =============================================================================
# THE PRINCIPLE, RESTATED
# -----------------------
# Correlation between two long/short books comes from overlapping positions.
# Universe splitting worked because the halves hold disjoint names. But
# splitting on a CHARACTERISTIC (size, beta, price) is only one way to make
# two books hold different names.
#
# You can also change which names a book holds by changing the TRANSFORM
# applied to the signal. A monotone rank puts the biggest positions at the
# extremes of the signal distribution. A non-monotone transform puts them in
# the middle. Those two books share almost no positions in the same direction
# -- from the SAME signal, with no universe restriction and no numstk cost.
#
# And a book built on whether two signals AGREE is not directional in either
# signal, so it is structurally orthogonal to both parents in the way M4's
# unsigned efficiency was orthogonal to directional alphas.
#
# Everything here reuses signals you have already proven. No new data, no new
# research -- just different books over the same information.
#
# HELPERS
#   cs_rank returns [1,2].  cs_rank(x) - 1.5  centres at 0, range [-0.5,+0.5].
#   at_signsqrt(x) * at_signsqrt(x) = |x|.
# =============================================================================

# ---- the proven cores, centred -----------------------------------------------
sk = ts_mean(skew_return_last_hour, 1)
A  = cs_rank(sk) - 1.5                                     # A2's core, centred

beta = ts_corr_binary(ret1, ret1_spx, 60)
res  = ret5 - (beta * ts_sum(ret1_spx, 5))
B    = cs_rank(-res / at_zero2nan(ts_std(ret1, 20))) - 1.5  # residual reversal

C = cs_rank(ts_max(ts_mean(ts_max(ts_corr_binary(
        (ts_mean(ts_std(ret1, 20), 20) - ts_zscore(ts_std(ret1, 20), 10)),
        ret20, 60), 10), 20), 5)
        - (close / at_zero2nan(ts_mean(close, 20)))) - 1.5  # A3's core

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
# TIER 1 — SIGNAL AGREEMENT.  Structurally orthogonal to both parents.
# =============================================================================
# The product of two CENTRED ranks is positive when both signals point the same
# way (both high or both low) and negative when they disagree. So this book is
# long names where two independent signals CONCUR and short names where they
# CONFLICT -- and it holds no view on the direction either signal points.
#
# That is why it should decorrelate: a blend a+b inherits both parents'
# directional exposure, while a*b inherits neither. It is the same structural
# trick that made M4's unsigned efficiency a characteristic rather than a
# forecast -- and M4 reached IR 0.103 on that basis before failing on sample
# robustness, so the construction type is known to carry signal here.
#
# ECONOMIC READING: two liquidity-provision signals agreeing means the flow
# reading is unambiguous -- the same conclusion from closing-hour shape and
# from residual price movement. Disagreement means the tape is mixed and the
# liquidity-provision premium is harder to collect. You are long clarity.

# --- G1. Agreement of A2 core and residual reversal --------------------------
score = A * B
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- G2. Same, event-gated ---------------------------------------------------
score = A * B
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- G3. Agreement of A2 core and A3 core ------------------------------------
# Different pair, so a different book. A3's core is slow and A2's is fast, so
# their agreement is a statement about signals at different horizons concurring
# -- arguably a stronger conviction reading than two fast signals agreeing.
score = A * C
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- G4. Three-way agreement -------------------------------------------------
# Positive only when all three concur or when exactly two dissent -- a sharper
# filter. Expect a more concentrated book, so CHECK numstk and effective
# concentration, not just the position count.
score = A * B * C
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)


# =============================================================================
# TIER 2 — NON-MONOTONE TRANSFORMS.  Hold the middle, short the tails.
# =============================================================================
# Every alpha you own is monotone in its signal: biggest longs at one extreme,
# biggest shorts at the other. A tent transform inverts that entirely -- the
# largest positions sit at the MEDIAN of the signal distribution and the tails
# are shorted.
#
# The two books share almost no positions in the same direction, so correlation
# should be very low. Full universe, no numstk cost.
#
# WHAT IT TESTS: whether the signal-return relationship is monotone. If the
# tent clears, the relationship is hump-shaped -- moderate signal values
# predict better than extreme ones, which would mean extreme readings are
# contaminated (news, halts, data errors) rather than being the strongest
# evidence. Lesmond's trade-size result is exactly this shape: information sits
# in medium trades, not at either end.
#
# HONEST PRIOR: low. Most signals are monotone and this will produce nothing.
# It is cheap, and a negative result is worth knowing before an interview.

# --- G5. Tent transform on A2's core -----------------------------------------
r     = cs_rank(sk) - 1.0                          # [0,1]
d     = r - 0.5
tent  = 1.0 - 2.0 * (at_signsqrt(d) * at_signsqrt(d))
score = tent - 0.5
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- G6. Inverted tent — hold the tails, short the middle --------------------
# The complement. If the relationship IS monotone, this should work BETTER than
# the plain rank, because it concentrates the book where the evidence is
# strongest and stands aside in the ambiguous middle. That makes G6 the
# higher-prior of this pair, despite being the less interesting idea.
r     = cs_rank(sk) - 1.0
d     = r - 0.5
score = at_signsqrt(d) * at_signsqrt(d)            # = |d|, V-shaped
alpha = at_zero2nan(cs_winsor(score - 0.25, 0.01, remove_extreme=False) * w)


# =============================================================================
# TIER 3 — ADAPTIVE SWITCHING.  D3 from file 28, still unrun.
# =============================================================================
# D2 (adaptive BLEND) qualified at IR 0.075. The SWITCH version was never
# tested, and it has better decorrelation odds by construction: it holds each
# component only on the stocks where that component currently has the higher
# rolling IC, so its exposure to each parent is roughly halved relative to a
# blend that always carries both.
#
# It is also a different book from D2 -- a partition of names rather than a
# weighted average over all of them.

# --- G7. Adaptive switch, A2 core vs residual reversal -----------------------
ic_a = ts_corr_binary(ts_delay(A, 1), ret1, 60)
ic_b = ts_corr_binary(ts_delay(B, 1), ret1, 60)

pick  = (ic_a > ic_b) * 1.0
score = (A * pick) + (B * (1.0 - pick))
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- G8. Adaptive switch, LOSER version --------------------------------------
# Hold whichever signal has been working WORSE. Sounds perverse, and it is the
# falsification test for the whole adaptive-weighting idea: if G8 performs as
# well as G7, then rolling IC carries no information and D2's qualification was
# luck rather than mechanism.
#
# This is the ablation I flagged as missing from note 30. It costs one run and
# it is the question an interviewer will ask about D2.
ic_a = ts_corr_binary(ts_delay(A, 1), ret1, 60)
ic_b = ts_corr_binary(ts_delay(B, 1), ret1, 60)

pick  = (ic_a < ic_b) * 1.0
score = (A * pick) + (B * (1.0 - pick))
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)


# =============================================================================
# TIER 4 — CONTINUOUS INTERACTION WEIGHTS.  Splitting without splitting.
# =============================================================================
# A hard split halves your booksize. A continuous weight tilts toward the same
# names while keeping the full universe -- same economic intent, no numstk cost
# and no booksize loss.
#
# Because the resulting book overlaps the parent's positions but reweights them
# substantially, correlation lands somewhere between "hard split" and "same
# alpha". Worth testing where the hard split has already worked, since it may
# capture the same effect with better coverage.

# --- G9. A2 core weighted by illiquidity -------------------------------------
# Amihud impact: |return| per unit volume, in the closing hour. Reversal and
# liquidity provision are paid most where absorbing flow is hardest, so tilt
# toward the names where it is. Continuous version of the vol/size splits.
absr  = at_signsqrt(return_last_hour) * at_signsqrt(return_last_hour)
imp   = absr / at_zero2nan(volume_last_hour)
iw    = cs_rank(imp / at_zero2nan(ts_mean(imp, 20)))       # [1,2]

score = A * iw
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- G10. A2 core weighted by intraday range ---------------------------------
# Relative daily range is a spread and uncertainty proxy that needs no volume
# data. Wide-range names are where liquidity provision is riskiest and best
# paid -- the same thesis through a different measurement, so if G9 works and
# G10 does not (or vice versa) that tells you which proxy is doing the work.
rng   = (high - low) / at_zero2nan(close)
rw    = cs_rank(rng / at_zero2nan(ts_mean(rng, 20)))

score = A * rw
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. G1, G2  -- signal agreement. Best decorrelation odds and the most
#               interesting construction. G1 ungated first, since ungated
#               already helps you against a pool that mostly wears the gate.
# 2. G8      -- the D2 ablation. Not primarily a submission: it is the answer
#               to the hardest question about an alpha you have ALREADY
#               submitted. Run it whatever else you do.
# 3. G7      -- adaptive switch.
# 4. G9, G10 -- continuous weights.
# 5. G3, G4, G6, G5.
#
# WATCH CONCENTRATION on tier 1. Products of centred ranks can put most of the
# book in a few names where both signals are extreme. numstk counts non-NaN
# positions and CANNOT see this -- it is the exact blindness that let four GP
# alphas report numstk ~400 while holding 99% of the book in one name. If
# booksize looks normal but returns are jumpy, suspect concentration.
#
# G8 IS THE PRIORITY IF YOU ONLY RUN ONE. Note 30 currently says "I have not
# run the control" as the answer to whether adaptive weighting does anything.
# One simulation replaces that with a number.
# =============================================================================
