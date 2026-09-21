# =============================================================================
# REGIME-MATCHED CONDITIONING — gate B and its complement, matched to mechanism
# =============================================================================
# THE IDEA
# --------
# Gate B fired on macro stress and every pairing with an EXISTING core
# correlated -- because the cores were the same. Pairing it with the slow and
# relational cores in file 23 changes both halves at once.
#
# But the important point is not "try gate B on more things." It is that
# DIFFERENT ALPHA FAMILIES WANT OPPOSITE REGIMES, and the reason is mechanical:
#
#   LIQUIDITY-PROVISION alphas (reversal, low-beta, hedging-demand) are paid
#   MORE in stress. Providers withdraw, funding tightens, the fee for supplying
#   immediacy rises. Nagel (2012); Frazzini & Pedersen (2014).
#       -> GATE B (active in stress)
#
#   MOMENTUM alphas are DESTROYED in stress. Daniel & Moskowitz, "Momentum
#   Crashes": after a market decline the momentum portfolio's beta turns
#   sharply negative (it is short the beaten-up high-beta names), so when the
#   market rebounds it is run over. Their central result is that these crashes
#   are FORECASTABLE from a bear-market indicator plus volatility -- which is
#   precisely what `regime` below measures.
#       -> GATE C (active in CALM, i.e. gate B inverted)
#
# So one regime variable yields two conditioners pointing opposite ways, each
# with an independent published mechanism. They trade near-disjoint day sets,
# which is the note-14 / note-15 relationship that produced two qualifying
# alphas from one thesis.
#
# HONEST WARNING -- READ BEFORE TRUSTING ANY RESULT HERE
# ------------------------------------------------------
# Gate B's day-set is CLUSTERED, not spread: 2008, 2011, 2015-16, 2018, 2020.
# A gate-B alpha is effectively fitted to about FIVE episodes, however many
# individual days that amounts to. The effective sample size is the number of
# distinct stress regimes, NOT the number of days -- so a strong gate-B IR is
# much weaker evidence than the same IR from gate A, whose days are spread
# evenly across fifteen years.
#
# Gate C does not have this problem (calm is most of the sample), which is a
# reason to trust the momentum pairings more than the stress pairings.
#
# Check numstk on every run. These are time-series gates so the full universe
# survives on active days, but gate B cuts the ACTIVE-DAY fraction hard, and
# numstk averages over the trailing 120 days.
# =============================================================================

# ---- REGIME (from file 21; ret-based only -- close_spx is stale past 2017) ---
vix    = at_zero2nan(close_vix)
mktvol = ts_std(ret1_spx, 20)

r1 = (vix > ts_mean(vix, 60))
r2 = (ts_zscore(vix, 20) > 1.0)
r3 = (mktvol > ts_mean(mktvol, 120))
r4 = (ts_sum(ret1_spx, 20) < 0)
r5 = (ts_sum(ret1_vix, 5) > 0)

regime = r1 + r2 + r3 + r4 + r5

wm = (regime > 0) * (1.0 + regime)      # GATE B  — stress-weighted
wc = (regime == 0) * 1.0                # GATE C  — calm only


# =============================================================================
# GROUP 1 — GATE C x MOMENTUM.  Daniel-Moskowitz crash avoidance.
# Run these FIRST: strongest mechanism, and gate C's day-set is broad enough
# that the result is not hostage to five episodes.
# =============================================================================

# --- P1. 12-1 momentum, crash-protected --------------------------------------
# The canonical version of the Daniel-Moskowitz trade. If momentum works at all
# on this universe, switching it off in panic states should raise IR AND cut
# drawdown -- and the drawdown improvement is the real tell. If IR rises but
# drawdown does not fall, the gate is not doing what the theory says.
mom   = ts_delay(close, 20) / at_zero2nan(ts_delay(close, 250))
score = cs_rank(mom) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wc)

# --- P2. 52-week high proximity, crash-protected -----------------------------
# George & Hwang's anchoring signal carries the same crash exposure as plain
# momentum -- it is long winners and short losers by another route.
prox  = close / at_zero2nan(ts_max(high, 250))
score = cs_rank(prox) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wc)

# --- P3. Information discreteness, crash-protected ---------------------------
sgn   = ret1 / at_zero2nan(at_signsqrt(ret1) * at_signsqrt(ret1))   # = sign(ret1), float
disc  = ts_mean(sgn, 250)
score = cs_rank(disc) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wc)

# --- P4. Idiosyncratic share, crash-protected --------------------------------
# In stress, cross-stock correlations converge toward 1 and idiosyncratic share
# collapses for everyone -- the signal loses its cross-sectional spread exactly
# when it is least reliable. Gating it to calm regimes is the same argument as
# for momentum, arrived at from dispersion rather than from beta.
b2    = ts_corr_binary(ret1, ret1_spx, 120)
idio  = 1.0 - (b2 * b2)
score = cs_rank(idio) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wc)


# =============================================================================
# GROUP 2 — GATE B x FUNDING-CONSTRAINT ALPHAS.  Nagel / Frazzini-Pedersen.
# =============================================================================

# --- P5. Low beta, stress-weighted -------------------------------------------
# Frazzini & Pedersen's mechanism is FUNDING CONSTRAINTS: investors who cannot
# borrow buy high-beta stocks instead of levering low-beta ones, bidding high
# beta up. When funding tightens, those positions unwind and BAB pays most.
# The paper conditions on the TED spread; VIX is the available proxy here.
# This is the single best-motivated pairing in the file.
beta  = ts_corr_binary(ret1, ret1_spx, 250)
score = -(cs_rank(beta) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wm)

# --- P6. Beta instability, stress-weighted -----------------------------------
# Unstable beta is unhedgeable exposure, and unhedgeable exposure is only
# expensive when hedging matters -- i.e. in stress. In calm markets nobody is
# paying for hedge reliability, so the premium should be near zero there.
# The mechanism predicts this pairing works and the ungated version does not,
# which makes it a sharper test than P5.
bser  = ts_corr_binary(ret1, ret1_spx, 60)
inst  = ts_std(bser, 120)
score = cs_rank(inst) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wm)

# --- P7. Gold beta, stress-weighted ------------------------------------------
# Safe-haven demand is a stress phenomenon by definition -- gold's hedging
# value is not being priced when nothing is going wrong. If gold beta matters
# anywhere it is here, and if the ungated R2 failed this is the version that
# should have worked.
gbeta = ts_corr_binary(ret1, ret1_gld, 120)
score = cs_rank(gbeta) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wm)


# =============================================================================
# GROUP 3 — THE DIAGNOSTIC. Run this even if it is not submittable.
# =============================================================================

# --- P8. Momentum with the WRONG gate ----------------------------------------
# Momentum under gate B -- stress-weighted, the opposite of what theory says.
# Daniel-Moskowitz predicts this should be BAD, and materially worse than P1.
#
# If P8 beats P1, the crash mechanism is not what is operating and every
# regime-matching argument in this file is post-hoc storytelling. That is worth
# knowing before any of it goes in an interview.
#
# This is the falsification test I have repeatedly written into the theory
# notes and never run. It costs one simulation.
mom   = ts_delay(close, 20) / at_zero2nan(ts_delay(close, 250))
score = cs_rank(mom) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * wm)


# =============================================================================
# IF A PAIRING CLEARS
# =============================================================================
# 1. Submit the GATED version, then test the UNGATED one separately. Gated and
#    ungated trade different day-weights and may both qualify -- but check the
#    50% correlation between them first, they share a core.
#
# 2. A gate-C alpha and a gate-B alpha are near-complementary by construction
#    (calm vs stress), so a P1-family and a P5-family alpha should be about as
#    decorrelated as anything you can build. That pair is the best shot at
#    adding TWO alphas rather than one.
#
# 3. Tune `regime` only AFTER something clears, and re-walk the first knob at
#    the end -- fomc_pre on A3 is still sitting at a value chosen when total IR
#    was 0.057 (note 18, section 5, finding 3). Ranges:
#       r1  ts_mean(vix, 60)     -> 20, 40, 60, 120
#       r2  zscore threshold     -> 0.5, 1.0, 1.5
#       r3  ts_mean(mktvol, 120) -> 60, 120, 250
#       r4  ts_sum(ret1_spx, 20) -> 5, 10, 20, 60
#       r5  ts_sum(ret1_vix, 5)  -> 3, 5, 10
#
#    For gate C also try loosening the threshold: `(regime <= 1)` keeps more
#    days than `(regime == 0)` and may hold IR while improving numstk.
# =============================================================================
