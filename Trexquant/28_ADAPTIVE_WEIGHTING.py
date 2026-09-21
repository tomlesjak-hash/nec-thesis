# =============================================================================
# ADAPTIVE / SELF-REFERENTIAL ALPHAS — the one transferable idea from the
# 2024-2026 automated alpha-mining literature
# =============================================================================
# WHAT THE LITERATURE SEARCH ACTUALLY FOUND
# -----------------------------------------
# The current generation of automated alpha miners -- AlphaForge (AAAI 2025),
# AlphaAgent (KDD 2025), QuantFactor REINFORCE, AlphaQCM, alpha-gfn (GFlowNet),
# AlphaPROBE, QuantaAlpha -- are all GENERATE-AND-TEST loops. Every one of them
# needs a fast, RELIABLE backtest to score candidates against.
#
# That is precisely what you do not have. Your local panel's rank correlation
# with platform IR was -0.70. Bolting a better search algorithm onto a fitness
# function that is ANTI-correlated with the target makes things worse faster.
# None of these frameworks are implementable in your remaining time anyway.
#
# But AlphaForge's SECOND stage transfers, and it does not need a backtest.
#
# ALPHAFORGE'S ACTUAL CONTRIBUTION
# --------------------------------
# Its gain came less from generating better factors than from DYNAMICALLY
# COMBINING them -- re-weighting each component at every time slice according
# to how it has performed recently, rather than fixing weights once. The paper
# is explicit that the 'Dynamic' variant consistently beats the 'Static' one.
#
# WHY THIS SOLVES YOUR ACTUAL PROBLEM
# -----------------------------------
# Your constraint is correlation, not signal. An adaptively-weighted composite
# is decorrelated from its own components BY CONSTRUCTION: it only carries a
# given component when that component has been working, so it tracks each one
# part-time. A static blend of A and B correlates highly with both. A blend
# whose weights move correlates with neither.
#
# This is also a genuinely different STRUCTURE from anything in your pool --
# every alpha you hold is a fixed function of price data. These are functions
# of their own recent forecasting performance. That is a self-referential
# object, and no plain reversal signal shares it.
#
# THE KEY OPERATOR
# ----------------
#   ts_corr_binary(ts_delay(sig, 1), ret1, 60)
#
# = per-stock rolling correlation between yesterday's signal value and today's
# realised return, over 60 days. That is a PER-STOCK ROLLING IC: "has this
# signal been predicting THIS name lately?"
#
# NO LOOK-AHEAD: the window is trailing and ret1 is Delay-0 = Yes, so today's
# return is known when the position is formed. But this is exactly the kind of
# construction where look-ahead hides, so SLOW MODE IS THE TEST. If slow-mode
# IR collapses versus fast, something leaks and the alpha is dead -- do not
# rationalise it.
# =============================================================================


# =============================================================================
# D1 — SELF-CONDITIONED SINGLE SIGNAL.  Simplest version, run this first.
# =============================================================================
# Take one core and overweight it in the names where it has been working.
# Shown with A2's core; swap in whichever you like.
#
# The economic claim is not that the signal is universally true -- it is that
# its validity VARIES BY STOCK and that variation is persistent. Closing-hour
# flow should be more informative in names with heavy index/ETF membership than
# in names without, and that membership is stable over months.

sk    = ts_mean(skew_return_last_hour, 1)
base  = cs_rank(sk) - 1.0

eff   = ts_corr_binary(ts_delay(base, 1), ret1, 60)     # per-stock rolling IC
score = base * (cs_rank(eff) - 1.0)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# D2 — TWO-SIGNAL ADAPTIVE BLEND.  AlphaForge's dynamic weighting.
# =============================================================================
# Two cores, each weighted by its own recent per-stock efficacy. Where signal A
# has been working the book leans on A; where B has, it leans on B.
#
# Both legs are cs_rank'd before blending so they share a scale -- their raw
# units are unrelated and whichever had the wider range would otherwise
# dominate regardless of efficacy.

sk = ts_mean(skew_return_last_hour, 1)
a  = cs_rank(sk) - 1.0                                   # A2's core

beta = ts_corr_binary(ret1, ret1_spx, 60)
res  = ret5 - (beta * ts_sum(ret1_spx, 5))
b    = cs_rank(-res / at_zero2nan(ts_std(ret1, 20))) - 1.0   # H4's core

ic_a = cs_rank(ts_corr_binary(ts_delay(a, 1), ret1, 60)) - 1.0
ic_b = cs_rank(ts_corr_binary(ts_delay(b, 1), ret1, 60)) - 1.0

score = (a * ic_a) + (b * ic_b)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# D3 — ADAPTIVE SWITCH.  Harder version: hold only the better signal.
# =============================================================================
# Rather than blending, hold whichever of the two has the higher recent IC for
# that stock. Because it holds each component only part-time on a partition
# that moves, its correlation to either component should be roughly halved
# relative to D2 -- the best decorrelation odds in this file.
#
# The (ic_a > ic_b) comparison stays in plain arithmetic and never enters a
# ts_ operator, so no float32 problem.

sk = ts_mean(skew_return_last_hour, 1)
a  = cs_rank(sk) - 1.0

beta = ts_corr_binary(ret1, ret1_spx, 60)
res  = ret5 - (beta * ts_sum(ret1_spx, 5))
b    = cs_rank(-res / at_zero2nan(ts_std(ret1, 20))) - 1.0

ic_a = ts_corr_binary(ts_delay(a, 1), ret1, 60)
ic_b = ts_corr_binary(ts_delay(b, 1), ret1, 60)

pick_a = (ic_a > ic_b) * 1.0
score  = (a * pick_a) + (b * (1.0 - pick_a))

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# D4 — SIGNAL DECAY / FRESHNESS.  From AlphaAgent's alpha-decay framing.
# =============================================================================
# AlphaAgent's contribution is regularising toward factors that DECAY SLOWLY.
# Inverted, that is a tradable signal: rank stocks by whether the reversal
# effect is currently STRENGTHENING or WEAKENING for them.
#
# Fast IC minus slow IC = is this signal getting better or worse at this name?
# Long the names where it is strengthening. Note this holds NO view on the
# signal's level -- purely on its second derivative, which is why it should not
# correlate with the signal itself.

sk = ts_mean(skew_return_last_hour, 1)
a  = cs_rank(sk) - 1.0

ic_fast = ts_corr_binary(ts_delay(a, 1), ret1, 20)
ic_slow = ts_corr_binary(ts_delay(a, 1), ret1, 120)

score = a * (cs_rank(ic_fast - ic_slow) - 1.0)

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# D5 — PURE PREDICTABILITY ALPHA.  Most original idea here.
# =============================================================================
# Drop the signal entirely and trade the META-VARIABLE: how PREDICTABLE has
# each stock been?
#
# Thesis: stocks whose short-horizon returns are currently well-explained by a
# simple reversal rule are stocks where uninformed liquidity demand dominates
# the tape. That is a statement about who is trading the name, not about where
# it is going -- and it should identify where liquidity provision is paid best,
# right now, without assuming any particular signal is correct.
#
# This holds no directional view derived from price at all. It is the furthest
# structurally from anything in your pool, and correspondingly the most likely
# to produce nothing. Worth one run for exactly that reason.

rev  = -ret1 / at_zero2nan(ts_std(ret1, 20))
pred = ts_corr_binary(ts_delay(rev, 1), ret1, 60)

score = cs_rank(pred) - 1.0

alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER AND WARNINGS
# =============================================================================
# 1. D1  -- simplest, and tells you whether per-stock efficacy conditioning
#           carries any information at all before you build on it.
# 2. D3  -- best decorrelation odds (part-time exposure to each component).
# 3. D5  -- most original, lowest prior.
# 4. D2, D4.
#
# CHECK SLOW MODE ON ANY WINNER, WITHOUT EXCEPTION. Every construction here
# uses realised returns inside the signal. That is legitimate -- the window is
# trailing -- but it is the textbook place for look-ahead to hide, and fast
# mode will not reveal it. A2 gained slightly in slow mode (0.097 -> 0.102);
# anything here that LOSES materially should be discarded rather than explained.
#
# WATCH numstk: ts_corr_binary over 60 days needs 60 days of history per stock,
# so names with short histories drop out. Combined with the 250-day warm-ups
# you have seen elsewhere, expect a later start date and check the count.
#
# TUNE the IC window (60) last: 20, 60, 120. Short windows chase noise, long
# windows defeat the purpose of adapting.
# =============================================================================
