# =============================================================================
# HIGH-SHARPE FAMILIES ONLY — the constraint that explains every recent failure
# =============================================================================
# THE ARITHMETIC I SHOULD HAVE DONE FIRST
# ---------------------------------------
#   platform gate       IR 0.070 daily  =  annual Sharpe 1.11
#   your A1 / A2 / A3   IR 0.091/0.102/0.073  =  Sharpe 1.44 / 1.62 / 1.16
#
# Published gross long/short Sharpes:
#   Value HML 0.35 | Size 0.25 | Momentum 0.50 | Quality 0.60 | BAB 0.80
#   52-week high 0.55 | Accruals 0.45
#         ^ EVERY ONE OF THESE IS BELOW 1.11. They are mathematically incapable
#           of clearing the gate. Files 22 and 23 sent you after them; that was
#           my error, not a run of bad luck.
#
#   Short-term reversal 1.3 | RESIDUAL reversal ~2.0 | intraday micro 1.5-3
#         ^ these clear. Your three working alphas are all in this family.
#
# CONSEQUENCE: the entire slow-anomaly literature is off the table for this
# competition. Everything below is fast (1-10 day), and every entry has a
# documented Sharpe above the bar. The problem is no longer "find an effect" --
# it is "find a HIGH-SHARPE effect that decorrelates from three reversal
# alphas", which is a much narrower and better-defined search.
# =============================================================================


# =============================================================================
# TIER 1 — RESIDUAL REVERSAL.  Highest documented Sharpe in the literature.
# =============================================================================
# Blitz, Huij, Lansdorp & Verbeek, "Short-term residual reversal" (JFM 2013).
#
# THE INSIGHT: plain reversal is CONTAMINATED. When you short last week's
# winners you are also, unintentionally, shorting whatever factor happened to
# rally last week -- and factor moves do NOT revert, they persist. So a chunk
# of plain reversal's volatility is factor noise that carries no expected
# return, and removing it roughly DOUBLES the Sharpe (their result: ~1.0 -> ~2.0).
#
# WHY IT SHOULD ALSO DECORRELATE FROM A1/A3: the residual and the raw signal
# differ by exactly the factor component. A1 and A3 trade raw reversal, which
# means they are partly trading that factor component. This is not a tweak of
# the same alpha -- it is the orthogonal piece. Still CHECK the correlation;
# the shared reverting-residual core may dominate.

# --- H1. Industry-residual reversal, vol-scaled ------------------------------
# cs_indneut subtracts the industry mean each day, leaving the stock-specific
# move. Dividing by the stock's own vol standardises how UNUSUAL that residual
# is -- without it you rank by volatility rather than by surprise, which is the
# same normalisation that lifted A2.
res   = cs_indneut(ret5, industry)
sig   = -res / at_zero2nan(ts_std(ret1, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H2. Same, one-day horizon -----------------------------------------------
# Fastest version. Highest Sharpe in the Blitz tests, highest turnover too.
res   = cs_indneut(ret1, industry)
sig   = -res / at_zero2nan(ts_std(ret1, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H3. Same, ten-day horizon -----------------------------------------------
res   = cs_indneut(ret10, industry)
sig   = -res / at_zero2nan(ts_std(ret1, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H4. Beta-residual reversal ----------------------------------------------
# Removes MARKET exposure rather than industry exposure -- a different residual,
# so a different alpha. beta*market_return is this stock's expected move given
# the market; subtracting it leaves the part the market does not explain.
beta  = ts_corr_binary(ret1, ret1_spx, 60)
res   = ret5 - (beta * ts_sum(ret1_spx, 5))
sig   = -res / at_zero2nan(ts_std(ret1, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H5. Double-residual reversal --------------------------------------------
# Strip BOTH market and industry. The most orthogonal version to raw reversal,
# and therefore the best correlation odds against A1/A3 -- at the cost of
# stripping so much variance that little signal may survive. Worth one run.
beta  = ts_corr_binary(ret1, ret1_spx, 60)
mres  = ret5 - (beta * ts_sum(ret1_spx, 5))
res   = cs_indneut(mres, industry)
sig   = -res / at_zero2nan(ts_std(ret1, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — OVERNIGHT / INTRADAY DECOMPOSITION.  Different MECHANISM entirely.
# =============================================================================
# Lou, Polk & Skouras, "A Tug of War: Overnight Versus Intraday Expected
# Returns" (JFE 2019).
#
# THE FINDING: split the daily return into the overnight move (prior close ->
# open) and the intraday move (open -> close). These two components have
# OPPOSITE and PERSISTENT cross-sectional predictability. A stock with strong
# overnight returns keeps earning overnight; the intraday component works the
# other way.
#
# THE MECHANISM IS CLIENTELE, NOT REVERSAL -- which is exactly why it is worth
# your runs. Retail and news-driven flow concentrates at the open; institutional
# execution happens through the session. The two groups hold persistently
# different exposures, and that persistence is a different economic object from
# anything A1/A2/A3 trade. Best decorrelation prospect in this file.

on = (open - ts_delay(close, 1)) / at_zero2nan(ts_delay(close, 1))    # overnight
it = (close - open) / at_zero2nan(open)                              # intraday

# --- H6. Overnight persistence -----------------------------------------------
# LPS's headline result: the overnight component PERSISTS rather than reverting.
# Note the sign -- this is momentum-flavoured, opposite to everything you hold.
sig   = ts_mean(on, 20)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H7. The tug of war ------------------------------------------------------
# The spread between the two components -- LPS's actual construction. Long
# stocks whose returns arrive overnight, short those whose returns arrive
# intraday. Differencing removes whatever is common to both, isolating the
# clientele tilt.
sig   = ts_mean(on - it, 20)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H8. Overnight gap reversal, vol-scaled ----------------------------------
# The one-day version, and the opposite sign to H6: a single large gap is
# largely liquidity-driven and reverts, while the AVERAGE overnight tilt over
# 20 days is a clientele fact that persists. If H6 and H8 both clear, that
# distinction is real and it is a genuinely interesting thing to be able to
# show -- same variable, opposite sign, different horizon.
sig   = -on / at_zero2nan(ts_std(on, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — INDUSTRY LEAD-LAG.  Relational, fast, and not a reversal effect.
# =============================================================================
# Hou, "Industry Information Diffusion and the Lead-Lag Effect" (RFS 2007).
# Information diffuses within an industry gradually, so an industry's move
# predicts the returns of its slower-moving members.
#
# CONSTRUCTION NOTE: cs_indneut(x, industry) subtracts the industry mean, so
# `x - cs_indneut(x, industry)` recovers the industry mean itself, broadcast
# to every member. That is the industry return -- a variable you do not
# otherwise have access to.

# --- H9. Industry momentum ---------------------------------------------------
# Ranks stocks by how their INDUSTRY has done, not how they have done. Constant
# within industry, so this is an industry-rotation book wearing a stock-level
# alpha's clothes. numstk stays full; the effective bet count is the number of
# industries, so expect low effective N -- check it does not concentrate.
ind   = ret5 - cs_indneut(ret5, industry)
score = cs_rank(ind) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- H10. Laggard catch-up ---------------------------------------------------
# Long stocks that have UNDER-performed an industry that is moving. Hou's
# diffusion story says the laggards follow. The multiplication is the point:
# a laggard in a flat industry has nothing to catch up to, so the signal should
# only fire where both conditions hold.
ind   = ret5 - cs_indneut(ret5, industry)
lag   = cs_indneut(ret5, industry)
sig   = ind * (-lag)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. H7, H6  -- different MECHANISM (clientele, not reversal) = best
#               decorrelation odds against A1/A2/A3. Run these first even
#               though tier 1 has the higher Sharpe, because correlation is
#               your binding constraint, not IR.
# 2. H1, H2  -- highest documented Sharpe. If either clears, tune the horizon
#               (ret1/ret5/ret10) and the vol window (10/20/60).
# 3. H5      -- best correlation odds within tier 1.
# 4. H9, H10, H8, H3, H4.
#
# IF SOMETHING CLEARS: try the event gate on it. Every gated alpha you have has
# a fast core, and all of these are fast -- so unlike the file-23 signals, the
# gate mechanism actually applies.
#
# IF NOTHING HERE CLEARS EITHER: stop hunting alpha #4. The remaining lever is
# combined OS performance, which is the other half of the score -- and A3's
# fomc_pre is still sitting at a value chosen when total IR was 0.057, never
# re-walked at 0.073 (note 18, section 5, finding 3).
# =============================================================================
