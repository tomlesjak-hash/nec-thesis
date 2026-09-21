# =============================================================================
# FUNDAMENTAL SWEEP — the structural break
# =============================================================================
# WHY EVERYTHING CORRELATED
# -------------------------
# A1..A4 are all the same STRUCTURE with different signals plugged in:
#       cs_rank(fast technical signal) * event weight
# Identical horizon (1 day), identical day-weighting, identical transform.
# When the structure is fixed, the structure sets the return profile and the
# signal only perturbs it -- so new signals inside it keep colliding.
#
# This file changes the structure on three axes at once:
#   1. HORIZON   these fields refresh every 25-51 days, not daily. The ranking
#                barely moves day to day.
#   2. DATA TYPE fundamental, not technical. Nothing in A1-A4 touches it.
#   3. NO GATE   run these UNGATED first. If the shared event gate is what has
#                been correlating your pool, removing it is the direct fix, and
#                an ungated alpha has a completely different day-structure.
#
# EXPECTED SIDE EFFECT — very low turnover. A2 breaks even at 1.4bp and A4 at
# 2.1bp, which makes both paper alphas. A signal that only moves on filing
# dates should land at TVR 0.02-0.10, i.e. break-even in the tens of bp. These
# may be the first alphas in the pool that would survive real trading.
#
# TRAPS FROM THE DATA DICTIONARY -- ALL THREE ARE LIVE HERE
# ---------------------------------------------------------
# * UNITS ARE NOT CONSISTENT across raw fields (some +/-1e9, some +/-1e3).
#   NEVER add or subtract two raw fundamentals. Always ratio against a
#   same-statement denominator first, then cs_rank.
# * cs_rank is immune to the wild ranges (rank is monotone) -- so rank, do NOT
#   cs_zscore these. A single 1e10 outlier destroys a z-score.
# * COVERAGE. Every field below is >2000 of ~3250 names unless flagged.
#   Known decoys, do not use: tangible_equity (8 names),
#   working_capital_changes (24), price_to_book_ratio (54).
#   WATCH numstk ON EVERY RUN -- the 160 floor is the live risk here, not IR.
#
# EACH BLOCK: run ungated as written. If IR < 0.07, re-run with `* w` appended.
# If IR is negative, flip the sign. Value and quality both invert by regime.
# =============================================================================


# =============================================================================
# TIER 1 — CLASSIC FACTORS. Highest hit-rate, LOWEST originality.
# These are 40-year-old published factors. Fine for alpha COUNT, and they will
# decorrelate from your technical pool, but do NOT present them as original
# work -- an interviewer will name the paper. Submit them, don't headline them.
# The real risk is `max corr (others) < 0.90`: many competitors will submit
# something close to plain earnings yield.
# =============================================================================

# --- F1. Earnings yield -- value ---------------------------------------------
# Dictionary: "Bounded, high coverage. Best value factor in the set." (cov 2766)
# NOTE: this group is Delay 0 = No (price-dependent). Check it simulates.
score = cs_rank(earnings_yield) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F2. Return on assets -- quality/profitability ---------------------------
# cov 2578. Novy-Marx profitability territory.
score = cs_rank(return_on_assets) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F3. Net income growth -- growth -----------------------------------------
# cov 2659, the highest-coverage growth field.
score = cs_rank(net_income_growth_pct) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F4. Debt to market cap -- leverage, DAILY refresh -----------------------
# Update freq 1.02 -- price-driven, so it moves every day while the numerator
# is stale. That makes it a hybrid: fundamental numerator, technical timing.
# Expect turnover between the two families.
score = cs_rank(debt_to_market_cap) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — LESS-MINED FIELDS. Same construction, fields most competitors skip.
# Better originality story and better protection on max-corr(others).
# =============================================================================

# --- F5. Net income per employee -- labour productivity ----------------------
# cov 2534. Genuinely under-used: most fundamental screens never touch
# headcount. Productivity is economically distinct from margin -- two firms can
# share a margin with very different output per worker, and the efficient one
# has more operating leverage into a demand upturn.
score = cs_rank(net_income_per_employee) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F6. Cash to total assets -- balance-sheet liquidity ---------------------
# cov 2604. Not a standard risk factor. Interpretation cuts both ways, which is
# WHY it may be unmined: cash is optionality (can buy assets in a downturn) and
# also dead weight (management with nothing to reinvest in). If the sign is
# stable across the sample, one reading dominates -- worth knowing which.
score = cs_rank(cash_to_total_assets_ratio) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F7. Normalized ROE, faster refresh --------------------------------------
# cov 2095, update 25.3d -- twice the refresh rate of the 50.6d fields, so it
# sits at a horizon between F2 and your technical alphas.
score = cs_rank(normalized_roe) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — ORIGINAL CONSTRUCTIONS. Lower hit-rate, best interview material.
# These are not standard factors. Run them even if tier 1 fills your quota.
# =============================================================================

# --- F8. Valuation dispersion ------------------------------------------------
# The data dictionary flags this explicitly as "a free derived var" and I have
# not seen it as a named factor anywhere.
#
# high_PE and low_PE are the P/E computed at the period's high and low price.
# Their SPREAD measures how far the market repriced the name within the period
# while earnings were held fixed -- pure multiple volatility, with the earnings
# denominator differenced out. That is a disagreement / valuation-uncertainty
# measure, and it is built from a ratio of two published fields rather than
# from either field's level, so it is not a value factor in disguise.
#
# Scaled by the sum to make it dimensionless -- the raw difference inherits the
# +/-1.8e10 range and would be meaningless across names.
pe_hi = high_price_to_earnings_ratio
pe_lo = low_price_to_earnings_ratio
disp  = (pe_hi - pe_lo) / at_zero2nan(pe_hi + pe_lo)
score = cs_rank(disp) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F9. Reported profitability REVISION -------------------------------------
# Not the level of profitability -- the CHANGE in it between filings.
#
# The distinction from published revision factors matters: the analyst-revision
# literature uses changes in ESTIMATES, which is heavily mined because estimate
# data is what most shops buy. This uses the change in the REPORTED line item,
# which is a different quantity and far less traded.
#
# net_income / total_assets makes it dimensionless before differencing --
# differencing raw net_income across names would be meaningless given units
# vary by field. 90 days spans roughly two 50-day refresh cycles, so it
# captures the most recent filing's move.
roa_r = net_income / at_zero2nan(total_assets)
rev   = roa_r - ts_delay(roa_r, 90)
score = cs_rank(rev) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F10. Cash build rate ----------------------------------------------------
# Same revision logic on the balance sheet. Is the company ACCUMULATING or
# BURNING cash relative to its size? A level tells you how much cash exists;
# the change tells you what the business is doing right now. Free cash
# generation shows up here before it shows up in a margin.
csh  = cash_and_short_term_investments / at_zero2nan(total_assets)
bld  = csh - ts_delay(csh, 90)
score = cs_rank(bld) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F11. Cheap quality — composite ------------------------------------------
# Long high-ROA names that are NOT expensively priced. Both legs are ranked
# first so they share a scale before combining -- the units are unrelated and
# raw subtraction would be dominated by whichever has the wider range.
#
# CAUTION: this is close to published quality-value composites (QMJ, and the
# 'quality at a reasonable price' family). It is the least original entry in
# tier 3 -- included because two-legged composites often clear when neither leg
# does alone, and F1/F2 are already being run separately as the control.
q = cs_rank(return_on_assets) - 1.0
v = cs_rank(price_to_tangible_book_value_per_share) - 1.0
score = q - v
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- F12. Fundamental staleness ----------------------------------------------
# HIGHEST-RISK, MOST ORIGINAL IDEA HERE.
#
# These fields refresh on filing dates, so `x - ts_delay(x, 1)` is zero on
# every day EXCEPT the day new data landed. ts_sum of that indicator over 90
# days therefore counts how recently and how often this name has reported --
# a measure of information FRESHNESS rather than of any accounting quantity.
#
# Thesis: names whose fundamental data has gone stale are less covered, less
# updated in other people's models, and more likely to be mispriced. This
# trades the information environment, not the company.
#
# It may simply pick up sector filing conventions rather than anything about
# neglect -- if it works, check it survives industry neutralisation before
# believing the story.
# NOTE: booleans cannot be passed INTO a ts_ operator (float32 only). Build the
# indicator as a float first -- |delta|/|delta| is 1 on change days, 0 elsewhere.
d     = net_income - ts_delay(net_income, 1)
chg   = d / at_zero2nan(at_signsqrt(d) * at_signsqrt(d))
fresh = ts_sum(chg * chg, 90)
score = cs_rank(fresh) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# IF A SIGNAL CLEARS UNGATED — then and only then, try these two variants.
# =============================================================================
# 1. Add the event gate. Both versions can be submitted IF they clear the 50%
#    check against each other; gated and ungated versions of one signal trade
#    different day-weights, which is the note-14 / note-15 relationship again.
#
#       alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)
#
# 2. Industry-neutralise. Fundamental levels are dominated by sector -- banks
#    and software have structurally different ROA, so an un-neutralised quality
#    factor is substantially a sector bet. The neutralised version is a
#    genuinely different alpha, not a tweak, and it is usually the better one.
#
#       alpha = at_zero2nan(cs_winsor(cs_indneut(score, industry), 0.01,
#                                     remove_extreme=False))
#
#    (The sim's Neut=industry setting does the same thing -- do not apply both.)
# =============================================================================
