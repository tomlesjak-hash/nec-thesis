# =============================================================================
# GATE v3 — ten-term event stress, now including both earnings windows
# =============================================================================
# WHAT CHANGED FROM v2
#   + earn_pre   pre-announcement window  (Savor & Wilson announcement premium)
#   + earn_post  post-announcement window (Bernard & Thomas PEAD)
#   + fomc_pre widened 2 -> 10
#   + arithmetic float sum (numpy `+` on booleans is LOGICAL OR, so the `* 1.0`
#     on each term is required -- without it stress is a bool and the weight
#     collapses to {0, 2})
#
# WHY ADDING EARNINGS IS SAFE HERE BUT WAS NOT IN NOTE 16
# -------------------------------------------------------
# trading_days_until_next_earnings_announcement is CROSS-SECTIONAL -- it differs
# per stock. The other eight terms are TIME-SERIES, identical for every name on
# a given day.
#
# As a STANDALONE gate the earnings window removed ~80% of names from every
# day: numstk 156, disqualified (note 16, section 2).
#
# As one term inside the SUM it behaves completely differently. The calendar
# terms already make stress > 0 on nearly every day, so the earnings terms do
# not zero anything out -- they only RAISE the weight on names near their
# report. Full universe preserved, and the event-time information is still
# expressed.
#
# The distinction: a cross-sectional GATE costs coverage, a cross-sectional
# WEIGHT does not. Same variable, opposite consequence.
# =============================================================================

fomc_pre  = (days_until_next_fomc_meeting <= 10)
fomc_post = (days_since_last_fomc_meeting <= 12)
opex      = (days_until_next_monthly_options_expiration <= 10)
qopex     = (days_until_next_quarterly_options_expiration <= 1)
qend      = (days_until_last_trading_day_of_quarter <= 10)
mend      = (days_until_last_trading_day_of_month <= 13)
mstart    = (days_since_first_trading_day_of_month <= 0)
holiday   = (days_until_next_trading_holiday <= 5)

# --- the two earnings windows ------------------------------------------------
# PRE: du counts DOWN to the next report, so a small value means "about to
# report". <= 12 is the value note 14 tuned to by walking 90 -> 60 -> 18 -> 12
# with IR improving monotonically at every step. It includes DAY 0, which alone
# produced +0.025 -- the single largest jump in that entire exercise.
# Mechanism: elevated returns around scheduled announcements as compensation
# for bearing event risk (Savor & Wilson).
earn_pre  = (trading_days_until_next_earnings_announcement <= 12)

# POST: du RESETS to ~62 the day after a report, so a HIGH value means "just
# reported". This is the post-earnings-announcement-drift window -- prices
# under-react to earnings and drift in the direction of the surprise for weeks
# (Bernard & Thomas 1989).
# A distinct effect from the pre-announcement premium, and one your gate has
# never covered in any form.
earn_post = (trading_days_until_next_earnings_announcement >= 55)

stress = (fomc_pre * 1.0) + (fomc_post * 1.0) + (opex * 1.0) + (qopex * 1.0) \
       + (qend * 1.0) + (mend * 1.0) + (mstart * 1.0) + (holiday * 1.0) \
       + (earn_pre * 1.0) + (earn_post * 1.0)

w = (stress > 0) * (1.0 + stress)


# =============================================================================
# DROP-IN USAGE
# =============================================================================
#   alpha = at_zero2nan(cs_winsor(core, 0.01, remove_extreme=False) * w)
#
# Every core in the pool takes this unchanged.
#
# =============================================================================
# TUNING ORDER — one knob at a time, record IR / TVR / numstk each run
# =============================================================================
# 1. earn_pre    6, 12, 18, 26
#    The highest-value knob in the project's history: walking it was worth
#    +0.061 in note 14, against +0.006 from every internal window combined.
#    It has NEVER been tuned inside a graded weight.
#
# 2. earn_post   45, 50, 55, 58
#    Untested at any value. Lower = a wider post-report window. If 45 beats 55
#    the drift is long-lived; if only 58 works it is the announcement reaction
#    itself rather than drift, which is a different mechanism and worth knowing.
#
# 3. fomc_pre    2, 5, 10, 15
#    You widened this 2 -> 10 without testing the intermediate values.
#
# 4. RE-WALK earn_pre at the end. Coordinate descent gives no guarantee an
#    early choice survives later moves -- that is what hid the day-0 finding
#    for two days (note 14 section 4), and why A3's fomc_pre is still sitting
#    at a value chosen when total IR was 0.057.
#
# =============================================================================
# WATCH numstk
# =============================================================================
# The coverage argument above depends on the CALENDAR terms being active on
# nearly every day. That holds at the current loose windows (mend <= 13,
# opex <= 10, fomc_post <= 12) but would break if you tighten them -- at which
# point the earnings terms start excluding names rather than reweighting them,
# and you are back to the note-16 failure at numstk 156.
#
# If numstk falls below ~400, widen a calendar term rather than the earnings one.
#
# =============================================================================
# DIAGNOSTIC WORTH RUNNING ONCE
# =============================================================================
# Compare against the same core with the two earnings terms REMOVED. That
# isolates what earnings contributes inside a graded weight -- a number you do
# not currently have, since note 14 measured it as a standalone gate on a
# different core with a different structure entirely.
# =============================================================================
