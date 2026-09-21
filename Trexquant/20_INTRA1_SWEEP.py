# =============================================================================
# INTRA1 SWEEP — 17 candidates, event-weighted
# =============================================================================
# You have used 1 of 30 intra1 fields (skew_return_last_hour -> A2, IR 0.102).
# This sweeps the rest using the template that has now cleared three times.
#
# RULES LEARNED SO FAR — apply to every candidate below:
#   1. NO SMOOTHING. A2 went 0.061 (window 5) -> 0.088 (window 1). The signal
#      has a one-day half-life; averaging 5 days averages 5 unrelated events.
#   2. IF IR IS NEGATIVE, FLIP THE SIGN: score = -(cs_rank(x) - 1.0).
#      Three of three intraday signs were backwards from my reasoning.
#   3. EVERY SIGNAL MUST BE DIMENSIONLESS. Ranking a raw price or raw volume
#      gives a static size/liquidity ranking that never changes -- this is the
#      clone problem that killed the first GP run.
#   4. Run Compute Correlation vs A1/A2/A3 before submitting. With 3 alphas
#      sharing one gate, the 50% self-correlation check is now the binding
#      constraint, not IR. Only A3 (IR/sqrt(TVR) 0.145) can invoke the override.
#
# PASTE THE GATE BLOCK, THEN ONE CORE.
# =============================================================================

# ---- GATE (identical to submitted A3) ---------------------------------------
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

# =============================================================================
# TIER 1 — new families, lowest correlation risk against A2. START HERE.
# =============================================================================

# --- 1. Closing-hour information ratio ---------------------------------------
# Intra-hour return / intra-hour vol. Dictionary: "bounded, clean, almost
# certainly under-mined by competitors." Risk-adjusted version of the closing
# move -- a big move on low intra-hour vol means one-sided flow with no
# opposing interest. Different statistic from A2's skew, same economic family.
sig   = information_ratio_last_hour
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 2. Average trade size surprise, closing hour ----------------------------
# Dictionary: "NOT derivable from daily data" -> best protection against the
# max-corr(others) < 0.90 check, since no daily-bar competitor can build it.
# The ts_mean(.,20) division is load-bearing: raw trade size just ranks stocks
# by how big their typical trades are, which is a static size proxy.
ats   = volume_last_hour / at_zero2nan(number_of_trades_last_hour)
sig   = ats / at_zero2nan(ts_mean(ats, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 3. Volume concentration at the close ------------------------------------
# What share of bookend volume printed in the final hour. Naturally bounded
# [0,1] so it is dimensionless with no normalisation. High = the day's activity
# was back-loaded = index/ETF/MOC flow dominated.
sig   = volume_last_hour / at_zero2nan(volume_first_hour + volume_last_hour)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 4. Closing-hour VWAP displacement ---------------------------------------
# This is the note-15 alpha, which was designed but never actually submitted.
# How far the close printed from the hour's volume-weighted average -- direct
# measure of one-sided auction pressure. Normalised by own dispersion.
push  = (close - vwap_last_hour) / at_zero2nan(vwap_last_hour)
sig   = push / at_zero2nan(ts_std(push, 20))
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 5. Opening-hour information ratio ---------------------------------------
# Same statistic, opposite end of the day. Doubles as falsification test #1
# from note 17: if the opening hour works AS WELL as the closing hour, the
# closing-auction mechanism is NOT what drives A2 and the story needs rewriting.
sig   = information_ratio_first_hour
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# =============================================================================
# TIER 2 — intraday shape from the OHLC bars. Untouched territory.
# =============================================================================

# --- 6. Closing-hour range, normalised ---------------------------------------
# How wide the final hour traded relative to price. Intraday range is a
# liquidity-demand proxy: wide range on an ordinary day = thin book.
sig   = (high_price_last_hour - low_price_last_hour) / at_zero2nan(close_price_last_hour)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 7. Close position within the closing-hour range -------------------------
# A stochastic oscillator computed INSIDE the final hour. Where the hour
# settled relative to its own high/low. Bounded [0,1] by construction.
# The daily-bar version of this is the core of A1 -- this is its intraday twin.
rng   = at_zero2nan(high_price_last_hour - low_price_last_hour)
sig   = (close_price_last_hour - low_price_last_hour) / rng
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 8. Range asymmetry, close vs open hour ----------------------------------
# Did the day widen or calm into the close? Ratio of two ranges = dimensionless
# without any normalisation.
r_last  = (high_price_last_hour - low_price_last_hour) / at_zero2nan(close_price_last_hour)
r_first = (high_price_first_hour - low_price_first_hour) / at_zero2nan(open_price_first_hour)
sig     = r_last / at_zero2nan(r_first)
score   = cs_rank(sig) - 1.0
alpha   = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 9. Closing-hour drift ---------------------------------------------------
# Return WITHIN the final hour, from its open to its close. Distinct from
# return_last_hour if that field is measured against the prior close.
sig   = (close_price_last_hour - open_price_last_hour) / at_zero2nan(open_price_last_hour)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 10. Opening-hour reaction ----------------------------------------------
# How the first hour absorbed the overnight gap. Classic overnight-reversal
# territory, but measured on the hour rather than the day.
sig   = (close_price_first_hour - open_price_first_hour) / at_zero2nan(open_price_first_hour)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 11. Pure overnight gap, intraday-bar version ----------------------------
# Open of the first hour vs close of the PREVIOUS last hour. Isolates the
# non-trading period cleanly -- daily open/close mixes in auction effects.
prev  = at_zero2nan(ts_delay(close_price_last_hour, 1))
sig   = (open_price_first_hour - prev) / prev
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# =============================================================================
# TIER 3 — remaining shape statistics. HIGHER correlation risk vs A2:
# these are siblings of skew_return_last_hour. Check correlation carefully.
# =============================================================================

# --- 12. Closing-hour volume skew -------------------------------------------
# Shape of VOLUME rather than returns. A few large prints in an otherwise quiet
# hour = block execution. Bounded ~+/-7.6, ranks directly.
sig   = skew_volume_last_hour
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 13. Closing-hour price skew --------------------------------------------
sig   = skew_close_last_hour
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 14. Skew asymmetry ------------------------------------------------------
# Differences out whatever is common to both hours, isolating what is specific
# to the CLOSE. Closer to A2's stated mechanism than A2 itself is.
sig   = skew_return_last_hour - skew_return_first_hour
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 15. Information ratio asymmetry -----------------------------------------
sig   = information_ratio_last_hour - information_ratio_first_hour
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 16. Closing-hour vol, own-history normalised ----------------------------
# std_dev_return_last_hour is a LEVEL -- ranking it raw just ranks volatile
# stocks, which is static. ts_zscore makes it "unusual FOR THIS STOCK".
sig   = ts_zscore(std_dev_return_last_hour, 20)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)

# --- 17. Trade-count asymmetry -----------------------------------------------
# Participation shift between the open and close auctions -- how many separate
# decisions were made, independent of size.
sig   = number_of_trades_last_hour / at_zero2nan(number_of_trades_first_hour)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False) * w)
