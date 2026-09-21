# =============================================================================
# INTRADAY PERIODICITY & EXECUTION FOOTPRINT
# =============================================================================
# THE FINDING THAT MATTERS
# ------------------------
# Heston, Korajczyk & Sadka, "Intraday Patterns in the Cross-Section of Stock
# Returns" (Journal of Finance 65, 2010, pp.1369-1407):
#
#   "a striking pattern of return CONTINUATION at half-hour intervals that are
#    EXACT MULTIPLES OF A TRADING DAY, and this effect lasts for at least
#    40 trading days."
#
# In plain terms: the return a stock earns during a given period of the day
# predicts the return it earns during THAT SAME PERIOD on subsequent days, and
# the effect persists for ~40 days. They further report that volume, order
# imbalance, volatility and spreads show the same periodicity but DO NOT
# EXPLAIN the return pattern -- so it is not a liquidity artifact.
#
# WHY THIS IS THE BEST REMAINING IDEA FOR YOUR POOL
# -------------------------------------------------
#   * You have exactly the right data. return_first_hour and return_last_hour,
#     daily, Delay-0 = Yes. The periodicity is at a one-day lag by construction.
#   * It is a CONTINUATION effect. Your five alphas are reversal/microstructure;
#     this runs the other way, which is what breaks correlation.
#   * It is a 40-DAY horizon on an INTRADAY variable -- a combination nothing in
#     your pool occupies. A2 uses the same data at a 1-day horizon and no
#     smoothing; this is the opposite corner of that space.
#   * The mechanism is institutional order-splitting: a large order worked over
#     days gets executed in the same time window each day, because that is how
#     execution algorithms are scheduled.
#
# SUPPORTING RESULT: Gao, Han, Li & Zhou, "Market Intraday Momentum" (JFE 2018)
# find first-half-hour return predicts last-half-hour return, and that the
# effect is STRONGER on high-volume days, high-volatility days, and macro news
# release days. That is your event gate, arrived at independently -- worth
# citing in the theory note whichever way these runs go.
# =============================================================================


# =============================================================================
# TIER 1 — RETURN PERIODICITY.  Direct implementation of HKS.
# =============================================================================

# --- Q1. Closing-hour return periodicity -------------------------------------
# The stock's average last-hour return over 40 days. HKS says this predicts the
# next last-hour return. Long the names that consistently rise into the close.
#
# NOTE the sign: POSITIVE. This is continuation, not reversal -- deliberately
# opposite to your pool. cs_rank is immune to the field's extreme tail
# (documented range runs to 366.8, almost certainly a data artifact).
sig   = ts_mean(return_last_hour, 40)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- Q2. Opening-hour return periodicity -------------------------------------
# Same effect, opposite end of the session. The open is where retail and
# overnight-news flow concentrates, so if both Q1 and Q2 work they are picking
# up different clienteles and should decorrelate from each other.
sig   = ts_mean(return_first_hour, 40)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- Q3. Periodicity spread --------------------------------------------------
# Long names that consistently gain into the close AND fade at the open.
# Differencing removes whatever is common to both windows -- market drift,
# beta, general momentum -- leaving only the time-of-day tilt, which is the
# quantity HKS actually identify.
sig   = ts_mean(return_last_hour - return_first_hour, 40)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- Q4. Short-window version ------------------------------------------------
# HKS say the effect lasts AT LEAST 40 days; they do not say 40 is optimal.
# 10 days is a much fresher read on which names currently have an order being
# worked. Higher turnover, likely higher IR if the order-splitting story is right.
sig   = ts_mean(return_last_hour, 10)
score = cs_rank(sig) - 1.0
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- Q5. Volume periodicity --------------------------------------------------
# HKS report volume shows the same daily periodicity. Persistent elevated
# closing-hour volume share is the direct footprint of a multi-day order being
# worked to the close -- a cleaner read on "unfinished institutional demand"
# than A2's skew, which infers it from one day's return shape.
vshare = volume_last_hour / at_zero2nan(volume_first_hour + volume_last_hour)
sig    = ts_mean(vshare, 40)
score  = cs_rank(sig) - 1.0
alpha  = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 2 — EXECUTION FOOTPRINT.  Most original idea in the file.
# =============================================================================
# The iceberg / hidden-order literature (Zotikov, "CME Iceberg Order Detection
# and Prediction", and the order-flow practitioner literature) detects large
# concealed orders by their EXECUTION SIGNATURE: an algorithm working a big
# order produces small, evenly-sized, regularly-spaced fills, while natural
# flow is lumpy and irregular.
#
# That entire literature is order-book and futures based. Nobody applies it as
# a daily CROSS-SECTIONAL EQUITY signal, because almost nobody has per-stock
# trade counts in a daily panel. You do -- number_of_trades_first_hour and
# number_of_trades_last_hour. The dictionary flags them as "NOT derivable from
# daily data", which is also your best protection on max-corr(others).
#
# A directly related 2025 result (arXiv 2512.15720, "Hidden Order in Trades
# Predicts the Size of Price Moves") finds that low ENTROPY in trade sequences
# -- i.e. unusual regularity -- predicts larger subsequent absolute returns,
# conditioning on entropy below the 5th percentile raising 5-minute absolute
# returns by 2.89x. Regularity is the tell; this builds the daily analogue.

# --- X1. Execution regularity ------------------------------------------------
# Dispersion of volume within the closing hour, scaled by that hour's total.
# LOW dispersion = evenly-paced prints = algorithmic execution in progress.
#
# Sign is written for "regular execution predicts continuation" -- an order
# being worked is an order not yet finished. If IR is negative, the reading is
# that regularity marks completed passive supply instead; flip and note which.
reg   = std_volume_last_hour / at_zero2nan(volume_last_hour)
score = -(cs_rank(reg) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- X2. Persistent execution regularity -------------------------------------
# A single regular hour is noise; regularity sustained over 10 days is a large
# order being worked across sessions. This is X1 crossed with the HKS
# persistence result, and it is the sharpest statement of the mechanism.
reg   = std_volume_last_hour / at_zero2nan(volume_last_hour)
sig   = ts_mean(reg, 10)
score = -(cs_rank(sig) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))

# --- X3. Trade-size regularity -----------------------------------------------
# Same idea via a different channel: average print size that is stable day to
# day indicates a scheduled algorithm; variable print size indicates
# discretionary flow. ts_std over the ratio, scaled by its own mean so it is
# dimensionless.
ats   = volume_last_hour / at_zero2nan(number_of_trades_last_hour)
stab  = ts_std(ats, 20) / at_zero2nan(ts_mean(ats, 20))
score = -(cs_rank(stab) - 1.0)
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# TIER 3 — NON-MONOTONE TRADE SIZE.  Nothing in your pool has this shape.
# =============================================================================
# Lesmond, "Trade Size and the Cross-Section of Stock Returns" (EFMA 2024):
# the pricing information sits in MEDIUM-sized trades (roughly 500-9,999
# shares), not at either extreme. Very large trades are blocks and negotiated;
# very small trades are retail noise. Informed traders hide in the middle.
#
# That implies a NON-MONOTONE signal -- and every alpha you hold is monotone in
# its input, so this is a structurally different object regardless of whether
# the underlying effect replicates.
#
# THE TENT TRANSFORM: at_signsqrt(x)*at_signsqrt(x) = |x|, so
# 1 - 2*|r - 0.5| peaks at the median rank and falls to 0 at both tails.
ats = volume_last_hour / at_zero2nan(number_of_trades_last_hour)
r   = cs_rank(ats) - 1.0                       # [0, 1]
d   = r - 0.5
mid = 1.0 - 2.0 * (at_signsqrt(d) * at_signsqrt(d))    # tent, peak at median

score = mid - 0.5
alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))


# =============================================================================
# RUN ORDER
# =============================================================================
# 1. Q1, Q3  -- direct HKS implementation, continuation effect, best
#               decorrelation prospects. Q3 is the cleaner isolation.
# 2. X1, X2  -- most original, and the best story if they work. X2 is the one
#               that states the mechanism properly.
# 3. Q4, Q5, Q2, X3, and the tent.
#
# IF SOMETHING CLEARS: add the event gate. Gao-Han-Li-Zhou found intraday
# momentum is stronger on high-volume, high-volatility and macro-news days,
# which is an independent derivation of your gate -- so unlike the slow signals
# in file 23, the gate mechanism genuinely applies to these.
#
# CORRELATION NOTE: Q1/Q3 use the same data group as A2 but at a 40-day horizon
# with the opposite sign. Different moment (level vs skew), different horizon,
# opposite direction -- the odds are good, but check rather than assume. The
# 40-day smoothing is doing the decorrelation work, so if correlation comes
# back high, LENGTHEN the window rather than abandoning the idea.
# =============================================================================
