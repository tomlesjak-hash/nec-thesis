# Universe coverage — should you pad to 500, or buy data?

**Short answer: neither. Fix the two real bugs, accept ~430 names/day, and spend
the saved day on Track A.** Reasoning below.

---

## 1. What the 211 failures actually are

796 unique S&P 500 members over 2006–2022; **588 downloaded, 211 failed (26.5%)**.
Every one is a name Yahoo has purged because it no longer trades. Not a bug, not
a rate limit — the 26.5% *is* your survivorship bias, measured.

**And look at which names are missing:**

| Ticker | What happened |
|---|---|
| `LEH` | Lehman Brothers — bankrupt Sept 2008 |
| `CFC` | Countrywide Financial — fire-sold to BofA 2008 |
| `FNM` / `FRE` | Fannie Mae / Freddie Mac — conservatorship 2008 |
| `ABK` | Ambac — monoline insurer, collapsed 2008 |
| `EK` | Eastman Kodak — bankrupt 2012 |
| `SBNY` | Signature Bank — failed March 2023 |
| `FRC` | First Republic — failed May 2023 |
| `SIVB` | Silicon Valley Bank — failed March 2023 |

**The gap is not random.** It is concentrated in the exact stocks that went to
zero. For a window that includes 2008, that is the worst possible bias: your
data has the survivors of the financial crisis and none of the casualties.

Any reversal or value alpha will look better than it was, because the "buy the
losers" leg is missing its worst outcomes.

---

## 2. Why padding to 500 makes it WORSE

You asked: if a name is delisted, replace it with the next stock that qualified.

The trap is that **these names were not delisted at the time.** Lehman was a live
S&P 500 member on 2008-06-30 with a $20bn market cap. It's missing from your
panel only because Yahoo purges tickers *today*.

So "replace it with the next qualifying stock" means substituting a company that
survived to 2026 for one that didn't. That's not filling a hole — it's
**deliberately selecting on survival**, which is precisely the bias you're
trying to limit.

| Approach | Names/day | Bias |
|---|---|---|
| Membership ∩ available data | ~430 | Under-represents failures ⚠️ |
| Pad to 500 from survivors | 500 | Under-represents failures **and** adds non-members ⚠️⚠️ |
| Paid survivorship-free source | ~500 | Correct ✅ |

The flag exists if you want it — `--no-membership` — but it buys a cosmetic
number at the cost of a worse sample.

**~430 vs 500 is statistically irrelevant for GP search.** Cross-sectional
ranks over 430 names are as stable as over 500; you lose about 8% of an
already-large cross-section. The number of names is not what's limiting you.

---

## 3. Should you buy a subscription?

**No — not with 6 days left.** The reasoning is allocation, not cost:

1. **The local panel serves Track B, which is your *secondary* track.** Track A
   (hand-crafted alphas in Trexsim) is half the score, needs no local data at
   all, and you haven't started it.
2. **Integration costs about a day** — new API, new auth, new loader, re-run,
   re-validate. That's ~17% of your remaining time, spent on the optional half.
3. **Trexsim is a free unbiased validator.** Local data is for *search*; the
   platform is for *truth*. A biased search space produces some candidates that
   die on the platform — irritating, not fatal, and the platform check is cheap.
4. **Bias affects levels more than rankings.** If survivorship inflates every
   reversal alpha similarly, the *ordering* of your GP candidates is roughly
   preserved — and ordering is all you need from a search stage.

**When buying would be right:** if this were your thesis, or if you had 3+ weeks,
Sharadar SEP (active + delisted back to the 1990s, explicitly survivorship-free)
is the correct answer. Revisit after the competition — it's the right foundation
for the thesis work, just not for this week.

---

## 4. What to do instead — mitigations that cost minutes

- **Weight training toward 2011+.** Post-2011 attrition is far lower than the
  2006–2010 block, which is missing the entire crisis-casualty cohort.
- **Be suspicious of alphas that only work in 2006–2010.** That window has both
  the sparse-membership problem *and* the missing-failures problem.
- **Validate on Trexsim before trusting any GP result.** Free and unbiased.
- **Keep the failure list.** 211 named tickers is a documented, quantified
  residual bias — far more defensible than an unstated one.

---

## 5. Bugs fixed in this pass

| Reported | Verdict |
|---|---|
| `no in-universe name lacks a price` — 159 orphan cells | **Real bug, fixed.** Membership is picked at the month anchor and held, so a name that stopped trading mid-month stayed flagged with no price behind it. `top_n_mask` now takes `price=` and masks those out. Verified 36 → 0 on a synthetic delisting. |
| `ratio -> 1.0 at series end` — median 1.0565 | **Not a bug — my check was wrong.** Yahoo anchors Adj Close so the factor is 1.0 at *today*, not at your window end. Your panel ends 2022-12-31, so the factor still carries every dividend paid 2023–2026. 1.0565 over ~3.6 years ≈ 1.5%/yr yield — exactly right. Check now scales its tolerance by the gap. |
| `35 bars \|ret1\| > 90%` | **Expected.** 585 names × 17 years including 2008 and 2020. Buyouts, crisis moves, biotech readouts. Stays a WARN. |
| `open` min $0.0064, `high` max $110,900 | **Worth eyeballing.** Deep-split names legitimately produce tiny adjusted prices, but sub-penny values wreck GP terminals. New WARN checks flag bars below $0.05 and above $50,000 with ticker counts. Consider excluding those names from GP terminals. |
