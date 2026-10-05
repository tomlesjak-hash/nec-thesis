# Code change brief 08: market-neutral target, the Q26 feature set, the window-average gate

Date: 2026-10-05. Five parts, done in order: A (target), B (gate weight and likelihood notes), C (data
layer: CRSP re-extract, GICS, Compustat point-in-time), D (the feature set), E (docs, registry,
real-data scripts, final checks). D builds on C. Each lettered part ends in its own commit.

Repository `Quant Model/nec_baseline`, package `nec_moe`; the git root is `Quant Model/`. Same rules as
briefs 02 to 07:

- **Every numeric quantity is a config field** with a documented default. No hardcoded constant: every
  window, threshold and cap below is a field.
- Every claim gets a test that pins it.
- The brief contains only decisions Tom has made. Where something is still open it says so, and the
  code must not pick an answer.

The decisions, with their full reasoning, are in `Master Thesis/Advisor_Questions.md` under RESOLVED:

- Q21: horizon;
- Q24: likelihood;
- Q25: target;
- Q26: features;
- Q27: gate weight.

**Read those five entries before starting.** `Master Thesis/Progress_Tracker.md` has the summary
table.

---

## Licence rules (unchanged from briefs 06 and 07)

- CRSP, Compustat and anything derived from them stays inside `Quant Model/Data/`. That includes every
  per-security output. `Data/` is gitignored and is not in your checkout: **you cannot see the real
  data**. Build against the schemas in this brief and test on invented fixtures.
- `results/` holds aggregate outputs only.
- Test fixtures use invented numbers. Tests that need `Data/` skip without it.
- No out-of-sample performance number on the real panel is printed, reported or read.
- Never commit anything under `Data/`. Stage by explicit path. `Claude outputs/` stays untracked.

---

## 0. Read first

1. `Master Thesis/Advisor_Questions.md`: Q24, Q25, Q26 (all six parts), Q27, Q21.
2. `nec_moe/features.py` (whole file), `nec_moe/crsp.py` (`CRSPSpec`, `STOCK_COLUMNS`,
   `ADJ_FACTOR_COLUMNS`, `crsp_file_source`, `extract_crsp`, `crsp_daily_frames`, `build_crsp_panel`),
   `nec_moe/markov_gate.py` (`MarkovSwitchingRegimePrior`, `_filtered_log_prior`, `apply_causal`,
   `_row_stochastic_transition`), `nec_moe/priors.py` (`PrecomputedRegimePrior`, `HMMRegimePrior`),
   `nec_moe/losses.py` (`mixture_nll`), `nec_moe/config.py` (`DataConfig`, `MarkovGateConfig`,
   `target_horizon`), `nec_moe/registry.py`, `tests/crsp_fixture.py`, `tests/test_stage_b.py`,
   `tests/test_residual_target.py`.
3. `HANDBOOK.md` and `RUNBOOK.md`, for how runs are launched and recorded.

---

## A. The target: market-neutral return (Q25)

**A.1** Add `target_kind = "market_neutral"` to `StageBSpec` and make it the **default**. Keep `"raw"`
and `"residual"` as options; `"residual"` is not the decision, so leave it unchanged.

**A.2** Definition. For date t, horizon h (`StageBSpec.horizon`, 5), and the set of rows on date t with
a valid raw forward target:

```
y_mn[i,t] = fwd_ret_h[i,t] - mean_{j on date t with valid fwd_ret_h} fwd_ret_h[j,t]
```

- The mean is equal-weighted.
- `fwd_ret_h` is the existing raw target: the sum of the next h daily log returns, completed by the
  post-delisting fill as today.
- Demeaning is cross-sectional, so it must happen in `assemble_panel` after the per-stock frames are
  built and the date's rows are known. It cannot happen inside `stock_features`. The mean is taken over
  the date's universe rows with a valid target, before any row is dropped for a missing
  characteristic (part D no longer drops those).

**A.3** Name the column `fwd_mn_ret_{h}d`, so `config.target_horizon` still parses the horizon. Test
that `target_horizon("fwd_mn_ret_5d") == 5`.

**A.4** Reporting is market-neutral throughout (Q25 clarification, "no raw-return reporting").

- IC and calibration use the market-neutral target.
- For the long-short book, check whether it is dollar-neutral, meaning the weights sum to zero on
  every date. If so, add a test showing its return is unchanged when a common per-date constant is
  added to every stock's forward daily returns. If any reported return series is not invariant, put
  it on the cross-sectionally demeaned basis and say so in the commit message.

**A.5** Tests:

- On every date the target averages to zero, to float tolerance.
- Adding the same constant to every stock's returns on a date leaves the target unchanged.
- The no-lookahead test still passes.
- The trial registry records `target_kind`.

Commit: "Brief 08 A: market-neutral target (Q25)".

---

## B. The gate weight (Q27) and the likelihood notes (Q24)

**B.1** Add `MarkovGateConfig.gate_weight: Literal["filtered", "predicted", "window"] = "window"`.

Let xi_t be the filtered probability row at date t (canonically ordered) and A the row-stochastic
transition matrix in the same order (`_row_stochastic_transition`, permuted by the fit's canonical
permutation). The rows of the gate's log-prior table become:

- `"filtered"`: xi_t (today's behaviour).
- `"predicted"`: xi_t @ A.
- `"window"`: (1/h) * sum_{j=1..h} xi_t @ A^j, where h is the panel target's horizon
  (`DataConfig.horizon_periods`). Plumb h into the gate; do not add a second horizon field that can
  disagree. If the gate is built without a panel, raise rather than default.

Apply the same rule in `fit` and in `apply_causal`, so training and test dates get the same kind of
weight. Keep `prob_floor` and renormalisation after the transformation.

**B.2** Tests (pure numbers, no statsmodels fit needed). Use the example matrix
A = [[0.98, 0.02], [0.05, 0.95]]:

- With certain stress (xi = [0, 1]), `predicted` gives stress 0.95 and `window` with h = 5 gives
  stress 0.8632852 (to 1e-6).
- With certain calm (xi = [1, 0]), `window` with h = 5 gives stress 0.0547133 (to 1e-6).
- With h = 1, `window` equals `predicted`.
- With A = I, `window` equals `filtered`.
- Rows sum to 1.
- The trial registry records `gate_weight`.

**B.3** `HMMRegimePrior` (the backprop-HMM baseline) is **not** changed. Its latent is attached to
each date's target window, and its h-step predict from the last fully realised posterior is already
the consistent weight for that model. Add two sentences to its docstring saying so, with a pointer to
Q27.

**B.4** Gates still to come (jump model, Wasserstein, TVTP) will need a transition matrix to support
`gate_weight`. Write that requirement into the gate interface docstring
(`PrecomputedRegimePrior` or wherever the gate contract lives). Implement nothing for them.

**B.5** Q24: no change to the objective. The per-row mixture NLL stays the training objective. Add a
docstring paragraph to `mixture_nll` explaining the reading Tom adopted: each per-row term is the
exact one-stock marginal of the per-date model, so the mean over rows is that model's independence
(composite) likelihood. Point to Q24 and to Appendix D of `Master Thesis/MoE_HMM_Gate_Formulation.pdf`.

**B.6** Add the diagnostic Q24 asks for, aggregate-only, in `nec_moe/diagnostics.py`:
`expert_weight_diagnostics(responsibilities, gate_prob, abs_residual)`. Each argument covers the
training rows: the first two are (N, K), the third is (N,). For each expert k it returns:

- the correlation of responsibility_k with gate_prob_k;
- the correlation of responsibility_k with abs_residual;
- the mean responsibility_k within each quintile of gate_prob_k (quintile edges a config field).

Wire it into the end of training so the numbers are written to the run's aggregate metrics. Test on a
synthetic case where the responsibilities equal the gate probabilities: the first correlation is 1.

Commit: "Brief 08 B: window-average gate weight (Q27); composite-likelihood note and expert-weight
diagnostic (Q24)".

---

## C. Data layer

### C.1 CRSP daily: re-extract with prices, quotes and the adjustment fields

The current extract streams `StkDlySecurityPrimaryData`, which lacks open, high, low, bid and ask.
Switch the default `CRSPSpec.stock_file` to `StkDlySecurityData` (same release, `ciz202512`; about
42 GB uncompressed; it streams the same way). Its header, pipe-delimited, is:

```
PERMNO|YYYYMMDD|DlyCalDt|DlyDelFlg|DlyPrc|DlyPrcFlg|DlyCap|DlyCapFlg|DlyPrevPrc|DlyPrevPrcFlg|DlyPrevDt|
DlyPrevCap|DlyPrevCapFlg|DlyRet|DlyRetx|DlyRetI|DlyRetMissFlg|DlyRetDurFlg|DlyOrdDivAmt|DlyNonOrdDivAmt|
DlyFacPrc|DlyDistRetFlg|DlyVol|DlyClose|DlyLow|DlyHigh|DlyBid|DlyAsk|DlyOpen|DlyNumTrd|DlyMMCnt|DlyPrcVol
```

- Add `DlyOpen`, `DlyHigh`, `DlyLow`, `DlyClose`, `DlyBid` and `DlyAsk` to `STOCK_COLUMNS`. Tom
  measured that they are filled on 100% of S&P 500 member stock-days in 2015-2024.
- Do **not** use `DlyNumTrd`: it is filled on only 28% of those stock-days (Nasdaq only).
- `ADJ_FACTOR_COLUMNS` already carries `DlyShrOut`, `DlyCumFacPr` and `DlyCumFacShr`.
- Verify the units of `DlyCap`, `DlyShrOut` and `DlyVol` from `MetaItemInfo.dat` (same release). Put
  them in the module docstring; Compustat is in $ millions.

**History.** Some features need about five years of past data (beta: five-year correlation;
seasonality: years 2 to 5 back). Make the extract's lookback derive from the longest feature window
instead of the fixed `extract_lookback_days = 550`. Add a helper that converts trading days to
calendar days with a margin (a config field), and assert in `build_crsp_panel` that the extract
covers the warm-up. The training window stays 2015-2024 (`start`/`end` unchanged); the extra history
feeds features only.

**Sharing the existing extract.** The extract folder name encodes release and window. A changed stock
file or lookback must produce a **new** folder, never overwrite the old one. Make the stock file and
lookback part of the folder name or of `extract.json`, with a check that refuses a mismatch.

Extend `tests/crsp_fixture.py` with the new columns (invented numbers).

### C.2 Sectors: GICS from Compustat (Q26 part 4, changed from ICB on 2026-10-05)

CRSP's ICB field is "NOAVAIL" for every S&P 500 member from October 2023, so the sector source is
Compustat's dated GICS history. It lives in the CRSP/Compustat Merged release `cfz202607`. The
file-location convention is the same as CIZ: either `Data/crspdata/cfz202607_ascii/<file>.dat` or
the zip `Data/cfz202607_ascii.zip` with member `crspdata/cfz202607_ascii/<file>.dat`.

- Generalise `crsp_file_source` / `open_crsp_file` so a spec can name the release (`ciz202512`,
  `cfz202607`).
- Add a `CompustatSpec` (config: `crsp_dir`, `release = "cfz202607"`), separate from `CRSPSpec`.
- Note that Tom also unpacked copies into differently named folders (`Data/crspdata 3/...`). Do not
  rely on those. Use the standard locations above, with the release as a config field.

`gicshistory.dat` header:

```
KYGVKEY|KEYSET|INDFROM|INDTHRU|lpermno|lpermco|LinkRangeTypeCd|GGROUPH|GINDH|GSECTORH|GSUBINDH
```

- Dates are ISO `YYYY-MM-DD`. `INDTHRU = 9999-12-31` means current.
- `GSECTORH` is the 2-digit GICS sector: 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60.
- Link GVKEY to PERMNO through the CCM link history (C.3), valid on the date. Do not trust the file's
  own `lpermno` column alone; check against it in a test.
- On a date where a stock has no sector, all dummies are 0. Report the share of such rows by year,
  aggregate-only.

### C.3 The CCM link

`linkhistory.dat` (in `cfz202607`, also in `clz202607`) header:

```
KYGVKEY|LINKDT|LINKENDDT|LPERMNO|LPERMCO|LIID|LINKTYPE|LINKPRIM
```

- Dates are ISO; an open end date is a far-future date.
- Keep `LINKTYPE in {LC, LU}` and `LINKPRIM in {P, C}` (config fields).
- A link is valid on date d when `LINKDT <= d <= LINKENDDT`.
- Test that no PERMNO maps to two GVKEYs on one date under these filters; the test must report any
  ambiguity, not drop it silently.

### C.4 Compustat quarterly, point-in-time

Files (`cfz202607`), all pipe-delimited with a header. Numeric fields are empty when missing. `KEYSET`
selects the data format; use **KEYSET 1** (industrial, consolidated, standardised), per
`d_keysetinfo.dat`. Each file also carries `lpermno`/`lpermco` and `LinkRangeTypeCd`; ignore those in
favour of C.3.

- `perioddescriptorquarterly.dat`: `KYGVKEY, KEYSET, FYYYYQ` (fiscal year and quarter, e.g. `20151`),
  `fyrq`, ..., `DATACQTR` (calendar quarter, e.g. `2015Q1`), `DATAFQTR`, `FQTR`, `FYEARQ`,
  `RDQ` (report date as `YYYYMMDD`, or `0` when missing), `FDATEQ`, `PDATEQ` (ISO).
- `incomestatementquarterly.dat`: `SALEQ, REVTQ, COGSQ, XSGAQ, IBQ, NIQ, OIADPQ, EPSPXQ, DPQ, XINTQ,
  TXTQ, PIQ` and more. Items are keyed by `KYGVKEY, KEYSET, FYYYYQ`; `_DC` and `_FN` columns are
  data-code and footnote columns.
- `balancesheetquarterly.dat`: `ATQ, ACTQ, CHEQ, LCTQ, DLCQ, DLTTQ, LTQ, CEQQ, SEQQ, PSTKQ, TXDITCQ,
  IVAOQ, IVSTQ, MIBQ, INVTQ, RECTQ, PPENTQ, CSHOQ`.
- `cashflowyeartodate.dat`: year-to-date items `OANCFY, IVNCFY, FINCFY, CAPXY, SSTKY, PRSTKCY, DVY`.
  Convert them to quarterly flows by differencing within the fiscal year (Q1 equals its YTD value).
- `filingdates.dat`: `KYGVKEY, FDATADATE` (period end, ISO), `LPERMNO, LPERMCO, LinkRangeTypeCd,
  FCONSOL, FPOPSRC, SRCTYPE` (filing type, e.g. 10-Q/10-K), `FILEDATE` (ISO), `FILEDATETIME`.
- The fiscal period end date (`DATADATE`) for a `FYYYYQ` is in `fiscalmarketdataquarterly.dat`
  (`KYGVKEY, DATADATE, ..., KEYSET`) and in `d_fiscalperiod.dat`. Find the reliable mapping, document
  it, and test it on the fixture.

**Availability rule** (Q26 part 5, decided as the brief's proposal):

- A quarter's numbers become usable **one trading day after the later of its RDQ and its first 10-Q
  or 10-K `FILEDATE`** for that period end.
- If RDQ is missing, use the filing date. If both are missing, use the period end plus a lag (config,
  default 90 calendar days).
- The earnings and revenue surprises (D.2, Profit Growth) use **RDQ + 1 trading day**, falling back
  to the rule above when RDQ is missing.

After availability, a value is carried forward daily until the next quarter becomes available, for at
most `fundamental_max_staleness_days` (default 365) calendar days after its availability date.

**Extraction.** Stream the needed files, filtered to GVKEYs linked to any universe PERMNO, into a
columnar extract under `Data/derived/compustat_<release>_<window>/`, with an `extract.json` like the
CRSP one. Fundamentals must start early enough for 12-quarter growth (`debt_gr3`) and 8-quarter
volatility (`niq_su`) before 2015. Derive the start from the feature definitions, about 2011.

**Restatements.** Standard Compustat values can be restated after first release. The file has
keysets 3 ("PRES") and 8 ("PRE"): summary data collected before a company amendment. Do **not**
switch to them. Count, aggregate-only, how many universe firm-quarters have a KEYSET 8 row, and write
that count to the extract report so Tom can judge later.

Tests on an invented cfz fixture covering every file above:

- a value is invisible before its availability date and visible from it;
- carry-forward stops at the staleness cap;
- YTD-to-quarter differencing is correct across a fiscal-year boundary;
- a GVKEY with two links over time maps to the right PERMNO on each date;
- the GICS dummies switch on the `INDFROM` date.

Commit: "Brief 08 C: CRSP re-extract with OHLC and quotes, GICS sectors, CCM link, Compustat
point-in-time pipeline".

---

## D. The feature set (Q26)

Add a `FeatureSpec` (or extend `StageBSpec`) holding every window, threshold and cap below as fields,
plus `feature_set: Literal["q26", "legacy14"] = "q26"`. `legacy14` is the current 14-feature set,
kept only for comparison and existing tests; `q26` is the decision. Record `feature_set` in every
trial row.

**Conventions:**

- r = daily log return (`ret` in the daily frame).
- R = daily simple return.
- m = the market's daily log return (`market_indno`, as today).
- Every window is in trading days and is trailing up to the close of t.
- P = a split- and distribution-consistent **price index**, the cumulative product of
  (1 + `DlyRetx`). Never use raw price levels across days.
- ME = company market equity at t: the sum of `DlyCap` over all PERMNOs of the same PERMCO.
- Units must be reconciled with Compustat's $ millions (C.1).
- Value at t uses fundamentals available at t (C.4).

### D.1 The JKP block: 26 characteristics (13 themes × 2)

| # | Name | Theme | Definition (defaults are config fields) |
|---|---|---|---|
| 1 | `ret_5d` | Short-Term Reversal | sum of r over 5 days |
| 2 | `ret_20d` | Short-Term Reversal | sum of r over 20 days |
| 3 | `mom_12_1` | Momentum | sum of r from t-251 to t-21 (skip the last 21 days) |
| 4 | `prc_highprc_252d` | Momentum | P_t / max(P over 252 days) |
| 5 | `rvol_21d` | Low Risk | std of r over 21 days |
| 6 | `beta_bab` | Low Risk | Frazzini-Pedersen: corr(3-day overlapping summed r_i, same for m) over 1,260 days (min 750) × std(r_i, 252) / std(m, 252) |
| 7 | `log_me` | Size | log(ME) |
| 8 | `ami_126d` | Size | mean over 126 days of \|R\| / dollar volume ($ millions), days with zero volume excluded |
| 9 | `seas_2_5an` | Seasonality | mean of the stock's monthly simple returns in the calendar month of t+1, over the years 2-5 back (needs 5 years) |
| 10 | `coskew_21d` | Seasonality (JKP cluster) | x = r_i - mean, y = m - mean over 21 days: mean(x·y²) / (sqrt(mean(x²))·mean(y²)) |
| 11 | `be_me` | Value | book equity / ME |
| 12 | `ni_me` | Value | trailing-4-quarter IBQ / ME |
| 13 | `niq_be` | Profitability | latest IBQ / book equity of the previous quarter |
| 14 | `ocf_at` | Profitability | trailing-4-quarter operating cash flow / ATQ |
| 15 | `gp_at` | Quality | trailing-4-quarter (SALEQ - COGSQ) / ATQ |
| 16 | `ni_inc8q` | Quality | number of consecutive quarters, up to 8, with IBQ above the same quarter a year earlier |
| 17 | `at_gr1` | Investment | ATQ / ATQ four quarters earlier - 1 |
| 18 | `sale_gr1` | Investment | trailing-4-quarter SALEQ / the same a year earlier - 1 |
| 19 | `oaccruals_at` | Accruals | (trailing-4-quarter IBQ - trailing-4-quarter OANCF) / ATQ |
| 20 | `taccruals_at` | Accruals | Richardson et al. (2005): change over 4 quarters in (WC + NCO + FIN) / average ATQ. WC = (ACTQ - CHEQ) - (LCTQ - DLCQ); NCO = (ATQ - ACTQ - IVAOQ) - (LTQ - LCTQ - DLTTQ); FIN = (IVSTQ + IVAOQ) - (DLTTQ + DLCQ + PSTKQ) |
| 21 | `debt_gr3` | Debt Issuance | (DLTTQ + DLCQ) / same 12 quarters earlier - 1 |
| 22 | `noa_at` | Debt Issuance | Hirshleifer et al. (2004): [(ATQ - CHEQ) - (ATQ - DLCQ - DLTTQ - MIBQ - PSTKQ - CEQQ)] / ATQ four quarters earlier |
| 23 | `netdebt_me` | Low Leverage | (DLTTQ + DLCQ - CHEQ) / ME |
| 24 | `cash_at` | Low Leverage | CHEQ / ATQ |
| 25 | `niq_su` | Profit Growth | (IBQ_q - IBQ_{q-4}) / std of that difference over the last 8 quarters (min 6); available at RDQ + 1 |
| 26 | `saleq_su` | Profit Growth | the same with SALEQ |

**Book equity** = SEQQ + TXDITCQ (0 if missing) - PSTKQ (0 if missing). If SEQQ is missing, use
CEQQ + PSTKQ; if that is missing too, use ATQ - LTQ. Book equity ≤ 0 gives a missing `be_me`.

**Trailing-4-quarter flows** need four consecutive available quarters; otherwise missing.

**Financials.** Where a definition does not apply (banks have no COGSQ, current assets and so on),
the value is missing, not zero. The missing flag (D.4) handles it.

### D.2 The short-horizon market block: 12 characteristics (6 themes × 2)

| # | Name | Theme | Definition |
|---|---|---|---|
| 27 | `ivol_capm_21d` | Volatility dynamics | std of the residuals of OLS r_i on m (with intercept) over 21 days |
| 28 | `vol_shock` | Volatility dynamics | std(r, 5) / std(r, 60) |
| 29 | `rskew_21d` | Tails | sample skewness of r over 21 days |
| 30 | `rmax1_21d` | Tails | max of R over 21 days |
| 31 | `volume_z` | Volume and liquidity | the existing split-invariant `volume_z_20d` |
| 32 | `qspread_21d` | Volume and liquidity | mean over 21 days of (DlyAsk - DlyBid) / ((DlyAsk + DlyBid)/2), using only days with 0 < bid ≤ ask |
| 33 | `overnight_20d` | Price path | sum over 20 days of (r - log(DlyClose / DlyOpen)): the total log return minus the same-day intraday part, which is split- and dividend-safe |
| 34 | `ma50_gap` | Price path | P_t / mean(P over 50 days) - 1 |
| 35 | `earn_next5` | Earnings timing | 1 if an announcement is expected within the next 5 trading days, else 0. Expected date = the earliest (d + 364 calendar days), over the stock's past RDQs d ≤ t, that falls after t. Uses only RDQs known at t. Not ranked (0/1). |
| 36 | `days_since_earn` | Earnings timing | trading days since the last RDQ + 1 trading day (the event is usable from RDQ + 1, because announcements are often after the close) |
| 37 | `var_ratio_60d` | Return dynamics | var(5-day overlapping summed r over 60 days) / (5 · var(r over 60 days)) (Lo & MacKinlay 1988) |
| 38 | `ret_vol_corr_60d` | Return dynamics | corr over 60 days of r and log(split-adjusted volume), with the adjustment as in `_split_invariant_volume_z` |

### D.3 The industry block: 17 inputs

- **11 GICS sector dummies** (0/1, not ranked), from C.2.
- `ind_mom_12_1`: the stock's sector's `mom_12_1`, value-weighted by ME at t (Moskowitz & Grinblatt
  1999). Make the weighting a config field `industry_weighting: "value" | "equal"`, default
  `"value"`.
- `ret_20d_ind_rel`: `ret_20d` minus its sector's `ret_20d`, same weighting.
- **Within-sector ranks** of `be_me`, `ni_me`, `niq_be` and `ocf_at`: ranked inside the date's sector
  instead of across the universe. If the sector has fewer than `min_sector_names` (default 5) valid
  values that date, fall back to the universe rank. These four are extra inputs; the universe-ranked
  versions stay.

Sector aggregates use the date's universe rows only, and each stock's own value is included.

### D.4 Missing values, ranking, flags (Q26 parts 2 and 3)

1. **Rolling windows** require at least `min_obs_frac` (default 0.8) of the window's days, rounded
   up; otherwise the value is missing. This replaces the full-window requirement.
2. **Fundamentals** are carried forward per C.4, with the staleness cap.
3. **Ranking.** On each date, rank each characteristic over the date's universe rows that have a
   value. Ties get the average rank. Map linearly to [-1, 1]:
   ```
   x = 2 * (rank - 1) / (n - 1) - 1
   ```
   where n is the number of non-missing values that date. If n = 1, set the value to 0. This replaces
   the [-0.5, 0.5] mapping in `_cross_sectional_rank` for `q26`; `legacy14` keeps the old mapping.
4. **Fill.** After ranking, missing values are 0, the cross-sectional median.
5. **Flags** (0/1, not ranked):
   - `flag_price_missing` = 1 if any of characteristics 1-10 or 27-38 that come from CRSP was filled;
   - `flag_fund_missing` = 1 if any of 11-26 was filled.
6. **Rows** are no longer dropped for a missing characteristic. A row is still dropped if:
   - the target is missing;
   - the day is not tradable;
   - the date's cross-section is thinner than `min_names_per_date`;
   - in `snapshot_plus_hidden` input mode only, the sequence window is incomplete (keep the existing
     `window_ok` logic for the sequence channels).
7. **No-lookahead test for every new characteristic**: the value at t computed from data truncated at
   t equals the value from the full data. For fundamentals, truncate both the CRSP series and the
   Compustat availability dates at t.

**Total inputs: 26 + 12 + 2 + 4 + 11 + 2 = 57.** Assert this in a test for `feature_set="q26"`.
`ExpertConfig` / `DataConfig` must pick up the new input dimension from the panel schema, not from a
constant.

**Sequence channels** (`SEQUENCE_FEATURES`, for the encoder input mode) are unchanged.

**Tests on the fixtures:**

- every characteristic is hand-checked on a tiny invented example;
- ranks lie in [-1, 1] and average 0 on each date (with no ties);
- the 0-fill and flags behave as specified;
- within-sector ranks fall back correctly;
- `earn_next5` uses only past RDQs.

Commit: "Brief 08 D: the Q26 feature set (57 inputs), [-1, 1] ranks, causal missing-value handling,
GICS industry block".

---

## E. Docs, registry, real-data scripts, final checks

**E.1** Registry: every trial row records `target_kind`, `feature_set`, `gate_weight`, the CRSP stock
file, `compustat_release` and `sector_source = "gics"`.

**E.2** Update the module docstrings in `features.py` and `crsp.py`, plus `HANDBOOK.md` and
`RUNBOOK.md`, for:

- the new target;
- the feature list;
- the data layout;
- the extract commands.

Do not edit anything under `Master Thesis/`; Tom keeps those documents.

**E.3** Real-data scripts for Tom to run locally (you cannot run them). Put them in `scripts/`:

1. `extract_crsp_v2`: the C.1 re-extract.
2. `extract_compustat`: the C.4 extract.
3. `feature_coverage_report`: prints and writes to `results/feature_coverage/`, **aggregate-only**, the
   share of universe rows per year with each characteristic present before the 0-fill, the flag
   rates, the sector-missing share and the KEYSET 8 count.
   - It must not print any model result or return statistic.
   - It must refuse to run if anything it would write contains per-security values.

Real-data tests skip without `Data/`.

**E.4** Final checks from `nec_baseline/`: pytest (all), ruff, mypy, as in brief 07. Report:

- the counts;
- anything skipped and why;
- for each part, what you could not verify without the real data.

Commit: "Brief 08 E: registry fields, docs, real-data scripts". Push to `main` after E. Then report
what Tom must run locally, in order, with the exact commands.

---

## What this brief does not decide (do not pick answers)

- Q20 (error function), Q22 (gate filter through the purge gap), Q23 (gate series), Q18 (K), Q8/Q17
  (architecture widths and depth), Q6/Q14 (size of the corrections), Q16 (training window). Keep the
  existing defaults and seams.
- The sample start stays 2015-01-01; the extra history is for features only.
- The long-short book's `portfolio_scheme` default stays as it is.
