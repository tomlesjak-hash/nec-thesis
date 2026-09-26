# Code change brief 06: CRSP replaces the free data, three small decisions, commit everything

Date: 2026-09-26. Do the steps in order: step 0, then A to G. Each lettered section ends in its own
commit (section G).

Repository `Quant Model/nec_baseline`, package `nec_moe`; the git root is `Quant Model/`. Same rules
as briefs 02 to 05: **every numeric quantity is a config field with a documented default and no
hardcoded constant**, and every claim gets a test that pins it. This brief contains only decisions
Tom has made. Where something is still open it says so, and the code must not pick an answer.

---

## Licence rules (read before touching the data)

CRSP and Compustat are licensed to the university, and redistribution is prohibited.

- `Quant Model/Data/` is gitignored. Everything derived from it goes **inside** `Data/`: extracts,
  parquet files, built panels. Never write it to `data_cache/`, `results/`, `figs/` or `Master Thesis/`.
- **Test fixtures are invented.** Write small CIZ-format files with the real headers and made-up
  numbers. Never copy real rows into a fixture, a docstring, a test or a report.
- Committed reports may contain **aggregate statistics only**: counts, date ranges, means, rates,
  convergence results. No raw rows, and no per-security values.
- Tests that need the real files are marked and **skip** when `Data/` is absent, so CI still passes.

---

## 0. Read first

1. `Code_Audit_2026-09-25.md`, sections 0, 1 (data), 2b (X-1), 2d (G-5), 4 (E-1) and 10 (the critical
   fixes already in the working tree).
2. The CRSP metadata files inside `Data/ciz202512_ascii.zip`: `MetaColumnInfo.dat` (what every
   column means), `MetaFlagInfo.dat` and `MetaFlagType.dat` (flag codes such as `DlyPrcFlg` and
   `DlyRetMissFlg`), and `MetaSIZtoCIZ.dat` (how the legacy names `RET`, `PRC`, `DLRET` map to CIZ
   names). Everything in section A that depends on a column's meaning must be checked against these
   files, not assumed.
3. Shumway, T. (1997), "The delisting bias in CRSP data", *Journal of Finance* 52(1), and Shumway, T.
   and Warther, V. A. (1999), "The delisting bias in CRSP's Nasdaq data and its implications for the
   size effect", *Journal of Finance* 54(6). Why delisting returns must not be dropped.
4. `nec_moe/features.py` (the anti-leakage timing rule and the feature list), `nec_moe/universe.py`,
   `nec_moe/market_data.py`, `scripts/build_pit_panel.py`, and `run_experiment.py`'s settings block.
5. Q21 and Q22 in `Advisor_Questions.md`. Both are open and both bear on this brief; neither may be
   answered in code.

---

## Step 0. Commit the work already in the tree

The five critical fixes (audit section 10) and several thesis documents are uncommitted. Before
changing anything:

1. Run the full suite, ruff and mypy. Record the counts; audit section 10 says 272 passed, 3 skipped,
   3 xfailed.
2. Commit the code, in one commit: every modified file under `nec_baseline/` plus the new
   `tests/test_overlap_inference.py` and `Master Thesis/Code_Audit_2026-09-25.md`. Message: "Audit
   critical fixes: D-1, D-2, E-1, E-2, M-1".
3. Commit the documents, in one commit: `.gitignore`, and in `Master Thesis/`: `Advisor_Questions.md`,
   `Progress_Tracker.md`, `Model_Derivation_BaseCase.md`, `Code_Change_Brief_04_Objective_Starts_Smoke.md`,
   `Code_Audit_Brief_05.md` and this brief. Message: "Thesis docs: derivation, briefs 04 to 06, Q21,
   Q22, progress tracker".

Section G's staging rules apply here too.

---

## A. The CRSP data layer

### A.1 What it replaces

| Now (free) | After |
|---|---|
| yfinance / Stooq daily bars, keyed by ticker | CRSP daily stock file, keyed by **PERMNO** (CRSP's permanent security identifier) |
| SPY as the market series | A CRSP index series, chosen by `INDNO` |
| Wikipedia S&P 500 change history | CRSP index membership (`StkIndMembership.dat`) |
| No delisting returns | CRSP delisting returns |

**Unchanged:** the feature list, the timing rule, the target definition (`target_kind`, horizon), the
`Panel` contract, and everything downstream of the panel.

### A.2 Source files, all in `Data/ciz202512_ascii.zip`, pipe-delimited with a header row

- `StkDlySecurityPrimaryData.dat` (14 GB unzipped): `PERMNO|DlyCalDt|DlyDelFlg|DlyPrc|DlyPrcFlg|DlyCap|DlyCapFlg|DlyRet|DlyRetx|DlyRetMissFlg|DlyDistRetFlg|DlyVol`.
  Use this rather than the 42 GB `StkDlySecurityData.dat`, unless a needed column is only in the
  larger file. If so, say which column and why.
- `StkDelists.dat`: delisting date, `DelRet`, `DelDlyDt`.
- `IndDlySeriesData.dat`: `INDNO|YYYYMMDD|DlyCalDt|DlyTotRet|...`; `IndSeriesInfoHdr.dat` names each `INDNO`.
- `StkIndMembership.dat`: `PERMNO|INDNO|MbrStartDt|MbrEndDt|MbrFlg|INDFAM`.
- `StkSecurityInfoHist.dat`: ticker, name, share type and exchange history per PERMNO. Used only for
  readable labels in reports.

Stream from the zip. Never unzip the whole archive; it is 66 GB.

### A.3 Config

Add a `CRSPSpec` (or extend `StageBSpec`) with, at minimum:

- `crsp_dir`: path to `Data/`, default relative to the repo.
- `stock_file`: default `"StkDlySecurityPrimaryData"`.
- `market_indno`: the market series. Default `1000500`, "CRSP Value-Weighted Index of the S&P 500
  Universe", total return (`DlyTotRet`). This is the like-for-like replacement of SPY for an S&P 500
  universe. It is a replacement, not a new modelling choice. Keep it configurable, and list
  `1000200` (the NYSE/NYSEMKT/Nasdaq/Arca value-weighted market) in the docstring as the alternative
  for a wider universe.
- `universe`: `"sp500"`, the only value implemented. The membership `INDNO` is a config field.
  Determine from the metadata which `INDNO` carries S&P 500 constituent membership; `1000502` "S&P
  500 Composite" in family `1100502` "S&P 500 Universe" is the likely one. Verify it on the real data
  (next point) and record the evidence in the docstring.
- `start`, `end`: default `2015-01-01` and `2024-12-31`, unchanged, so the CRSP panel is comparable
  with the old one. The window is part of Q16, which is open.

### A.4 Construction rules, each a correctness requirement with its own test

1. **Returns.** Daily log return `log(1 + DlyRet)`. `DlyRet` includes dividends, so it replaces the
   adjusted close. A missing return (check `DlyRetMissFlg` in the metadata) is **missing**, never
   zero. Rows whose features or target need it are invalid, through the existing `_valid_rows`.
2. **Delisting returns.** Establish from `MetaColumnInfo.dat` whether CIZ `DlyRet` already
   incorporates the delisting return. Then confirm it on the real data: for a sample of delisted
   PERMNOs, compare `DlyRet` on `DelDlyDt` with `DelRet`. Implement the correct rule:
   - if `DlyRet` already includes it, do nothing;
   - otherwise, compound `DelRet` into the return on the delisting date.

   Either way the target of a stock that delists inside its forward window includes the delisting
   return. A delisting is never silently dropped. Report what was found and the evidence.
3. **Price.** Take the absolute value of `DlyPrc` if negative values occur (the legacy bid/ask-average
   convention), and check `DlyPrcFlg` against `MetaFlagInfo.dat`.
4. **Price levels used as features** (drawdown against the 60-day high) come from a total-return
   index built by compounding `DlyRet` **within the trailing window only**. That makes each value
   depend on information through `t` alone, which is the same property the D-2 fix established.
5. **Dollar volume** at `t` is `|DlyPrc_t| x DlyVol_t`: raw price times raw shares on the same day,
   which is the dollars actually traded. No adjustment is needed.
6. **Share volume across a split.** Features that compare share volume across days (the 20-day volume
   z-score, the volume sequence channel) must not jump at a split. Adjust share volume within the
   trailing window, using only split factors dated inside the window (`DlyFacPrc`, or ratios of
   `StkDlyCumulativeAdjFactor`, whichever the metadata supports). Never use a factor anchored to the
   end of the sample. Test: a synthetic series with a 2-for-1 split gives the same z-score as the
   same series without the split.
7. **Market series.** The market daily log return is `log(1 + DlyTotRet)` of `market_indno`. Market
   features, rolling beta and the residual target use it where they used SPY.
8. **Universe.** A PERMNO is in the panel on date `t` only if a membership spell covers `t` (the
   `MbrStartDt`/`MbrEndDt` bounds are inclusive; confirm this from the metadata). Features may use a
   stock's history before it joined. The point-in-time filter and its re-rank (the D-1 fix) are
   applied exactly as now. Real-data test: the member count per date lies in a narrow band around
   500 across the whole window; report the minimum and maximum.
9. **Entity labels** become PERMNO. Build a date-aware PERMNO-to-ticker lookup from
   `StkSecurityInfoHist.dat` for readable report labels only; nothing keys on tickers.

### A.5 Extraction and caching

- `scripts/extract_crsp.py`: stream the stock file once, keep rows for the PERMNOs that were ever
  members in the window, from `start` minus the longest feature lookback to `end` plus the maximum
  horizon, and write a columnar extract to `Data/derived/`. Make it resumable. If a dependency such
  as `pyarrow` is needed, add it as an optional extra `crsp` in `pyproject.toml`.
- `scripts/build_pit_panel.py` builds from the extract, not from the network, and writes the panel to
  `Data/derived/pit_panel_crsp_<start>_<end>.pt`. Its coverage table (members with no usable return
  data per year) goes to `Data/derived/` too. Its aggregate numbers may be quoted in the section F
  report.
- Record the source release (`ciz202512`) inside the panel metadata.

### A.6 Provenance

Add `data_source` to every `TrialRegistry` entry and every sweep row: `"crsp_ciz202512"` for this
panel, `"synthetic"` for synthetic panels. This closes WORK_QUEUE item 7's requirement that no result
can be mistaken for another source's.

### A.7 Tests (fixture-based unless marked)

Delisting return reaches the target; a missing return is never zero; the split-invariant volume
z-score; no look-ahead (the existing `test_no_lookahead` pattern on a CRSP fixture); membership
inclusive bounds; PERMNO labels; `data_source` in registry rows; the market series comes from the
configured `INDNO`. Real data, skipped without `Data/`: the member-count band, the delisting check
from A.4.2, and panel build succeeding for a one-year window.

---

## B. Retire the free data

Tom's decision: CRSP replaces the free price and universe data everywhere in the pipeline.

- Remove the yfinance and Stooq loaders and the Wikipedia universe from the pipeline, the build
  script and `run_experiment.py`, together with their tests and the `yahoo` and `universe` extras.
  Git history keeps them.
- If anything else depends on them, stop and report rather than keep a hidden fallback.
- **Keep** `context_data.py` (VIX and the French factors). These are diagnostics only, never a
  training signal, and have no CRSP equivalent. Say so in its docstring.
- Do not rebuild the old free-data panel. `data_cache/pit_panel_2015_2024.pt` is obsolete; leave it
  untracked and unused.

---

## C. E-1 follow-up: report the volatility-only gain

Tom's decision: report the variance part of the old "NLL improvement" as its own number. Three
negative log likelihoods per evaluation:

- `NLL_single`: the base scored as a **single Gaussian**, exactly the pre-fix definition in
  `base_and_correction` at commit `9a29368` (the prior-weighted sigma);
- `NLL_base`: the base under the mixture density with every correction at zero (the current fix);
- `NLL_full`: the full model.

Report:

$$\text{nll\_variance\_gain} = \text{NLL}_{single} - \text{NLL}_{base}$$

$$\text{nll\_improvement} = \text{NLL}_{base} - \text{NLL}_{full} \quad \text{(unchanged)}$$

$$\text{nll\_total\_gain} = \text{NLL}_{single} - \text{NLL}_{full}$$

The docstring says the variance gain comes from the experts' per-regime noise scales weighted by the
gate, not from the corrections. It must not describe either gain as the better measure. Add all
three to harness output, sweep rows and the smoke script's columns.

Tests:
- `total == variance_gain + improvement` to float tolerance;
- `variance_gain == 0` when all sigma_k are equal;
- `total == variance_gain` when every correction is zero.

Do not assert a sign for the variance gain; it has none in general.

---

## D. X-1: a switch for how the experts start

Tom's decision: make it a config switch. Which setting is used is **not** decided.

- `ExpertConfig.hidden_init: Literal["diversified", "identical"]`, default `"diversified"`. The
  default is today's behaviour, kept so existing results reproduce; the docstring says the default is
  not a decision. `"identical"` gives every expert the same hidden-layer weights and the same initial
  `log_sigma`. Heads stay zero in both.
- Record `hidden_init` in every `TrialRegistry` entry.
- **Same seed, same start, in every arm.** For a given seed, expert initial weights must be
  bit-identical across gate arms (uniform, Hamilton, and so on), so that arms differ only in the gate.
  Check whether this already holds; audit finding B-1 (RNG order) may break it. Do **not** fix B-1
  here. If it breaks this property, report it.

Tests:
1. `identical`: all experts' parameters equal at step 0.
2. `diversified`: bit-identical to the current initialisation for the same seed (regression).
3. `identical` with a uniform prior and `dropout = 0`: the experts are still equal after N training
   steps. Symmetry is preserved because equal experts receive equal gradients.
4. `identical` with a fixed non-uniform prior that varies across dates: the experts separate.
5. Initial expert weights are bit-identical across two gate arms with the same seed, for both
   settings.

**Report, do not decide:** whether dropout masks are drawn independently per expert. If they are,
then under `identical` with `dropout > 0` the experts diverge even under a uniform gate, so the clean
"no regime information" control needs dropout off. State which it is.

---

## E. G-5: accepted

Tom accepts both start-value deviations from brief 04: the expanding volatility quantile (a) and the
1-nat clustering gap (b). No code change. In `markov_gate.py`, reword any docstring that presents
them as open deviations so it states them as the specification, "accepted 2026-09-26". Mark G-5 as
accepted in the audit's findings table.

---

## F. Integration check on the CRSP panel

Build the CRSP panel for the default window and run the existing smoke script with the same settings
as brief 04 C. Change only the data source. The purpose is to confirm the pipeline runs end to end on
CRSP. Write `Master Thesis/Smoke_Run_<date>_CRSP.md` containing **only**:

- **Data:** rows, dates, members per date (min, median, max), coverage by year, number of delistings
  inside the window, and the delisting finding from A.4.2.
- **Gate, per fold:** convergence, the start diagnostics from brief 04 B.3, stationary distribution,
  expected durations, and the filtered probability of the high-variance regime against VIX (Spearman
  correlation and the plot).
- **Experts:** that training completed, and the live parameter count.
- **Wall clock per fold**, split into gate, base and experts.

**Do not report any out-of-sample performance number**: no IC, ICIR, NLL or any gain from section C,
pooled or per fold. The pre-registration is not written yet (WORK_QUEUE item 8), and every look at
test-block performance on the real sample counts against it (Q15). Compute them if the script must,
but keep them out of the report and out of the chat.

Stop and report, rather than work around, if:
- the gate fails to converge on any fold;
- the fitted regimes are degenerate (a stationary probability below 0.02, or an expected duration
  below two days);
- the member count per date leaves the band in A.4.8.

Leave `Smoke_Run_2026-09-25.md` untouched; add one line at its top pointing to the new report and
saying it predates the critical fixes and the CRSP switch.

---

## G. Documentation and commits

**Documentation**
- `README.md` and `HANDBOOK.md`: the Stage B sections describe the CRSP source, the licence rules
  above, and `Data/derived/`. Remove the free-data instructions.
- `WORK_QUEUE.md`: item 3 is done (the free-data run, superseded by F). Item 7, the CRSP seam, is done
  by A. Update the "Last updated" line.
- `Progress_Tracker.md`: append one dated entry to the log for this brief, and update the Data,
  Universe and Evaluation rows in section 1.
- `Code_Audit_2026-09-25.md`: add a section 11 listing what this brief closed (E-1 follow-up, X-1
  switch, G-5 accepted) and what it did not.

**Commits.** One commit per section, A to G, after the suite, ruff and mypy pass. Rules:

- Stage files by explicit path. **Never** `git add -A`, `git add .` or `git commit -a`.
- Before every commit, run `git diff --cached --name-only` and confirm that nothing under `Data/`,
  `data_cache/` or `Claude outputs/` is staged, and no `.parquet`, `.pt` or `.csv` file derived from
  CRSP. `Claude outputs/` holds personal files and stays untracked.
- Do not push.

---

## Tests, summarised

A: A.7. B: removed tests go with their modules; the suite stays green. C: three identities. D: five
tests. E: none. F: the run is the test, and its report is the deliverable.

## Scope fence

Implement step 0 and A to G only. In particular, do not:

- answer Q21 (frequency and horizon): keep the daily panel and the configured horizon, and add no
  monthly path;
- answer Q22: the gate filter keeps skipping the purge gap;
- touch Q16 (time decay, crash weights) or B-2 (the base's validation tail);
- use Compustat: no fundamentals, and no link table;
- widen the universe beyond the S&P 500;
- fix B-1 or any other open audit finding;
- register any objective other than `"mixture_nll"` (Q20);
- tune any hyperparameter;
- report any out-of-sample performance number.
