# Trexquant Alpha Competition — Data Dictionary

**Saved:** 2026-08-07 · **Source:** PySim → Data Description (6 pages, ~255 variables)
**Universe:** top1000 US · **Neutralization:** industry · **Region:** us
**Competition deadline:** ~2026-08-29 (21d from save)

---

## How to read the columns

| Column | Meaning | Why it matters for alpha design |
|---|---|---|
| **Dimensions** | `2D` = stocks × dates matrix. `1D` = single time series broadcast to all stocks. | 1D vars have **zero cross-sectional variance** → useless alone (they get neutralized to nothing). Only useful as *conditioners* / regime switches multiplying a 2D signal. |
| **Value Type** | `single` (float) or `int` | ints (industry/sector/subindustry, calendar) are **grouping keys**, not signals. |
| **Data Type** | Technical / Fundamental / Seasonality / Macro / ETF | Cheap diversification axis: alphas from different data types tend to be naturally decorrelated. |
| **Adj Factor** | `CAX_ADJ_DIV_D1` (price-like, divide) · `CAX_ADJ_MUL_D1` (share/volume-like, multiply) · `—` (none) | **Critical.** Adjusted series are retro-adjusted for splits/dividends → fast-vs-slow mode divergence. Always make expressions *dimensionless* (ratios) to kill this. |
| **Delay 0** | `Yes` = same-day value usable; `No` = must be lagged | `No` on `mktcap`, `industry`, `sector`, `subindustry`, and all `*_d1` valuation ratios. Using them at delay 0 is forward bias. |
| **Coverage** | # of stocks with any data, **against the ~3250-name database — not against the top1000 universe** | A var with coverage 800 could be 800 names all inside the top1000 (large caps file more completely), or 800 microcaps outside it. Low coverage is a **flag to check empirically**, not an automatic disqualifier. Check realised `numstk` in the sim. |
| **Update Freq** | avg **days between value changes** | ≈1 = daily · ≈25–32 = TTM-refresh cadence · ≈50 = semi-annual-ish · ≈70–100 = quarterly · ≈1500 = static. **A fundamental with freq 70 changes ~4×/year — the raw level is a step function. Diff it or rank it, don't smooth it.** |
| **Empty Value** | value used for "missing" | `0` on group/calendar vars. Remember: **0 ≠ no position** in Trexsim (median gets subtracted). Use `np.nan`. |
| **Lower/Upper Bound** | ⚠️ **NOT a reliable observed min/max** — see "Bounds don't reconcile" below | Directionally useful for spotting heavy tails (1e13 in places), useless as an actual range. **Winsorize or rank before use** — `cs_rank`, `cs_winsor`, `ts_norm`. |

### ⚠️ Bounds don't reconcile to a single window

The Lower/Upper Bound column cannot be a plain observed min/max over one date range:

- `close` max **1092.34** — but NVR (~$6000), BKNG (~$3000) and others are top-1000 names. A true max would be far higher.
- `close_spx` max **2395.96** — SPX last traded there in **March 2017**; it ended 2021 at 4766.
- The FRED block, by contrast, fingerprints **2022 → ~mid-2023** unambiguously: CPI 288.66–301.81, WTI $73.28–114.84, U3 3.1–3.4%, GDP $24.4–26.1T.

So either the bounds are percentile-clipped, or they're vendor validation limits, or different groups were computed over different windows. **Treat the column as advisory only.** The FRED range is still a genuine clue that the competition's evaluation data may be more recent than the tutorial's 2006–2021 example — worth confirming in the UI.

---

## Group index — where the alpha actually is

| # | Group | Vars | Data type | Update freq | Verdict |
|---|---|---|---|---|---|
| 1 | `GROUP_backoffice` | 12 | Technical | daily | Core OHLCV. Everyone uses it → crowded, but the base for everything. |
| 2 | `GROUP_backoffice_derived` | 4 | Technical | daily | ret1/5/10/20. Reversal & momentum primitives. |
| 3 | `GROUP_edgar8kfs` | ~84 | Fundamental | 70–380d | Raw 8-K line items. Sparse coverage on many. **Surprise/revision alphas live here.** |
| 4 | `GROUP_edgar8kfs_ratio_derived` | ~66 | Fundamental | 25–51d | Pre-built ratios. Quality/profitability/growth factors. |
| 5 | `GROUP_edgar8kfs_ratio_derived_d1` | 11 | Fundamental | 28–51d | **Price-dependent** valuation (Delay0 = **No**). Value factors. |
| 6 | `GROUP_edgar8kfs_ratio_derived2` | 14 | Fundamental | 1–51d | More ratios incl. P/B, P/FCF, EV/FCF. |
| 7 | `GROUP_fomc_dates` | 2 | Seasonality | daily | 1D-like conditioner. FOMC drift. |
| 8 | `GROUP_intra1` | 30 | Technical | daily | **First-hour / last-hour bars.** Highest alpha density per unit of crowding — most competitors won't mine this properly. |
| 9 | `GROUP_trading_days_fast` | 1 | Seasonality | ~2d | Days to next earnings. **Best single conditioner in the whole dataset.** |
| 10 | `GROUP_days_to_calendar_events` | 15 | Seasonality | daily | Calendar effects: OpEx, month/quarter/year end, holidays. |
| 11 | `GROUP_daily_etf_returns` / `daily_idx` / `daily_idx_derived` | 5 | ETF / Macro | daily | GLD, SPX, VIX. 1D → betas & regime switches only. |
| 12 | `GROUP_fred` | 11 | Macro | monthly/quarterly | **1D only.** Regime conditioners, nothing more. |

---

## 1. GROUP_backoffice — daily core (12)

| Var | Adj | D0 | Cov | Freq | Range | Notes |
|---|---|---|---|---|---|---|
| `industry` | — | No | 3237 | 1476 | 0–84 | ~70 industries. int. **Neutralization group used by the sim.** |
| `sector` | — | No | 3237 | 1720 | 0–14 | ~13 sectors. int. |
| `subindustry` | — | No | 3237 | 1500 | 0–534 | ~500 subindustries. int. Finest grain → best for `cs_*_group` ops. |
| `cashdvd` | — | No | 1619 | 342 | 0–53 | Cash dividend; 0 when none declared. |
| `close` | DIV | Yes | 3251 | 1.03 | 0–1092 | Adjusted close. |
| `high` | DIV | Yes | 3251 | 1.03 | 0–1120 | Adjusted high. |
| `low` | DIV | Yes | 3251 | 1.03 | 0–1083 | Adjusted low. |
| `open` | DIV | Yes | 3251 | 1.03 | 0–1088 | Adjusted open. |
| `volume` | MUL | Yes | 3251 | 1.00 | 0–3.77e9 | Composite volume incl. open+close auctions. |
| `mktcap` | — | **No** | 3237 | 1.03 | 0–774692 | price × shsout. **Delay0 = No.** Size factor / normalizer. |
| `shsout` | MUL | No | 3237 | 58.96 | 0–29206 | Shares outstanding (millions). Freq 59 → **buyback signal via diff**. |
| `shsfloat` | MUL | No | 3011 | 15.64 | 0–27805 | Float shares (millions). Freq 15.6 — updates 4× faster than shsout. **`shsout − shsfloat` = insider/restricted holding.** |

## 2. GROUP_backoffice_derived — returns (4)

| Var | D0 | Cov | Freq | Range |
|---|---|---|---|---|
| `ret1` | Yes | 3250 | 1.03 | −0.93 … 74.55 |
| `ret5` | Yes | 3250 | 1.00 | −1.59 … 74.54 |
| `ret10` | Yes | 3250 | 1.01 | −0.97 … 75.80 |
| `ret20` | Yes | 3250 | 1.01 | −0.97 … 77.35 |

> Tutorial baseline alphas were `-ret1` (IR 0.025), `-ret5` (IR 0.043), `-ret20` (IR 0.026), with pairwise corr 0.46–0.77. Plain reversal is the crowded floor, not a submission.

## 8. GROUP_intra1 — first hour / last hour (30) ⭐

Every variable exists in a `_first_hour` and `_last_hour` pair. Coverage 3246–3249, Update freq ≈1.18 (daily), Delay 0 = **Yes** for all.

| Family | Vars | Adj | Notes |
|---|---|---|---|
| OHLC | `open_price_*`, `high_price_*`, `low_price_*`, `close_price_*` | DIV | 8 vars. Build intraday ranges, gaps, overnight vs intraday decomposition. |
| VWAP | `vwap_first_hour`, `vwap_last_hour` | DIV | Price-vs-VWAP = execution pressure proxy. |
| Volume | `volume_*`, `std_volume_*`, `skew_volume_*` | MUL / — | 6 vars. Volume *concentration* in open vs close auction window. |
| Trade count | `number_of_trades_*` | MUL | 2 vars. **`volume / number_of_trades` = average trade size → institutional vs retail flow proxy.** Not derivable from daily data. |
| Return | `return_first_hour` (−0.92…4472), `return_last_hour` (−1…366.8) | — | Raw hourly returns. |
| Vol of return | `std_dev_return_*` | — | Realized intraday vol. |
| Skew | `skew_close_*`, `skew_return_*`, `skew_volume_*` | — | 6 vars, bounded ≈ ±7.6. Already normalized → safe to use directly. |
| Info ratio | `information_ratio_first_hour` (−2.83…2.19), `information_ratio_last_hour` (−2.61…2.12) | — | "Return information ratio" within the hour = intra-hour return / vol. Bounded, clean, **almost certainly under-mined by competitors.** |

**Why this group is the edge:** it is the only place with sub-daily structure. Overnight return = `open − ts_delay(close,1)`; first-hour reaction = `close_price_first_hour − open`; close-auction pressure = `close − vwap_last_hour`. These decompose the daily return into behaviourally distinct pieces that daily-bar competitors cannot separate.

## 9. GROUP_trading_days_fast (1) ⭐

| Var | D0 | Cov | Freq | Range |
|---|---|---|---|---|
| `trading_days_until_next_earnings_announcement` | Yes | 7687 | 2.01 | 0–127 |

> Coverage 7687 (> universe) and daily-ish refresh. This is the single best **conditioner**: gates earnings-drift, pre-announcement drift, and post-earnings-announcement-drift (PEAD) windows. Any technical alpha × earnings-window mask is a genuinely new alpha, not a correlated clone.

## 10. GROUP_days_to_calendar_events (15)

All: Seasonality, Delay0 = **No**, Coverage 3237 (3277 for quarterly OpEx), Empty value 0.

| Var | Range | Freq |
|---|---|---|
| `day_of_the_week` | 1–5 (1=Mon) | 1.00 |
| `month_of_the_year` | 1–12 | 21.1 |
| `quarter_of_year` | 1–4 | 64.3 |
| `days_since_first_trading_day_of_month` | 0–22 | 1.05 |
| `days_until_last_trading_day_of_month` | 0–22 | 1.05 |
| `days_since_first_trading_day_of_quarter` | 0–63 | 1.02 |
| `days_until_last_trading_day_of_quarter` | 0–63 | 1.02 |
| `days_since_first_trading_day_of_year` | 0–252 | 1.00 |
| `days_until_last_trading_day_of_year` | 0–252 | 1.00 |
| `days_since_last_trading_holiday` | 1–62 | 1.00 |
| `days_until_next_trading_holiday` | 1–62 | 1.00 |
| `days_since_last_monthly_options_expiration` | 0–24 | 1.05 |
| `days_until_next_monthly_options_expiration` | 0–24 | 1.05 |
| `days_since_last_quarterly_options_expiration` | 0–63 | 1.02 |
| `days_until_next_quarterly_options_expiration` | 0–63 | 1.02 |

> These are **identical across all stocks on a given day** → cross-sectionally flat → they neutralize to zero on their own. Value is exclusively as **multiplicative gates** (turn a signal on/off around OpEx, month-end rebalance, turn-of-quarter window dressing, tax-loss selling in December).

## 7. GROUP_fomc_dates (2)

| Var | D0 | Cov | Freq | Range |
|---|---|---|---|---|
| `days_since_last_fomc_meeting` | No | 3237 | 1.03 | 0–38 |
| `days_until_next_fomc_meeting` | No | 3237 | 1.03 | 0–38 |

> Same gating logic. Known effect: pre-FOMC announcement drift; low-beta outperforms into the meeting.

## 11. Index / ETF (5) — 1D-behaving

| Var | Group | D0 | Cov | Freq | Range |
|---|---|---|---|---|---|
| `close_spx` | daily_idx | Yes | 3251 | 1.00 | 0–2396 |
| `close_vix` | daily_idx | Yes | 3251 | 1.01 | 0–80.86 |
| `ret1_spx` | daily_idx_derived | Yes | 3237 | 1.00 | −0.09…0.12 |
| `ret1_vix` | daily_idx_derived | Yes | 3237 | 1.00 | −0.30…0.64 |
| `ret1_gld` | daily_etf_returns | Yes | 3791 | 1.01 | −0.03…0.03 |

> Uses: rolling beta of each stock to SPX/VIX/GLD (`ts_corr(ret1, ret1_spx, N)`) is a genuine **cross-sectional** signal even though the index itself is 1D. Also VIX-level regime switching.

## 12. GROUP_fred — macro, **1D only** (11)

`cpi_total_all_items_us` (0.37–2.97, growth rate) · `cpi_all_urban_consumers` (288.66–301.81, SA, 1982-84=100) · `cpi_all_urban_consumers_ex_food_energy` (290.45–305.24, core) · `cpi_all_urban_consumers_purchasing_power` (33.1–34.6) · `gdp_seasonally_adjusted_annual_rate` ($bn, 24383–26132) · `real_gdp_seasonally_adjusted_annual_rate` (chained 2009 $bn, 19682–20198) · `gdp_implicit_price_deflator` (123.54–129.38) · `gdp_now_annualized_percent_change` (0.67–3.07) · `unemployment_rate_20_years_and_over` (3.1–3.4%) · `unemployment_level_seasonally_adjusted` (5670–6059 k) · `wti_spot_crude_oil_price` ($73.28–114.84/bbl)

> All Delay0 = No, Update freq 1, Dimensions **1D**. Note the tight observed ranges — this looks like a short macro sample. Treat as **regime dummies only**; do not try to build a factor from them directly. `wti_spot_crude_oil_price` × energy-sector mask is the one with real cross-sectional bite.

---

## 3. GROUP_edgar8kfs — raw 8-K line items (~84)

All Delay0 = **Yes**, Data type Fundamental, no adjustment factor unless noted. Format below: `var` — coverage / update-freq days.

### Income statement
`net_sales` 559/69 · `sales_revenue_turnover` 1275/73 · `gross_profit` 966/71 · `cost_of_goods_sold_to_fixed_expenses_and_production_and_general` 1310/71 · `operating_income` 2068/71 · `pre_tax_income` 1958/84 · `net_income` **2585/71** · `net_income_before_minority_interest` 2531/71 · `adjusted_net_income_as_reported` 387/165 · `general_and_admin_expense` 1873/71 · `total_operating_expenses` 870/75 (also 298/86 dup) · `research_and_development_expense` 776/73 · `personnel_expenses` 302/70 · `income_tax_expenses` 2376/73 · `interest_expense` 1433/83 · `net_interest_expense` 518/97 · `interest_income` 939/94 · `depreciation_amortization` 928/77 (also 552/77 dup) · `amortization_of_intangible_assets` 361/97 · `stock_based_compensation` 990/77 · `operating_profits_discontinued_operations` 176/376 · `minority_non_controlling_interest` 475/101 · `minority_non_controlling_interests_credits` 423/126

### Banks / financials specific
`net_interest_income` 402/97 · `net_interest_income_after_provision` 202/74 · `non_interest_income` 244/69 · `non_interest_expense` 248/69 · `total_deposits` 204/70

### Per-share
`basic_eps` 2459/76 (adj DIV) · `diluted_eps` **2541/77** (adj DIV) · `dividend_per_share` 453/329 (adj DIV split-only) · `average_number_of_shares_for_eps` 1676/83 (adj MUL) · `diluted_weighted_average_shares` 1749/81 (adj MUL)

### Balance sheet — assets
`total_assets` **2336/70** · `current_assets_report` 1768/70 · `balance_sheet_cash_near_cash_items` **2448/70** · `short_term_investments` 304/126 · `marketable_securities_and_other_short_term_investments` 166/131 · `accounts_notes_receivable` 298/110 · `accounts_receivable_excluding_notes` 718/77 · `inventories` 1202/71 · `other_current_assets` 1488/71 · `goodwill` 1592/130 · `intangible_and_other_assets` 1003/78 · `other_assets` 1489/75 · `total_other_property_and_equipment` 1763/71 · `accumulated_depreciation` 113/105 · `deferred_tax_assets_long_term` 145/126 · `collaterals_cash_market_short_term_investments` 421/185

### Balance sheet — liabilities & equity
`total_liabilities` 1683/70 · `current_liabilities` 1778/70 · `non_current_liabilities` 220/83 · `accounts_payable` 1598/71 · `accounts_payable_and_accruals` 503/76 · `other_short_term_liabilities` 1305/72 · `other_long_term_liabilities` 659/77 · `long_term_debt` 1140/90 · `long_term_borrowings` 1220/91 · `short_term_borrowings` 276/117 · `current_portion_of_long_term_debt` 384/150 · `deferred_tax_liabilities` 951/93 · `total_shareholders_equity` 1870/77 · `total_equity_partnership_capital` 471/75 · `total_liabilities_and_equity` 1854/75 · `tangible_equity` **8**/279 ⚠️ unusable · `additional_paid_in_capital` 1129/72 · `common_stocks` 515/224 · `treasury_stock_amount` 315/147 · `pure_retained_earnings` 984/75 · `retained_earnings_and_other_equity` 955/76 · `accumulated_other_comprehensive_income` 1122/79

### Cash flow
`cash_from_operations` **1277/75** · `cash_flow_net_income` 1280/76 · `cash_from_investing_activities` 1154/77 · `cash_from_financing_activities` 1137/79 · `net_change_in_cash` 1225/76 · `capital_expenditures_property_additions` 398/77 · `dividends_paid` 313/83 · `change_in_inventories` 399/82 · `deferred_income_taxes` 722/88 · `working_capital_changes` **24**/172 ⚠️ unusable · `effect_of_exchange_rate_changes_on_cash` 567/80

> **Coverage warning:** bolded vars are the only ones with coverage comfortably above ~1200 *of the ~3250-name database*. Under ~800 is a flag to check realised `numstk` in-sim, not an automatic reject — large caps file more completely, so a low-coverage field may still be dense inside the top1000. `tangible_equity` (8 stocks) and `working_capital_changes` (24 stocks) are genuine decoys.

> **Units warning:** bounds run to ±1e9 in some fields and ±1e3 in others — the reporting units are *not* consistent across variables. Never add two raw fundamentals; always ratio them against a same-statement denominator (`total_assets`, `sales_revenue_turnover`, `mktcap`) then `cs_rank`.

---

## 4. GROUP_edgar8kfs_ratio_derived (~66)

All Delay0 = Yes, Fundamental. Update freq mostly **50.6** (≈ semi-annual refresh), some **25.3 / 28.6 / 31.6** (faster TTM refresh).

### Profitability & margins
`ebit_margin` 1360/31.6 · `ebitda_margin` 1361/31.6 · `operating_margin` 1393/50.6 · `pretax_margin` 1285/50.6 · `profit_margin` 1601/50.6 · `ebit_to_net_sales_ratio` 1393/50.6 · `ebitda_to_revenue_pct` 1393/50.6 · `trailing_12_month_operating_margin_pct` 1360/31.6 · `return_on_assets` **2578**/50.6 · `roa_based_on_bottom_eps` 2561/50.6 · `return_on_equity` 2175/50.6 · `ann_return_on_equity` 1312/50.6 · `normalized_roe` 2095/25.3 · `net_income_per_employee` 2534/42.2 · `sell_and_admin_exp_to_net_sales_ratio` 1156/50.6

### Growth (YoY %) — **the highest-signal subset here**
`eps_growth` **2539**/50.6 · `net_income_growth_pct` **2659**/50.6 · `operating_income_growth` 2143/50.6 · `ebit_year_over_year_growth` 2143/50.6 · `ebitda_growth_pct` 2153/50.6 · `pretax_income_year_growth_pct` 1937/50.6 · `cash_flow_growth` 1560/50.6 · `accounts_receivable_growth_pct` 268/50.6 · `inventory_growth_pct_change` 1183/50.6 · `income_tax_expense_year_growth_pct` 2325/50.6 · `interest_expense_year_growth_pct` 1321/50.6 · `net_change_long_term_debt` 1127/50.6

### Cash flow quality
`free_cash_flow` 1745/50.6 · `free_cash_flow_per_share` 1213/50.6 (adj DIV) · `cash_flow_per_share` 1190/50.6 (adj DIV) · `trailing_12_month_cash_flow_per_share` 1184/28.6 (adj DIV) · `trailing_12_months_cash_from_operations` 1713/31.6 · `cash_from_operations_to_sales_pct` 1039/50.6 · `cash_from_operations_to_total_debt` 961/50.6 · `cash_generation_to_cash_requirements_ratio` 1295/50.6 · `cash_flow_to_total_liabilities` 1271/50.6 · `operating_income_per_share` 1507/50.6 (adj DIV) · `pretax_income_per_share` 1491/50.6 (adj DIV) · `ebitda_per_diluted_share` 1567/50.6 (adj DIV) · `reinvested_earnings` 1689/50.6

### Liquidity / balance-sheet
`cash_ratio` 2030/50.6 · `quick_ratio` 2031/50.6 · `cash_and_short_term_investments` **2740**/50.6 · `cash_and_marketable_securities_per_share` 2740/25.3 (adj DIV) · `cash_to_total_assets_ratio` **2604**/50.6 · `cash_and_equivalents_to_total_liabilities_pct` 2066/50.6 · `cash_and_investments_to_current_assets_pct` 2021/50.6 · `current_assets_to_total_assets_ratio` 1997/50.6 (bounded 0–1.28 ✔) · `non_cash_working_capital` 2002/50.6 · `inventory_to_current_assets_pct` / `inventory_to_current_assets_ratio` 1222/50.6 (dupes, bounded 0–95.4) · `inventory_to_total_assets_ratio` 1245/50.6 (0–87.7) · `sales_to_cash_ratio` 1488/50.6 · `other_assets_turnover_ratio` 831/50.6

### Leverage
`total_debt` 1275/50.6 · `net_debt` 1310/50.6 · `net_debt_to_ebitda` 1122/50.6 · `net_debt_to_ebitda_after_capex` 1122/50.6 · `net_debt_to_shareholders_equity` 997/50.6 · `debt_to_market_cap` 1275/**1.02** ⭐ (price-driven → refreshes daily) · `operating_income_to_total_debt` 1082/50.6 · `operating_income_to_long_term_debt` 1081/50.6 · `operating_income_to_current_liabilities` 1925/50.6 · `retained_earnings_to_total_liabilities_and_equity` 972/50.6 (bounded −2.17…2.51 ✔)

### TTM EPS
`t12m_diluted_eps_continuing_operations` **2837**/50.6 (adj DIV) · `trailing_12_month_diluted_eps` **2743**/50.6 (adj DIV) · `price_to_tangible_book_value_per_share` 2231/**1.02** ⭐ (−11.89…105.84, bounded ✔)

---

## 5. GROUP_edgar8kfs_ratio_derived_d1 — valuation (11) ⚠️ Delay 0 = **No**

Price-dependent, so they refresh with price (freq ≈28–51 shown but driven by both).

| Var | Cov | Freq | Range | Notes |
|---|---|---|---|---|
| `earnings_yield` | 2766 | 50.6 | −7.82…7.46 | **Bounded, high coverage. Best value factor in the set.** |
| `earnings_yield_history` | 2742 | 50.6 | −4401…745.7 | Needs winsorizing. |
| `ebit_yield` | 1487 | 31.6 | −55163…407037 | Heavy tails. |
| `enterprise_value_to_market_cap_ratio` | 392 | 28.1 | −0.84…220 | Low coverage. |
| `ev_to_t12m_cash_flow` | 990 | 28.1 | −33210…2.3e7 | |
| `ev_to_t12m_ebit` | 1117 | 30.0 | −67937…1.58e7 | |
| `ev_to_t12m_net_income` | 1289 | 28.1 | −54587…2.4e7 | |
| `high_price_to_earnings_ratio` | 2766 | 50.6 | ±1.8e10 | P/E using period high. |
| `low_price_to_earnings_ratio` | 2742 | 50.6 | −1.8e10…2.5e9 | P/E using period low. **`high_PE − low_PE` = valuation dispersion, a free derived var.** |
| `price_to_cash_flow_ratio` | 1185 | 50.6 | −6.1e7…1.3e9 | |
| `price_to_sales_ratio` | 1585 | 50.6 | −330…3.7e6 | |

## 6. GROUP_edgar8kfs_ratio_derived2 (14)

| Var | D0 | Cov | Freq | Range |
|---|---|---|---|---|
| `asset_to_equity_ratio` | Yes | 2112 | 50.6 | −1021…2.63e7 |
| `cash_flow_to_interest_expense_ratio` | Yes | 853 | 50.6 | ±~1e7 |
| `cash_flow_to_net_income_ratio` | Yes | 1686 | 50.6 | ±1e13 ⚠️ |
| `times_interest_earned_ratio` | Yes | 1494 | 50.6 | −1.38e6…3.93e6 |
| `ev_to_t12m_free_cash_flow` | Yes | 413 | **1.02** | ±1e13 ⚠️ |
| `free_cash_flow_to_total_debt_ttm` | Yes | 1017 | 31.6 | −13280…1.58e6 |
| `inventories_to_working_capital_pct` | Yes | 1206 | 50.6 | **0–1.58** ✔ clean |
| `price_to_book_ratio` | Yes | **54** | 1.02 | 0–6.99 ⚠️ unusable coverage |
| `price_to_ebitda` | Yes | 1613 | 1.02 | ±~3.5e9 |
| `price_to_free_cash_flow` | Yes | 1224 | 1.02 | −1.3e8…5.86e9 |
| `sales_to_marketable_securities` | Yes | 1461 | 50.6 | ±~1.3e7 |
| `sales_to_total_assets_ratio` | Yes | 1398 | 50.6 | −751…328356 |
| `total_debt_to_enterprise_value` | Yes | 406 | 1.01 | **−0.65…2.57** ✔ clean but low coverage |

---

## Traps to remember

1. **`0` is not "no position"** — the median is subtracted from the alpha matrix, so zeros become live long/short bets. Use `np.nan`.
2. **1D vars neutralize to zero.** FRED macro and all calendar vars only work multiplicatively.
3. **`Delay 0 = No`** on `mktcap`, `industry`, `sector`, `subindustry`, all `_d1` valuation ratios, all calendar/FOMC vars. Lag them or you have forward bias in slow mode.
4. **Retro-adjustment bias.** `CAX_ADJ_DIV_D1` / `CAX_ADJ_MUL_D1` series are adjusted backwards for splits — a raw level alpha like `-close` behaves differently in fast vs slow mode. Ratios fix this.
5. **Fundamental step functions.** Update freq 50–100d means the value is piecewise-constant. `ts_delta` on it fires only on the update day; that's a *feature* (an event) if intentional, a bug if not.
6. **Coverage < ~1000** → the alpha can't populate a top1000 universe.
7. **Mixed units.** Bounds show 1e3 to 1e13 across the fundamental set. Rank or normalize before combining anything.
8. **Duplicate variable names** exist across groups (`depreciation_amortization`, `total_operating_expenses`, `inventory_to_current_assets_*`) with different coverage — check which one the sim resolves to.
