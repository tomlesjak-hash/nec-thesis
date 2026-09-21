# Setup & Run — macOS terminal

## Step 1 — Python 3.14 is fine ✅

Verified against PyPI on 2026-08-07 — every dependency ships CPython 3.14 wheels:

| Package | Latest | 3.14 wheel? |
|---|---|---|
| `pyarrow` | 25.0.0 | ✅ cp314 |
| `pandas` | 3.0.5 | ✅ cp314 |
| `numpy` | 2.5.1 | ✅ cp314 |
| `lxml` | 6.1.1 | ✅ cp314 |
| `yfinance` | 1.5.2 | ✅ pure Python (`py3-none-any`) |

Keep your global Python at 3.14. No downgrade needed.

```bash
python3 --version
which python3
```

Make sure you are **not** inside the venv — if your prompt shows `(.venv)`, run `deactivate` first.

### ⚠️ Two real caveats (neither is about Python 3.14)

**1. `geppy` is dead — use the DEAP path only.**
Your QIP `genetic_factor.py` has both `deap_factor()` and `geppy_factor()`. Only DEAP is viable:

| GP library | Last release | Python support |
|---|---|---|
| `deap` | **2026-04-17** (v1.4.4) | pure Python, actively maintained ✅ |
| `gplearn` | **2026-01-07** (v0.4.3) | `requires_python >= 3.11`, classifiers through 3.14 ✅ |
| `geppy` | **2021-09-21** (v0.1.3) | classifiers stop at **3.9**, no wheels ❌ |

`geppy` is five years stale and will not install cleanly on 3.14. That's no loss — DEAP is the path that produced your existing results.

**2. pandas 3.0 is a major version, and that's the actual compatibility risk.**
This pipeline was tested on **pandas 2.3.3 / Python 3.10** (my sandbox). On 3.14 you'd get **pandas 3.0.5**, where Copy-on-Write is the default rather than opt-in and several long-deprecated APIs are gone.

- The **download pipeline** (`tq_data.py`, `tq_calendar.py`) uses only stable APIs — direct `.loc` assignment on owned frames, `groupby`, `rolling`, `to_parquet`. Low risk.
- The **QIP GP code** is the real exposure: it was written for pandas 1.x/2.x and does things like in-place mutation inside operator functions (`y[y==0] = 1e-10` in `div`). Copy-on-Write can change that silently.

**Plan:** install latest, run the smoke test. If the GP port later misbehaves in ways that smell like silent no-op assignments, pin `pandas<3` in a venv **for the GP only** — the panel is parquet on disk, so the two halves don't need matching pandas versions.

---

## Step 2 — install

```bash
python3 -m pip install --upgrade pip
python3 -m pip install yfinance pandas pyarrow lxml
```

Later, for the GP port (not needed for the download):

```bash
python3 -m pip install deap gplearn scikit-learn attrs    # NOT geppy
```

### If you get `error: externally-managed-environment`

Normal on Homebrew Python 3.12+. It's PEP 668 protecting the system Python. Since you want a global install:

```bash
python3 -m pip install --break-system-packages yfinance pandas pyarrow lxml
```

### If you get `Permission denied`

```bash
python3 -m pip install --user yfinance pandas pyarrow lxml
```

> Use `python3 -m pip`, not bare `pip3` — it guarantees the packages land in the same interpreter that will run the script. Mismatched `pip3`/`python3` is the single most common cause of "I installed it but it says ModuleNotFoundError".

---

## Step 3 — verify before running anything

```bash
python3 -c "import yfinance, pandas, pyarrow, lxml; print('all four OK')"
```

Must print `all four OK`. If not, fix it here — don't proceed.

---

## Step 4 — smoke test (2 minutes, do NOT skip)

Twenty tickers, no shares, no sectors. Proves the whole path works before you commit to ~900 tickers.

```bash
cd ~/Desktop/"Quant Model"/Trexquant/pipeline
python3 run_download.py --max-tickers 20 --no-shares --no-sectors
```

**What good output looks like:**

```
[1] migrated N tickers from nec_baseline cache (not re-downloaded)
[2] fetching S&P 500 point-in-time membership…
    503 current members, ~400 changes, 20 unique tickers over the window
[3] downloading 20 equities + 3 context…
    20 ok, 0 failed
[4] building wide panel…
    ~3900 dates x 20 tickers
[5] computing calendar + FOMC features…
    17 seasonality variables
[6] building top-500-by-ADV mask (monthly rebalance)…
    median names/day: 20
[7] saving…
done in ~60s — ~5 MB
```

**If `[2]` returns 0 tickers** → Wikipedia fetch or table parse failed. Delete `Trexquant/cache/sp500_wiki.html` and retry.
**If `[3]` shows many failures** → Yahoo is rate-limiting. Wait 10 minutes, rerun; the cache means you only refetch what's missing.

---

## Step 5 — the real run

```bash
python3 run_download.py
```

Defaults: `--start 2010-01-01`, `--end` today, `--top-n 500`, shares + sectors on.

| Stage | Time |
|---|---|
| Migrate nec_baseline cache | seconds |
| Download ~900 tickers (batched 100 at a time) | 10–20 min |
| Shares outstanding (per-ticker, slow) | 10–15 min |
| Sector/industry (per-ticker) | 3–5 min |
| Panel + calendar + mask + save | 1–2 min |

**~30–45 min total.** Resumable — everything is cached, so a re-run only fetches what's missing.

To skip the two slow per-ticker loops:

```bash
python3 run_download.py --no-shares --no-sectors
```

But **don't skip sectors for the real run** — Trexsim neutralizes by industry, so your local fitness needs it or your local IR won't predict the platform's.

---

## Step 6 — sanity-check the output

```bash
cd ~/Desktop/"Quant Model"/Trexquant
python3 -c "
import pandas as pd, glob, os
for f in sorted(glob.glob('panel/*.parquet')):
    d = pd.read_parquet(f)
    print(f'{os.path.basename(f):24s} {d.shape[0]:5d}d x {d.shape[1]:4d}tk  '
          f'{100*d.notna().mean().mean():5.1f}% filled')
u = pd.read_parquet('panel/universe.parquet')
print('\nuniverse names/day: median', int(u.sum(1).median()),
      'min', int(u.sum(1).min()), 'max', int(u.sum(1).max()))
c = pd.read_parquet('panel/close.parquet')
print('dates', c.index[0].date(), '->', c.index[-1].date())
"
```

**What to check:**
- `close` / `open` / `high` / `low` / `volume` — should be **60–90% filled** (names not yet listed or already delisted are legitimately NaN).
- `universe` — median names/day should be near your `--top-n`.
- `mktcap` / `shsout` — **expect low fill, concentrated post-2022.** That's the Yahoo share-history limitation, not a bug. Use `adv60` as the size proxy in the GP.
- `industry` — should be ~100% filled for names Yahoo still serves.
- Date range should start 2010-01-xx.

---

## Then what?

The panel is the input to Track B. Next step is porting `QIP/data_and_models/quant_lib/genetic_factor.py`:

1. **Data layer** — replace the three-chunk `read_pickle()` CN loader with `load_panel()` on `Trexquant/panel/`. Its existing `universe` parameter points at `universe.parquet`.
2. **Terminals** — replace raw `closes/highs/lows/opens/amounts` with dimensionless ratios. *This is the change that stops the clone problem.*
3. **Fitness** — swap quintile long/short at `lag_periods=2` for industry-neutral, rank-weighted, dollar-neutral **IR/√TVR**, with per-variable lags.
4. **Drop** `vwap` and `amount` terminals — no daily VWAP exists in Trexsim.

**Meanwhile, Track A runs in parallel and is still the priority** — hand-crafted alphas in Trexsim need none of this and guarantee a non-zero score. Start those while the download runs.
