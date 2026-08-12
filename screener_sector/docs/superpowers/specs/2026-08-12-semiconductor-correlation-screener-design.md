# Semiconductor Sector Correlation & Rebound Screener — Design

**Date:** 2026-08-12
**Status:** Approved design, pending implementation plan

## 1. Purpose

Screen a semiconductor / AI / optical-device equity universe to answer four questions in sequence:

1. Which names are in a strong trend on a short and mid-term horizon?
2. Within those trends, which names move together, and how strongly?
3. Within a correlated group, which names resist falls best and rise hardest?
4. When is a correlated group bottoming — and can we raise a rebound alarm with measurable, validated precision?

The system is a research and backtest toolkit first. Live daily screening is the same code run with `--as-of` set to today.

## 2. Non-goals

- No order execution, broker integration, or position sizing.
- No intraday or tick data. Daily bars only.
- No portfolio optimization or risk budgeting.
- No investment advice. The output is a ranked, annotated screen; interpretation is the user's.
- No export/import archive tooling. `data/` is relocatable by copy (see §5).

## 3. Data source

Yahoo Finance via `yfinance`.

Google Finance is not used: it has no public API, and `GOOGLEFINANCE()` is a Google Sheets formula, not a service. Yahoo's endpoint is unofficial and unsupported — it changes shape without notice and rate-limits aggressively. The local cache (§5) is the mitigation: the network is touched only to extend history, never to re-read it.

Universe seed lists come from the NASDAQ Trader symbol files (`nasdaqlisted.txt`, `otherlisted.txt`), which are public, stable, and free.

## 4. Container and layout

Everything runs in Docker. The host needs only Docker; no local Python setup is used.

```
Dockerfile              python:3.12-slim, pinned requirements
docker-compose.yml      service `screener`, bind-mounts DATA_DIR -> /data, ./out, ./config
.env.example            DATA_DIR=./data, PROFILE=dev
config/
  params.yaml           windows, weights, thresholds (per profile)
  universe.yaml         theme keyword sets, industry allow-list, liquidity filters
  universe.dev.yaml     ~30 checked-in tickers for the dev profile
src/screener_sector/
  config.py             typed config loading, profile resolution
  paths.py              the ONLY module that resolves filesystem paths
  universe/
    symbols.py          NASDAQ Trader symbol files
    enrich.py           Yahoo profile fields, cached permanently, resumable
    classify.py         theme keyword matching
    build.py            filters -> universe.csv
  data/
    store.py            parquet cache, incremental refresh
    calendar.py         trading-day alignment, NaN policy
  features/
    trend.py
    correlation.py
    strength.py
    rebound.py
  backtest/
    labels.py           forward-return bottom labeling
    walkforward.py      expanding-window splits
    evaluate.py         classification + economic metrics
  report/
    render.py           HTML + CSV output
  cli.py                subcommands
tests/
docs/superpowers/specs/
```

Invocation: `docker compose run --rm screener <command> [--profile dev|prod] [--as-of YYYY-MM-DD]`.

## 5. Data layer — portable by design

`data/` is the complete persistent state of the system, self-contained and relocatable. Copy it to any machine at any path, set `DATA_DIR`, and work continues.

```
data/
  manifest.json         schema_version, per-stage last_refresh, profiles run
  universe.csv          screened universe with theme tags and filter reasons
  meta/
    symbols.parquet     raw US-listed symbol table
    info.parquet        cached Yahoo profile fields (longName, industry, summary)
    failures.csv        quarantined tickers with reasons
  prices/
    <TICKER>.parquet    adjusted OHLCV, one file per ticker, maximum history
  derived/<profile>/    computed features keyed by as-of date (regenerable)
```

Three enforced rules:

1. **Single path root.** Every filesystem path is resolved through `paths.py` from `DATA_DIR` (default `./data`). No module constructs a path from `__file__` or the CWD. A test runs the pipeline against a temp dir at a different path and diffs the results.
2. **Versioned schema.** `manifest.json` carries `schema_version`. On startup the code refuses to run against a folder written by an incompatible version rather than corrupting it.
3. **`derived/` is disposable.** Deleting it costs CPU, never data. A partial or stale copy of `data/` is never wrong, only slower.

In compose, `DATA_DIR` is read from `.env` and bind-mounted to `/data`. Pointing it at an external drive, a NAS, or shared storage requires no code change.

## 6. Profiles

Two profiles, selected with `--profile` or `PROFILE`, defaulting to `dev`. Profiles change **configuration only, never code paths** — a dev run exercises exactly the code prod will run.

| | `dev` (default) | `prod` |
|---|---|---|
| Purpose | debug methodology, fast iteration | the real answer |
| Date range | 2022-01-01 → now | 2006-01-01 → now |
| Universe | ~30 checked-in tickers, discovery skipped | full discovery, 150–300 names |
| Walk-forward | 2 folds (test 2024, test 2025) | 12 folds, 1-year steps from 2015 |
| Runtime target | < ~2 minutes end to end | hours, dominated by one-time enrichment |

Two rules prevent contamination between modes:

1. **Price files always store maximum available history regardless of profile.** A profile narrows the analysis window, not what is cached. A dev-built cache is a valid subset for prod; switching profiles never invalidates a cache.
2. **Derived artifacts and reports are namespaced by profile** (`data/derived/<profile>/`, `out/<profile>/`). Dev and prod results coexist and can never be confused.

Handoff to the prod location: copy `data/` (or point `DATA_DIR` at shared storage) and run with `--profile prod`. The manifest records which profiles have run and through what date.

**Interpretation rule.** The dev window covers roughly 1.5 semiconductor drawdown cycles. It suffices to prove the code is correct and that signals fire where expected. It does **not** suffice to conclude the method works. Only prod results are evidence about the method. The report header states this explicitly on every dev report.

## 7. Pipeline

Seven stages, each a CLI subcommand writing a typed artifact the next stage reads. Any stage runs standalone.

| Stage | Command | Output |
|---|---|---|
| Universe discovery | `build-universe` | `universe.csv` |
| Price cache | `fetch` | `prices/*.parquet` |
| Trend | `trend --as-of` | trend scores per ticker |
| Correlation | `cluster --as-of` | cluster assignments, matrices |
| Relative strength | `strength --as-of` | capture ratios, ranks |
| Rebound alarm | `rebound --as-of` | alarm scores, fired signals |
| Validation | `backtest` | metrics, curves |
| Reporting | `report --as-of` | `out/<profile>/YYYY-MM-DD.html` + CSVs |

### 7.1 Universe discovery

Seed from NASDAQ Trader symbol files plus holdings of SOXX, SMH, BOTZ, AIQ as high-precision seeds. Enrich each symbol once with Yahoo's `longName`, `sector`, `industry`, `longBusinessSummary`; cache permanently in `meta/info.parquet`. The pass is slow (hours, throttled) but one-time and resumable.

Keep a ticker if **either** its industry is in the industry allow-list (Semiconductors, Semiconductor Equipment & Materials, Communication Equipment, Electronic Components, …) **or** its name/summary matches a theme keyword set in `config/universe.yaml`: semiconductor, wafer, foundry, photonic, silicon photonics, optical transceiver, lithography, EDA, GPU, accelerator, HBM, ASIC, interconnect, advanced packaging, and similar.

Then apply liquidity filters: US-listed common stock or ADR (so TSM, ASML, UMC are included); price > $2; median 60-day dollar volume > $5M. Foreign-only listings are excluded — Yahoo covers them unevenly and their timezone-misaligned bars corrupt correlation.

Expected result: 150–300 names. Every row carries its matched themes, which serve as human-readable labels alongside the statistical clusters. Rows rejected by filters are retained with a reason column rather than silently dropped.

### 7.2 Trend

Per ticker, per window (short = 20 trading days, mid = 60):

- Slope of an OLS fit on log price, normalized by realized volatility.
- R² of that fit — trend *quality*, distinguishing a clean advance from a drift with the same net move.
- ADX.
- Moving-average stack alignment.

Combined into a signed 0–100 composite score with configurable weights.

### 7.3 Correlation and clustering

Correlate rolling **log returns**, not prices; price correlation is spurious when everything trends together. Produce two matrices:

- **Raw** correlation — what moves together.
- **Residual** correlation, after regressing out the SOXX factor — what moves together for reasons other than shared sector beta.

Hierarchical clustering (average linkage) on the distance `sqrt(2 * (1 - rho))` computed from the residual matrix discovers groups without pre-naming them. A cluster is reported as strongly correlated only if mean intra-cluster rho clears a threshold **and** it has at least 3 members. Clusters are formed only from names with complete data over the correlation window.

### 7.4 Relative strength

Within each cluster, define group-up and group-down days from the cluster's equal-weight daily return.

- **Downside capture** = ticker mean return on group-down days / group mean on those days. Lower is better.
- **Upside capture** = the same on group-up days. Higher is better.
- Max drawdown vs. the group, and recovery-to-prior-high speed in days.

Ranked and percentiled within the cluster. Low downside capture with high upside capture identifies the leader — the name that resists falls and rises hardest.

### 7.5 Rebound alarm

Fires at cluster level first, then per ticker. Weighted score over:

- **Breadth washout** — percent of cluster members oversold and below their mid-window mean.
- **Price stretch** — distance below the mid-window mean in sigma units.
- **Oscillator** — RSI / Williams %R oversold, with bullish divergence.
- **Volume** — capitulation spike followed by dry-up.
- **Confirmation** — close above the prior bar's high, or reclaim of the short MA.

Cluster-level washout **gates** the per-ticker signals. The output is "this correlated group is bottoming, and these are its strongest members," not forty uncorrelated pings.

## 8. Validation

### 8.1 Labels

Date `t` is labeled a bottom if the low at `t` is the minimum over `[t - k, t + k]` **and** the forward 20-day return exceeds a threshold. `k` and the threshold are configurable.

### 8.2 Metrics

- **Classification:** precision, recall, F1, and lead/lag in days between alarm and true bottom.
- **Economic:** mean and median forward returns at 5 / 10 / 20 days, hit rate.
- **Baseline:** the same economic metrics for random entries on the same universe and dates. This comparison is what determines whether the signal has any edge; absolute hit rates alone are uninformative in a rising market.

### 8.3 Walk-forward

Expanding-window splits. Prod: fit 2010–2014 → test 2015; refit 2010–2015 → test 2016; continuing through the current year for 12 out-of-sample periods (2015–2025 full years plus 2026 year-to-date) and 12 independent parameter fits. Dev: fit 2022–2023 → test 2024; refit through 2024 → test 2025. The final fold is always partial-year and is reported separately so a short, unrepresentative stub is not averaged in with full years.

Parameter stability across folds is reported as a first-class diagnostic. Thresholds that thrash from fold to fold indicate overfitting more reliably than any single aggregate score.

**Point-in-time correctness:** all features are computed strictly from data at or before `t`. Only labels look forward, and only for evaluation. This is enforced by test (§10).

### 8.4 History depth

Fetch maximum available history back to 2006. Backtest spans 2010 → present, with 2006–2009 as a warm-up buffer so indicators at the start of 2010 are computed from full windows rather than truncated ones. Storage is negligible: ~300 tickers × 20 years ≈ 1.5M rows.

The span deliberately covers six distinct semiconductor drawdown regimes — 2011, the 2015–16 downturn, Q4 2018, COVID 2020, the 2022 bear, and the 2024–25 AI-cycle corrections. A shorter window risks tuning the alarm entirely on the post-2020 regime, in which nearly every dip bounced, which would flatter the signal badly.

A ticker needs at least 250 trading days of history to enter the backtest.

### 8.5 Known limitations — stated, not hidden

- **Survivorship bias.** The universe is built from today's listings, so delisted semiconductor companies are absent. This inflates backtest results, and the distortion grows the further back the test runs. Reports print this in the header, along with per-fold ticker counts so the thinness of early folds is visible.
- **Restated history.** Yahoo's adjusted prices are revised over time, so backtests are not bit-reproducible across re-fetches.
- **Short-history names.** Many target AI/optical names (ALAB, CRDO, and similar) have only 1–3 years of history and contribute to recent folds only.

## 9. Error handling

- Per-ticker network failures never kill a run: retry with exponential backoff, then quarantine to `meta/failures.csv` with a reason and continue.
- Tickers with insufficient history are excluded with a stated reason column, never silently dropped.
- Delisted tickers retain their cached data.
- A corrupt or partially written parquet file is detected on read and re-fetched rather than propagating NaNs.
- Schema-version mismatch aborts before any write.

## 10. Testing

Pytest. **Tests never touch the network** — a small frozen parquet fixture plus synthetic series with known analytic properties.

- A pure exponential uptrend must score R² ≈ 1 and a high positive trend score.
- Two series constructed with a known rho must recover it within tolerance.
- A synthetic V-bottom must fire the rebound alarm within a bounded lag.
- A flat series must fire nothing.
- Clustering on three synthetic blocks must recover the three blocks.
- **Lookahead test:** mutating data after `t` must not change any feature value at `t`. This is the single most important test in the suite.
- **Relocation test:** running the pipeline against a temp `DATA_DIR` at a different absolute path must produce byte-identical derived output.

## 11. Open items for the implementation plan

None blocking. Parameter defaults (weights, thresholds, window lengths) are starting points in `config/params.yaml` and are expected to move during walk-forward fitting.
