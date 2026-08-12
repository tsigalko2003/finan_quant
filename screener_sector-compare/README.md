# Sector correction and rebound screener

A containerized research pipeline for multi-ticker sector analysis. It begins with semiconductors and accepts a curated project industry, a cached Nasdaq stock-screener industry query, or a locally installed Qlib stock pool.

The pipeline separates network I/O from analysis:

1. `download` incrementally fills a range-aware per-ticker Parquet cache.
2. `analyze` runs fully offline from that cache and produces breadth, rolling correlation, PCA, causal HMM regime, relative fall/rise strength, walk-forward XGBoost validation, and rebound-alert artifacts.

This is an idea-screening and monitoring tool, not an investment recommendation or execution system.

## Interactive web console

Start the visualization and job-control server:

```bash
docker compose up -d --build web
open http://localhost:8002
```

The responsive web console provides:

- searchable catalog, cached Nasdaq-export industry, and POC/production selection;
- optional dates, ticker additions/exclusions, cache-tail refresh, and universe cap;
- guarded trend, correlation, PCA, HMM, rebound-label, coverage, and alert parameters;
- distinct **Download / refresh**, **Analyze cached**, and **Run both stages** actions;
- persisted queued/running/completed/failed job status;
- cache-hit and failure details from download jobs;
- an archive of completed runs with alert and walk-forward validation snapshots;
- sandboxed inline visualization of each generated HTML report, plus a full-report link.

The server listens on port `8002` inside the container and publishes it as host port `8002` by default. Override only the host port when needed:

```bash
SCREENER_WEB_PORT=8090 docker compose up -d web
```

Health endpoints are `/health/live` and `/health/ready`; interactive API documentation is available at `/api/docs`. The web server has network access for explicit download jobs. Analysis still calls the cache-only analysis path and never downloads implicitly. For hard network isolation in scripted workflows, continue using the `screener-offline` Compose service.

Job execution is intentionally bounded to one background worker for a single-user local deployment. Identical active submissions are deduplicated, job history is persisted in `outputs/web_jobs.json`, and a restart marks interrupted jobs failed instead of falsely reporting success.

## Quick POC

```bash
docker compose build
docker compose up -d web
docker compose run --rm screener list-industries
docker compose run --rm screener download --stage poc --industry semiconductor
docker compose run --rm screener-offline analyze --stage poc --industry semiconductor
```

The POC is hard-capped at 6 semiconductor tickers and roughly 900 calendar days. Its purpose is plumbing and signal validation, not a statistically conclusive backtest.

Repeat the `download` command to verify caching. A fully covered run reports `remote_fetches: 0` and makes no historical-data request. To deliberately refresh provider revisions in only the latest tail:

```bash
docker compose run --rm screener download --stage poc --industry semiconductor --refresh-tail 5
```

## Production-style job

```bash
docker compose run --rm screener download --stage prod --industry semiconductor
docker compose run --rm screener-offline analyze --stage prod --industry semiconductor
```

The production profile uses the full resolved universe, 15 years of requested history, stricter minimum history, more walk-forward folds, and a larger XGBoost model. The first run is intentionally much heavier; subsequent runs fetch only uncovered date ranges.

## Cache contract

Cache keys include provider, ticker, interval, and adjustment policy. Each ticker has:

- normalized OHLCV Parquet data;
- a JSON manifest with successful half-open coverage ranges (`start` inclusive, `end` exclusive), downloads, row count, and SHA-256;
- a per-key file lock and atomic replacement on successful writes.

An empty or failed provider response is never marked covered. `analyze` never downloads implicitly and fails if the cache does not meet row and universe-coverage gates. `--force` and `--refresh-tail N` are explicit exceptions to no-duplicate downloading.

## Industry, Nasdaq export, and Qlib universe selection

Project choices live in `config/industries.yaml`:

```bash
docker compose run --rm screener resolve --stage poc --industry semiconductor
docker compose run --rm screener resolve --stage prod --industry semiconductor_equipment
```

Pull and cache the full Nasdaq stock-screener export, inspect its industry taxonomy, and resolve every eligible common-equity/ordinary-share/ADR match for semiconductor:

```bash
docker compose run --rm screener refresh-universe --source nasdaq
docker compose run --rm screener list-industries
docker compose run --rm screener resolve --stage prod --industry nasdaq:semiconductor
```

The export snapshot is cached under `cache/universes/nasdaq/` with its retrieval time and checksums. A repeat within the 24-hour TTL is a cache hit; use `--force` only when you deliberately want a new export. Raw snapshots are retained for auditability. Nasdaq industry matching is a transparent case-insensitive slug substring, so `nasdaq:semiconductor` matches the Nasdaq label `Semiconductors`.

Run the full dynamic production universe with:

```bash
make prod-nasdaq
```

In the web console, click **Load Nasdaq universe**, then type or select `nasdaq:semiconductors`. POC keeps only the six largest matches by current Nasdaq market capitalization; production uses all eligible matches unless **Maximum tickers** is set. Warrants, rights, units, preferreds, notes, and bonds are excluded. Nasdaq's classifications are accepted as supplied and can contain broad or surprising members, so review the resolved list before production analysis.

Qlib does not provide one universal U.S. industry taxonomy; it exposes stock pools installed under a provider's `instruments/` directory. Discover and select those without installing Qlib into the base image:

```bash
screener list-industries --qlib-data-dir ~/.qlib/qlib_data/us_data
screener resolve --industry qlib:sp500 --qlib-data-dir ~/.qlib/qlib_data/us_data --stage poc
```

When containerized, mount that directory and pass its container path. The optional `pyqlib` extra is available for future Qlib-native feature/provider adapters; the primary raw-price cache remains independent because Qlib's ready-made U.S. snapshot is not an authoritative incremental feed.

Yahoo Finance is the default source. Google Finance has no supported official bulk historical market-data API, so `provider: google` fails explicitly instead of silently scraping an unstable page. A licensed provider can be added behind `MarketDataProvider`.

## Signal pipeline

- **Trend:** volatility-normalized 20- and 60-session equal-weight sector returns.
- **Breadth:** advancing fraction, fractions above moving averages, positive 5/20-day breadth, thrust, and dispersion.
- **Correlation/PCA:** rolling median pairwise correlation plus sign-anchored first principal component score and explained variance.
- **HMM:** causally refit Gaussian HMM; each timestamp uses only observations available through that timestamp. State IDs are mapped from training-only return summaries.
- **Relative strength:** separate downside-capture/fall-resistance and upside-capture/rise-strength ranks. They are intentionally not collapsed into one opaque score.
- **Rebound model:** XGBoost on correction-eligible dates, with an event label requiring the upside threshold before the adverse threshold over the configured horizon.
- **Validation:** purged expanding walk-forward folds with an embargo equal to the label horizon. Output includes PR-AUC, ROC-AUC, Brier score, precision, and recall when enough positive/negative events exist.
- **Alarm:** a correction watch requires drawdown plus multi-ticker correlation. A rebound trigger additionally requires the probability and breadth-thrust thresholds. Every close-based alert says it is actionable no earlier than the next session.

Each run writes `report.html`, `alert.json`, `validation.json`, `sector_features.csv`, `ticker_rankings.csv`, and `run_manifest.json` beneath `outputs/`. Dynamic-universe manifests record the Nasdaq snapshot ID, retrieval time, membership checksum, matched industry labels, full eligible count, and selected count.

## Configuration and overrides

Tune defaults in `config/base.yaml`, stage bounds in `config/poc.yaml` and `config/prod.yaml`, and universes in `config/industries.yaml`. Date overrides preserve the same cache:

```bash
docker compose run --rm screener download --stage poc --industry semiconductor --start 2023-01-01 --end 2025-01-01
docker compose run --rm screener-offline analyze --stage poc --industry semiconductor --start 2023-01-01 --end 2025-01-01
```

`end` is exclusive, matching `yfinance.download` behavior.

## Local development

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

On macOS, native XGBoost also needs `brew install libomp`. The container installs its Linux OpenMP runtime and is the canonical test/runtime path: `docker compose run --rm test`.

The default tests use deterministic synthetic/fake data and make no network calls. They verify cache hits and tail-only downloads, Qlib-pool resolution, causal feature construction, separate relative-strength ranks, and a complete offline analysis artifact set.

## Known research limitations

- The curated catalog is a current snapshot. Historical analysis can have survivorship bias unless supplied with point-in-time Qlib pool membership.
- Yahoo data can be revised and may contain omissions; use `--refresh-tail` deliberately and inspect cache/run manifests.
- High crash correlation and PC1 concentration identify a common move, not a bottom by themselves.
- HMM state labels and XGBoost probabilities require out-of-sample monitoring and periodic recalibration. An `insufficient_data` validation status is an abstention, not evidence for a signal.
- Transaction costs, next-session execution, index weights, liquidity, earnings/event gaps, and market-factor controls need to be added before using results for portfolio decisions.

References: [Qlib data API](https://qlib.readthedocs.io/en/stable/start/getdata.html), [Qlib data layer and stock pools](https://qlib.readthedocs.io/en/latest/component/data.html), [Qlib Yahoo collector notes](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo), [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), and [yfinance cache documentation](https://ranaroussi.github.io/yfinance/advanced/caching.html).
