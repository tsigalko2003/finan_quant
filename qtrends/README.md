# qtrends

`qtrends` is a containerized, leakage-aware research pipeline for identifying a common trend in a predefined group of public-equity tickers. It combines:

1. multi-ticker market breadth;
2. market-beta residual returns;
3. rolling PCA for common-factor strength and group coherence;
4. walk-forward Gaussian HMM filtering for bear/neutral/bull regimes; and
5. walk-forward XGBoost classification of positive forward group excess returns.

The project uses Microsoft Qlib as a native market-data provider when a Qlib dataset is available. Because Qlib's official downloadable dataset is currently disabled, Yahoo and long-form CSV adapters are also included so arbitrary ticker groups work immediately.

The Docker build installs Qlib `v0.9.7` from Microsoft's tagged source archive. This also supports ARM64 systems where PyPI does not publish a compatible `pyqlib` wheel.

> This is research software, not an investment recommendation or execution system.

For a copy-paste walkthrough of the complete pipeline, see [`docs/USAGE.md`](docs/USAGE.md).

## Quick start

Run the deterministic offline demo:

```bash
cd /Users/taosong/Projects/qtrends
make build
make demo
```

The demo generates `data/sample_prices.csv`, trains the models, and writes artifacts to `outputs/demo/`.

Run tests:

```bash
make test
```

## Complete incremental-universe example

Run the complete, deterministic example:

```bash
cd /Users/taosong/Projects/qtrends
make example
```

This example exercises the whole solution without depending on a live market-data endpoint:

1. builds the Docker image;
2. generates deterministic multi-ticker daily prices;
3. loads an initial Nasdaq-style `Semiconductors` export containing four tickers;
4. loads a later export that adds a fifth ticker, `EPSILON`;
5. persists the incremental membership manifest and gives `EPSILON` its discovery date as `effective_from`;
6. runs breadth, rolling PCA, walk-forward HMM, and walk-forward XGBoost in the container; and
7. validates and prints the final artifacts.

The walkthrough, inputs, configuration, and result interpretation are under `examples/semiconductors/`. Generated example state is isolated under `data/example_*`, `data/universes/example_semiconductors.csv`, and `outputs/example_semiconductors/`.

## Analyze a real ticker group

`configs/semiconductors.yaml` starts from the Nasdaq Screener industry `Semiconductors` rather than a manually maintained ticker list. First inspect or refresh the universe:

```bash
make universe
```

Then run the complete model:

```bash
make live
```

Or invoke the container directly:

```bash
docker compose run --rm qtrends run \
  --config configs/semiconductors.yaml \
  --output-dir outputs/semiconductors
```

The Yahoo adapter is convenient but is not a point-in-time institutional data source. For defensible historical tests, use a survivorship-bias-free Qlib dataset or a CSV export containing historical group membership.

## Nasdaq industry universe

The universe configuration supports Nasdaq's current Screener snapshot or a CSV downloaded from the [Nasdaq Stock Screener](https://www.nasdaq.com/market-activity/stocks/screener):

```yaml
universe:
  source: nasdaq_screener
  industry: Semiconductors
  sector: Technology
  export_path: null
  manifest_path: data/universes/semiconductors.csv
  refresh_on_run: true
  min_market_cap: 5000000000
  max_symbols: null
  min_snapshot_retention: 0.80
  countries: []
  exclude_name_contains: [Warrant, Right, Unit, Preferred]
  initial_effective_from: "2015-01-01"
```

With `export_path: null`, `qtrends` retrieves Nasdaq's live Screener export. To use a file downloaded manually, mount it under `data/` and set, for example:

```yaml
export_path: data/nasdaq_screener.csv
```

The match on `industry` and optional `sector` is case-insensitive but exact. Market-cap, country, security-name, and maximum-universe filters are applied after normalization.

### Incremental membership behavior

Membership is persisted in `data/universes/semiconductors.csv`:

- the first sync seeds the group and uses the later of `initial_effective_from` and January 1 of the reported IPO year;
- a ticker appearing in a later Screener export is added with `effective_from` equal to that discovery date;
- a ticker missing from a later snapshot becomes inactive but remains in the manifest for auditability;
- a snapshot that retains less than `min_snapshot_retention` of the previous active universe is rejected without modifying the manifest;
- `run` uses active tickers and masks each ticker before its recorded `effective_from`;
- `run --no-universe-refresh` reuses the existing manifest without contacting Nasdaq.

Nasdaq Screener is a current snapshot, not historical industry-membership data. The initial bootstrap can therefore still contain survivorship/classification bias. After monitoring begins, incremental additions are handled point-in-time. Use a licensed historical constituent source when full historical membership accuracy is required.

Useful commands:

```bash
# Only refresh membership
docker compose run --rm qtrends sync-universe \
  --config configs/semiconductors.yaml

# Model using the existing manifest without a refresh
docker compose run --rm qtrends run \
  --config configs/semiconductors.yaml \
  --output-dir outputs/semiconductors \
  --no-universe-refresh
```

## Native Qlib provider

Place an existing US Qlib binary dataset under `data/qlib/us_data`, then run:

```bash
docker compose run --rm qtrends run \
  --config configs/qlib_provider.yaml \
  --output-dir outputs/qlib-run
```

The adapter initializes Qlib with `qlib.init(provider_uri=..., region=REG_US)` and retrieves `$close` and `$volume` through `qlib.data.D.features`.

## CSV schema

CSV inputs are long-form and require these columns:

```text
date,ticker,close,volume
2024-01-02,NVDA,48.16,411254000
2024-01-02,SOXX,576.95,4321000
```

Prices should be consistently adjusted for splits and distributions. Ticker membership should be point-in-time when the group changes historically.

## Outputs

Each run produces:

- `summary.md`: first-read result and evaluation summary;
- `latest_signal.json`: latest live regime and probability;
- `signals.csv`: features, HMM probabilities, live/OOS XGBoost probabilities, and signals;
- `metrics.json`: walk-forward evaluation metrics;
- `feature_importance.csv`: final XGBoost feature importances;
- `resolved_config.json`: exact run configuration;
- `hmm_model.joblib` and `xgboost_model.joblib`: fitted research models.

`xgb_probability_oos` is used for historical evaluation. `xgb_probability` is produced by the final model trained on all currently label-observable rows and is used for the latest signal. Keeping these separate prevents in-sample probabilities from being reported as backtest results.

## Modeling and leakage controls

- Rolling beta, breadth, correlation, and PCA features use data available through date `t` only.
- PCA is refit separately for every rolling window; it is never fit on the full history.
- HMM probabilities are forward-filtered. Test-period observations are not backward-smoothed using later test data.
- HMM and XGBoost models are refit on expanding windows.
- XGBoost training uses a forecast-horizon embargo so labels overlapping the prediction block are excluded.
- Model evaluation uses only `xgb_probability_oos`.

The default label is whether the equal-weighted group residual return is positive over the next 20 trading days. Residual returns remove a rolling-beta estimate of the configured benchmark.

## Signal rule

A bullish result requires all of:

- XGBoost probability at or above the configured bullish threshold;
- HMM bull probability at or above the configured HMM threshold;
- group breadth above the short moving average at or above the configured breadth threshold; and
- positive medium-horizon relative return.

The bearish rule is symmetric. Anything else is neutral. Thresholds are research parameters and should be validated without repeatedly optimizing against the same history.
