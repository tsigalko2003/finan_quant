# How to use the qtrends pipeline

This guide shows three ways to use qtrends:

1. run the deterministic end-to-end example to learn the workflow;
2. run the live Nasdaq Semiconductor universe; or
3. create a new industry or fixed-ticker configuration.

All commands below run from the repository root:

```bash
cd /Users/taosong/Projects/qtrends
```

## Prerequisites

You need Docker Desktop, including Docker Compose. Confirm that Docker is running:

```bash
docker version
docker compose version
```

## 1. Run the complete deterministic example

The fastest way to understand the system is:

```bash
make example
```

This single command:

1. builds `qtrends:local`;
2. creates deterministic prices for five fictional semiconductor tickers and one benchmark;
3. imports a first Nasdaq-style industry snapshot with four tickers;
4. imports a second snapshot that adds `EPSILON` on `2021-07-01`;
5. records the incremental membership change;
6. masks `EPSILON` before its discovery date;
7. builds breadth and rolling-PCA features;
8. runs the walk-forward HMM and XGBoost models; and
9. validates the generated manifest, signal, metrics, and model files.

Expected final output looks like:

```text
Validated end-to-end example
  Active universe: ALPHA, BETA, DELTA, EPSILON, GAMMA
  Incremental addition: EPSILON (effective 2021-07-01)
  Membership masking: 4 constituents before discovery, 5 afterward
  Latest observation: 2022-10-17
  Composite signal: neutral
  Walk-forward observations: 547
```

The exact deterministic example is implemented in:

```text
examples/semiconductors/config.yaml
examples/semiconductors/nasdaq_snapshot_1.csv
examples/semiconductors/nasdaq_snapshot_2.csv
examples/semiconductors/run.sh
examples/semiconductors/inspect_results.py
```

### Run the same example one step at a time

Build the image and create market data:

```bash
docker compose build

docker compose run --rm qtrends generate-sample \
  --output data/example_semiconductors_prices.csv \
  --periods 1250 \
  --seed 42
```

Seed the first industry snapshot:

```bash
rm -f data/universes/example_semiconductors.csv
cp examples/semiconductors/nasdaq_snapshot_1.csv \
  data/example_nasdaq_semiconductors.csv

docker compose run --rm qtrends sync-universe \
  --config examples/semiconductors/config.yaml \
  --snapshot-date 2020-01-02
```

Apply the second snapshot containing the new ticker:

```bash
cp examples/semiconductors/nasdaq_snapshot_2.csv \
  data/example_nasdaq_semiconductors.csv

docker compose run --rm qtrends sync-universe \
  --config examples/semiconductors/config.yaml \
  --snapshot-date 2021-07-01
```

Run the model using the persisted membership:

```bash
docker compose run --rm qtrends run \
  --config examples/semiconductors/config.yaml \
  --output-dir outputs/example_semiconductors \
  --no-universe-refresh
```

Validate the result:

```bash
docker compose run --rm --entrypoint python qtrends \
  examples/semiconductors/inspect_results.py
```

## 2. Run the live Semiconductor pipeline

The live configuration uses:

- the Nasdaq Screener for current `Semiconductors` membership;
- a minimum market capitalization of $5 billion;
- security-name filtering for warrants, rights, units, and preferred shares;
- Yahoo adjusted daily market data;
- `SOXX` as the benchmark; and
- history beginning in 2015.

Build once:

```bash
make build
```

Refresh and inspect the ticker universe:

```bash
make universe

sed -n '1,20p' data/universes/semiconductors.csv
```

Run the complete live pipeline:

```bash
make live
```

`make live` refreshes the universe again because `refresh_on_run` is enabled. To freeze membership at the already-saved manifest and only rerun the models:

```bash
docker compose run --rm qtrends run \
  --config configs/semiconductors.yaml \
  --output-dir outputs/semiconductors \
  --no-universe-refresh
```

Read the first-level results:

```bash
cat outputs/semiconductors/summary.md
cat outputs/semiconductors/latest_signal.json
cat outputs/semiconductors/metrics.json
```

## 3. Create a configuration for another industry

Copy the live configuration. In the Nasdaq Screener, select the desired sector/industry and use the industry text from its downloaded export exactly.

```bash
cp configs/semiconductors.yaml configs/software_infrastructure.yaml
```

Edit these fields in `configs/software_infrastructure.yaml`:

```yaml
data:
  provider: yahoo
  tickers: []
  benchmark: IGV
  start: "2015-01-01"

universe:
  source: nasdaq_screener
  industry: REPLACE_WITH_EXACT_NASDAQ_INDUSTRY
  sector: Technology
  export_path: null
  manifest_path: data/universes/software_infrastructure.csv
  refresh_on_run: true
  min_market_cap: 5000000000
  min_snapshot_retention: 0.80
```

Use a unique manifest and output directory for every industry. Do not point a new industry at `data/universes/semiconductors.csv`.

Validate, sync, and run:

```bash
docker compose run --rm qtrends validate-config \
  --config configs/software_infrastructure.yaml

docker compose run --rm qtrends sync-universe \
  --config configs/software_infrastructure.yaml

docker compose run --rm qtrends run \
  --config configs/software_infrastructure.yaml \
  --output-dir outputs/software_infrastructure
```

If you download the Screener CSV manually, place it under `data/` and replace the live API setting:

```yaml
universe:
  export_path: data/nasdaq_screener.csv
```

## 4. Use a fixed ticker group instead of an industry

Remove the entire `universe:` section and list at least two group tickers:

```yaml
data:
  provider: yahoo
  tickers: [NVDA, AMD, AVGO, MU, MRVL]
  benchmark: SOXX
  start: "2015-01-01"
  end: null
```

Then run:

```bash
docker compose run --rm qtrends run \
  --config configs/my_fixed_group.yaml \
  --output-dir outputs/my_fixed_group
```

This fixed-list mode does not create incremental membership history. Use the industry manifest mode when additions and removals must be recorded over time.

## 5. Understand the outputs

Every model run creates the following files in its output directory:

| File | Purpose |
| --- | --- |
| `summary.md` | First-read signal and walk-forward evaluation |
| `latest_signal.json` | Machine-readable latest signal and component values |
| `signals.csv` | Daily features, probabilities, regimes, targets, and signals |
| `metrics.json` | Out-of-sample evaluation metrics |
| `feature_importance.csv` | Final XGBoost feature importance |
| `resolved_config.json` | Exact ticker list and settings used for the run |
| `hmm_model.joblib` | Final fitted HMM bundle |
| `xgboost_model.joblib` | Final fitted XGBoost model |

Important interpretation fields:

- `signal`: combined `bullish`, `bearish`, or `neutral` classification;
- `xgb_probability`: final-model probability used only for the latest signal;
- `xgb_probability_oos`: walk-forward probability used for historical evaluation;
- `hmm_*_probability`: filtered probability of each market regime;
- `breadth_above_ma20`: fraction of available active constituents above their 20-day moving average;
- `pca_explained_variance`: fraction of cross-sectional variation explained by the first rolling PCA factor;
- `constituent_coverage`: fraction of active constituents with usable observations; and
- `roc_auc` and `brier_score`: out-of-sample discrimination and probability-calibration measures.

Do not interpret the latest signal in isolation. If walk-forward ROC AUC, Brier score, conditional returns, or coverage are weak, treat the signal as an unvalidated research flag.

## 6. Normal operating sequence

For an established industry configuration, the regular workflow is:

```bash
# 1. Refresh and audit membership
docker compose run --rm qtrends sync-universe --config configs/semiconductors.yaml

# 2. Run without a second membership refresh
docker compose run --rm qtrends run \
  --config configs/semiconductors.yaml \
  --output-dir outputs/semiconductors \
  --no-universe-refresh

# 3. Review the signal and evaluation
cat outputs/semiconductors/summary.md

# 4. Run the regression tests after code/config changes
make test
```

The membership manifest is durable state. Back it up, review additions and deactivations, and do not delete it during routine runs.

## Common problems

### Docker is not running

If `docker compose` cannot connect to the daemon, start Docker Desktop and rerun the command.

### Nasdaq snapshot rejected

The pipeline refuses a snapshot that retains less than `min_snapshot_retention` of the previous active universe. This protects the manifest from a partial or malformed response. Inspect the export/API response before changing the threshold.

### A Yahoo ticker has no data

Review the active manifest for renamed, delisted, foreign, or unsupported symbols. Either correct the symbol mapping or use CSV/Qlib market data with the required history.

### The model produces a neutral signal

Neutral is a valid result. It means the XGBoost probability, HMM regime, breadth, and relative-return conditions did not all satisfy either the bullish or bearish rule.

## Research limitations

Nasdaq Screener is a current universe snapshot, not a historical constituent database. The first bootstrap therefore has survivorship and classification bias. Incremental changes become point-in-time only after monitoring begins. Yahoo is also a convenient research source rather than an institutional historical market-data source. Use licensed point-in-time constituents and validated price data before relying on the results for investment decisions.
