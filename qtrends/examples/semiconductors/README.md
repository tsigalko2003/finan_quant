# End-to-end Semiconductor example

This example demonstrates the complete containerized qtrends workflow with deterministic inputs. It is deliberately small enough to inspect while exercising the same universe and modeling code used by the live configuration.

## Run it

From the repository root:

```bash
make example
```

For a command-by-command walkthrough, live-market instructions, configuration examples, and output interpretation, see [`../../docs/USAGE.md`](../../docs/USAGE.md).

The runner performs four stages:

1. Generate 1,250 business days of deterministic prices for `ALPHA`, `BETA`, `GAMMA`, `DELTA`, `EPSILON`, and the `MARKET` benchmark.
2. Synchronize `nasdaq_snapshot_1.csv` on `2020-01-02`. Exact industry and sector matching selects four semiconductor common stocks and rejects the unrelated software row.
3. Synchronize `nasdaq_snapshot_2.csv` on `2021-07-01`. `EPSILON` is added with that discovery date; its warrant is excluded by the security-name filter.
4. Reuse the persisted manifest to run multi-ticker breadth, rolling PCA, walk-forward HMM, and walk-forward XGBoost, then validate every output.

## Data flow

```text
Nasdaq-style exports
        |
        v
industry/sector/market-cap/name filters
        |
        v
incremental membership manifest
        |
        +---- effective_from masking ----+
        |                                |
        v                                v
deterministic daily prices       active ticker group
        |                                |
        +---------------+----------------+
                        v
        breadth + residual returns + rolling PCA
                        |
                        v
             walk-forward Gaussian HMM
                        |
                        v
               walk-forward XGBoost
                        |
                        v
       signal, evaluation, models, and audit files
```

## Files to inspect

- `config.yaml`: complete example configuration.
- `nasdaq_snapshot_1.csv`: initial Screener-style export.
- `nasdaq_snapshot_2.csv`: later export with an incremental addition.
- `run.sh`: complete orchestration.
- `inspect_results.py`: deterministic artifact and membership validation.
- `data/universes/example_semiconductors.csv`: generated membership history.
- `outputs/example_semiconductors/summary.md`: first-read result.
- `outputs/example_semiconductors/signals.csv`: full feature/probability/signal history.
- `outputs/example_semiconductors/resolved_config.json`: exact resolved configuration and ticker list.

## What the result means

The composite signal is a research classification for the group, not a recommendation for any constituent. Historical model quality must be judged from the walk-forward fields, especially `roc_auc`, `brier_score`, and the conditional forward-return statistics. A strong latest probability does not compensate for weak out-of-sample evaluation.

This example uses synthetic prices and fictional tickers, so its numerical signal has no market meaning. Its purpose is to prove orchestration, incremental membership, point-in-time masking, leakage controls, and artifact generation. The live configuration replaces these inputs with the current Nasdaq Screener universe and Yahoo market data.
