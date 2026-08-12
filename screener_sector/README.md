# screener_sector

Semiconductor / AI / optical equity screener: trend scoring, correlation
clustering, relative strength, and rebound alarms, validated walk-forward.

## Quick start

```bash
cp .env.example .env
docker compose build
docker compose run --rm screener info
```

## Commands

All commands take `--profile dev|prod` and `--as-of YYYY-MM-DD`.

```bash
docker compose run --rm screener build-universe --profile prod
docker compose run --rm screener fetch --profile dev
docker compose run --rm screener screen --profile dev
docker compose run --rm screener backtest --profile dev
docker compose run --rm screener report --profile dev
```

## Profiles

`dev` (default) runs 2022→now over ~30 checked-in tickers in about two minutes.
It exists to debug the methodology. **Dev results are not evidence that the
method works** — the window covers roughly one and a half drawdown cycles.

`prod` runs 2006→now over the full discovered universe with 12 walk-forward
folds. `build-universe` in prod takes hours; it is resumable and cached.

## Relocating the data directory

`data/` is self-contained: no absolute paths, no external references. To move it:

```bash
cp -r data /Volumes/external/screener-data
# then in .env:
DATA_DIR=/Volumes/external/screener-data
```

Price files always hold maximum available history regardless of profile, so a
cache built under `dev` is a valid starting point for `prod`.

## Known limitations

- **Survivorship bias.** The universe comes from currently listed securities;
  delisted companies are absent, which inflates backtest results, more so the
  further back the test runs.
- **Forward-looking selection bias.** The universe is also screened on current
  liquidity and price; historical folds run on companies selected for what they
  became, not what they were — an additional bias beyond the missing delistings.
- **Restated history.** Yahoo revises adjusted prices, so backtests are not
  bit-reproducible across re-fetches.
- **Unofficial data source.** `yfinance` uses an unsupported endpoint that can
  change or rate-limit without notice.
- Not investment advice.
