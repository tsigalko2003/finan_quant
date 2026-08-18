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
docker compose run --rm screener rebound-leaders --profile dev
```

## Profiles

`dev` (default) runs 2022→now over ~30 checked-in tickers in about two minutes.
It exists to debug the methodology. **Dev results are not evidence that the
method works** — the window covers roughly one and a half drawdown cycles.

`prod` runs 2006→now over the full discovered universe with 12 walk-forward
folds. `build-universe` in prod enriches ~400–800 candidate symbols rather than
all ~8,000 listed to avoid Yahoo rate-limiting. Discovery selects symbols whose
name matches a theme keyword (e.g., "semiconductor") or ticker is in a curated
seed list; this covers all major players and companies with opaque names like
Coherent and Lumentum. It takes hours; it is resumable and cached.

## Relocating the data directory

`data/` is self-contained: no absolute paths, no external references. To move it:

```bash
cp -r data /Volumes/external/screener-data
# then in .env:
DATA_DIR=/Volumes/external/screener-data
```

Price files always hold maximum available history regardless of profile, so a
cache built under `dev` is a valid starting point for `prod`.

## Rebound leadership

The `rebound-leaders` command identifies which tickers in each cluster bounce
back strongest from shared drawdowns. It measures leadership per cluster by:

1. Finding every date the cluster's equal-weight index bottomed (a local minimum
   with a forward return exceeding the configured threshold).
2. Computing each member's forward return at 5, 10, and 20 days from each trough.
3. Ranking members on **rebound ratio** (their 20-day return ÷ cluster median
   20-day return) and **consistency** (fraction of troughs where they beat the
   cluster median). A ratio > 1.0 means the ticker recovers harder than its
   peers; consistency shows how often it wins at each trough.
4. Computing **recovery efficiency** (20-day return ÷ drawdown depth) to normalize
   for how far each name fell. This prevents the ranking from being dominated by
   names that simply fell furthest.

Importantly, every member is measured from the **group's trough date**, not its
own individual low. This matters because in live trading, you see the cluster's
turn in real time; measuring each name from its personal optimal low flatters
everything and is not actionable.

Unlike **relative strength** (which measures up/down capture across all bars),
rebound leadership focuses only on the critical post-trough period when recovery
behavior is most informative. A ticker with high rebound ratio and high
consistency is a strong candidate to lead the cluster out of the next drawdown.

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
