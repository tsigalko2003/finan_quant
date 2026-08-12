# Residual Risks and Known Weaknesses

Recorded 2026-08-12 at the end of the initial 21-task build. Everything here
survived the final whole-branch review as a *known* issue rather than a fixed
one. Read this before trusting a prod number.

## Fixed during final review — context worth keeping

**The walk-forward once fitted nothing.** Every fold's fit window began on the
same date as the earliest cached bar, so it contained zero rows: no clusters
formed, every candidate gate scored F1 = 0, and `fit_alarm_gate` silently
returned the first candidate. `fitted_gates.csv` — the file the README tells
you to read first to detect overfitting — was a constant column reporting
perfect stability by construction. Fixed by separating the data floor from the
fold floor (`Config.fetch_start` = `start` minus `warmup_years`). Prod now
fetches from 2006 and runs folds from 2010.

The lesson generalizes: a diagnostic that cannot fail is worse than no
diagnostic. If `fitted_gates.csv` ever again shows a single repeated value
across every fold, suspect the fit window before believing the stability.

## Live limitations

**Forward-looking universe selection.** The universe is screened on *today's*
liquidity and price, then that same list is used for every historical fold. A
2011 fold therefore runs on companies chosen for what they became. This is
distinct from the delisting survivorship also disclosed in the report, and the
lookahead integration test cannot catch it because that test passes a static
ticker list and never exercises universe construction. Fixing it properly means
making universe construction date-parameterized so each fold screens the
universe as it actually looked then.

**Overlapping forward returns are counted as independent.** Consecutive alarms
a few bars apart produce heavily overlapping 20-day forward returns, all counted
separately in `n` and `hit_rate`, while the random baseline's draws are
well separated. Effective sample size is far below reported `n`, and there is
no significance test anywhere in the codebase. Treat a small positive
`mean_edge` as suggestive, not established.

**Pooled economics are dominated by late folds.** Metrics pool across folds by
true observation count, which is correct for a total-`n` statistic but means the
headline edge mostly describes recent years, when the universe is fattest. The
per-fold table is the only place that asymmetry is visible — which is why the
per-fold ticker count matters more than its one-line implementation suggests.

**Gate ties break toward firing more.** `fit_alarm_gate` maximizes pooled F1
with a strict `>`, so when candidates tie it keeps the first, which is the
loosest gate. On data where no candidate separates, the backtest silently
evaluates the most permissive threshold.

**Baseline sample size is not what it looks like.** `baseline.csv` reports `n`
pooled across all 50 draws per ticker, roughly 50x the signal-side `n`. Means
and hit rates are correct; the `n` column is not comparable side by side with
the signal file.

## Things that are genuinely verified

- No module under `features/` or `pipeline.py` reads past the evaluation bar.
  `backtest/labels.py` is the sole sanctioned exception. The integration test
  proving this was confirmed non-vacuous by deliberately sabotaging the
  truncation in `pipeline.py` and observing the test fail.
- Cached data is never re-downloaded: refresh fills only missing leading and
  trailing gaps, a later start date cannot truncate history, a failed fetch
  preserves the existing cache, and enrichment never re-requests a cached
  ticker.
- `screen`, `backtest`, and `report` make zero network calls, enforced by a test
  that swaps in a fetcher raising on any request.
- The data directory is relocatable: copying it to a different absolute path
  produces byte-identical output with an empty fetcher.

## Reading a backtest result

1. Open `fitted_gates.csv` first. A gate that swings widely across folds means
   the alarm is fitting noise; a gate that never moves means either genuine
   stability or a broken fit window — check the per-fold `signals` column to
   tell those apart.
2. Then `edges.csv`. Absolute hit rate is meaningless in a sector that rose over
   most of the sample; only `mean_edge` and `hit_rate_edge` versus random entry
   carry information.
3. Then the per-fold `tickers` column, to see how thin the early folds are.
4. A small positive edge is a good result. A large one on the first run should
   be treated as a suspected bug until explained.
