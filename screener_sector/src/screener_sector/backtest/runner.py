"""Walk-forward backtest.

Exactly one parameter is fitted per fold: the alarm gate. Everything else is
held at its configured value. A single fitted dimension is what makes the
fold-to-fold stability table readable — a dozen jointly-fitted parameters
would produce a diagnostic nobody can interpret.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from screener_sector.backtest.evaluate import (
    edge_table,
    forward_return_samples,
    match_counts,
    pool_samples,
)
from screener_sector.backtest.labels import label_bottoms
from screener_sector.backtest.walkforward import Fold, expanding_folds
from screener_sector.config import Config
from screener_sector.data.store import PriceStore
from screener_sector.features.correlation import cluster_universe
from screener_sector.features.rebound import (
    WASHOUT_GATE,
    cluster_washout,
    confirmation,
    ticker_alarm,
)
from screener_sector.pipeline import load_frames

PER_FOLD_COLUMNS = [
    "fold",
    "partial",
    "test_start",
    "test_end",
    "tickers",
    "gate",
    "precision",
    "recall",
    "f1",
    "signals",
    "labels",
    "mean_lead_days",
]


@dataclass(frozen=True)
class BacktestResult:
    per_fold: pd.DataFrame
    economics: pd.DataFrame
    baseline: pd.DataFrame
    edges: pd.DataFrame
    fitted_gates: pd.DataFrame


def alarm_series(
    store: PriceStore,
    tickers: Sequence[str],
    config: Config,
    start: date,
    end: date,
    alarm_gate: float,
) -> dict[str, pd.Series]:
    """Boolean alarm signals per ticker across [start, end].

    Clusters are fitted once on data up to `start` and then held fixed for the
    window, so no signal inside the window depends on data from later in it.
    """
    fit_frames = load_frames(store, list(tickers) + [config.benchmark], start)
    benchmark_fit = fit_frames.pop(config.benchmark, None)
    if not fit_frames:
        return {}

    fit_panel = pd.DataFrame(
        {t: f["close"] for t, f in fit_frames.items()}
    ).sort_index()
    clusters = cluster_universe(
        fit_panel,
        benchmark_fit["close"] if benchmark_fit is not None else None,
        config.corr_threshold,
        config.min_cluster_size,
        config.windows.corr,
    )
    if not clusters.clusters:
        return {}

    full_frames = load_frames(store, list(tickers) + [config.benchmark], end)
    full_frames.pop(config.benchmark, None)
    full_panel = pd.DataFrame(
        {t: f["close"] for t, f in full_frames.items()}
    ).sort_index()

    out: dict[str, pd.Series] = {}
    window = (full_panel.index >= pd.Timestamp(start)) & (
        full_panel.index <= pd.Timestamp(end)
    )

    for cluster in clusters.clusters:
        members = [m for m in cluster.members if m in full_frames]
        if not members:
            continue
        washout = cluster_washout(full_panel, members, config.windows.mid)
        for ticker in members:
            frame = full_frames[ticker]
            alarm = ticker_alarm(frame, washout, config.rebound_weights, config.windows)
            confirmed = confirmation(frame, config.windows.short)
            fired = (
                (washout.reindex(frame.index).fillna(0.0) > WASHOUT_GATE)
                & (alarm > alarm_gate)
                & confirmed
            )
            out[ticker] = fired.reindex(full_panel.index).fillna(False)[window]
    return out


def _labels_for(
    store: PriceStore, ticker: str, config: Config, start: date, end: date
) -> pd.Series:
    frame = store.load(ticker)
    labels = label_bottoms(
        frame,
        config.backtest.label_k,
        config.backtest.label_forward_days,
        config.backtest.label_min_return,
    )
    mask = (labels.index >= pd.Timestamp(start)) & (labels.index <= pd.Timestamp(end))
    return labels[mask]


def _score(
    store: PriceStore,
    tickers: Sequence[str],
    config: Config,
    start: date,
    end: date,
    gate: float,
) -> tuple[float, float, float, int, int, float]:
    """Score a gate by pooling counts across tickers within the fold.

    Precision, recall, and f1 are computed from pooled counts, not averaged.
    mean_lead_days is averaged over matched signals.
    """
    signals = alarm_series(store, tickers, config, start, end, gate)

    # Pool counts across all tickers
    pooled_true_positives = 0
    pooled_signals = 0
    pooled_matched_labels = 0
    pooled_labels = 0
    all_leads: list[float] = []

    for ticker, series in signals.items():
        labels = _labels_for(store, ticker, config, start, end).reindex(
            series.index, fill_value=False
        )
        counts = match_counts(series, labels, tolerance_days=config.backtest.label_k)
        pooled_true_positives += counts.true_positives
        pooled_signals += counts.signals
        pooled_matched_labels += counts.matched_labels
        pooled_labels += counts.labels
        all_leads.extend(counts.leads)

    # Compute metrics from pooled counts
    if pooled_signals == 0 or pooled_labels == 0:
        precision = recall = f1 = 0.0
    else:
        precision = pooled_true_positives / pooled_signals
        recall = pooled_matched_labels / pooled_labels
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

    mean_lead = float(np.mean(all_leads)) if all_leads else float("nan")

    return (
        float(precision),
        float(recall),
        float(f1),
        pooled_signals,
        pooled_labels,
        mean_lead,
    )


def fit_alarm_gate(
    store: PriceStore,
    tickers: Sequence[str],
    config: Config,
    fold: Fold,
    candidates: Sequence[float],
) -> float:
    best_gate = float(candidates[0])
    best_f1 = -1.0
    for gate in candidates:
        _, _, f1, signals, _, _ = _score(
            store, tickers, config, fold.fit_start, fold.fit_end, float(gate)
        )
        if f1 > best_f1:
            best_f1, best_gate = f1, float(gate)

    # Verify fit window has data: if every candidate produced zero signals AND the
    # fit window has no data at all, something went wrong (likely config.start ==
    # fold.fit_start with insufficient warmup).
    first_score = _score(store, tickers, config, fold.fit_start, fold.fit_end, float(candidates[0]))
    if first_score[3] == 0:  # signals == 0 for first candidate
        # Load frames to check if fit window had any data
        fit_frames = load_frames(store, list(tickers), fold.fit_end)
        if not fit_frames:
            raise RuntimeError(
                f"Fold {fold.index}: fit window [{fold.fit_start}, {fold.fit_end}] "
                f"has no data. Fit data may be missing or cache not yet populated."
            )

    return best_gate


DEFAULT_GATES = (40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0)


def _generate_baseline_samples(
    close: pd.Series,
    n_signals: int,
    horizons: Sequence[int],
    seed: int,
    draws: int,
    eligible: pd.DatetimeIndex,
) -> dict[int, list[float]]:
    """Generate raw baseline samples for pooling (used internally by run_backtest)."""
    from screener_sector.backtest.labels import forward_return

    samples: dict[int, list[float]] = {h: [] for h in horizons}
    rng = np.random.default_rng(seed)

    # Compute the cutoff: exclude dates without room for the longest horizon
    cutoff_index = max(len(close) - max(horizons), 1)
    default_eligible = close.index[:cutoff_index]

    # Intersect with provided eligible window
    eligible_set = set(eligible)
    final_eligible = pd.DatetimeIndex([d for d in default_eligible if d in eligible_set])

    if len(final_eligible) == 0:
        return samples

    for _ in range(draws):
        size = min(n_signals, len(final_eligible))
        picks = rng.choice(len(final_eligible), size=size, replace=False)
        dates = [final_eligible[int(p)] for p in picks]

        for horizon in horizons:
            forward = forward_return(close, horizon)
            values = forward.reindex(pd.DatetimeIndex(dates)).dropna()
            samples[horizon].extend(values.tolist())

    return samples


def run_fold(
    store: PriceStore, tickers: Sequence[str], config: Config, fold: Fold
) -> pd.DataFrame:
    gate = fit_alarm_gate(store, tickers, config, fold, DEFAULT_GATES)
    precision, recall, f1, signals, labels, lead = _score(
        store, tickers, config, fold.test_start, fold.test_end, gate
    )

    # Count tickers with data in the test window, excluding benchmark (F6)
    test_tickers_count = 0
    test_start_ts = pd.Timestamp(fold.test_start)
    test_end_ts = pd.Timestamp(fold.test_end)
    for ticker in tickers:
        if ticker == config.benchmark:
            continue
        if not store.has(ticker):
            continue
        frame = store.load(ticker)
        mask = (frame.index >= test_start_ts) & (frame.index <= test_end_ts)
        if mask.any():
            test_tickers_count += 1

    return pd.DataFrame(
        [
            {
                "fold": fold.index,
                "partial": fold.partial,
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "tickers": test_tickers_count,
                "gate": gate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "signals": signals,
                "labels": labels,
                "mean_lead_days": lead,
            }
        ],
        columns=PER_FOLD_COLUMNS,
    )


def run_backtest(
    store: PriceStore, tickers: Sequence[str], config: Config, end: date
) -> BacktestResult:
    folds = expanding_folds(
        config.start,
        end,
        config.backtest.initial_fit_years,
        config.backtest.step_years,
    )

    fold_rows: list[pd.DataFrame] = []
    signal_samples: dict[int, list[float]] = {h: [] for h in config.backtest.horizons}
    baseline_samples: dict[int, list[float]] = {h: [] for h in config.backtest.horizons}

    for fold in folds:
        row = run_fold(store, tickers, config, fold)
        fold_rows.append(row)

        gate = float(row["gate"].iloc[0])
        signals = alarm_series(
            store, tickers, config, fold.test_start, fold.test_end, gate
        )

        # Extract eligible dates for this fold's test window
        test_window_start = pd.Timestamp(fold.test_start)
        test_window_end = pd.Timestamp(fold.test_end)

        for ticker, series in signals.items():
            dates = list(series[series].index)
            if not dates:
                continue
            close = store.load(ticker)["close"]

            # Compute per-ticker seed deterministically (F3)
            seed = (fold.index * 1_000_003 + zlib.crc32(ticker.encode())) % (2**32)

            # Eligible dates are those in the fold's test window
            eligible_mask = (close.index >= test_window_start) & (close.index <= test_window_end)
            eligible_dates = close.index[eligible_mask]

            # Collect raw samples for pooled aggregation (F4)
            signal_samples_ticker = forward_return_samples(
                close, dates, config.backtest.horizons
            )
            for horizon in config.backtest.horizons:
                signal_samples[horizon].extend(signal_samples_ticker[horizon])

            # Collect baseline samples via regenerated raw draws
            baseline_samples_ticker = _generate_baseline_samples(
                close, len(dates), config.backtest.horizons, seed, 50, eligible_dates
            )
            for horizon in config.backtest.horizons:
                baseline_samples[horizon].extend(baseline_samples_ticker[horizon])

    per_fold = (
        pd.concat(fold_rows, ignore_index=True)
        if fold_rows
        else pd.DataFrame(columns=PER_FOLD_COLUMNS)
    )

    # Pool samples for final aggregation (F4)
    economics = pool_samples(signal_samples)
    baseline = pool_samples(baseline_samples)

    edges = (
        edge_table(economics, baseline)
        if not economics.empty and not baseline.empty
        else pd.DataFrame(columns=["horizon", "mean_edge", "hit_rate_edge"])
    )
    fitted_gates = (
        per_fold[["fold", "gate"]].copy()
        if not per_fold.empty
        else pd.DataFrame(columns=["fold", "gate"])
    )
    return BacktestResult(per_fold, economics, baseline, edges, fitted_gates)
