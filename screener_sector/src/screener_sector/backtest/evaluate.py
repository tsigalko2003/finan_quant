"""Scoring the alarm as a classifier and as an entry rule.

Absolute hit rates are uninformative in a market that rose over the sample:
buying at random would also have 'worked'. The random baseline is what turns a
number into evidence of edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener_sector.backtest.labels import forward_return

ECONOMIC_COLUMNS = ["horizon", "n", "mean_return", "median_return", "hit_rate"]


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    signals: int
    labels: int
    mean_lead_days: float


@dataclass(frozen=True)
class MatchCounts:
    """Raw counts for pooling across tickers."""
    true_positives: int
    signals: int
    matched_labels: int
    labels: int
    leads: tuple[float, ...]


def match_counts(
    signals: pd.Series, labels: pd.Series, tolerance_days: int
) -> MatchCounts:
    """Raw counts for pooling across tickers.

    Returns MatchCounts with true_positives, signals, matched_labels, labels, and leads.
    This function contains the matching logic; classification_metrics delegates here.
    """
    signal_positions = np.flatnonzero(signals.to_numpy(dtype=bool))
    label_positions = np.flatnonzero(labels.to_numpy(dtype=bool))

    if signal_positions.size == 0 or label_positions.size == 0:
        return MatchCounts(
            true_positives=0,
            signals=int(signal_positions.size),
            matched_labels=0,
            labels=int(label_positions.size),
            leads=(),
        )

    matched_labels: set[int] = set()
    true_positives = 0
    leads: list[float] = []

    for position in signal_positions:
        distances = np.abs(label_positions - position)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tolerance_days:
            true_positives += 1
            matched_labels.add(int(label_positions[nearest]))
            leads.append(float(label_positions[nearest] - position))

    return MatchCounts(
        true_positives=true_positives,
        signals=int(signal_positions.size),
        matched_labels=len(matched_labels),
        labels=int(label_positions.size),
        leads=tuple(leads),
    )


def classification_metrics(
    signals: pd.Series, labels: pd.Series, tolerance_days: int
) -> ClassificationMetrics:
    """Match signals to labels within +/- tolerance_days bars.

    Exact-day matching would score a signal one bar early as a complete miss,
    which does not reflect how the alarm is used.
    """
    counts = match_counts(signals, labels, tolerance_days)

    if counts.signals == 0 or counts.labels == 0:
        return ClassificationMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            signals=counts.signals,
            labels=counts.labels,
            mean_lead_days=float("nan"),
        )

    precision = counts.true_positives / counts.signals
    recall = counts.matched_labels / counts.labels
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        signals=counts.signals,
        labels=counts.labels,
        mean_lead_days=float(np.mean(counts.leads)) if counts.leads else float("nan"),
    )


def forward_return_samples(
    close: pd.Series, signal_dates: Sequence[pd.Timestamp], horizons: Sequence[int]
) -> dict[int, list[float]]:
    """Raw forward returns per horizon, NaNs dropped, for pooling across tickers.

    Returns a dict mapping horizon -> list of forward returns (empty list if none).
    """
    samples: dict[int, list[float]] = {h: [] for h in horizons}
    for horizon in horizons:
        forward = forward_return(close, horizon)
        values = forward.reindex(pd.DatetimeIndex(signal_dates)).dropna()
        samples[horizon] = values.tolist()
    return samples


def pool_samples(samples: dict[int, list[float]]) -> pd.DataFrame:
    """Aggregate pooled raw samples into ECONOMIC_COLUMNS.

    Takes a dict mapping horizon -> list[float] (from multiple tickers/folds)
    and computes n, mean_return, median_return, hit_rate across all observations.
    """
    rows = []
    for horizon in sorted(samples.keys()):
        values = samples[horizon]
        if not values:
            rows.append(
                {
                    "horizon": horizon,
                    "n": 0,
                    "mean_return": float("nan"),
                    "median_return": float("nan"),
                    "hit_rate": float("nan"),
                }
            )
        else:
            arr = np.array(values)
            rows.append(
                {
                    "horizon": horizon,
                    "n": int(len(values)),
                    "mean_return": float(arr.mean()),
                    "median_return": float(np.median(arr)),
                    "hit_rate": float((arr > 0).mean()),
                }
            )
    return pd.DataFrame(rows, columns=ECONOMIC_COLUMNS)


def economic_metrics(
    close: pd.Series, signal_dates: Sequence[pd.Timestamp], horizons: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        forward = forward_return(close, horizon)
        values = forward.reindex(pd.DatetimeIndex(signal_dates)).dropna()
        rows.append(
            {
                "horizon": horizon,
                "n": int(len(values)),
                "mean_return": float(values.mean()) if len(values) else float("nan"),
                "median_return": float(values.median()) if len(values) else float("nan"),
                "hit_rate": float((values > 0).mean()) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=ECONOMIC_COLUMNS)


def random_baseline(
    close: pd.Series,
    n_signals: int,
    horizons: Sequence[int],
    seed: int,
    draws: int = 200,
    eligible: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Average economic metrics over `draws` random entry sets of the same size.

    When eligible is None, sample from all dates except those without enough bars
    to compute the longest horizon. When provided, sample only from those dates
    that fall within the eligible window (still respecting the horizon cutoff).
    """
    rng = np.random.default_rng(seed)
    accumulated: list[pd.DataFrame] = []

    # Compute the cutoff: exclude dates without room for the longest horizon
    cutoff_index = max(len(close) - max(horizons), 1)
    default_eligible = close.index[:cutoff_index]

    # If eligible is provided, intersect with the default cutoff
    if eligible is not None:
        eligible_set = set(eligible)
        final_eligible = pd.DatetimeIndex([d for d in default_eligible if d in eligible_set])
    else:
        final_eligible = default_eligible

    if len(final_eligible) == 0:
        return pd.DataFrame(columns=ECONOMIC_COLUMNS)

    for _ in range(draws):
        size = min(n_signals, len(final_eligible))
        picks = rng.choice(len(final_eligible), size=size, replace=False)
        dates = [final_eligible[int(p)] for p in picks]
        accumulated.append(economic_metrics(close, dates, horizons))

    if not accumulated:
        return pd.DataFrame(columns=ECONOMIC_COLUMNS)

    stacked = pd.concat(accumulated)
    return (
        stacked.groupby("horizon", as_index=False)
        .mean(numeric_only=True)[ECONOMIC_COLUMNS]
        .reset_index(drop=True)
    )


def edge_table(
    signal_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame
) -> pd.DataFrame:
    merged = signal_metrics.merge(
        baseline_metrics, on="horizon", suffixes=("", "_baseline")
    )
    merged["mean_edge"] = merged["mean_return"] - merged["mean_return_baseline"]
    merged["hit_rate_edge"] = merged["hit_rate"] - merged["hit_rate_baseline"]
    return merged
