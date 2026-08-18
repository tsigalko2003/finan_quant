"""Rebound leadership ranking within clusters, measured from group troughs.

Forward-looking module: identifies troughs by inspecting price movement after
them, so it is not importable from features/. This lives in backtest/ even though
it serves screening, because trough detection requires knowing what happened next.

POINT-IN-TIME DISCIPLINE: label_bottoms ensures a trough too recent to have a
full forward window is never yielded False. When this runs on a truncated panel
at an as_of date, no troughs are invented from incomplete data. The user sees
only events with complete information.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from screener_sector.backtest.labels import label_bottoms
from screener_sector.features.correlation import Cluster


REBOUND_COLUMNS = [
    "ticker",
    "cluster",
    "events",
    "median_rebound_5d",
    "median_rebound_10d",
    "median_rebound_20d",
    "rebound_ratio_20d",
    "recovery_efficiency",
    "consistency",
    "median_days_to_recover",
    "rank_in_cluster",
]

MIN_EVENTS = 3


def group_trough_dates(
    group_close: pd.Series, k: int, forward_days: int, min_return: float
) -> list[pd.Timestamp]:
    """Dates where the cluster's equal-weight index bottomed.

    Reuses label_bottoms so trough definition stays consistent with the
    backtest. Build the OHLCV frame it expects with low and close both set to
    the group index.

    Args:
        group_close: equal-weight index of a cluster, indexed by date.
        k: half-window for local minimum detection (center window = 2k+1).
        forward_days: bars forward to check for recovery.
        min_return: minimum forward return to qualify as a tradeable bottom.

    Returns:
        List of dates where the cluster bottomed, sorted ascending.
    """
    ohlcv = pd.DataFrame(
        {
            "open": group_close.shift(1).fillna(group_close.iloc[0]),
            "high": group_close,
            "low": group_close,
            "close": group_close,
            "volume": 1.0,
        }
    )
    labels = label_bottoms(ohlcv, k, forward_days, min_return)
    return sorted(labels[labels].index.tolist())


def rebound_leaders(
    panel: pd.DataFrame,
    clusters: Sequence[Cluster],
    k: int,
    forward_days: int,
    min_return: float,
    horizons: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Per-ticker rebound leadership, ranked within each cluster.

    For every date the cluster's equal-weight index bottomed, measure each
    member's forward return at multiple horizons. Aggregate metrics over all
    bottoms, then rank within cluster.

    Args:
        panel: DataFrame with index=dates, columns=ticker, values=close prices.
               Must be truncated to the evaluation date (no forward-looking data).
        clusters: sequence of Cluster objects defining members of each group.
        k: half-window for local minimum detection (center window = 2k+1).
        forward_days: bars forward to check for recovery (also the base for
                      rebound_ratio and recovery_efficiency).
        min_return: minimum forward return to qualify as a tradeable bottom.
        horizons: list of days forward to report median rebound (default 5/10/20).

    Returns:
        DataFrame with REBOUND_COLUMNS. One row per (cluster, ticker) pair.
        Empty DataFrame with correct columns when there are no clusters or no troughs.
        A ticker with fewer than MIN_EVENTS events gets rank_in_cluster as pd.NA.
    """
    if not clusters or panel.empty:
        return pd.DataFrame(columns=REBOUND_COLUMNS)

    rows = []

    for cluster in clusters:
        # Compute the cluster's equal-weight index.
        members_in_panel = [t for t in cluster.members if t in panel.columns]
        if not members_in_panel:
            continue

        cluster_index = panel[members_in_panel].mean(axis=1)
        trough_dates = group_trough_dates(cluster_index, k, forward_days, min_return)
        if not trough_dates:
            continue

        # Compute cluster's median 20d rebound at each trough (used for ratio and consistency).
        cluster_rebounds_20d = []
        for trough_date in trough_dates:
            if trough_date not in panel.index:
                continue
            idx = panel.index.get_loc(trough_date)
            horizon_idx = idx + forward_days
            if horizon_idx >= len(panel):
                continue
            trough_close = cluster_index.iloc[idx]
            forward_close = cluster_index.iloc[horizon_idx]
            if trough_close > 0:
                cluster_rebounds_20d.append(forward_close / trough_close - 1.0)

        cluster_median_20d = np.median(cluster_rebounds_20d) if cluster_rebounds_20d else np.nan

        # Per-ticker metrics.
        for ticker in cluster.members:
            if ticker not in panel.columns:
                continue

            ticker_data = panel[ticker]
            events = []
            rebounds_5d = []
            rebounds_10d = []
            rebounds_20d = []
            efficiencies = []
            beat_cluster = 0
            days_to_recover = []

            for trough_date in trough_dates:
                if trough_date not in panel.index:
                    continue

                idx = panel.index.get_loc(trough_date)

                # Check we have a complete forward window.
                if idx + forward_days >= len(panel):
                    continue

                trough_close = ticker_data.iloc[idx]
                if trough_close <= 0:
                    continue

                # Rebounds at each horizon.
                for horizon in horizons:
                    horizon_idx = idx + horizon
                    if horizon_idx < len(panel):
                        forward_close = ticker_data.iloc[horizon_idx]
                        rebound = forward_close / trough_close - 1.0
                        if horizon == 5:
                            rebounds_5d.append(rebound)
                        elif horizon == 10:
                            rebounds_10d.append(rebound)
                        elif horizon == 20:
                            rebounds_20d.append(rebound)

                # 20d rebound for ratio and consistency.
                horizon_20_idx = idx + forward_days
                forward_close_20 = ticker_data.iloc[horizon_20_idx]
                rebound_20 = forward_close_20 / trough_close - 1.0

                # Recovery efficiency: rebound / abs(drawdown_depth).
                # Running peak is the maximum price before the trough.
                running_peak = ticker_data.iloc[:idx].max()
                if running_peak > 0:
                    drawdown_depth = trough_close / running_peak - 1.0
                    if drawdown_depth < 0:  # Only process if there was a drawdown.
                        efficiency = rebound_20 / abs(drawdown_depth)
                        efficiencies.append(efficiency)

                # Consistency: did this ticker beat the cluster median at this event?
                if np.isfinite(cluster_median_20d):
                    if rebound_20 > cluster_median_20d:
                        beat_cluster += 1

                # Days to recover: bars from trough to first bar >= running peak.
                if running_peak > 0:
                    recovery_window = ticker_data.iloc[idx + 1 : idx + forward_days + 1]
                    recovered_idx = (recovery_window >= running_peak).argmax()
                    if recovered_idx > 0 or recovery_window.iloc[0] >= running_peak:
                        # Found recovery within the forward window.
                        days_to_recover.append(recovered_idx if recovered_idx > 0 else 0)
                    # else: never recovered, use sentinel -1 which we exclude from median.

                events.append(None)  # Dummy append to count valid events.

            n_events = len(events)
            if n_events == 0:
                continue

            median_rebound_5d = np.median(rebounds_5d) if rebounds_5d else np.nan
            median_rebound_10d = np.median(rebounds_10d) if rebounds_10d else np.nan
            median_rebound_20d = np.median(rebounds_20d) if rebounds_20d else np.nan

            # Rebound ratio: ticker's median 20d rebound / cluster's median 20d rebound.
            if np.isfinite(cluster_median_20d) and cluster_median_20d != 0:
                rebound_ratio = median_rebound_20d / cluster_median_20d if np.isfinite(median_rebound_20d) else np.nan
            else:
                rebound_ratio = np.nan

            # Recovery efficiency: median over events.
            recovery_efficiency = np.median(efficiencies) if efficiencies else np.nan

            # Consistency: fraction beating cluster median.
            consistency = beat_cluster / n_events if n_events > 0 else np.nan

            # Median days to recover, excluding -1 sentinel.
            if days_to_recover:
                median_days_to_recover = np.median(days_to_recover)
            else:
                median_days_to_recover = -1.0

            rows.append(
                {
                    "ticker": ticker,
                    "cluster": cluster.label,
                    "events": n_events,
                    "median_rebound_5d": median_rebound_5d,
                    "median_rebound_10d": median_rebound_10d,
                    "median_rebound_20d": median_rebound_20d,
                    "rebound_ratio_20d": rebound_ratio,
                    "recovery_efficiency": recovery_efficiency,
                    "consistency": consistency,
                    "median_days_to_recover": median_days_to_recover,
                    "rank_in_cluster": np.nan,  # Placeholder, filled below.
                }
            )

    df = pd.DataFrame(rows, columns=REBOUND_COLUMNS) if rows else pd.DataFrame(columns=REBOUND_COLUMNS)

    # Rank within each cluster.
    if not df.empty:
        for cluster_label in df["cluster"].unique():
            cluster_mask = df["cluster"] == cluster_label
            cluster_indices = df.loc[cluster_mask].index.tolist()

            # Only rank members with MIN_EVENTS or more.
            rankable_mask = (df.loc[cluster_mask, "events"] >= MIN_EVENTS).values
            rankable_indices = [idx for idx, is_rankable in zip(cluster_indices, rankable_mask) if is_rankable]

            if rankable_indices:
                # Sort by rebound_ratio_20d desc, tie-break by consistency desc.
                rankable_df = df.loc[rankable_indices].sort_values(
                    ["rebound_ratio_20d", "consistency"],
                    ascending=[False, False],
                    na_position="last",
                )
                for rank, idx in enumerate(rankable_df.index, start=1):
                    df.loc[idx, "rank_in_cluster"] = rank

    return df[REBOUND_COLUMNS]
