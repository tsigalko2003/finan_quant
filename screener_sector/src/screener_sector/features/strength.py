"""Relative strength within a correlated group.

Splitting days by the group's own direction answers the question directly:
which member falls least when the group falls, and rises most when it rises?
A single blended relative-strength number cannot separate those two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener_sector.features.correlation import Cluster, log_returns


def group_return(returns: pd.DataFrame, members: Sequence[str]) -> pd.Series:
    present = [m for m in members if m in returns.columns]
    return returns[present].mean(axis=1, skipna=True)


@dataclass(frozen=True)
class Capture:
    up: float
    down: float
    up_days: int
    down_days: int


def capture_ratios(
    ticker_returns: pd.Series, group_returns: pd.Series
) -> Capture:
    pair = pd.concat(
        [ticker_returns.rename("t"), group_returns.rename("g")], axis=1
    ).dropna()
    up_mask = pair["g"] > 0
    down_mask = pair["g"] < 0

    def ratio(mask: pd.Series) -> float:
        if not mask.any():
            return float("nan")
        denominator = pair.loc[mask, "g"].mean()
        if denominator == 0:
            return float("nan")
        return float(pair.loc[mask, "t"].mean() / denominator)

    return Capture(
        up=ratio(up_mask),
        down=ratio(down_mask),
        up_days=int(up_mask.sum()),
        down_days=int(down_mask.sum()),
    )


def max_drawdown(close: pd.Series) -> float:
    series = close.dropna()
    if series.empty:
        return 0.0
    running_peak = series.cummax()
    return float((series / running_peak - 1.0).min())


def recovery_days(close: pd.Series) -> int:
    """Bars from the deepest trough back to the peak that preceded it."""
    series = close.dropna()
    if series.empty:
        return -1
    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    trough_position = int(drawdown.to_numpy().argmin())
    peak_level = float(running_peak.iloc[trough_position])
    after = series.iloc[trough_position:]
    recovered = after[after >= peak_level]
    if recovered.empty:
        return -1
    return int(series.index.get_loc(recovered.index[0]) - trough_position)


STRENGTH_COLUMNS = [
    "ticker",
    "cluster",
    "up_capture",
    "down_capture",
    "capture_spread",
    "max_drawdown",
    "recovery_days",
    "rank_in_cluster",
]


def strength_table(
    panel: pd.DataFrame, clusters: Sequence[Cluster], window: int
) -> pd.DataFrame:
    if not clusters:
        return pd.DataFrame(columns=STRENGTH_COLUMNS)

    returns = log_returns(panel).tail(window)
    prices = panel.tail(window)
    frames: list[pd.DataFrame] = []

    for cluster in clusters:
        members = [m for m in cluster.members if m in returns.columns]
        if not members:
            continue
        benchmark = group_return(returns, members)
        rows = []
        for ticker in members:
            capture = capture_ratios(returns[ticker], benchmark)
            spread = capture.up - capture.down
            rows.append(
                {
                    "ticker": ticker,
                    "cluster": cluster.label,
                    "up_capture": capture.up,
                    "down_capture": capture.down,
                    "capture_spread": spread,
                    "max_drawdown": max_drawdown(prices[ticker]),
                    "recovery_days": recovery_days(prices[ticker]),
                }
            )
        block = pd.DataFrame(rows)
        block["rank_in_cluster"] = (
            block["capture_spread"].rank(ascending=False, method="min").astype(int)
        )
        frames.append(block)

    if not frames:
        return pd.DataFrame(columns=STRENGTH_COLUMNS)
    return pd.concat(frames, ignore_index=True)[STRENGTH_COLUMNS]
