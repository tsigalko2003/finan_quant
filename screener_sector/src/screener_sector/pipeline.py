"""Screen orchestration.

Truncation at `as_of` happens here, once, before any feature runs. That makes
point-in-time correctness a structural property of the pipeline rather than a
discipline each feature function has to remember.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from screener_sector.config import Config
from screener_sector.data.store import PriceStore
from screener_sector.features.correlation import cluster_universe
from screener_sector.features.rebound import rebound_table
from screener_sector.features.strength import strength_table
from screener_sector.features.trend import trend_table
from screener_sector.paths import Paths
from screener_sector.report.render import ScreenOutput, write_csvs


def load_frames(
    store: PriceStore, tickers: Sequence[str], as_of: date
) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(as_of)
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if not store.has(ticker):
            continue
        frame = store.load(ticker)
        truncated = frame.loc[frame.index <= cutoff]
        if truncated.empty:
            continue
        frames[ticker] = truncated
    return frames


def run_screen(
    store: PriceStore, tickers: Sequence[str], config: Config, as_of: date
) -> ScreenOutput:
    frames = load_frames(store, list(tickers) + [config.benchmark], as_of)

    benchmark_frame = frames.pop(config.benchmark, None)
    benchmark = benchmark_frame["close"] if benchmark_frame is not None else None

    if not frames:
        empty_clusters = cluster_universe(
            pd.DataFrame(), None, config.corr_threshold, config.min_cluster_size, 1
        )
        return ScreenOutput(as_of, pd.DataFrame(), empty_clusters, pd.DataFrame(), pd.DataFrame())

    trend = trend_table(frames, config.windows, config.trend_weights)

    panel = pd.DataFrame({t: f["close"] for t, f in frames.items()}).sort_index()
    clusters = cluster_universe(
        panel,
        benchmark,
        config.corr_threshold,
        config.min_cluster_size,
        config.windows.corr,
    )

    strength = strength_table(panel, clusters.clusters, config.windows.corr)
    rebound = rebound_table(
        panel,
        frames,
        clusters.clusters,
        config.rebound_weights,
        config.windows,
        as_of=pd.Timestamp(as_of) if pd.Timestamp(as_of) in panel.index else None,
    )

    return ScreenOutput(
        as_of=as_of,
        trend=trend,
        clusters=clusters,
        strength=strength,
        rebound=rebound,
    )


def save_screen(paths: Paths, output: ScreenOutput, profile: str) -> Path:
    directory = paths.derived_dir(profile) / output.as_of.isoformat()
    write_csvs(output, directory)
    return directory
