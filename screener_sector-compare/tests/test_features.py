from __future__ import annotations

import numpy as np
import pandas as pd

from sector_screener.features import build_sector_features, rank_tickers

PARAMS = {
    "short_window": 10,
    "mid_window": 30,
    "long_window": 60,
    "correlation_window": 20,
    "pca_window": 20,
}


def synthetic_prices(rows: int = 180, tickers: int = 6):
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-02", periods=rows, freq="B")
    common = rng.normal(0.0004, 0.012, rows)
    returns = np.column_stack(
        [common * (0.7 + number * 0.08) + rng.normal(0, 0.005, rows) for number in range(tickers)]
    )
    close = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=index,
        columns=[f"T{number}" for number in range(tickers)],
    )
    volume = pd.DataFrame(1_000_000, index=index, columns=close.columns)
    return close, volume


def test_features_are_causal_and_pca_is_anchored():
    close, volume = synthetic_prices()
    original, _ = build_sector_features(close, volume, PARAMS)
    modified = close.copy()
    modified.iloc[-20:] *= 3
    changed, _ = build_sector_features(modified, volume, PARAMS)
    pd.testing.assert_frame_equal(original.iloc[:-20], changed.iloc[:-20])
    assert 0 <= original["pc1_explained_variance"].dropna().iloc[-1] <= 1


def test_relative_strength_rankings_are_separate():
    close, volume = synthetic_prices()
    features, _ = build_sector_features(close, volume, PARAMS)
    ranking = rank_tickers(close, features, 60)
    assert {"fall_resistance_rank", "rise_strength_rank"}.issubset(ranking.columns)
    assert ranking["fall_resistance_rank"].notna().all()
