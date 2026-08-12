"""Synthetic data builders shared by every test.

Tests never touch the network. Series here have known analytic properties so
assertions can be exact rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_days(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def exponential_trend(
    n: int, daily_rate: float, noise: float = 0.0, seed: int = 0
) -> pd.Series:
    """Price series compounding at `daily_rate` with optional lognormal noise."""
    idx = trading_days(n)
    drift = np.exp(np.arange(n) * daily_rate)
    if noise:
        rng = np.random.default_rng(seed)
        drift = drift * np.exp(rng.normal(0.0, noise, n))
    return pd.Series(100.0 * drift, index=idx, name="close")


def v_bottom(n_down: int, n_up: int, depth: float = 0.30) -> pd.Series:
    """Linear decline to a trough at index n_down-1, then a linear recovery."""
    idx = trading_days(n_down + n_up)
    trough = 100.0 * (1.0 - depth)
    down = np.linspace(100.0, trough, n_down)
    up = np.linspace(trough, 100.0, n_up + 1)[1:]
    return pd.Series(np.concatenate([down, up]), index=idx, name="close")


def correlated_returns(n: int, rho: float, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Two return series with population correlation `rho`."""
    rng = np.random.default_rng(seed)
    idx = trading_days(n)
    z1 = rng.normal(0.0, 0.01, n)
    z2 = rng.normal(0.0, 0.01, n)
    mixed = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
    return pd.Series(z1, index=idx, name="a"), pd.Series(mixed, index=idx, name="b")


def flat_series(n: int, level: float = 100.0) -> pd.Series:
    return pd.Series(np.full(n, level), index=trading_days(n), name="close")


def make_ohlcv(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Wrap a close series into a full OHLCV frame with consistent bounds."""
    prev = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([close, prev], axis=1).max(axis=1) * 1.01
    low = pd.concat([close, prev], axis=1).min(axis=1) * 0.99
    if volume is None:
        volume = pd.Series(1_000_000.0, index=close.index)
    return pd.DataFrame(
        {
            "open": prev.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
        },
        index=close.index,
    )
