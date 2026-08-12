import numpy as np
import pandas as pd

from conftest import (
    correlated_returns,
    exponential_trend,
    flat_series,
    make_ohlcv,
    trading_days,
    v_bottom,
)


def test_trading_days_excludes_weekends():
    idx = trading_days(10)
    assert len(idx) == 10
    assert all(d.weekday() < 5 for d in idx)


def test_exponential_trend_is_monotonic_without_noise():
    s = exponential_trend(50, daily_rate=0.002)
    assert s.is_monotonic_increasing
    assert len(s) == 50


def test_exponential_trend_is_reproducible_with_seed():
    a = exponential_trend(50, daily_rate=0.002, noise=0.01, seed=7)
    b = exponential_trend(50, daily_rate=0.002, noise=0.01, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_v_bottom_has_minimum_in_the_middle():
    s = v_bottom(30, 30, depth=0.30)
    assert s.idxmin() == s.index[29]
    assert s.iloc[29] < s.iloc[0] * 0.75


def test_correlated_returns_recovers_rho():
    a, b = correlated_returns(4000, rho=0.8, seed=1)
    assert abs(a.corr(b) - 0.8) < 0.05


def test_flat_series_has_zero_variance():
    assert flat_series(40).std() == 0.0


def test_make_ohlcv_has_expected_columns_and_bounds():
    close = exponential_trend(30, 0.001, noise=0.02, seed=3)
    df = make_ohlcv(close)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()
    assert isinstance(df.index, pd.DatetimeIndex)
