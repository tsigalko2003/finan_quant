import numpy as np
import pytest

from conftest import exponential_trend, flat_series, make_ohlcv, v_bottom
from screener_sector.config import TrendWeights, Windows
from screener_sector.features.trend import (
    adx,
    log_slope_r2,
    ma_stack_score,
    trend_score,
    trend_table,
)

WEIGHTS = TrendWeights(slope=0.40, r2=0.30, adx=0.15, ma_stack=0.15)
WINDOWS = Windows(short=20, mid=60, corr=120)


def test_pure_exponential_trend_has_r2_near_one():
    close = exponential_trend(60, daily_rate=0.002)
    slope, r2 = log_slope_r2(close, window=60)
    assert r2 > 0.999
    assert slope > 0


def test_noisy_trend_has_lower_r2_than_clean_trend():
    clean = exponential_trend(60, 0.002)
    noisy = exponential_trend(60, 0.002, noise=0.05, seed=5)
    _, clean_r2 = log_slope_r2(clean, 60)
    _, noisy_r2 = log_slope_r2(noisy, 60)
    assert clean_r2 > noisy_r2


def test_downtrend_has_negative_slope():
    close = exponential_trend(60, daily_rate=-0.002)
    slope, r2 = log_slope_r2(close, 60)
    assert slope < 0
    assert r2 > 0.999


def test_flat_series_has_zero_slope():
    slope, r2 = log_slope_r2(flat_series(60), 60)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_slope_uses_only_the_last_window_bars():
    close = exponential_trend(200, 0.002)
    full = log_slope_r2(close, 60)
    tail = log_slope_r2(close.tail(60), 60)
    assert full == pytest.approx(tail)


def test_adx_is_higher_for_trending_than_choppy():
    trending = make_ohlcv(exponential_trend(100, 0.004))
    choppy = make_ohlcv(flat_series(100) + np.tile([1.0, -1.0], 50))
    assert adx(trending) > adx(choppy)


def test_ma_stack_positive_when_price_above_rising_mas():
    close = exponential_trend(150, 0.003)
    assert ma_stack_score(close, 20, 60) > 0.5


def test_ma_stack_negative_in_downtrend():
    close = exponential_trend(150, -0.003)
    assert ma_stack_score(close, 20, 60) < -0.5


def test_uptrend_scores_strongly_positive():
    frame = make_ohlcv(exponential_trend(150, 0.003))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert result.score > 50
    assert result.r2 > 0.9


def test_downtrend_scores_strongly_negative():
    frame = make_ohlcv(exponential_trend(150, -0.003))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert result.score < -50


def test_flat_series_scores_near_zero():
    frame = make_ohlcv(flat_series(150))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert abs(result.score) < 15


def test_score_is_bounded():
    frame = make_ohlcv(exponential_trend(150, 0.05))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert -100.0 <= result.score <= 100.0


def test_trend_table_reports_both_windows():
    frames = {
        "UP": make_ohlcv(exponential_trend(150, 0.003)),
        "DOWN": make_ohlcv(exponential_trend(150, -0.003)),
    }
    table = trend_table(frames, WINDOWS, WEIGHTS).set_index("ticker")
    assert list(table.columns) == [
        "short_score", "mid_score", "short_r2", "mid_r2", "adx", "ma_stack",
    ]
    assert table.loc["UP", "mid_score"] > 0
    assert table.loc["DOWN", "mid_score"] < 0


def test_trend_table_skips_tickers_with_insufficient_bars():
    frames = {
        "OK": make_ohlcv(exponential_trend(150, 0.003)),
        "SHORT": make_ohlcv(exponential_trend(10, 0.003)),
    }
    table = trend_table(frames, WINDOWS, WEIGHTS)
    assert list(table["ticker"]) == ["OK"]
