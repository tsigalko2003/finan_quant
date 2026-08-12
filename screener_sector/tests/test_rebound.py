import numpy as np
import pandas as pd
import pytest

from conftest import (
    exponential_trend,
    flat_series,
    make_ohlcv,
    trading_days,
    v_bottom,
)
from screener_sector.config import ReboundWeights, Windows
from screener_sector.features.correlation import Cluster
from screener_sector.features.rebound import (
    bullish_divergence,
    cluster_washout,
    confirmation,
    rebound_table,
    rsi,
    stretch_z,
    ticker_alarm,
    volume_signal,
    williams_r,
)

WEIGHTS = ReboundWeights(
    breadth=0.25, stretch=0.20, oscillator=0.25, volume=0.15, confirmation=0.15
)
WINDOWS = Windows(short=20, mid=60, corr=120)


def test_rsi_is_bounded():
    values = rsi(exponential_trend(200, 0.001, noise=0.03, seed=1)).dropna()
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_rsi_is_high_in_uptrend_and_low_in_downtrend():
    up = rsi(exponential_trend(100, 0.004)).iloc[-1]
    down = rsi(exponential_trend(100, -0.004)).iloc[-1]
    assert up > 70
    assert down < 30


def test_williams_r_is_negative_bounded():
    values = williams_r(make_ohlcv(exponential_trend(100, 0.002))).dropna()
    assert values.min() >= -100.0
    assert values.max() <= 0.0


def test_stretch_z_is_negative_below_mean():
    close = pd.concat([flat_series(80), pd.Series([70.0] * 5)], ignore_index=True)
    close.index = trading_days(85)
    assert stretch_z(close, 60).iloc[-1] < 0


def test_stretch_z_is_zero_for_flat_series():
    assert stretch_z(flat_series(100), 60).iloc[-1] == pytest.approx(0.0)


def test_volume_signal_rewards_spike_then_dryup():
    idx = trading_days(60)
    volume = pd.Series([1_000_000.0] * 60, index=idx)
    volume.iloc[-6] = 6_000_000.0           # capitulation spike
    volume.iloc[-5:] = 400_000.0            # dry-up
    frame = make_ohlcv(flat_series(60), volume)
    assert volume_signal(frame).iloc[-1] > 0.6


def test_volume_signal_is_low_for_constant_volume():
    frame = make_ohlcv(flat_series(60))
    assert volume_signal(frame).iloc[-1] < 0.4


def test_bullish_divergence_detects_lower_low_with_higher_oscillator():
    idx = trading_days(60)
    close = pd.Series(
        np.concatenate([np.linspace(100, 80, 30), np.linspace(80, 78, 30)]), index=idx
    )
    oscillator = pd.Series(
        np.concatenate([np.linspace(50, 20, 30), np.linspace(20, 35, 30)]), index=idx
    )
    assert bool(bullish_divergence(close, oscillator, lookback=20).iloc[-1])


def test_no_divergence_in_clean_downtrend():
    idx = trading_days(60)
    close = pd.Series(np.linspace(100, 60, 60), index=idx)
    oscillator = pd.Series(np.linspace(60, 15, 60), index=idx)
    assert not bool(bullish_divergence(close, oscillator, lookback=20).iloc[-1])


def test_confirmation_fires_on_close_above_prior_high():
    close = pd.Series(
        list(np.linspace(100, 80, 40)) + [88.0], index=trading_days(41)
    )
    frame = make_ohlcv(close)
    assert bool(confirmation(frame, short_window=20).iloc[-1])


def test_confirmation_does_not_fire_mid_decline():
    frame = make_ohlcv(pd.Series(np.linspace(100, 80, 41), index=trading_days(41)))
    assert not bool(confirmation(frame, short_window=20).iloc[-1])


def test_cluster_washout_is_high_when_all_members_oversold():
    idx = trading_days(120)
    declining = pd.Series(np.linspace(100, 60, 120), index=idx)
    panel = pd.DataFrame({f"T{i}": declining * (1 + 0.01 * i) for i in range(4)})
    washout = cluster_washout(panel, [f"T{i}" for i in range(4)], window=60)
    assert washout.iloc[-1] > 0.8


def test_cluster_washout_is_low_when_members_are_strong():
    idx = trading_days(120)
    rising = pd.Series(np.linspace(60, 100, 120), index=idx)
    panel = pd.DataFrame({f"T{i}": rising * (1 + 0.01 * i) for i in range(4)})
    washout = cluster_washout(panel, [f"T{i}" for i in range(4)], window=60)
    assert washout.iloc[-1] < 0.2


def test_alarm_fires_near_a_v_bottom():
    close = v_bottom(80, 40, depth=0.35)
    frame = make_ohlcv(close)
    washout = pd.Series(1.0, index=close.index)
    alarm = ticker_alarm(frame, washout, WEIGHTS, WINDOWS)
    trough = close.idxmin()
    trough_position = close.index.get_loc(trough)
    nearby = alarm.iloc[trough_position : trough_position + 10]
    assert nearby.max() > 60


def test_alarm_stays_low_on_flat_series():
    frame = make_ohlcv(flat_series(200))
    washout = pd.Series(0.0, index=frame.index)
    alarm = ticker_alarm(frame, washout, WEIGHTS, WINDOWS)
    assert alarm.dropna().max() < 40


def test_alarm_stays_low_in_steady_uptrend():
    frame = make_ohlcv(exponential_trend(200, 0.002))
    washout = pd.Series(0.0, index=frame.index)
    alarm = ticker_alarm(frame, washout, WEIGHTS, WINDOWS)
    assert alarm.dropna().max() < 50


def test_rebound_table_gates_on_cluster_washout():
    close = v_bottom(80, 40, depth=0.35)
    frames = {f"T{i}": make_ohlcv(close * (1 + 0.01 * i)) for i in range(3)}
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    clusters = [Cluster(0, tuple(frames), 0.95)]

    as_of = close.index[80]  # one bar past the trough
    table = rebound_table(panel, frames, clusters, WEIGHTS, WINDOWS, as_of=as_of)
    assert list(table.columns) == [
        "ticker", "cluster", "alarm", "washout", "stretch_z", "rsi",
        "volume", "divergence", "confirmed", "fired",
    ]
    assert table["washout"].max() > 0.5


def test_rebound_table_fires_nothing_in_uptrend():
    frames = {
        f"T{i}": make_ohlcv(exponential_trend(200, 0.002) * (1 + 0.01 * i))
        for i in range(3)
    }
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    clusters = [Cluster(0, tuple(frames), 0.95)]
    table = rebound_table(panel, frames, clusters, WEIGHTS, WINDOWS)
    assert not table["fired"].any()
