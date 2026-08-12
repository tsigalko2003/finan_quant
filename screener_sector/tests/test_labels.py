import numpy as np
import pandas as pd
import pytest

from conftest import exponential_trend, flat_series, make_ohlcv, trading_days, v_bottom
from screener_sector.backtest.labels import forward_return, label_bottoms


def test_v_bottom_is_labeled_at_the_trough():
    close = v_bottom(40, 40, depth=0.40)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert bool(labels.loc[close.idxmin()])


def test_only_the_trough_region_is_labeled():
    close = v_bottom(40, 40, depth=0.40)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert labels.sum() <= 3


def test_uptrend_has_no_bottom_labels():
    labels = label_bottoms(
        make_ohlcv(exponential_trend(200, 0.002)), k=10, forward_days=20, min_return=0.10
    )
    assert not labels.any()


def test_flat_series_has_no_bottom_labels():
    labels = label_bottoms(
        make_ohlcv(flat_series(200)), k=10, forward_days=20, min_return=0.10
    )
    assert not labels.any()


def test_shallow_bounce_below_threshold_is_not_labeled():
    close = v_bottom(40, 40, depth=0.03)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert not labels.any()


def test_no_label_within_forward_window_of_the_end():
    close = v_bottom(40, 10, depth=0.40)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert not labels.tail(20).any()


def test_forward_return_matches_hand_computation():
    close = pd.Series([100.0, 110.0, 121.0], index=trading_days(3))
    assert forward_return(close, 2).iloc[0] == pytest.approx(0.21)


def test_forward_return_is_nan_at_the_tail():
    close = pd.Series([100.0, 110.0, 121.0], index=trading_days(3))
    assert np.isnan(forward_return(close, 2).iloc[-1])
