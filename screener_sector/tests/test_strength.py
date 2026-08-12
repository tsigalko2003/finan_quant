import numpy as np
import pandas as pd
import pytest

from conftest import trading_days
from screener_sector.features.correlation import Cluster
from screener_sector.features.strength import (
    capture_ratios,
    group_return,
    max_drawdown,
    recovery_days,
    strength_table,
)


def panel_from_returns(columns: dict[str, pd.Series]) -> pd.DataFrame:
    return 100.0 * np.exp(pd.DataFrame(columns).cumsum())


def test_group_return_is_equal_weight_mean():
    idx = trading_days(5)
    returns = pd.DataFrame(
        {"A": pd.Series([0.02] * 5, index=idx), "B": pd.Series([0.00] * 5, index=idx)}
    )
    assert group_return(returns, ["A", "B"]).iloc[0] == pytest.approx(0.01)


def test_group_return_ignores_missing_members():
    idx = trading_days(3)
    returns = pd.DataFrame(
        {"A": pd.Series([0.02, 0.02, 0.02], index=idx),
         "B": pd.Series([np.nan, 0.00, 0.00], index=idx)}
    )
    assert group_return(returns, ["A", "B"]).iloc[0] == pytest.approx(0.02)


def test_defensive_name_has_low_down_capture():
    idx = trading_days(200)
    rng = np.random.default_rng(1)
    group = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
    # falls half as much on down days, matches on up days
    ticker = group.where(group > 0, group * 0.5)
    capture = capture_ratios(ticker, group)
    assert capture.down < 0.7
    assert capture.up == pytest.approx(1.0, abs=0.05)


def test_aggressive_name_has_high_up_capture():
    idx = trading_days(200)
    rng = np.random.default_rng(2)
    group = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
    ticker = group.where(group < 0, group * 1.8)
    capture = capture_ratios(ticker, group)
    assert capture.up > 1.5
    assert capture.down == pytest.approx(1.0, abs=0.05)


def test_capture_counts_up_and_down_days():
    idx = trading_days(10)
    group = pd.Series([0.01] * 6 + [-0.01] * 4, index=idx)
    capture = capture_ratios(group.copy(), group)
    assert capture.up_days == 6
    assert capture.down_days == 4


def test_capture_is_nan_when_no_down_days():
    idx = trading_days(5)
    group = pd.Series([0.01] * 5, index=idx)
    assert np.isnan(capture_ratios(group.copy(), group).down)


def test_max_drawdown_matches_hand_computation():
    close = pd.Series([100.0, 120.0, 60.0, 90.0], index=trading_days(4))
    assert max_drawdown(close) == pytest.approx(-0.5)


def test_recovery_days_counts_bars_back_to_prior_peak():
    close = pd.Series([100.0, 80.0, 90.0, 101.0, 105.0], index=trading_days(5))
    assert recovery_days(close) == 2


def test_recovery_days_is_minus_one_when_never_recovered():
    close = pd.Series([100.0, 80.0, 85.0], index=trading_days(3))
    assert recovery_days(close) == -1


def test_strength_table_ranks_leader_first():
    idx = trading_days(250)
    rng = np.random.default_rng(6)
    driver = pd.Series(rng.normal(0.0, 0.02, 250), index=idx)
    columns = {
        "LEADER": driver.where(driver < 0, driver * 1.5).where(driver > 0, driver * 0.5),
        "LAGGARD": driver.where(driver > 0, driver * 1.5).where(driver < 0, driver * 0.5),
        "MIDDLE": driver.copy(),
    }
    clusters = [Cluster(0, ("LEADER", "LAGGARD", "MIDDLE"), 0.9)]
    table = strength_table(panel_from_returns(columns), clusters, window=250)
    table = table.set_index("ticker")
    assert list(table.columns) == [
        "cluster", "up_capture", "down_capture", "capture_spread",
        "max_drawdown", "recovery_days", "rank_in_cluster",
    ]
    assert table.loc["LEADER", "rank_in_cluster"] == 1
    assert table.loc["LAGGARD", "rank_in_cluster"] == 3
    assert table.loc["LEADER", "down_capture"] < table.loc["LAGGARD", "down_capture"]


def test_strength_table_is_empty_without_clusters():
    idx = trading_days(100)
    columns = {"A": pd.Series(np.zeros(100), index=idx)}
    table = strength_table(panel_from_returns(columns), [], window=100)
    assert table.empty
