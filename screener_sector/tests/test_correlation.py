import numpy as np
import pandas as pd
import pytest

from conftest import correlated_returns, exponential_trend, trading_days
from screener_sector.features.correlation import (
    cluster_universe,
    correlation_distance,
    correlation_matrix,
    log_returns,
    residualize,
)


def panel_from_returns(columns: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    return 100.0 * np.exp(frame.cumsum())


def test_log_returns_shape_and_first_row_dropped():
    prices = pd.DataFrame(
        {"A": exponential_trend(50, 0.001), "B": exponential_trend(50, 0.002)}
    )
    returns = log_returns(prices)
    assert len(returns) == 49
    assert list(returns.columns) == ["A", "B"]


def test_correlation_recovers_known_rho():
    a, b = correlated_returns(3000, rho=0.75, seed=11)
    corr = correlation_matrix(pd.DataFrame({"A": a, "B": b}))
    assert corr.loc["A", "B"] == pytest.approx(0.75, abs=0.05)


def test_correlation_of_prices_would_differ_from_returns():
    """Two independent uptrends have near-zero return correlation but very
    high price correlation. This is why the pipeline uses returns."""
    a = exponential_trend(500, 0.001, noise=0.02, seed=1)
    b = exponential_trend(500, 0.001, noise=0.02, seed=2)
    prices = pd.DataFrame({"A": a, "B": b})
    price_corr = prices.corr().loc["A", "B"]
    return_corr = correlation_matrix(log_returns(prices)).loc["A", "B"]
    assert price_corr > 0.9
    assert abs(return_corr) < 0.2


def test_correlation_requires_minimum_overlap():
    idx = trading_days(100)
    a = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
    b = a.copy()
    b.iloc[:90] = np.nan
    corr = correlation_matrix(pd.DataFrame({"A": a, "B": b}), min_overlap=30)
    assert np.isnan(corr.loc["A", "B"])


def test_residualize_removes_common_factor():
    rng = np.random.default_rng(3)
    idx = trading_days(1000)
    factor = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    a = 1.2 * factor + pd.Series(rng.normal(0, 0.002, 1000), index=idx)
    b = 0.8 * factor + pd.Series(rng.normal(0, 0.002, 1000), index=idx)
    returns = pd.DataFrame({"A": a, "B": b})

    before = correlation_matrix(returns).loc["A", "B"]
    after = correlation_matrix(residualize(returns, factor)).loc["A", "B"]
    assert before > 0.9
    assert abs(after) < 0.2


def test_correlation_distance_is_zero_for_perfect_correlation():
    corr = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert correlation_distance(corr)[0] == pytest.approx(0.0)


def test_correlation_distance_is_two_for_perfect_anticorrelation():
    corr = pd.DataFrame(
        [[1.0, -1.0], [-1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert correlation_distance(corr)[0] == pytest.approx(2.0)


def test_clustering_recovers_three_synthetic_blocks():
    rng = np.random.default_rng(9)
    idx = trading_days(600)
    columns: dict[str, pd.Series] = {}
    for block in range(3):
        driver = rng.normal(0, 0.012, 600)
        for member in range(4):
            noise = rng.normal(0, 0.003, 600)
            columns[f"B{block}_{member}"] = pd.Series(driver + noise, index=idx)

    result = cluster_universe(
        panel_from_returns(columns),
        benchmark=None,
        threshold=0.6,
        min_size=3,
        window=600,
    )
    assert len(result.clusters) == 3
    for cluster in result.clusters:
        prefixes = {name.split("_")[0] for name in cluster.members}
        assert len(prefixes) == 1
        assert cluster.mean_correlation > 0.6


def test_clustering_drops_groups_below_min_size():
    rng = np.random.default_rng(4)
    idx = trading_days(600)
    driver = rng.normal(0, 0.012, 600)
    columns = {
        "A": pd.Series(driver + rng.normal(0, 0.002, 600), index=idx),
        "B": pd.Series(driver + rng.normal(0, 0.002, 600), index=idx),
        "LONER": pd.Series(rng.normal(0, 0.012, 600), index=idx),
    }
    result = cluster_universe(
        panel_from_returns(columns), None, threshold=0.6, min_size=3, window=600
    )
    assert result.clusters == ()
    assert set(result.assignments) == {-1}


def test_cluster_result_exposes_both_matrices():
    rng = np.random.default_rng(5)
    idx = trading_days(400)
    factor = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    columns = {
        f"T{i}": pd.Series(factor + rng.normal(0, 0.003, 400), index=idx)
        for i in range(4)
    }
    panel = panel_from_returns(columns)
    benchmark = 100.0 * np.exp(factor.cumsum())
    result = cluster_universe(panel, benchmark, threshold=0.6, min_size=3, window=400)
    assert result.raw_corr.shape == (4, 4)
    assert result.residual_corr.shape == (4, 4)
    raw_mean = result.raw_corr.to_numpy()[np.triu_indices(4, 1)].mean()
    residual_mean = result.residual_corr.to_numpy()[np.triu_indices(4, 1)].mean()
    assert raw_mean > residual_mean
