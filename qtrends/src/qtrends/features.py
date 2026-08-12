from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qtrends.config import FeatureConfig
from qtrends.data import MarketData


def _compound(values: np.ndarray) -> float:
    return float(np.prod(1.0 + values) - 1.0)


def _rolling_average_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index, name="average_pairwise_correlation")
    for end in range(window - 1, len(returns)):
        sample = returns.iloc[end - window + 1 : end + 1].dropna(axis=1, thresh=window // 2)
        if sample.shape[1] < 2:
            continue
        correlation = sample.corr().to_numpy()
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        finite = upper[np.isfinite(upper)]
        if finite.size:
            output.iloc[end] = float(finite.mean())
    return output


def _rolling_pca(residual_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    result = pd.DataFrame(
        index=residual_returns.index,
        columns=["pca_factor", "pca_explained_variance"],
        dtype=float,
    )
    for end in range(window - 1, len(residual_returns)):
        sample = residual_returns.iloc[end - window + 1 : end + 1]
        sample = sample.dropna(axis=1, thresh=int(window * 0.8))
        if sample.shape[1] < 2:
            continue
        sample = sample.fillna(sample.mean())
        scale = sample.std(ddof=0).replace(0.0, np.nan)
        standardized = ((sample - sample.mean()) / scale).dropna(axis=1)
        if standardized.shape[1] < 2:
            continue
        model = PCA(n_components=1, svd_solver="full")
        scores = model.fit_transform(standardized)
        sign = 1.0 if model.components_[0].sum() >= 0 else -1.0
        result.iloc[end, result.columns.get_loc("pca_factor")] = float(scores[-1, 0] * sign)
        result.iloc[end, result.columns.get_loc("pca_explained_variance")] = float(
            model.explained_variance_ratio_[0]
        )
    return result


def build_features(
    market: MarketData,
    tickers: list[str],
    benchmark: str,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = market.close[tickers + [benchmark]].replace([np.inf, -np.inf], np.nan)
    volume = market.volume.reindex(index=close.index, columns=tickers + [benchmark])
    returns = close.pct_change(fill_method=None)
    stock_returns = returns[tickers]
    benchmark_return = returns[benchmark]

    benchmark_variance = benchmark_return.rolling(config.beta_lookback).var()
    beta = stock_returns.apply(
        lambda series: series.rolling(config.beta_lookback).cov(benchmark_return)
        / benchmark_variance
    )
    residual_returns = stock_returns - beta.mul(benchmark_return, axis=0)

    membership = (
        market.membership.reindex(index=close.index, columns=tickers).fillna(False)
        if market.membership is not None
        else pd.DataFrame(True, index=close.index, columns=tickers)
    )
    active_count = membership.sum(axis=1)
    required_count = np.ceil(active_count * config.min_constituent_coverage).clip(lower=2)
    available_count = residual_returns.notna().sum(axis=1)
    eligible = available_count >= required_count

    features = pd.DataFrame(index=close.index)
    features["group_return"] = stock_returns.mean(axis=1).where(eligible)
    features["group_excess_return"] = residual_returns.mean(axis=1).where(eligible)
    features["breadth_positive_1d"] = residual_returns.gt(0).mean(axis=1).where(eligible)
    features["cross_sectional_dispersion"] = residual_returns.std(axis=1).where(eligible)
    features["active_constituents"] = active_count
    features["constituent_coverage"] = available_count / active_count.replace(0, np.nan)

    for window in config.breadth_ma_windows:
        moving_average = close[tickers].rolling(window).mean()
        valid = close[tickers].notna() & moving_average.notna()
        denominator = valid.sum(axis=1).replace(0, np.nan)
        features[f"breadth_above_ma{window}"] = (
            (close[tickers] > moving_average).where(valid).sum(axis=1) / denominator
        )

    high_window = min(config.trend_lookbacks)
    rolling_high = close[tickers].rolling(high_window).max()
    valid_high = close[tickers].notna() & rolling_high.notna()
    features[f"breadth_new_high_{high_window}"] = (
        (close[tickers] >= rolling_high).where(valid_high).sum(axis=1)
        / valid_high.sum(axis=1).replace(0, np.nan)
    )

    volume_ratio = volume[tickers] / volume[tickers].rolling(20).mean()
    valid_volume = volume_ratio.notna() & stock_returns.notna()
    features["advancing_volume_breadth"] = (
        ((stock_returns > 0) & (volume_ratio > 1.0)).where(valid_volume).sum(axis=1)
        / valid_volume.sum(axis=1).replace(0, np.nan)
    )
    features["median_volume_ratio"] = volume_ratio.median(axis=1)

    for window in config.trend_lookbacks:
        features[f"relative_return_{window}"] = features["group_excess_return"].rolling(window).apply(
            _compound, raw=True
        )

    features["realized_volatility"] = (
        features["group_return"].rolling(config.volatility_lookback).std() * np.sqrt(252.0)
    )
    features["average_pairwise_correlation"] = _rolling_average_correlation(
        stock_returns, config.correlation_lookback
    )
    features = features.join(_rolling_pca(residual_returns, config.pca_lookback))
    features = features.astype(float).replace([np.inf, -np.inf], np.nan)
    return features, residual_returns


def forward_compound_return(series: pd.Series, horizon: int) -> pd.Series:
    """Forward return from t+1 through t+horizon, aligned to information at t."""
    compounded = (1.0 + series).rolling(horizon).apply(np.prod, raw=True) - 1.0
    return compounded.shift(-horizon)
