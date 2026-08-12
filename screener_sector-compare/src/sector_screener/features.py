from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _rolling_pairwise_median(returns: pd.DataFrame, window: int) -> pd.Series:
    values = pd.Series(index=returns.index, dtype=float)
    for position in range(window - 1, len(returns)):
        sample = returns.iloc[position - window + 1 : position + 1].dropna(
            axis=1, thresh=window // 2
        )
        if sample.shape[1] < 3:
            continue
        corr = sample.corr(min_periods=max(10, window // 2)).to_numpy()
        triangle = corr[np.triu_indices_from(corr, k=1)]
        finite = triangle[np.isfinite(triangle)]
        if finite.size:
            values.iloc[position] = float(np.median(finite))
    return values


def _rolling_pca(returns: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    explained = pd.Series(index=returns.index, dtype=float)
    score = pd.Series(index=returns.index, dtype=float)
    for position in range(window - 1, len(returns)):
        sample = returns.iloc[position - window + 1 : position + 1]
        sample = sample.dropna(axis=1, thresh=max(20, int(window * 0.8)))
        sample = sample.dropna(axis=0)
        if sample.shape[0] < max(20, window // 2) or sample.shape[1] < 3:
            continue
        scale = sample.std(ddof=0).replace(0, np.nan)
        normalized = ((sample - sample.mean()) / scale).dropna(axis=1)
        if normalized.shape[1] < 3:
            continue
        model = PCA(n_components=1)
        transformed = model.fit_transform(normalized).ravel()
        basket = sample[normalized.columns].mean(axis=1).to_numpy()
        if np.corrcoef(transformed, basket)[0, 1] < 0:
            transformed *= -1
        explained.iloc[position] = float(model.explained_variance_ratio_[0])
        score.iloc[position] = float(transformed[-1])
    return explained, score


def build_sector_features(
    close: pd.DataFrame, volume: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.Series]:
    close = close.sort_index().ffill(limit=2)
    returns = close.pct_change(fill_method=None)
    sector_return = returns.mean(axis=1, skipna=True)
    sector_index = (1.0 + sector_return.fillna(0.0)).cumprod()
    short = int(params["short_window"])
    mid = int(params["mid_window"])
    long = int(params["long_window"])

    features = pd.DataFrame(index=close.index)
    features["sector_return_1d"] = sector_return
    for window in (5, short, mid):
        features[f"sector_return_{window}d"] = sector_index.pct_change(window)
    features["realized_vol_20d"] = sector_return.rolling(short).std() * np.sqrt(252)
    features["drawdown_60d"] = sector_index / sector_index.rolling(mid).max() - 1.0
    features["drawdown_long"] = sector_index / sector_index.rolling(long).max() - 1.0
    features["advance_fraction"] = (returns > 0).sum(axis=1) / returns.notna().sum(axis=1)
    features["breadth_above_short_ma"] = (close > close.rolling(short).mean()).sum(
        axis=1
    ) / close.notna().sum(axis=1)
    features["breadth_above_mid_ma"] = (close > close.rolling(mid).mean()).sum(
        axis=1
    ) / close.notna().sum(axis=1)
    features["breadth_positive_5d"] = (close.pct_change(5) > 0).sum(axis=1) / close.notna().sum(
        axis=1
    )
    features["breadth_positive_20d"] = (close.pct_change(short) > 0).sum(
        axis=1
    ) / close.notna().sum(axis=1)
    features["breadth_thrust_5d"] = features["breadth_positive_5d"].diff(5)
    features["cross_section_dispersion"] = returns.std(axis=1)
    features["median_pairwise_correlation"] = _rolling_pairwise_median(
        returns, int(params["correlation_window"])
    )
    explained, score = _rolling_pca(returns, int(params["pca_window"]))
    features["pc1_explained_variance"] = explained
    features["pc1_score"] = score
    dollar_volume = close * volume.reindex_like(close)
    features["median_dollar_volume"] = dollar_volume.median(axis=1)
    features["eligible_ticker_fraction"] = close.notna().sum(axis=1) / close.shape[1]

    vol = features["realized_vol_20d"].replace(0, np.nan)
    features["short_trend_score"] = features[f"sector_return_{short}d"] / vol
    features["mid_trend_score"] = features[f"sector_return_{mid}d"] / vol
    return features.replace([np.inf, -np.inf], np.nan), sector_index


def rank_tickers(close: pd.DataFrame, features: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = close.pct_change(fill_method=None)
    sector_returns = returns.mean(axis=1)
    sample = returns.iloc[-window:]
    sector_sample = sector_returns.iloc[-window:]
    sector_index = (1 + sector_sample.fillna(0)).cumprod()
    rows: list[dict] = []
    for ticker in close.columns:
        stock = sample[ticker]
        negative = sector_sample < 0
        positive = sector_sample > 0
        downside_denominator = sector_sample[negative].mean()
        upside_denominator = sector_sample[positive].mean()
        downside = stock[negative].mean() / downside_denominator if downside_denominator else np.nan
        upside = stock[positive].mean() / upside_denominator if upside_denominator else np.nan
        stock_index = (1 + stock.fillna(0)).cumprod()
        relative_drawdown = (stock_index / stock_index.cummax() - 1).min() - (
            sector_index / sector_index.cummax() - 1
        ).min()
        momentum_20d = close[ticker].pct_change(20).iloc[-1]
        correlation = stock.corr(sector_sample)
        rows.append(
            {
                "ticker": ticker,
                "downside_capture": downside,
                "upside_capture": upside,
                "relative_drawdown": relative_drawdown,
                "momentum_20d": momentum_20d,
                "sector_correlation": correlation,
            }
        )
    ranking = pd.DataFrame(rows).set_index("ticker")
    ranking["fall_resistance_score"] = -ranking["downside_capture"].rank(pct=True) + ranking[
        "relative_drawdown"
    ].rank(pct=True)
    ranking["rise_strength_score"] = ranking["upside_capture"].rank(pct=True) + ranking[
        "momentum_20d"
    ].rank(pct=True)
    ranking["fall_resistance_rank"] = ranking["fall_resistance_score"].rank(
        ascending=False, method="min"
    )
    ranking["rise_strength_rank"] = ranking["rise_strength_score"].rank(
        ascending=False, method="min"
    )
    return ranking.sort_values(["fall_resistance_rank", "rise_strength_rank"])
