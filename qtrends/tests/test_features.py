from pathlib import Path

import numpy as np

from qtrends.config import load_config
from qtrends.data import generate_synthetic_csv, load_market_data
from qtrends.features import build_features, forward_compound_return


def test_feature_pipeline_is_point_in_time(tmp_path: Path) -> None:
    path = generate_synthetic_csv(tmp_path / "prices.csv", periods=500)
    config = load_config("configs/sample.yaml")
    config.data.csv_path = str(path)
    market = load_market_data(config.data)
    features, residuals = build_features(
        market, config.data.tickers, config.data.benchmark, config.features
    )
    assert features.index.equals(market.close.index)
    assert residuals.shape[1] == len(config.data.tickers)
    assert features["pca_explained_variance"].dropna().between(0.0, 1.0).all()
    assert features["breadth_above_ma20"].dropna().between(0.0, 1.0).all()

    truncated_market = type(market)(
        close=market.close.iloc[:-10],
        volume=market.volume.iloc[:-10],
    )
    truncated, _ = build_features(
        truncated_market, config.data.tickers, config.data.benchmark, config.features
    )
    common = truncated.index[-20:]
    np.testing.assert_allclose(
        truncated.loc[common, "pca_factor"],
        features.loc[common, "pca_factor"],
        equal_nan=True,
    )


def test_forward_return_uses_only_future_values() -> None:
    import pandas as pd

    returns = pd.Series([0.10, 0.20, -0.10, 0.05])
    result = forward_compound_return(returns, horizon=2)
    assert np.isclose(result.iloc[0], (1.20 * 0.90) - 1.0)
    assert np.isclose(result.iloc[1], (0.90 * 1.05) - 1.0)
    assert result.iloc[-1] != result.iloc[-1]  # NaN

