from pathlib import Path

import pandas as pd

from qtrends.config import load_config
from qtrends.data import generate_synthetic_csv
from qtrends.pipeline import run_pipeline


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    config = load_config("configs/sample.yaml")
    data_path = generate_synthetic_csv(tmp_path / "sample.csv", periods=850)
    config.data.csv_path = str(data_path)
    config.features.pca_lookback = 126
    config.hmm.min_train_size = 180
    config.hmm.refit_every = 63
    config.hmm.n_iter = 75
    config.hmm.seeds = [7]
    config.xgboost.min_train_size = 180
    config.xgboost.refit_every = 63
    config.xgboost.n_estimators = 40
    output = tmp_path / "output"

    artifacts = run_pipeline(config, output)

    assert artifacts.latest_signal["signal"] in {"bullish", "bearish", "neutral"}
    for expected in [
        "signals.csv",
        "metrics.json",
        "latest_signal.json",
        "feature_importance.csv",
        "summary.md",
        "xgboost_model.joblib",
        "hmm_model.joblib",
    ]:
        assert (output / expected).exists()
    signals = pd.read_csv(output / "signals.csv")
    assert signals["xgb_probability_oos"].notna().any()

