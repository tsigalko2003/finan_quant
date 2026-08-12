from pathlib import Path

from qtrends.config import load_config


def test_sample_config_is_valid() -> None:
    config = load_config(Path("configs/sample.yaml"))
    assert config.data.provider == "csv"
    assert config.data.benchmark == "MARKET"
    assert len(config.data.tickers) == 5

