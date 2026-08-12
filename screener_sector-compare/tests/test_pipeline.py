from __future__ import annotations

import numpy as np
import pandas as pd

from sector_screener.config import Settings
from sector_screener.pipeline import analyze_stage
from sector_screener.universe import Universe


class OfflineCache:
    def __init__(self, panels):
        self.panels = panels

    def load(self, ticker, start, end):
        frame = self.panels[ticker]
        return frame.loc[(frame.index >= start) & (frame.index < end)]


def test_offline_analysis_writes_complete_artifacts(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-02", periods=430, freq="B")
    common = rng.normal(0.0003, 0.014, len(dates))
    common[180:220] -= 0.012
    common[220:250] += 0.009
    panels = {}
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for number, ticker in enumerate(tickers):
        returns = common * (0.8 + number * 0.05) + rng.normal(0, 0.004, len(dates))
        close = 100 * np.cumprod(1 + returns)
        panels[ticker] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000 + number,
            },
            index=dates.rename("date"),
        )

    raw = {
        "stage": "poc",
        "seed": 42,
        "universe": {"max_tickers": 6},
        "data": {
            "provider": "yahoo",
            "interval": "1d",
            "auto_adjust": True,
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "outputs"),
            "min_coverage": 0.8,
            "min_rows": 200,
        },
        "analysis": {
            "short_window": 20,
            "mid_window": 60,
            "long_window": 120,
            "correlation_window": 40,
            "pca_window": 40,
            "hmm_states": 3,
            "hmm_min_train": 80,
            "hmm_refit_every": 40,
            "capture_window": 60,
            "rebound_horizon": 10,
            "rebound_return": 0.04,
            "max_adverse_return": -0.03,
            "correction_drawdown": -0.05,
            "minimum_correlation": 0.5,
            "rebound_probability": 0.7,
            "minimum_breadth_thrust": 0.1,
            "xgb_estimators": 10,
            "validation_splits": 2,
        },
    }
    settings = Settings(raw=raw, config_dir=tmp_path)
    universe = Universe("synthetic", tickers, "test", "test", {})
    monkeypatch.setattr("sector_screener.pipeline.make_cache", lambda _: OfflineCache(panels))
    result = analyze_stage(
        settings, universe, pd.Timestamp("2020-01-02"), pd.Timestamp("2022-01-01")
    )
    from pathlib import Path

    run_dir = Path(result["run_dir"])
    assert (run_dir / "report.html").exists()
    assert (run_dir / "alert.json").exists()
    assert (run_dir / "validation.json").exists()
    assert (run_dir / "ticker_rankings.csv").exists()
