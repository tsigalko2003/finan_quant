from datetime import date
from pathlib import Path

import pytest

from screener_sector.config import Config

CONFIG_DIR = Path("/app/config")


def test_dev_profile_loads():
    cfg = Config.load(CONFIG_DIR, "dev")
    assert cfg.profile == "dev"
    assert cfg.start == date(2022, 1, 1)
    assert cfg.end is None
    assert cfg.universe_mode == "static"


def test_prod_profile_loads():
    cfg = Config.load(CONFIG_DIR, "prod")
    assert cfg.start == date(2010, 1, 1)
    assert cfg.universe_mode == "discover"


def test_defaults_are_shared_across_profiles():
    dev = Config.load(CONFIG_DIR, "dev")
    prod = Config.load(CONFIG_DIR, "prod")
    assert dev.windows == prod.windows
    assert dev.benchmark == prod.benchmark == "SOXX"


def test_profile_overrides_nested_default():
    dev = Config.load(CONFIG_DIR, "dev")
    prod = Config.load(CONFIG_DIR, "prod")
    assert dev.backtest.initial_fit_years == 2
    assert prod.backtest.initial_fit_years == 5
    # unrelated backtest values still come from defaults
    assert dev.backtest.label_k == prod.backtest.label_k == 10


def test_dev_profile_loads_static_tickers():
    cfg = Config.load(CONFIG_DIR, "dev")
    assert "NVDA" in cfg.static_tickers
    assert len(cfg.static_tickers) >= 25


def test_prod_profile_has_no_static_tickers():
    cfg = Config.load(CONFIG_DIR, "prod")
    assert cfg.static_tickers == ()


def test_trend_weights_sum_to_one():
    cfg = Config.load(CONFIG_DIR, "dev")
    w = cfg.trend_weights
    assert abs(w.slope + w.r2 + w.adx + w.ma_stack - 1.0) < 1e-9


def test_rebound_weights_sum_to_one():
    w = Config.load(CONFIG_DIR, "dev").rebound_weights
    total = w.breadth + w.stretch + w.oscillator + w.volume + w.confirmation
    assert abs(total - 1.0) < 1e-9


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        Config.load(CONFIG_DIR, "nope")


def test_config_is_frozen():
    cfg = Config.load(CONFIG_DIR, "dev")
    with pytest.raises(Exception):
        cfg.profile = "prod"


def test_fetch_start_is_warmup_years_before_start():
    dev = Config.load(CONFIG_DIR, "dev")
    assert dev.fetch_start == date(2020, 1, 1)
    prod = Config.load(CONFIG_DIR, "prod")
    assert prod.fetch_start == date(2006, 1, 1)


def test_fetch_start_handles_feb_29_safely():
    """Fetch start must handle Feb 29 in leap years without crashing.

    When start is Feb 29 and target year is not a leap year, clamp to Feb 28.
    """
    from screener_sector.config import BacktestParams, Windows, TrendWeights, ReboundWeights, UniverseFilters, NetworkParams

    # Create a config with Feb 29 start and warmup_years that would make target non-leap
    cfg = Config(
        profile="test",
        start=date(2020, 2, 29),  # Leap year
        end=None,
        universe_mode="static",
        static_tickers=(),
        benchmark="SOXX",
        windows=Windows(short=10, mid=20, corr=60),
        trend_weights=TrendWeights(slope=0.25, r2=0.25, adx=0.25, ma_stack=0.25),
        rebound_weights=ReboundWeights(breadth=0.2, stretch=0.2, oscillator=0.2, volume=0.2, confirmation=0.2),
        corr_threshold=0.7,
        min_cluster_size=2,
        filters=UniverseFilters(min_price=5.0, min_dollar_volume=10_000_000, min_history_days=100),
        backtest=BacktestParams(warmup_years=5, label_k=10, label_forward_days=20, label_min_return=0.05, initial_fit_years=2, step_years=1, horizons=(5, 10, 20)),
        network=NetworkParams(enrich_pause_seconds=1.5, rate_limit_backoff_seconds=(60.0, 180.0, 420.0)),
    )

    # 2020 is a leap year, so 2015 is not a leap year
    fetch_start = cfg.fetch_start
    # Should be Feb 28, 2015 (clamped from Feb 29)
    assert fetch_start == date(2015, 2, 28)
