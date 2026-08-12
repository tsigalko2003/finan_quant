from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv, trading_days
from screener_sector.config import Config
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths
from screener_sector.pipeline import load_frames, run_screen, save_screen

CONFIG_DIR = Path("/app/config")


@pytest.fixture
def env(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    rng = np.random.default_rng(12)
    idx = trading_days(400)
    driver = rng.normal(0.0005, 0.015, 400)
    frames = {}
    for i in range(5):
        noise = rng.normal(0, 0.004, 400)
        close = pd.Series(100.0 * np.exp(np.cumsum(driver + noise)), index=idx)
        frames[f"T{i}"] = make_ohlcv(close)
    frames["SOXX"] = make_ohlcv(
        pd.Series(100.0 * np.exp(np.cumsum(driver)), index=idx)
    )
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2015, 1, 1))
    return paths, store, list(frames), idx


def test_load_frames_truncates_at_as_of(env):
    paths, store, tickers, idx = env
    as_of = idx[300].date()
    frames = load_frames(store, tickers, as_of)
    for frame in frames.values():
        assert frame.index.max().date() <= as_of


def test_run_screen_produces_all_sections(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers, cfg, idx[350].date())
    assert not output.trend.empty
    assert output.clusters.raw_corr.shape[0] > 0
    assert output.as_of == idx[350].date()


def test_run_screen_excludes_benchmark_from_the_screen(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers, cfg, idx[350].date())
    assert "SOXX" not in set(output.trend["ticker"])


def test_run_screen_is_point_in_time(env):
    """Mutating data after as_of must not change any computed value."""
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    as_of = idx[300].date()
    before = run_screen(store, tickers, cfg, as_of)

    for ticker in tickers:
        frame = store.load(ticker)
        mask = frame.index > pd.Timestamp(as_of)
        frame.loc[mask, ["open", "high", "low", "close"]] *= 3.0
        frame.to_parquet(paths.price_file(ticker))

    after = run_screen(store, tickers, cfg, as_of)
    pd.testing.assert_frame_equal(before.trend, after.trend)
    pd.testing.assert_frame_equal(before.strength, after.strength)
    pd.testing.assert_frame_equal(before.rebound, after.rebound)


def test_run_screen_ignores_tickers_with_no_data(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers + ["MISSING"], cfg, idx[350].date())
    assert "MISSING" not in set(output.trend["ticker"])


def test_save_screen_writes_under_profile_namespace(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers, cfg, idx[350].date())
    directory = save_screen(paths, output, "dev")
    assert directory.is_relative_to(paths.derived_dir("dev"))
    assert (directory / "trend.csv").exists()
    assert (directory / "rebound.csv").exists()
