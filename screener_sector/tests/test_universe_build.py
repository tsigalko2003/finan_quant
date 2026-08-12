from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.config import UniverseFilters
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths
from screener_sector.universe.build import (
    build_universe,
    liquidity_stats,
    load_universe,
    save_universe,
)
from screener_sector.universe.classify import ThemeRules

CONFIG_DIR = Path("/app/config")
FILTERS = UniverseFilters(
    min_price=2.0, min_dollar_volume=5_000_000.0, min_history_days=250
)


def ohlcv(n=300, price_scale=1.0, volume=1_000_000.0):
    close = exponential_trend(n, 0.0005, noise=0.01, seed=4) * price_scale
    vol = pd.Series(volume, index=close.index)
    return make_ohlcv(close, vol)


@pytest.fixture
def env(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    frames = {
        "NVDA": ohlcv(),
        "PENNY": ohlcv(price_scale=0.01),
        "THIN": ohlcv(volume=100.0),
        "NEW": ohlcv(n=100),
        "KO": ohlcv(),
    }
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2015, 1, 1))
    symbols = pd.DataFrame(
        {
            "ticker": list(frames),
            "name": ["NVIDIA", "Penny Chips", "Thin Optics", "New Silicon", "Coca-Cola"],
            "exchange": ["NASDAQ"] * 5,
            "etf": [False] * 5,
        }
    )
    info = pd.DataFrame(
        {
            "ticker": list(frames),
            "long_name": symbols["name"],
            "sector": ["Technology"] * 4 + ["Consumer Defensive"],
            "industry": ["Semiconductors"] * 4 + ["Beverages"],
            "summary": [
                "GPU and semiconductor processor designer.",
                "Makes semiconductor parts.",
                "Optical transceiver maker.",
                "Wafer processing.",
                "Sells soft drinks.",
            ],
            "quote_type": ["EQUITY"] * 5,
            "fetched_at": ["2026-08-12"] * 5,
        }
    )
    return paths, store, symbols, info


def test_liquidity_stats_computes_median_dollar_volume():
    frame = ohlcv(n=100, volume=1_000_000.0)
    median_dv, last_close, days = liquidity_stats(frame, window=60)
    assert days == 100
    assert last_close == pytest.approx(frame["close"].iloc[-1])
    assert median_dv > 0


def test_included_ticker_passes_all_filters(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["NVDA", "included"]) is True
    assert df.loc["NVDA", "reason"] == ""


def test_low_price_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["PENNY", "included"]) is False
    assert "price" in df.loc["PENNY", "reason"]


def test_illiquid_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["THIN", "included"]) is False
    assert "dollar_volume" in df.loc["THIN", "reason"]


def test_short_history_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["NEW", "included"]) is False
    assert "history" in df.loc["NEW", "reason"]


def test_off_theme_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["KO", "included"]) is False
    assert "theme" in df.loc["KO", "reason"]


def test_rejected_rows_are_retained_not_dropped(env):
    paths, store, symbols, info = env
    df = build_universe(paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS)
    assert len(df) == 5


def test_themes_are_recorded_as_pipe_delimited(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert "semiconductor" in df.loc["NVDA", "themes"]


def test_save_and_load_included_only(env):
    paths, store, symbols, info = env
    df = build_universe(paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS)
    save_universe(paths, df)
    assert len(load_universe(paths, included_only=True)) == 1
    assert len(load_universe(paths, included_only=False)) == 5
