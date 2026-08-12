from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from screener_sector.backtest.runner import (
    alarm_series,
    fit_alarm_gate,
    run_backtest,
    run_fold,
)
from screener_sector.backtest.walkforward import Fold
from screener_sector.config import Config
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths

CONFIG_DIR = Path("/app/config")


def cyclical_panel(n_years: int = 8, seed: int = 3) -> dict[str, pd.DataFrame]:
    """A correlated group with repeated drawdown-and-recovery cycles.

    The group factor is deliberately absent from the benchmark, so it survives
    residualization and the members actually cluster. Building the members as
    benchmark-plus-noise would leave nothing but independent noise after the
    SOXX factor is regressed out, and no cluster would ever form.
    """
    rng = np.random.default_rng(seed)
    n = 252 * n_years
    idx = pd.bdate_range("2018-01-01", periods=n)
    cycle = np.sin(np.arange(n) * 2 * np.pi / 252) * 0.004
    market = cycle + rng.normal(0.0003, 0.010, n)   # this is SOXX
    group = rng.normal(0.0, 0.008, n)               # shared, NOT in SOXX
    frames = {}
    for i in range(5):
        noise = rng.normal(0, 0.003, n)
        close = pd.Series(100.0 * np.exp(np.cumsum(market + group + noise)), index=idx)
        frames[f"T{i}"] = make_ohlcv(close)
    frames["SOXX"] = make_ohlcv(
        pd.Series(100.0 * np.exp(np.cumsum(market)), index=idx)
    )
    return frames


@pytest.fixture
def env(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    frames = cyclical_panel(n_years=8)
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2017, 1, 1))
    return paths, store, [t for t in frames if t != "SOXX"]


def test_alarm_series_returns_boolean_series_per_ticker(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    series = alarm_series(
        store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), alarm_gate=60.0
    )
    assert set(series) <= set(tickers)
    for value in series.values():
        assert value.dtype == bool


def test_alarm_series_is_confined_to_the_requested_window(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    series = alarm_series(
        store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), alarm_gate=60.0
    )
    for value in series.values():
        assert value.index.min() >= pd.Timestamp("2021-01-01")
        assert value.index.max() <= pd.Timestamp("2021-12-31")


def test_lower_gate_produces_at_least_as_many_signals(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    loose = alarm_series(store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), 40.0)
    tight = alarm_series(store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), 80.0)
    assert sum(s.sum() for s in loose.values()) >= sum(s.sum() for s in tight.values())


def test_fit_alarm_gate_returns_a_candidate(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    fold = Fold(0, date(2018, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31), False)
    candidates = [40.0, 50.0, 60.0, 70.0]
    assert fit_alarm_gate(store, tickers, cfg, fold, candidates) in candidates


def test_run_fold_reports_expected_columns(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    fold = Fold(0, date(2018, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31), False)
    row = run_fold(store, tickers, cfg, fold)
    assert list(row.columns) == [
        "fold", "partial", "test_start", "test_end", "tickers", "gate",
        "precision", "recall", "f1", "signals", "labels", "mean_lead_days",
    ]
    assert row["fold"].iloc[0] == 0


def test_run_fold_never_fits_on_test_data(env):
    """Gate fitted for a fold must not change when test-window prices change."""
    paths, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    fold = Fold(0, date(2018, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31), False)
    candidates = [40.0, 50.0, 60.0, 70.0]
    before = fit_alarm_gate(store, tickers, cfg, fold, candidates)

    for ticker in tickers:
        frame = store.load(ticker)
        mask = frame.index >= pd.Timestamp("2021-01-01")
        frame.loc[mask, ["open", "high", "low", "close"]] *= 0.5
        frame.to_parquet(paths.price_file(ticker))

    assert fit_alarm_gate(store, tickers, cfg, fold, candidates) == before


def test_run_backtest_produces_all_result_frames(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    result = run_backtest(store, tickers, cfg, end=date(2025, 12, 31))
    assert not result.per_fold.empty
    assert list(result.fitted_gates.columns) == ["fold", "gate"]
    assert "mean_edge" in result.edges.columns


def test_run_backtest_marks_partial_final_fold(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    result = run_backtest(store, tickers, cfg, end=date(2025, 6, 30))
    assert bool(result.per_fold["partial"].iloc[-1]) is True


def test_backtest_actually_exercises_the_signal_path(env):
    """Guards against a fixture that silently produces no clusters, which
    would let every other backtest test pass without running the alarm."""
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    signals = alarm_series(
        store, tickers, cfg, date(2024, 1, 1), date(2025, 12, 31), alarm_gate=40.0
    )
    assert signals, "alarm_series returned no tickers - no clusters formed"
    assert sum(int(s.sum()) for s in signals.values()) > 0, "no alarm ever fired"
