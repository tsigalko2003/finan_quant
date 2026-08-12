from datetime import date

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.data.fetcher import (
    FakeFetcher,
    FetchError,
    YFinanceFetcher,
    normalize_frame,
)


def test_normalize_lowercases_yahoo_columns():
    raw = make_ohlcv(exponential_trend(10, 0.001))
    raw.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = normalize_frame(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_rejects_missing_columns():
    df = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
    with pytest.raises(FetchError):
        normalize_frame(df)


def test_normalize_drops_rows_with_null_close():
    df = make_ohlcv(exponential_trend(5, 0.001))
    df.iloc[2, df.columns.get_loc("close")] = None
    assert len(normalize_frame(df)) == 4


def test_fake_fetcher_returns_configured_data():
    df = make_ohlcv(exponential_trend(10, 0.001))
    fetcher = FakeFetcher({"NVDA": df})
    out = fetcher.history("NVDA", date(2020, 1, 1), None)
    assert len(out) == 10


def test_fake_fetcher_raises_for_configured_failures():
    fetcher = FakeFetcher({}, fail={"BAD"})
    with pytest.raises(FetchError):
        fetcher.history("BAD", date(2020, 1, 1), None)


def test_fake_fetcher_slices_by_start_date():
    df = make_ohlcv(exponential_trend(20, 0.001))
    fetcher = FakeFetcher({"NVDA": df})
    cutoff = df.index[10].date()
    out = fetcher.history("NVDA", cutoff, None)
    assert out.index.min().date() >= cutoff


def test_yfinance_fetcher_retries_then_raises():
    calls = []

    class AlwaysFails:
        def history(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("boom")

    fetcher = YFinanceFetcher(
        sleep=lambda _: None,
        max_retries=3,
        ticker_factory=lambda symbol: AlwaysFails(),
    )
    with pytest.raises(FetchError):
        fetcher.history("NVDA", date(2020, 1, 1), None)
    assert len(calls) == 3


def test_yfinance_fetcher_succeeds_on_second_attempt():
    frame = make_ohlcv(exponential_trend(10, 0.001))
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    attempts = {"n": 0}

    class FlakyOnce:
        def history(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("rate limited")
            return frame

    fetcher = YFinanceFetcher(
        sleep=lambda _: None,
        max_retries=3,
        ticker_factory=lambda symbol: FlakyOnce(),
    )
    out = fetcher.history("NVDA", date(2020, 1, 1), None)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert attempts["n"] == 2
