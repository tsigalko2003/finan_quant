from datetime import date

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.data.fetcher import (
    FakeFetcher,
    FetchError,
    RateLimited,
    YFinanceFetcher,
    _is_rate_limited,
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


def test_is_rate_limited_detects_429():
    """Test detection of 429 status code in error message."""
    exc = RuntimeError("429 Client Error: Too Many Requests")
    assert _is_rate_limited(exc)


def test_is_rate_limited_detects_too_many_requests():
    """Test detection of 'too many requests' text."""
    exc = RuntimeError("Too Many Requests for url: https://query2.finance.yahoo.com/...")
    assert _is_rate_limited(exc)


def test_is_rate_limited_rejects_other_errors():
    """Test that non-rate-limit errors are not detected as rate limits."""
    exc = RuntimeError("Connection refused")
    assert not _is_rate_limited(exc)


def test_yfinance_fetcher_uses_long_backoff_for_rate_limits():
    """Rate-limited errors should use the long backoff sequence."""
    frame = make_ohlcv(exponential_trend(10, 0.001))
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    attempts = []

    class TracksBackoff:
        def history(self, **kwargs):
            if len(attempts) < 2:
                # Raise a 429 error
                attempts.append(None)
                raise RuntimeError("429 Client Error: Too Many Requests")
            return frame

    sleep_calls = []
    fetcher = YFinanceFetcher(
        sleep=lambda x: sleep_calls.append(x),
        max_retries=3,
        ticker_factory=lambda symbol: TracksBackoff(),
        rate_limit_backoff_seconds=(100.0, 200.0, 300.0),
    )
    out = fetcher.history("NVDA", date(2020, 1, 1), None)
    # Should have slept with the long backoff sequence: 100, 200
    assert sleep_calls == [100.0, 200.0]


def test_yfinance_fetcher_uses_short_backoff_for_normal_errors():
    """Non-rate-limit errors should use the short exponential backoff."""
    frame = make_ohlcv(exponential_trend(10, 0.001))
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    attempts = []

    class TrackBackoff:
        def history(self, **kwargs):
            if len(attempts) < 2:
                attempts.append(None)
                raise RuntimeError("Connection timeout")
            return frame

    sleep_calls = []
    fetcher = YFinanceFetcher(
        sleep=lambda x: sleep_calls.append(x),
        max_retries=3,
        ticker_factory=lambda symbol: TrackBackoff(),
    )
    out = fetcher.history("NVDA", date(2020, 1, 1), None)
    # Should have slept with the short exponential backoff: 2^0=1, 2^1=2
    assert sleep_calls == [1.0, 2.0]


def test_yfinance_fetcher_raises_rate_limited_after_max_retries():
    """Exhausting retries on a rate limit should raise RateLimited."""

    class AlwaysRateLimited:
        def history(self, **kwargs):
            raise RuntimeError("429 Client Error: Too Many Requests")

    fetcher = YFinanceFetcher(
        sleep=lambda _: None,
        max_retries=3,
        ticker_factory=lambda symbol: AlwaysRateLimited(),
        rate_limit_backoff_seconds=(60.0, 180.0, 420.0),
    )
    with pytest.raises(RateLimited) as exc_info:
        fetcher.history("NVDA", date(2020, 1, 1), None)
    assert "rate-limited" in str(exc_info.value).lower()
    assert "resumable" in str(exc_info.value).lower()
    assert "420" in str(exc_info.value)
