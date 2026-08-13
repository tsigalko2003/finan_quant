"""Price retrieval behind a protocol.

Yahoo's endpoint is unofficial and rate-limits aggressively, so every call is
retried with backoff and failures are raised as FetchError for the caller to
quarantine. Tests use FakeFetcher and never open a socket.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Protocol

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class FetchError(RuntimeError):
    """A ticker could not be retrieved or its data was unusable."""


class NoNewData(FetchError):
    """The cache is already current; no new data to fetch. Not a failure."""


class RateLimited(FetchError):
    """Yahoo Finance has rate-limited the request. The run is resumable after a wait."""


class PriceFetcher(Protocol):
    def history(
        self, ticker: str, start: date, end: date | None
    ) -> pd.DataFrame: ...


def _is_rate_limited(exc: Exception) -> bool:
    """Detect if an exception represents a 429 rate limit response."""
    exc_str = str(exc).lower()
    return "429" in exc_str or "too many requests" in exc_str


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, verify the schema, and drop unusable rows."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise FetchError(f"missing columns: {missing}")
    out = out[REQUIRED_COLUMNS].astype(float)
    out = out.dropna(subset=["close"])
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out.sort_index()


class YFinanceFetcher:
    """Adapter over yfinance with bounded retries and rate-limit awareness."""

    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        ticker_factory: Callable[[str], object] | None = None,
        rate_limit_backoff_seconds: tuple[float, ...] = (60.0, 180.0, 420.0),
    ) -> None:
        self._sleep = sleep
        self._max_retries = max_retries
        self._ticker_factory = ticker_factory or _default_ticker_factory
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds

    def history(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                handle = self._ticker_factory(ticker)
                raw = handle.history(
                    start=start.isoformat(),
                    end=end.isoformat() if end else None,
                    interval="1d",
                    auto_adjust=True,
                )
                if raw is None or len(raw) == 0:
                    raise NoNewData(f"empty history for {ticker}")
                return normalize_frame(raw)
            except NoNewData as exc:
                # No new data is not a retryable error; raise immediately
                raise exc from None
            except Exception as exc:  # noqa: BLE001 - deliberate: retry anything
                last_error = exc
                if attempt < self._max_retries - 1:
                    # Use rate-limit-aware backoff
                    if _is_rate_limited(exc):
                        wait_time = self._rate_limit_backoff_seconds[attempt]
                        # Honor Retry-After header if present and longer
                        if hasattr(exc, "response") and exc.response is not None:
                            retry_after = exc.response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    retry_after_secs = float(retry_after)
                                    wait_time = max(wait_time, retry_after_secs)
                                except (ValueError, TypeError):
                                    pass
                        self._sleep(wait_time)
                    else:
                        self._sleep(2.0**attempt)
        # If we exhausted retries on a rate limit, raise RateLimited
        if _is_rate_limited(last_error):
            raise RateLimited(
                f"Yahoo Finance rate-limited ticker {ticker} after {self._max_retries} attempts. "
                f"The run is resumable. Wait at least {self._rate_limit_backoff_seconds[-1]} seconds "
                f"before retrying."
            ) from last_error
        raise FetchError(f"failed to fetch {ticker}: {last_error}") from last_error


def _default_ticker_factory(symbol: str):
    import yfinance

    return yfinance.Ticker(symbol)


class FakeFetcher:
    """In-memory PriceFetcher for tests."""

    def __init__(
        self, data: dict[str, pd.DataFrame], fail: set[str] | None = None
    ) -> None:
        self._data = data
        self._fail = fail or set()
        self.calls: list[tuple[str, date, date | None]] = []

    def history(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        if ticker in self._fail:
            raise FetchError(f"configured failure for {ticker}")
        if ticker not in self._data:
            raise FetchError(f"no data for {ticker}")
        frame = self._data[ticker]
        sliced = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            sliced = sliced.loc[sliced.index <= pd.Timestamp(end)]
        if sliced.empty:
            raise NoNewData(f"empty slice for {ticker}")
        return sliced.copy()
