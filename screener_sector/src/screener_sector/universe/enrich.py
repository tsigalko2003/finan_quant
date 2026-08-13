"""Company profile enrichment, cached permanently and resumable.

For prod this is thousands of throttled requests taking hours. It writes
partial results every `batch_flush` tickers so an interrupted run resumes
almost where it stopped rather than starting over.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import pandas as pd

from screener_sector.data.fetcher import RateLimited as _RateLimited
from screener_sector.paths import Paths

def _is_rate_limited(exc: Exception) -> bool:
    """Detect if an exception represents a 429 rate limit response."""
    exc_str = str(exc).lower()
    return "429" in exc_str or "too many requests" in exc_str


INFO_COLUMNS = [
    "ticker",
    "long_name",
    "sector",
    "industry",
    "summary",
    "quote_type",
    "fetched_at",
]


class InfoLookupError(RuntimeError):
    """Profile fields for a ticker could not be retrieved."""


class RateLimited(InfoLookupError):
    """Yahoo Finance has rate-limited the request. The run is resumable after a wait."""


class InfoSource(Protocol):
    def info(self, ticker: str) -> dict[str, object]: ...


class YFinanceInfoSource:
    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        pause: float = 1.5,
        ticker_factory: Callable[[str], object] | None = None,
        rate_limit_backoff_seconds: tuple[float, ...] = (60.0, 180.0, 420.0),
    ) -> None:
        self._sleep = sleep
        self._max_retries = max_retries
        self._pause = pause
        self._ticker_factory = ticker_factory or _default_ticker_factory
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds

    def info(self, ticker: str) -> dict[str, object]:
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                payload = self._ticker_factory(ticker).info
                if not payload:
                    raise InfoLookupError(f"empty info for {ticker}")
                self._sleep(self._pause)
                return dict(payload)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - retry anything transient
                last = exc
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
        if _is_rate_limited(last):
            raise RateLimited(
                f"Yahoo Finance rate-limited ticker {ticker} after {self._max_retries} attempts. "
                f"The run is resumable. Wait at least {self._rate_limit_backoff_seconds[-1]} seconds "
                f"before retrying."
            ) from last
        raise InfoLookupError(f"failed info for {ticker}: {last}") from last


def _default_ticker_factory(symbol: str):
    import yfinance

    return yfinance.Ticker(symbol)


class FakeInfoSource:
    def __init__(
        self, data: dict[str, dict[str, object]], fail: set[str] | None = None
    ) -> None:
        self._data = data
        self._fail = fail or set()
        self.calls: list[str] = []

    def info(self, ticker: str) -> dict[str, object]:
        self.calls.append(ticker)
        if ticker in self._fail or ticker not in self._data:
            raise InfoLookupError(f"no info for {ticker}")
        return dict(self._data[ticker])


def load_info(paths: Paths) -> pd.DataFrame:
    if not paths.info_parquet.exists():
        return pd.DataFrame(columns=INFO_COLUMNS)
    return pd.read_parquet(paths.info_parquet)


def _save_info(paths: Paths, df: pd.DataFrame) -> None:
    paths.ensure()
    df[INFO_COLUMNS].to_parquet(paths.info_parquet, index=False)


def _row(ticker: str, payload: dict[str, object], now: str) -> dict[str, object]:
    def text(key: str) -> str:
        value = payload.get(key)
        return "" if value is None else str(value)

    return {
        "ticker": ticker,
        "long_name": text("longName"),
        "sector": text("sector"),
        "industry": text("industry"),
        "summary": text("longBusinessSummary"),
        "quote_type": text("quoteType"),
        "fetched_at": now,
    }


def enrich(
    paths: Paths,
    tickers: Sequence[str],
    source: InfoSource,
    now: str,
    batch_flush: int = 25,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    cached = load_info(paths)
    known = set(cached["ticker"]) if not cached.empty else set()
    pending = [t for t in tickers if t not in known]

    rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    interrupted = False

    def flush() -> pd.DataFrame:
        nonlocal cached, rows
        if rows:
            cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            rows = []
            _save_info(paths, cached)
        if on_progress is not None:
            on_progress(len(cached), len(tickers))
        return cached

    try:
        for index, ticker in enumerate(pending, start=1):
            try:
                rows.append(_row(ticker, source.info(ticker), now))
            except InfoLookupError as exc:
                failures[ticker] = str(exc)
            if index % batch_flush == 0:
                flush()
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if not interrupted:
            flush()
        if failures:
            _record_failures(paths, failures)

    return cached


def _record_failures(paths: Paths, failures: dict[str, str]) -> None:
    path = paths.failures_csv
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["ticker", "reason"])
        for ticker, reason in failures.items():
            writer.writerow([ticker, reason])
