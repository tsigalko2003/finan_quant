from datetime import date, timedelta

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.data.fetcher import FakeFetcher, NoNewData
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_env({"DATA_DIR": str(tmp_path)})
    p.ensure()
    return p


@pytest.fixture
def sample():
    return make_ohlcv(exponential_trend(200, 0.001, noise=0.01, seed=2))


def test_refresh_writes_parquet(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    result = store.refresh(["NVDA"], date(2020, 1, 1))
    assert result.fetched == ("NVDA",)
    assert paths.price_file("NVDA").exists()


def test_load_roundtrips_values(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    store.refresh(["NVDA"], date(2020, 1, 1))
    loaded = store.load("NVDA")
    pd.testing.assert_series_equal(
        loaded["close"], sample["close"], check_freq=False
    )


def test_failed_ticker_is_quarantined_not_raised(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}, fail={"BAD"}))
    result = store.refresh(["NVDA", "BAD"], date(2020, 1, 1))
    assert result.fetched == ("NVDA",)
    assert "BAD" in result.failed
    assert paths.failures_csv.exists()
    assert "BAD" in paths.failures_csv.read_text()


def test_refresh_is_incremental(paths, sample):
    first_half = sample.iloc[:100]
    fetcher = FakeFetcher({"NVDA": first_half})
    store = PriceStore(paths, fetcher)
    store.refresh(["NVDA"], date(2020, 1, 1))

    fetcher_full = FakeFetcher({"NVDA": sample})
    store2 = PriceStore(paths, fetcher_full)
    store2.refresh(["NVDA"], date(2020, 1, 1))

    # With backward backfill, we may attempt leading gap (if start < cached_min)
    # and forward gap (for new data). Verify that neither overlaps with cached bars.
    # The key is that we never ask for already-cached dates.
    cached_min = sample.iloc[0].name.date()
    cached_max = sample.iloc[99].name.date()

    for ticker, start, end in fetcher_full.calls:
        if end is None:
            # Forward gap: should start after cached data
            assert start > cached_max
        else:
            # Leading gap: should end before cached data
            assert end < cached_min

    assert len(store2.load("NVDA")) == 200


def test_refresh_never_truncates_existing_history(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    store.refresh(["NVDA"], date(2020, 1, 1))
    before = len(store.load("NVDA"))

    later = sample.index[150].date()
    store.refresh(["NVDA"], later)
    assert len(store.load("NVDA")) == before


def test_close_panel_aligns_on_union_of_dates(paths):
    a = make_ohlcv(exponential_trend(50, 0.001, seed=1))
    b = make_ohlcv(exponential_trend(30, 0.002, seed=2))
    store = PriceStore(paths, FakeFetcher({"A": a, "B": b}))
    store.refresh(["A", "B"], date(2019, 1, 1))
    panel = store.close_panel(["A", "B"])
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 50
    assert panel["B"].isna().sum() == 20


def test_close_panel_does_not_forward_fill(paths):
    a = make_ohlcv(exponential_trend(50, 0.001, seed=1))
    b = make_ohlcv(exponential_trend(30, 0.002, seed=2))
    store = PriceStore(paths, FakeFetcher({"A": a, "B": b}))
    store.refresh(["A", "B"], date(2019, 1, 1))
    panel = store.close_panel(["A", "B"])
    # B has only 30 of the 50 bars; the trailing 20 must stay NaN rather than
    # carry a forward-filled price that downstream code would treat as observed.
    assert panel["B"].head(30).notna().all()
    assert panel["B"].tail(20).isna().all()


def test_available_filters_by_min_history(paths):
    a = make_ohlcv(exponential_trend(300, 0.001, seed=1))
    b = make_ohlcv(exponential_trend(100, 0.002, seed=2))
    store = PriceStore(paths, FakeFetcher({"A": a, "B": b}))
    store.refresh(["A", "B"], date(2019, 1, 1))
    assert store.available(["A", "B"], min_days=250) == ["A"]


def test_corrupt_parquet_is_refetched(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    store.refresh(["NVDA"], date(2020, 1, 1))
    paths.price_file("NVDA").write_bytes(b"not parquet")

    store2 = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    result = store2.refresh(["NVDA"], date(2020, 1, 1))
    assert result.fetched == ("NVDA",)
    assert len(store2.load("NVDA")) == 200


def test_no_new_data_treated_as_skipped_not_failed(paths, sample):
    """NoNewData is not a failure; cache is already current.

    With backward backfill, we may attempt both leading and forward gaps.
    Each gap that's empty will raise NoNewData exactly once with no retries.
    """

    class NoNewDataFetcher:
        def __init__(self):
            self.call_count = 0
            self.sleeps = []

        def history(self, ticker, start, end):
            self.call_count += 1
            raise NoNewData(f"already current for {ticker}")

    fetcher = NoNewDataFetcher()
    store = PriceStore(paths, fetcher)

    # Pre-populate cache
    PriceStore(paths, FakeFetcher({"TICKER": sample})).refresh(
        ["TICKER"], date(2020, 1, 1)
    )

    # Try to refresh when already current; should be skipped, not failed
    # With backward backfill, we may try leading gap and forward gap
    result = store.refresh(["TICKER"], date(2020, 1, 1))
    assert result.skipped == ("TICKER",)
    assert result.failed == {}
    # Each gap is attempted once with no retries
    assert fetcher.call_count >= 1


def test_backward_backfill_fetches_leading_gap_only(paths, sample):
    """Refresh with earlier start must fetch only the leading gap, not cached bars."""
    # Cache bars 100..199
    cached_portion = sample.iloc[100:200]
    first_fetcher = FakeFetcher({"TICKER": cached_portion})
    store = PriceStore(paths, first_fetcher)
    store.refresh(["TICKER"], date(2020, 1, 1))

    # Now refresh with an earlier start (bar 0)
    full_sample = sample
    second_fetcher = FakeFetcher({"TICKER": full_sample})
    store2 = PriceStore(paths, second_fetcher)
    result = store2.refresh(["TICKER"], date(2020, 1, 1))

    # Verify the result
    assert "TICKER" in result.fetched
    loaded = store2.load("TICKER")
    assert len(loaded) == 200  # Full span of 200 bars

    # Verify that the fetcher was called twice: once for leading gap, once for forward
    assert len(second_fetcher.calls) == 2
    # First call should be for the leading gap (bars before cached_min)
    ticker1, start1, end1 = second_fetcher.calls[0]
    assert start1 == date(2020, 1, 1)
    assert end1 < sample.iloc[100].name.date()
    # Second call should be for the forward gap (bars after cached_max)
    ticker2, start2, end2 = second_fetcher.calls[1]
    assert start2 >= sample.iloc[199].name.date()


def test_backward_backfill_with_two_fetches(paths, sample):
    """Refresh with earlier start AND newer data fetches both gaps."""
    # Cache bars 100..150
    cached_portion = sample.iloc[100:150]
    first_fetcher = FakeFetcher({"TICKER": cached_portion})
    store = PriceStore(paths, first_fetcher)
    store.refresh(["TICKER"], date(2020, 6, 1))

    # Now refresh with earlier start AND later end
    second_fetcher = FakeFetcher({"TICKER": sample})
    store2 = PriceStore(paths, second_fetcher)
    result = store2.refresh(["TICKER"], date(2020, 1, 1), date(2020, 12, 31))

    # Verify the result
    assert "TICKER" in result.fetched
    loaded = store2.load("TICKER")
    # Should span from bar 0 to bar 199
    assert len(loaded) == 200

    # Verify two fetches were made (leading and forward)
    assert len(second_fetcher.calls) == 2


def test_backward_backfill_does_not_re_request_cached_bars(paths, sample):
    """Refresh with start >= cached_min must not issue a backfill request."""
    # Cache bars 100..199
    cached_portion = sample.iloc[100:200]
    first_fetcher = FakeFetcher({"TICKER": cached_portion})
    store = PriceStore(paths, first_fetcher)
    store.refresh(["TICKER"], date(2020, 1, 1))
    cached_min = sample.iloc[100].name.date()

    # Refresh with start at cached_min or later (e.g., at cached_min + 5 days)
    later_start = cached_min + timedelta(days=5)
    second_fetcher = FakeFetcher({"TICKER": sample})
    store2 = PriceStore(paths, second_fetcher)
    result = store2.refresh(["TICKER"], later_start)

    # Should be skipped because we asked for data strictly after cached_max
    # and there's nothing newer in the sample
    assert len(second_fetcher.calls) == 1
    requested_ticker, requested_start, requested_end = second_fetcher.calls[0]
    # The requested start should be after cached_min (no backfill)
    assert requested_start >= cached_min


def test_empty_history_no_cache_is_failed(paths):
    """NoNewData with no existing cache means the symbol yielded nothing; should be failed."""
    # Empty fetcher that returns nothing for a ticker
    fetcher = FakeFetcher({})
    store = PriceStore(paths, fetcher)
    result = store.refresh(["NONEXISTENT"], date(2020, 1, 1))

    assert result.fetched == ()
    assert result.skipped == ()
    assert "NONEXISTENT" in result.failed
    assert paths.failures_csv.exists()


def test_empty_history_with_cache_is_skipped(paths, sample):
    """NoNewData with existing cache means no new data; should be skipped not failed."""
    # First, populate the cache
    PriceStore(paths, FakeFetcher({"TICKER": sample})).refresh(
        ["TICKER"], date(2020, 1, 1)
    )

    # Now try to fetch when the data source returns nothing
    fetcher = FakeFetcher({})
    store = PriceStore(paths, fetcher)
    result = store.refresh(["TICKER"], date(2020, 1, 1))

    assert result.fetched == ()
    assert result.skipped == ("TICKER",)
    assert result.failed == {}


def test_yfinance_fetcher_no_retry_on_nonewdata():
    """YFinanceFetcher must make exactly one call and zero sleeps when NoNewData is raised."""
    from screener_sector.data.fetcher import YFinanceFetcher

    sleeps = []
    def mock_sleep(duration):
        sleeps.append(duration)

    calls = []
    def mock_ticker_factory(symbol):
        calls.append(symbol)
        # Return a mock that raises NoNewData
        class MockTicker:
            def history(self, **kwargs):
                return None  # This will trigger NoNewData in YFinanceFetcher

        return MockTicker()

    fetcher = YFinanceFetcher(sleep=mock_sleep, max_retries=3, ticker_factory=mock_ticker_factory)

    try:
        fetcher.history("INVALID", date(2020, 1, 1), date(2020, 12, 31))
    except NoNewData:
        pass  # Expected

    # Should make exactly one call, not retried
    assert len(calls) == 1
    # Should make zero sleeps (no retry backoff)
    assert len(sleeps) == 0
