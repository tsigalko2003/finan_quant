from __future__ import annotations

import pandas as pd

from sector_screener.cache import MarketDataCache, merge_intervals, missing_intervals
from sector_screener.providers.base import MarketDataProvider


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self):
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def download(self, ticker, start, end, interval, auto_adjust):
        self.calls.append((ticker, start, end))
        index = pd.date_range(start, end - pd.offsets.Day(1), freq="B")
        values = pd.Series(range(len(index)), index=index, dtype=float) + 100
        return pd.DataFrame(
            {
                "open": values,
                "high": values + 1,
                "low": values - 1,
                "close": values + 0.5,
                "volume": 1_000,
            },
            index=index.rename("date"),
        )


def test_interval_math():
    intervals = merge_intervals(
        [
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")),
            (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-10")),
        ]
    )
    assert intervals == [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))]
    gaps = missing_intervals((pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15")), intervals)
    assert gaps == [(pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-15"))]


def test_cache_avoids_duplicate_downloads_and_fetches_only_tail(tmp_path):
    provider = FakeProvider()
    cache = MarketDataCache(tmp_path, provider, "1d", True)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")

    first = cache.fetch("TEST", start, end)
    assert not first.cache_hit
    assert len(provider.calls) == 1

    second = cache.fetch("TEST", start, end)
    assert second.cache_hit
    assert len(provider.calls) == 1
    pd.testing.assert_frame_equal(first.frame, second.frame, check_freq=False)

    extended = cache.fetch("TEST", start, pd.Timestamp("2024-02-10"))
    assert not extended.cache_hit
    assert len(provider.calls) == 2
    assert provider.calls[-1][1:] == (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-10"))


def test_narrow_refresh_does_not_delete_older_cache(tmp_path):
    provider = FakeProvider()
    cache = MarketDataCache(tmp_path, provider, "1d", True)
    cache.fetch("TEST", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"))
    cache.fetch(
        "TEST",
        pd.Timestamp("2023-12-01"),
        pd.Timestamp("2024-01-01"),
        refresh_tail_days=5,
    )
    historical = cache.load("TEST", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"))
    assert not historical.empty


def test_corrupt_cache_is_quarantined_and_refetched(tmp_path):
    provider = FakeProvider()
    cache = MarketDataCache(tmp_path, provider, "1d", True)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")
    cache.fetch("TEST", start, end)
    data_path = tmp_path / "fake" / "1d" / "adjusted" / "TEST.parquet"
    data_path.write_bytes(b"corrupt")

    result = cache.fetch("TEST", start, end)
    assert len(provider.calls) == 2
    assert not result.frame.empty
    assert list(data_path.parent.glob("TEST.corrupt-*.parquet"))
