import pytest

from screener_sector.paths import Paths
from screener_sector.universe.enrich import (
    FakeInfoSource,
    RateLimited,
    YFinanceInfoSource,
    _is_rate_limited,
    enrich,
    load_info,
)


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_env({"DATA_DIR": str(tmp_path)})
    p.ensure()
    return p


def info_for(name, industry, summary, quote_type="EQUITY"):
    return {
        "longName": name,
        "sector": "Technology",
        "industry": industry,
        "longBusinessSummary": summary,
        "quoteType": quote_type,
    }


def test_enrich_writes_expected_columns(paths):
    source = FakeInfoSource({"NVDA": info_for("NVIDIA", "Semiconductors", "GPUs.")})
    df = enrich(paths, ["NVDA"], source, now="2026-08-12T00:00:00")
    assert list(df.columns) == [
        "ticker", "long_name", "sector", "industry", "summary",
        "quote_type", "fetched_at",
    ]
    assert df.iloc[0]["long_name"] == "NVIDIA"


def test_enrich_is_resumable(paths):
    source = FakeInfoSource(
        {
            "NVDA": info_for("NVIDIA", "Semiconductors", "GPUs."),
            "AMD": info_for("AMD", "Semiconductors", "CPUs."),
        }
    )
    enrich(paths, ["NVDA"], source, now="2026-08-12T00:00:00")
    source.calls.clear()
    enrich(paths, ["NVDA", "AMD"], source, now="2026-08-12T00:00:00")
    assert source.calls == ["AMD"]
    assert len(load_info(paths)) == 2


def test_enrich_records_failures_without_stopping(paths):
    source = FakeInfoSource(
        {"NVDA": info_for("NVIDIA", "Semiconductors", "GPUs.")}, fail={"BAD"}
    )
    df = enrich(paths, ["BAD", "NVDA"], source, now="2026-08-12T00:00:00")
    assert set(df["ticker"]) == {"NVDA"}
    assert "BAD" in paths.failures_csv.read_text()


def test_enrich_flushes_partially_on_interruption(paths):
    payloads = {f"T{i}": info_for(f"T{i}", "Semiconductors", "chips.") for i in range(5)}

    class ExplodesAtIndexThree(FakeInfoSource):
        def info(self, ticker):
            if ticker == "T3":
                raise KeyboardInterrupt
            return super().info(ticker)

    source = ExplodesAtIndexThree(payloads)
    with pytest.raises(KeyboardInterrupt):
        enrich(
            paths,
            [f"T{i}" for i in range(5)],
            source,
            now="2026-08-12T00:00:00",
            batch_flush=2,
        )
    assert len(load_info(paths)) == 2


def test_missing_fields_become_empty_strings(paths):
    source = FakeInfoSource({"XYZ": {"longName": "XYZ Corp"}})
    df = enrich(paths, ["XYZ"], source, now="2026-08-12T00:00:00")
    row = df.iloc[0]
    assert row["industry"] == ""
    assert row["summary"] == ""


def test_is_rate_limited_detects_429():
    """Test detection of 429 status code in error message."""
    exc = RuntimeError("429 Client Error: Too Many Requests")
    assert _is_rate_limited(exc)


def test_is_rate_limited_detects_too_many_requests():
    """Test detection of 'too many requests' text."""
    exc = RuntimeError("Too Many Requests for url: https://query2.finance.yahoo.com/...")
    assert _is_rate_limited(exc)


def test_yfinance_info_source_uses_long_backoff_for_rate_limits():
    """Rate-limited errors should use the long backoff sequence."""
    attempts = []

    class TracksBackoff:
        @property
        def info(self):
            if len(attempts) < 2:
                attempts.append(None)
                raise RuntimeError("429 Client Error: Too Many Requests")
            return {"longName": "Test"}

    sleep_calls = []
    source = YFinanceInfoSource(
        sleep=lambda x: sleep_calls.append(x),
        max_retries=3,
        ticker_factory=lambda symbol: TracksBackoff(),
        rate_limit_backoff_seconds=(100.0, 200.0, 300.0),
    )
    result = source.info("NVDA")
    # Should have slept with the long backoff sequence: 100, 200, then pause
    assert 100.0 in sleep_calls
    assert 200.0 in sleep_calls


def test_yfinance_info_source_uses_short_backoff_for_normal_errors():
    """Non-rate-limit errors should use the short exponential backoff."""
    attempts = []

    class TrackBackoff:
        @property
        def info(self):
            if len(attempts) < 2:
                attempts.append(None)
                raise RuntimeError("Connection timeout")
            return {"longName": "Test"}

    sleep_calls = []
    source = YFinanceInfoSource(
        sleep=lambda x: sleep_calls.append(x),
        max_retries=3,
        ticker_factory=lambda symbol: TrackBackoff(),
    )
    result = source.info("NVDA")
    # Should have slept with the short exponential backoff: 2^0=1, 2^1=2, then pause
    assert 1.0 in sleep_calls
    assert 2.0 in sleep_calls


def test_yfinance_info_source_raises_rate_limited_after_max_retries():
    """Exhausting retries on a rate limit should raise RateLimited."""

    class AlwaysRateLimited:
        @property
        def info(self):
            raise RuntimeError("429 Client Error: Too Many Requests")

    source = YFinanceInfoSource(
        sleep=lambda _: None,
        max_retries=3,
        ticker_factory=lambda symbol: AlwaysRateLimited(),
        rate_limit_backoff_seconds=(60.0, 180.0, 420.0),
    )
    with pytest.raises(RateLimited) as exc_info:
        source.info("NVDA")
    assert "rate-limited" in str(exc_info.value).lower()
    assert "resumable" in str(exc_info.value).lower()
    assert "420" in str(exc_info.value)
