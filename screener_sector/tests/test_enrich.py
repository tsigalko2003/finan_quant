import pytest

from screener_sector.paths import Paths
from screener_sector.universe.enrich import (
    FakeInfoSource,
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
