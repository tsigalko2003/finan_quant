from __future__ import annotations

import json

import pytest

from sector_screener import nasdaq_universe
from sector_screener.nasdaq_universe import NasdaqUniverseCache, normalize_export
from sector_screener.universe import UniverseCatalog


def payload(rows):
    return {"data": {"rows": rows}, "status": {"rCode": 200}}


def row(symbol, name, industry="Semiconductors", market_cap="100"):
    return {
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "sector": "Technology",
        "country": "United States",
        "marketCap": market_cap,
    }


@pytest.fixture(autouse=True)
def small_export(monkeypatch):
    monkeypatch.setattr(nasdaq_universe, "MIN_EXPORT_ROWS", 3)


def test_normalizes_and_excludes_non_common_instruments():
    result = normalize_export(
        payload(
            [
                row("AAA", "Alpha Common Stock", market_cap="10"),
                row("BBB", "Beta Ordinary Shares", market_cap="30"),
                row("CCC", "Gamma American Depositary Shares", market_cap="20"),
                row("BBBW", "Beta Warrants", market_cap="1"),
                row("UMC", "United Microelectronics Corporation Common Stock", market_cap="40"),
            ]
        )
    )
    assert result["normalized_rows"] == 5
    assert sum(item["eligible_common_equity"] for item in result["rows"]) == 4


def test_cache_avoids_duplicate_export_download_and_queries_by_industry(tmp_path):
    calls = 0

    def download():
        nonlocal calls
        calls += 1
        return payload(
            [
                row("AAA", "Alpha Common Stock", market_cap="10"),
                row("BBB", "Beta Ordinary Shares", market_cap="30"),
                row("CCC", "Gamma Common Stock", market_cap="20"),
            ]
        )

    cache = NasdaqUniverseCache(tmp_path, downloader=download)
    first = cache.ensure()
    second = cache.ensure()
    assert calls == 1
    assert not first["cache_hit"]
    assert second["cache_hit"]
    assert cache.query("semiconductor")[0] == ["BBB", "CCC", "AAA"]
    assert cache.describe()[0]["name"] == "nasdaq:semiconductors"


def test_failed_refresh_preserves_last_known_good_snapshot(tmp_path):
    good = lambda: payload(
        [
            row("AAA", "Alpha Common Stock"),
            row("BBB", "Beta Common Stock"),
            row("CCC", "Gamma Common Stock"),
        ]
    )
    cache = NasdaqUniverseCache(tmp_path, downloader=good)
    original = cache.ensure()

    def fail():
        raise RuntimeError("upstream unavailable")

    stale = NasdaqUniverseCache(tmp_path, downloader=fail).ensure(refresh=True)
    assert stale["stale_cache_used"]
    assert stale["snapshot_id"] == original["snapshot_id"]


def test_catalog_resolves_cached_nasdaq_query(tmp_path):
    config = tmp_path / "industries.yaml"
    config.write_text("industries: {}\n", encoding="utf-8")
    cache = NasdaqUniverseCache(
        tmp_path,
        downloader=lambda: payload(
            [
                row("AAA", "Alpha Common Stock", market_cap="10"),
                row("BBB", "Beta Common Stock", market_cap="30"),
                row("CCC", "Gamma Common Stock", market_cap="20"),
            ]
        ),
    )
    cache.ensure()
    universe = UniverseCatalog(config).resolve(
        "nasdaq:semiconductor", max_tickers=3, nasdaq_cache_dir=tmp_path
    )
    assert universe.tickers == ["BBB", "CCC", "AAA"]
    assert universe.metadata["full_eligible_count"] == 3
    assert universe.source == "nasdaq-export"


def test_corrupt_latest_snapshot_is_rejected(tmp_path):
    cache = NasdaqUniverseCache(
        tmp_path,
        downloader=lambda: payload(
            [
                row("AAA", "Alpha Common Stock"),
                row("BBB", "Beta Common Stock"),
                row("CCC", "Gamma Common Stock"),
            ]
        ),
    )
    cache.ensure()
    latest = json.loads(cache.latest_path.read_text(encoding="utf-8"))
    latest["rows"][0]["symbol"] = "TAMPERED"
    cache.latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="cache is missing"):
        cache.load()
