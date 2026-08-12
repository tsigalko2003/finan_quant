from pathlib import Path

import pytest

from sector_screener.universe import UniverseCatalog


def test_catalog_and_qlib_pool_resolution(tmp_path):
    catalog_file = tmp_path / "industries.yaml"
    catalog_file.write_text(
        "industries:\n  semi:\n    description: test\n    tickers: [AAA, BBB, CCC, DDD]\n",
        encoding="utf-8",
    )
    catalog = UniverseCatalog(catalog_file)
    universe = catalog.resolve("semi", max_tickers=3)
    assert universe.tickers == ["AAA", "BBB", "CCC"]

    instruments = tmp_path / "qlib" / "instruments"
    instruments.mkdir(parents=True)
    (instruments / "sp500.txt").write_text(
        "AAPL\t2000-01-01\t2099-12-31\nMSFT\t2000-01-01\t2099-12-31\nNVDA\t2000-01-01\t2099-12-31\n",
        encoding="utf-8",
    )
    assert catalog.qlib_pools(tmp_path / "qlib") == ["sp500"]
    qlib = catalog.resolve("qlib:sp500", qlib_data_dir=tmp_path / "qlib")
    assert qlib.tickers == ["AAPL", "MSFT", "NVDA"]


def test_catalog_rejects_too_small_universe(tmp_path: Path):
    path = tmp_path / "industries.yaml"
    path.write_text("industries:\n  tiny:\n    tickers: [AAA, BBB]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least three"):
        UniverseCatalog(path).resolve("tiny")


def test_yaml_on_ticker_is_a_string():
    catalog = UniverseCatalog(Path(__file__).parents[1] / "config" / "industries.yaml")
    universe = catalog.resolve("semiconductor")
    assert "ON" in universe.tickers
    assert "TRUE" not in universe.tickers
