from pathlib import Path

from screener_sector.paths import Paths, VALID_TICKER_PATTERN
from screener_sector.universe.symbols import VALID_TICKER_PATTERN as SYMBOLS_PATTERN


def test_from_env_uses_data_dir():
    paths = Paths.from_env({"DATA_DIR": "/tmp/somewhere"})
    assert paths.root == Path("/tmp/somewhere")


def test_from_env_defaults_to_local_data():
    paths = Paths.from_env({})
    assert paths.root == Path("data")


def test_all_paths_are_under_root(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    for candidate in [
        paths.manifest_file,
        paths.universe_csv,
        paths.meta_dir,
        paths.symbols_parquet,
        paths.info_parquet,
        paths.failures_csv,
        paths.prices_dir,
        paths.price_file("NVDA"),
        paths.derived_dir("dev"),
    ]:
        assert candidate.is_relative_to(tmp_path)


def test_price_file_is_ticker_named(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    assert paths.price_file("NVDA").name == "NVDA.parquet"


def test_price_file_rejects_path_traversal(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    for bad in ["../etc", "a/b", ""]:
        try:
            paths.price_file(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_derived_dir_is_profile_namespaced(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    assert paths.derived_dir("dev") != paths.derived_dir("prod")
    assert paths.derived_dir("dev").name == "dev"


def test_ensure_creates_directories(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path / "fresh")})
    paths.ensure()
    assert paths.meta_dir.is_dir()
    assert paths.prices_dir.is_dir()


def test_ticker_patterns_agree():
    """Paths and symbols modules must use identical ticker validation patterns."""
    # Test valid symbols
    valid_tickers = ["NVDA", "BRK.A", "RDS-A", "T", "SOXX", "A0", "TSM"]
    for ticker in valid_tickers:
        assert VALID_TICKER_PATTERN.match(ticker), f"Paths rejects valid ticker {ticker}"
        assert SYMBOLS_PATTERN.match(ticker), f"Symbols rejects valid ticker {ticker}"

    # Test invalid symbols
    invalid_tickers = ["", "A" * 16, "NVDA@", "NVDA$", " ", "NVDA ", " NVDA"]
    for ticker in invalid_tickers:
        assert not VALID_TICKER_PATTERN.match(ticker), f"Paths accepts invalid ticker {ticker}"
        assert not SYMBOLS_PATTERN.match(ticker), f"Symbols accepts invalid ticker {ticker}"
