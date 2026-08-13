import pytest

from screener_sector.universe.symbols import (
    FakeTextSource,
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    fetch_symbols,
    load_symbols,
    parse_nasdaq_listed,
    parse_other_listed,
    save_symbols,
)
from screener_sector.paths import Paths

NASDAQ_TEXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N
AMAT|Applied Materials Inc. - Common Stock|Q|N|N|100|N|N
ZTEST|Test Issue Corp - Common Stock|Q|Y|N|100|N|N
SOXX|iShares Semiconductor ETF|G|N|N|100|Y|N
File Creation Time: 0812202617:30|||||||
"""

OTHER_TEXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
TSM|Taiwan Semiconductor Manufacturing Company Ltd.|N|TSM|N|100|N|TSM
SMH|VanEck Semiconductor ETF|P|SMH|Y|100|N|SMH
File Creation Time: 0812202617:30||||||||
"""


def test_parse_nasdaq_drops_footer_and_test_issues():
    df = parse_nasdaq_listed(NASDAQ_TEXT)
    assert set(df["ticker"]) == {"NVDA", "AMAT", "SOXX"}


def test_parse_nasdaq_flags_etfs():
    df = parse_nasdaq_listed(NASDAQ_TEXT).set_index("ticker")
    assert bool(df.loc["SOXX", "etf"]) is True
    assert bool(df.loc["NVDA", "etf"]) is False


def test_parse_nasdaq_sets_exchange():
    df = parse_nasdaq_listed(NASDAQ_TEXT)
    assert set(df["exchange"]) == {"NASDAQ"}


def test_parse_other_maps_exchange_codes():
    df = parse_other_listed(OTHER_TEXT).set_index("ticker")
    assert df.loc["TSM", "exchange"] == "NYSE"
    assert df.loc["SMH", "exchange"] == "NYSE ARCA"


def test_fetch_symbols_combines_both_files():
    source = FakeTextSource(
        {NASDAQ_LISTED_URL: NASDAQ_TEXT, OTHER_LISTED_URL: OTHER_TEXT}
    )
    df = fetch_symbols(source)
    assert set(df["ticker"]) == {"NVDA", "AMAT", "SOXX", "TSM", "SMH"}
    assert list(df.columns) == ["ticker", "name", "exchange", "etf"]


def test_fetch_symbols_deduplicates():
    source = FakeTextSource(
        {NASDAQ_LISTED_URL: NASDAQ_TEXT, OTHER_LISTED_URL: OTHER_TEXT}
    )
    df = fetch_symbols(source)
    assert df["ticker"].is_unique


def test_save_and_load_symbols_roundtrip(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    source = FakeTextSource(
        {NASDAQ_LISTED_URL: NASDAQ_TEXT, OTHER_LISTED_URL: OTHER_TEXT}
    )
    df = fetch_symbols(source)
    save_symbols(paths, df)
    assert len(load_symbols(paths)) == len(df)


def test_parse_nasdaq_drops_blank_ticker():
    """Blank tickers should be filtered out."""
    text = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
|Empty Corp - Common Stock|Q|N|N|100|N|N
NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N
File Creation Time: 0812202617:30|||||||
"""
    df = parse_nasdaq_listed(text)
    assert "NVDA" in df["ticker"].values
    assert len(df[df["ticker"] == ""]) == 0


def test_parse_nasdaq_drops_whitespace_only_ticker():
    """Whitespace-only tickers should be filtered out."""
    text = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
   |Whitespace Corp - Common Stock|Q|N|N|100|N|N
NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N
File Creation Time: 0812202617:30|||||||
"""
    df = parse_nasdaq_listed(text)
    assert set(df["ticker"]) == {"NVDA"}


def test_parse_nasdaq_drops_invalid_characters():
    """Tickers with invalid characters should be filtered out."""
    text = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
NVDA@|Invalid Char Corp - Common Stock|Q|N|N|100|N|N
NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N
File Creation Time: 0812202617:30|||||||
"""
    df = parse_nasdaq_listed(text)
    assert set(df["ticker"]) == {"NVDA"}


def test_parse_nasdaq_keeps_valid_symbols_with_dots_and_hyphens():
    """Valid symbols with dots and hyphens should be kept."""
    text = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
BRK.A|Berkshire Hathaway A|Q|N|N|100|N|N
RDS-A|Royal Dutch Shell A|Q|N|N|100|N|N
NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N
File Creation Time: 0812202617:30|||||||
"""
    df = parse_nasdaq_listed(text)
    assert set(df["ticker"]) == {"BRK.A", "RDS-A", "NVDA"}


def test_parse_other_listed_drops_blank_ticker():
    """Blank tickers in other_listed should be filtered out."""
    text = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
|Empty Corp|N|TSM|N|100|N|TSM
TSM|Taiwan Semiconductor Manufacturing Company Ltd.|N|TSM|N|100|N|TSM
File Creation Time: 0812202617:30||||||||
"""
    df = parse_other_listed(text)
    assert "TSM" in df["ticker"].values
    assert len(df[df["ticker"] == ""]) == 0


def test_parse_other_listed_drops_whitespace_only_ticker():
    """Whitespace-only tickers in other_listed should be filtered out."""
    text = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
   |Whitespace Corp|N|TSM|N|100|N|TSM
TSM|Taiwan Semiconductor Manufacturing Company Ltd.|N|TSM|N|100|N|TSM
File Creation Time: 0812202617:30||||||||
"""
    df = parse_other_listed(text)
    assert set(df["ticker"]) == {"TSM"}


def test_parse_other_listed_keeps_valid_symbols_with_dots_and_hyphens():
    """Valid symbols with dots and hyphens in other_listed should be kept."""
    text = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.A|Berkshire Hathaway A|N|BRK.A|N|100|N|BRK.A
RDS-A|Royal Dutch Shell A|N|RDS-A|N|100|N|RDS-A
TSM|Taiwan Semiconductor Manufacturing Company Ltd.|N|TSM|N|100|N|TSM
File Creation Time: 0812202617:30||||||||
"""
    df = parse_other_listed(text)
    assert set(df["ticker"]) == {"BRK.A", "RDS-A", "TSM"}
