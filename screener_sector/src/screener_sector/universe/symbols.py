"""US-listed symbol universe from the NASDAQ Trader public files.

These files are the seed for discovery. They are free, stable, and require no
API key. Both end with a 'File Creation Time' footer line that is not data.
"""

from __future__ import annotations

import io
from typing import Protocol

import pandas as pd

from screener_sector.paths import Paths, VALID_TICKER_PATTERN

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_EXCHANGE_CODES = {
    "A": "NYSE MKT",
    "N": "NYSE",
    "P": "NYSE ARCA",
    "Z": "BATS",
    "V": "IEX",
}

COLUMNS = ["ticker", "name", "exchange", "etf"]


class TextSource(Protocol):
    def get(self, url: str) -> str: ...


class HttpTextSource:
    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def get(self, url: str) -> str:
        import requests

        response = requests.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.text


class FakeTextSource:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def get(self, url: str) -> str:
        return self._pages[url]


def _read_pipe_table(text: str) -> pd.DataFrame:
    lines = text.strip().split('\n')
    filtered_lines = [line for line in lines if not line.startswith("File Creation Time")]
    text_filtered = '\n'.join(filtered_lines)
    df = pd.read_csv(io.StringIO(text_filtered), sep="|", dtype=str).fillna("")
    return df


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    df = _read_pipe_table(text)
    df = df[df["Test Issue"] == "N"]
    result = pd.DataFrame(
        {
            "ticker": df["Symbol"].str.strip(),
            "name": df["Security Name"].str.strip(),
            "exchange": "NASDAQ",
            "etf": df["ETF"].str.strip() == "Y",
        }
    ).reset_index(drop=True)
    # Filter out empty and invalid ticker symbols
    result = result[result["ticker"].str.len() > 0]
    result = result[result["ticker"].str.match(VALID_TICKER_PATTERN)]
    return result.reset_index(drop=True)


def parse_other_listed(text: str) -> pd.DataFrame:
    df = _read_pipe_table(text)
    df = df[df["Test Issue"] == "N"]
    result = pd.DataFrame(
        {
            "ticker": df["ACT Symbol"].str.strip(),
            "name": df["Security Name"].str.strip(),
            "exchange": df["Exchange"].str.strip().map(_EXCHANGE_CODES).fillna("OTHER"),
            "etf": df["ETF"].str.strip() == "Y",
        }
    ).reset_index(drop=True)
    # Filter out empty and invalid ticker symbols
    result = result[result["ticker"].str.len() > 0]
    result = result[result["ticker"].str.match(VALID_TICKER_PATTERN)]
    return result.reset_index(drop=True)


def fetch_symbols(source: TextSource) -> pd.DataFrame:
    nasdaq = parse_nasdaq_listed(source.get(NASDAQ_LISTED_URL))
    other = parse_other_listed(source.get(OTHER_LISTED_URL))
    combined = pd.concat([nasdaq, other], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    return combined[COLUMNS].sort_values("ticker").reset_index(drop=True)


def save_symbols(paths: Paths, df: pd.DataFrame) -> None:
    paths.ensure()
    df.to_parquet(paths.symbols_parquet)


def load_symbols(paths: Paths) -> pd.DataFrame:
    return pd.read_parquet(paths.symbols_parquet)
