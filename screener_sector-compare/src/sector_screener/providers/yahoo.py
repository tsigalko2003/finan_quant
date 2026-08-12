from __future__ import annotations

import pandas as pd
import yfinance as yf

from .base import MarketDataProvider


class YahooFinanceProvider(MarketDataProvider):
    name = "yahoo"

    def download(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        frame = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=interval,
            auto_adjust=auto_adjust,
            actions=True,
            progress=False,
            threads=False,
            timeout=30,
        )
        if isinstance(frame.columns, pd.MultiIndex):
            if ticker in frame.columns.get_level_values(-1):
                frame = frame.xs(ticker, axis=1, level=-1)
            else:
                frame.columns = frame.columns.get_level_values(0)
        return normalize_bars(frame)


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result.columns = [str(c).strip().lower().replace(" ", "_") for c in result.columns]
    result.index = pd.to_datetime(result.index, utc=True).tz_convert(None).normalize()
    result.index.name = "date"
    result = result[~result.index.duplicated(keep="last")].sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Provider response missing OHLCV columns: {sorted(missing)}")
    numeric = [c for c in result.columns if c in required | {"dividends", "stock_splits"}]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    return result
