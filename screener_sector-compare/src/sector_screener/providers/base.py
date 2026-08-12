from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def download(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        """Return normalized OHLCV indexed by date for the half-open range [start, end)."""
