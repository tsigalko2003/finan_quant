"""Parquet price cache.

One file per ticker holding maximum available history, independent of the
active profile. Refresh is incremental: only bars after the cached maximum are
requested, and an existing file is never truncated by a later start date.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from screener_sector.data.fetcher import FetchError, NoNewData, PriceFetcher
from screener_sector.paths import Paths


@dataclass(frozen=True)
class RefreshResult:
    fetched: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: dict[str, str]


class PriceStore:
    def __init__(self, paths: Paths, fetcher: PriceFetcher) -> None:
        self._paths = paths
        self._fetcher = fetcher
        self._paths.ensure()

    def has(self, ticker: str) -> bool:
        return self._paths.price_file(ticker).exists()

    def load(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self._paths.price_file(ticker))

    def refresh(
        self, tickers: Sequence[str], start: date, end: date | None = None
    ) -> RefreshResult:
        fetched: list[str] = []
        skipped: list[str] = []
        failed: dict[str, str] = {}

        for ticker in tickers:
            existing = self._read_existing(ticker)

            # Determine if we need to fetch leading or forward gaps
            leading_range = None
            forward_range = None

            if existing is not None and not existing.empty:
                cached_min = existing.index.min().date()
                cached_max = existing.index.max().date()

                if end is not None and cached_max >= end:
                    skipped.append(ticker)
                    continue

                # Check for leading gap: if start < cached_min, fetch the gap
                if start < cached_min:
                    leading_range = (start, cached_min - timedelta(days=1))

                # Check for forward gap: if there may be newer bars
                forward_range = (cached_max + timedelta(days=1), end)
            else:
                # No existing cache; fetch the full range as forward
                forward_range = (start, end)

            # Fetch leading gap if needed
            leading_data = None
            leading_failed = False
            if leading_range is not None:
                try:
                    leading_data = self._fetcher.history(
                        ticker, leading_range[0], leading_range[1]
                    )
                except NoNewData:
                    # Leading gap is empty; that's okay
                    pass
                except FetchError as exc:
                    # Leading gap fetch failed; only mark as failed if no existing cache
                    if existing is None or existing.empty:
                        failed[ticker] = str(exc)
                        continue
                    # With existing cache, just skip the leading gap
                    leading_failed = True

            # Fetch forward gap if needed
            forward_data = None
            if forward_range is not None:
                try:
                    forward_data = self._fetcher.history(
                        ticker, forward_range[0], forward_range[1]
                    )
                except NoNewData:
                    # No new data in forward gap
                    if existing is None or existing.empty:
                        # No cache and no data at all = failed (unless we got leading data)
                        if leading_data is None:
                            failed[ticker] = "empty history"
                            continue
                except FetchError as exc:
                    # Forward gap fetch failed
                    if existing is None or existing.empty:
                        failed[ticker] = str(exc)
                        continue
                    # With existing cache, we can skip

            # Decide outcome based on what we got
            new_data = leading_data is not None or forward_data is not None

            if new_data:
                # Merge all pieces: existing, leading, forward
                combined = existing
                if leading_data is not None:
                    combined = self._merge(combined, leading_data)
                if forward_data is not None:
                    combined = self._merge(combined, forward_data)
                combined.to_parquet(self._paths.price_file(ticker))
                fetched.append(ticker)
            elif existing is not None and not existing.empty:
                # Have existing cache but no new data
                skipped.append(ticker)
            else:
                # No existing cache and no new data
                failed[ticker] = "empty history"

        if failed:
            self._record_failures(failed)
        return RefreshResult(tuple(fetched), tuple(skipped), failed)

    def close_panel(
        self,
        tickers: Sequence[str],
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Wide close-price frame. No forward fill: gaps stay NaN so downstream
        code cannot silently treat a stale price as a real observation."""
        columns: dict[str, pd.Series] = {}
        for ticker in tickers:
            if not self.has(ticker):
                continue
            frame = self.load(ticker)
            columns[ticker] = frame["close"]
        if not columns:
            return pd.DataFrame()
        panel = pd.DataFrame(columns).sort_index()
        if start is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start)]
        if end is not None:
            panel = panel.loc[panel.index <= pd.Timestamp(end)]
        return panel

    def available(self, tickers: Sequence[str], min_days: int) -> list[str]:
        out = []
        for ticker in tickers:
            if self.has(ticker) and len(self.load(ticker)) >= min_days:
                out.append(ticker)
        return out

    def _read_existing(self, ticker: str) -> pd.DataFrame | None:
        path = self._paths.price_file(ticker)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt file: treat as absent
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _merge(
        existing: pd.DataFrame | None, incoming: pd.DataFrame
    ) -> pd.DataFrame:
        if existing is None or existing.empty:
            return incoming.sort_index()
        combined = pd.concat([existing, incoming])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def _record_failures(self, failed: dict[str, str]) -> None:
        path = self._paths.failures_csv
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["ticker", "reason"])
            for ticker, reason in failed.items():
                writer.writerow([ticker, reason])
