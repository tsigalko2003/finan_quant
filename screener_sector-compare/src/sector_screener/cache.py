from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .providers.base import MarketDataProvider

Interval = tuple[pd.Timestamp, pd.Timestamp]


def _day(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize()


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted((_day(a), _day(b)) for a, b in intervals if _day(a) < _day(b))
    merged: list[list[pd.Timestamp]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def missing_intervals(requested: Interval, covered: Iterable[Interval]) -> list[Interval]:
    start, end = _day(requested[0]), _day(requested[1])
    cursor = start
    gaps: list[Interval] = []
    for left, right in merge_intervals(covered):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            gaps.append((cursor, min(left, end)))
        cursor = max(cursor, right)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


@dataclass
class CacheResult:
    ticker: str
    frame: pd.DataFrame
    requested_ranges: list[Interval]
    cache_hit: bool


class MarketDataCache:
    schema_version = 1

    def __init__(self, root: Path, provider: MarketDataProvider, interval: str, auto_adjust: bool):
        self.root = Path(root)
        self.provider = provider
        self.interval = interval
        self.auto_adjust = auto_adjust

    def _stem(self, ticker: str) -> Path:
        policy = "adjusted" if self.auto_adjust else "raw"
        return self.root / self.provider.name / self.interval / policy / ticker.upper()

    def _read_manifest(self, path: Path, ticker: str) -> dict:
        if not path.exists():
            return {
                "schema_version": self.schema_version,
                "provider": self.provider.name,
                "ticker": ticker.upper(),
                "interval": self.interval,
                "auto_adjust": self.auto_adjust,
                "coverage": [],
                "downloads": [],
            }
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _coverage(manifest: dict) -> list[Interval]:
        return [(_day(item["start"]), _day(item["end"])) for item in manifest["coverage"]]

    def fetch(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        refresh_tail_days: int = 0,
        force: bool = False,
    ) -> CacheResult:
        stem = self._stem(ticker)
        data_path = stem.with_suffix(".parquet")
        manifest_path = stem.with_suffix(".json")
        stem.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(stem.with_suffix(".lock")))
        with lock:
            manifest = self._read_manifest(manifest_path, ticker)
            coverage = [] if force else self._coverage(manifest)
            gaps = missing_intervals((_day(start), _day(end)), coverage)
            if refresh_tail_days > 0 and not force:
                tail_start = max(_day(start), _day(end) - pd.offsets.Day(refresh_tail_days))
                gaps = merge_intervals([*gaps, (tail_start, _day(end))])

            existing = pd.DataFrame()
            if data_path.exists() and not force:
                try:
                    expected_hash = manifest.get("sha256")
                    if (
                        expected_hash
                        and hashlib.sha256(data_path.read_bytes()).hexdigest() != expected_hash
                    ):
                        raise ValueError("cached Parquet checksum mismatch")
                    existing = pd.read_parquet(data_path)
                except Exception:  # noqa: BLE001 - quarantine any corrupt cache representation
                    quarantine = data_path.with_suffix(
                        f".corrupt-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.parquet"
                    )
                    os.replace(data_path, quarantine)
                    coverage = []
                    gaps = [(_day(start), _day(end))]
                    manifest["coverage"] = []
                    manifest["quarantined"] = str(quarantine)
            downloaded: list[pd.DataFrame] = []
            successful: list[Interval] = []
            for left, right in gaps:
                frame = self.provider.download(
                    ticker=ticker,
                    start=left,
                    end=right,
                    interval=self.interval,
                    auto_adjust=self.auto_adjust,
                )
                if frame.empty:
                    continue
                downloaded.append(frame)
                successful.append((left, right))
                manifest["downloads"].append(
                    {
                        "start": left.date().isoformat(),
                        "end": right.date().isoformat(),
                        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "rows": len(frame),
                    }
                )

            if downloaded:
                combined = pd.concat([existing, *downloaded]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                tmp = data_path.with_suffix(".tmp.parquet")
                combined.to_parquet(tmp)
                os.replace(tmp, data_path)
                coverage = merge_intervals([*coverage, *successful])
                manifest["coverage"] = [
                    {"start": a.date().isoformat(), "end": b.date().isoformat()}
                    for a, b in coverage
                ]
                manifest["rows"] = len(combined)
                manifest["sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
                tmp_manifest = manifest_path.with_suffix(".tmp.json")
                tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                os.replace(tmp_manifest, manifest_path)
                existing = combined
            elif not data_path.exists():
                raise RuntimeError(f"No data returned for {ticker} in {start.date()}..{end.date()}")

            frame = existing.loc[(_day(start) <= existing.index) & (existing.index < _day(end))]
            return CacheResult(
                ticker=ticker.upper(),
                frame=frame,
                requested_ranges=gaps,
                cache_hit=len(gaps) == 0,
            )

    def load(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        path = self._stem(ticker).with_suffix(".parquet")
        manifest_path = self._stem(ticker).with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(
                f"Cache missing for {ticker}: {path}. Run the download stage first."
            )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = manifest.get("sha256")
            if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(
                    f"Cache checksum mismatch for {ticker}; rerun download with --force"
                )
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        result = frame.loc[(_day(start) <= frame.index) & (frame.index < _day(end))]
        if result.empty:
            raise ValueError(f"Cached data for {ticker} does not cover the requested period")
        return result
