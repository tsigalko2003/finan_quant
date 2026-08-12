from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from filelock import FileLock

NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
)
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MIN_EXPORT_ROWS = 1_000
SCHEMA_VERSION = 3
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.^/-]{0,14}$")
NON_COMMON_PATTERN = re.compile(
    r"\b(warrants?|rights?|units?|preferred|preference|bonds?|debentures?)\b|\bnotes due\b",
    re.IGNORECASE,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def industry_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _eligible_name(name: str) -> bool:
    return NON_COMMON_PATTERN.search(name) is None


def _market_cap(value: Any) -> float | None:
    try:
        result = float(str(value).replace("$", "").replace(",", ""))
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def normalize_export(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise TypeError("Nasdaq response does not contain a stock row list")
    rows = data["rows"]
    if len(rows) < MIN_EXPORT_ROWS:
        raise ValueError(f"Nasdaq export is unexpectedly small ({len(rows)} rows)")

    normalized: list[dict[str, Any]] = []
    invalid = 0
    for raw in rows:
        if not isinstance(raw, dict):
            invalid += 1
            continue
        source_symbol = str(raw.get("symbol") or "").strip().upper()
        name = str(raw.get("name") or "").strip()
        industry = str(raw.get("industry") or "").strip()
        sector = str(raw.get("sector") or "").strip()
        if not TICKER_PATTERN.fullmatch(source_symbol) or not name:
            invalid += 1
            continue
        symbol = source_symbol.replace("/", "-").replace(".", "-")
        normalized.append(
            {
                "symbol": symbol,
                "source_symbol": source_symbol,
                "name": name,
                "sector": sector,
                "industry": industry,
                "country": str(raw.get("country") or "").strip(),
                "market_cap": _market_cap(raw.get("marketCap")),
                "eligible_common_equity": _eligible_name(name),
            }
        )
    if not normalized or invalid > len(rows) * 0.10:
        raise ValueError("Nasdaq export failed ticker/schema validation")

    by_symbol: dict[str, dict[str, Any]] = {}
    for row in normalized:
        current = by_symbol.get(row["symbol"])
        if current is None or (row["market_cap"] or -1) > (current["market_cap"] or -1):
            by_symbol[row["symbol"]] = row
    normalized = sorted(by_symbol.values(), key=lambda item: item["symbol"])
    rows_bytes = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    membership = json.dumps(
        [
            {
                key: row[key]
                for key in (
                    "symbol",
                    "source_symbol",
                    "sector",
                    "industry",
                    "eligible_common_equity",
                )
            }
            for row in normalized
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "rows": normalized,
        "source_rows": len(rows),
        "normalized_rows": len(normalized),
        "invalid_rows": invalid,
        "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "membership_sha256": hashlib.sha256(membership).hexdigest(),
    }


def download_nasdaq_export(timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(
        NASDAQ_SCREENER_URL,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
            "User-Agent": "Mozilla/5.0 (compatible; sector-screener/0.1)",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            raise ValueError("Nasdaq export returned a non-JSON response")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Nasdaq export exceeded the response-size limit")
    return json.loads(raw)


class NasdaqUniverseCache:
    def __init__(
        self,
        cache_dir: Path,
        ttl_hours: int = 24,
        downloader: Callable[[], dict[str, Any]] = download_nasdaq_export,
    ):
        self.root = Path(cache_dir) / "universes" / "nasdaq"
        self.latest_path = self.root / "latest.json"
        self.lock = FileLock(str(self.root / "refresh.lock"))
        self.ttl = timedelta(hours=ttl_hours)
        self.downloader = downloader

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _read_latest(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        try:
            snapshot = json.loads(self.latest_path.read_text(encoding="utf-8"))
            if snapshot.get("schema_version") != SCHEMA_VERSION:
                return None
            expected = snapshot.get("rows_sha256")
            actual = hashlib.sha256(
                json.dumps(snapshot["rows"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if not expected or expected != actual:
                return None
            return snapshot
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def is_fresh(self, snapshot: dict[str, Any]) -> bool:
        retrieved = datetime.fromisoformat(snapshot["retrieved_at"])
        return self._now() - retrieved <= self.ttl

    def load(self) -> dict[str, Any]:
        snapshot = self._read_latest()
        if snapshot is None:
            raise FileNotFoundError(
                "Nasdaq universe cache is missing. Refresh the Nasdaq universe before resolving it."
            )
        return snapshot

    def ensure(self, refresh: bool = False, force: bool = False) -> dict[str, Any]:
        current = self._read_latest()
        if current is not None and not force and not refresh and self.is_fresh(current):
            return {**current, "cache_hit": True, "stale_cache_used": False}
        with self.lock:
            current = self._read_latest()
            if current is not None and not force and not refresh and self.is_fresh(current):
                return {**current, "cache_hit": True, "stale_cache_used": False}
            try:
                raw_payload = self.downloader()
                normalized = normalize_export(raw_payload)
                if (
                    current is not None
                    and not force
                    and normalized["normalized_rows"] < current["normalized_rows"] * 0.70
                ):
                    raise ValueError("Nasdaq export membership dropped by more than 30%")
                retrieved_at = self._now().isoformat()
                snapshot_id = (
                    self._now().strftime("%Y%m%dT%H%M%SZ")
                    + "-"
                    + normalized["membership_sha256"][:12]
                )
                snapshot = {
                    "schema_version": SCHEMA_VERSION,
                    "source": "nasdaq-stock-screener-export",
                    "source_url": NASDAQ_SCREENER_URL,
                    "snapshot_id": snapshot_id,
                    "retrieved_at": retrieved_at,
                    **normalized,
                }
                self.root.mkdir(parents=True, exist_ok=True)
                snapshot_dir = self.root / "snapshots"
                raw_dir = self.root / "raw"
                snapshot_dir.mkdir(exist_ok=True)
                raw_dir.mkdir(exist_ok=True)
                raw_bytes = json.dumps(raw_payload, separators=(",", ":")).encode()
                raw_path = raw_dir / f"{snapshot_id}.json"
                if not raw_path.exists():
                    raw_path.write_bytes(raw_bytes)
                snapshot["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
                snapshot_path = snapshot_dir / f"{snapshot_id}.json"
                snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                tmp = self.latest_path.with_suffix(".tmp.json")
                tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                os.replace(tmp, self.latest_path)
                return {**snapshot, "cache_hit": False, "stale_cache_used": False}
            except Exception:
                if current is not None:
                    return {**current, "cache_hit": True, "stale_cache_used": True}
                raise

    def describe(self) -> list[dict[str, Any]]:
        try:
            snapshot = self.load()
        except FileNotFoundError:
            return []
        counts: dict[str, int] = {}
        for row in snapshot["rows"]:
            if row["industry"] and row["eligible_common_equity"]:
                counts[row["industry"]] = counts.get(row["industry"], 0) + 1
        return [
            {
                "name": f"nasdaq:{industry_slug(industry)}",
                "source": "nasdaq-export",
                "tickers": count,
                "description": f"Nasdaq export industry: {industry}",
                "snapshot_id": snapshot["snapshot_id"],
                "retrieved_at": snapshot["retrieved_at"],
            }
            for industry, count in sorted(counts.items())
        ]

    def query(self, query: str, max_tickers: int | None = None) -> tuple[list[str], dict[str, Any]]:
        snapshot = self.load()
        wanted = industry_slug(query)
        matches = [
            row
            for row in snapshot["rows"]
            if row["eligible_common_equity"] and wanted and wanted in industry_slug(row["industry"])
        ]
        matches.sort(key=lambda row: (-(row["market_cap"] or -1), row["symbol"]))
        full_count = len(matches)
        if max_tickers is not None:
            matches = matches[: int(max_tickers)]
        if len(matches) < 3:
            raise ValueError(f"Nasdaq industry query '{query}' returned fewer than three equities")
        industries = sorted({row["industry"] for row in matches})
        return [row["symbol"] for row in matches], {
            "snapshot_id": snapshot["snapshot_id"],
            "retrieved_at": snapshot["retrieved_at"],
            "membership_sha256": snapshot["membership_sha256"],
            "query": query,
            "matched_industries": industries,
            "full_eligible_count": full_count,
            "selected_count": len(matches),
            "selection": "market-cap descending",
        }
