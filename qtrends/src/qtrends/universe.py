from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from qtrends.config import PipelineConfig, UniverseConfig


NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
MANIFEST_COLUMNS = [
    "symbol",
    "name",
    "sector",
    "industry",
    "country",
    "ipo_year",
    "market_cap",
    "discovered_at",
    "effective_from",
    "last_seen",
    "active",
    "source",
]


@dataclass(frozen=True)
class UniverseSyncResult:
    manifest_path: Path
    active_symbols: list[str]
    added_symbols: list[str]
    deactivated_symbols: list[str]
    snapshot_date: str


def _fetch_nasdaq_rows(timeout: int = 30) -> list[dict[str, Any]]:
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
        ),
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
    }
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
        ),
    )
    response = session.get(
        NASDAQ_SCREENER_URL,
        params={"tableonly": "true", "download": "true"},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = ((payload.get("data") or {}).get("rows") or [])
    if not rows:
        raise RuntimeError("Nasdaq Screener returned no rows")
    return rows


def _read_export(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Nasdaq Screener export not found: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    normalized_columns = {
        column: column.strip().lower().replace(" ", "").replace("_", "")
        for column in frame.columns
    }
    frame = frame.rename(columns=normalized_columns)
    aliases = {
        "marketcap": "marketCap",
        "ipoyear": "ipoyear",
        "lastsale": "lastsale",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    return frame.to_dict(orient="records")


def _number(value: Any) -> float:
    text = str(value or "").strip().replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_rows(rows: list[dict[str, Any]], config: UniverseConfig) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    exclusions = [pattern.casefold() for pattern in config.exclude_name_contains]
    allowed_countries = {country.casefold() for country in config.countries}
    for raw in rows:
        symbol = str(raw.get("symbol", "")).strip().upper()
        name = str(raw.get("name", "")).strip()
        industry = str(raw.get("industry", "")).strip()
        sector = str(raw.get("sector", "")).strip()
        country = str(raw.get("country", "")).strip()
        market_cap = _number(raw.get("marketCap", raw.get("marketcap", 0)))
        if not symbol or industry.casefold() != config.industry.casefold():
            continue
        if config.sector and sector.casefold() != config.sector.casefold():
            continue
        if allowed_countries and country.casefold() not in allowed_countries:
            continue
        if market_cap < config.min_market_cap:
            continue
        if any(pattern in name.casefold() for pattern in exclusions):
            continue
        normalized.append(
            {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "industry": industry,
                "country": country,
                "ipo_year": str(raw.get("ipoyear", raw.get("ipoyear", ""))).strip(),
                "market_cap": market_cap,
            }
        )
    if not normalized:
        raise ValueError(f"No Nasdaq Screener rows matched industry={config.industry!r}")
    frame = pd.DataFrame(normalized).drop_duplicates("symbol", keep="first")
    frame = frame.sort_values(["market_cap", "symbol"], ascending=[False, True])
    if config.max_symbols:
        frame = frame.head(config.max_symbols)
    return frame.reset_index(drop=True)


def load_nasdaq_snapshot(config: UniverseConfig) -> pd.DataFrame:
    rows = _read_export(Path(config.export_path)) if config.export_path else _fetch_nasdaq_rows()
    return _normalize_rows(rows, config)


def sync_universe(
    config: PipelineConfig,
    snapshot_date: str | None = None,
) -> UniverseSyncResult:
    if config.universe is None:
        raise ValueError("No universe configuration is present")
    universe = config.universe
    observed_on = snapshot_date or date.today().isoformat()
    current = load_nasdaq_snapshot(universe)
    manifest_path = Path(universe.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    first_sync = not manifest_path.exists()
    existing = (
        pd.read_csv(manifest_path, dtype={"symbol": str})
        if not first_sync
        else pd.DataFrame(columns=MANIFEST_COLUMNS)
    )
    existing_records = {
        str(row["symbol"]).upper(): row.to_dict() for _, row in existing.iterrows()
    }
    previous_active = {
        symbol for symbol, row in existing_records.items() if _as_bool(row.get("active", False))
    }
    current_symbols = set(current["symbol"])
    if previous_active:
        retained_fraction = len(current_symbols.intersection(previous_active)) / len(previous_active)
        if retained_fraction < universe.min_snapshot_retention:
            raise RuntimeError(
                "Nasdaq snapshot retained only "
                f"{retained_fraction:.1%} of the prior active universe; "
                "manifest was not modified"
            )
    records: dict[str, dict[str, Any]] = dict(existing_records)

    for row in current.to_dict(orient="records"):
        symbol = row["symbol"]
        if symbol in records:
            preserved = records[symbol]
            row["discovered_at"] = preserved.get("discovered_at", observed_on)
            row["effective_from"] = preserved.get("effective_from", observed_on)
        else:
            row["discovered_at"] = observed_on
            if first_sync:
                bootstrap_start = universe.initial_effective_from or config.data.start
                ipo_year = str(row.get("ipo_year", ""))
                ipo_start = f"{ipo_year}-01-01" if ipo_year.isdigit() else bootstrap_start
                row["effective_from"] = max(bootstrap_start, ipo_start)
            else:
                row["effective_from"] = observed_on
        row.update({"last_seen": observed_on, "active": True, "source": universe.source})
        records[symbol] = row

    for symbol, row in records.items():
        if symbol not in current_symbols:
            row["active"] = False

    manifest = pd.DataFrame(records.values()).reindex(columns=MANIFEST_COLUMNS)
    manifest["symbol"] = manifest["symbol"].astype(str).str.upper()
    manifest = manifest.sort_values(["active", "market_cap", "symbol"], ascending=[False, False, True])
    manifest.to_csv(manifest_path, index=False)
    active = sorted(current_symbols)
    return UniverseSyncResult(
        manifest_path=manifest_path,
        active_symbols=active,
        added_symbols=sorted(current_symbols.difference(previous_active)),
        deactivated_symbols=sorted(previous_active.difference(current_symbols)),
        snapshot_date=observed_on,
    )


def _as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def active_symbols_from_manifest(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"symbol": str})
    active = frame[frame["active"].map(_as_bool)]
    return sorted(active["symbol"].astype(str).str.upper().unique().tolist())


def resolve_universe(config: PipelineConfig, refresh: bool | None = None) -> tuple[PipelineConfig, UniverseSyncResult | None]:
    if config.universe is None:
        return config, None
    resolved = config.model_copy(deep=True)
    universe = resolved.universe
    assert universe is not None
    manifest_path = Path(universe.manifest_path)
    should_refresh = universe.refresh_on_run if refresh is None else refresh
    result = sync_universe(resolved) if should_refresh or not manifest_path.exists() else None
    tickers = result.active_symbols if result else active_symbols_from_manifest(manifest_path)
    if len(tickers) < 2:
        raise ValueError("Resolved universe must contain at least two active symbols")
    resolved.data.tickers = tickers
    resolved.data.membership_manifest_path = str(manifest_path)
    return resolved, result
