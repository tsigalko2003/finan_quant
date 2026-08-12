from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .nasdaq_universe import NasdaqUniverseCache


@dataclass(frozen=True)
class Universe:
    name: str
    tickers: list[str]
    source: str
    description: str
    metadata: dict[str, Any]


class UniverseCatalog:
    """Versioned project industries plus optional locally installed Qlib pools."""

    def __init__(self, catalog_path: Path):
        with catalog_path.open("r", encoding="utf-8") as handle:
            self._industries = (yaml.safe_load(handle) or {}).get("industries", {})

    def names(self) -> list[str]:
        return sorted(self._industries)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "source": "catalog",
                "tickers": len(spec.get("tickers", [])),
                "description": spec.get("description", ""),
            }
            for name, spec in sorted(self._industries.items())
        ]

    def resolve(
        self,
        name: str,
        max_tickers: int | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        qlib_data_dir: Path | None = None,
        nasdaq_cache_dir: Path | None = None,
    ) -> Universe:
        if name.startswith("qlib:"):
            return self._resolve_qlib(name.split(":", 1)[1], qlib_data_dir, max_tickers)
        if name.startswith("nasdaq:"):
            return self._resolve_nasdaq(
                name.split(":", 1)[1], nasdaq_cache_dir, max_tickers, include, exclude
            )
        if name not in self._industries:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown industry '{name}'. Catalog choices: {available}")
        spec = self._industries[name]
        tickers = [str(t).upper() for t in spec.get("tickers", [])]
        excluded = {t.upper() for t in (exclude or [])}
        tickers = [t for t in tickers if t not in excluded]
        for ticker in include or []:
            ticker = ticker.upper()
            if ticker not in tickers and ticker not in excluded:
                tickers.append(ticker)
        if max_tickers is not None:
            tickers = tickers[: int(max_tickers)]
        if len(tickers) < 3:
            raise ValueError("A multi-ticker sector screen needs at least three tickers")
        return Universe(
            name=name,
            tickers=tickers,
            source="catalog",
            description=spec.get("description", ""),
            metadata={k: v for k, v in spec.items() if k != "tickers"},
        )

    @staticmethod
    def _resolve_nasdaq(
        query: str,
        nasdaq_cache_dir: Path | None,
        max_tickers: int | None,
        include: list[str] | None,
        exclude: list[str] | None,
    ) -> Universe:
        if not nasdaq_cache_dir:
            raise ValueError("A cache directory is required for a nasdaq:<industry> universe")
        tickers, metadata = NasdaqUniverseCache(nasdaq_cache_dir).query(query, max_tickers=None)
        excluded = {ticker.upper() for ticker in (exclude or [])}
        tickers = [ticker for ticker in tickers if ticker not in excluded]
        for ticker in include or []:
            normalized = ticker.upper()
            if normalized not in tickers and normalized not in excluded:
                tickers.append(normalized)
        if max_tickers is not None:
            tickers = tickers[: int(max_tickers)]
        if len(tickers) < 3:
            raise ValueError("A multi-ticker sector screen needs at least three tickers")
        metadata["selected_count"] = len(tickers)
        metadata["ticker_sha256"] = hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest()
        return Universe(
            name=f"nasdaq:{query}",
            tickers=tickers,
            source="nasdaq-export",
            description=f"Nasdaq stock-screener industry query '{query}'",
            metadata=metadata,
        )

    @staticmethod
    def qlib_pools(qlib_data_dir: Path | None) -> list[str]:
        if not qlib_data_dir:
            return []
        instruments = Path(qlib_data_dir).expanduser() / "instruments"
        if not instruments.exists():
            return []
        return sorted(path.stem for path in instruments.glob("*.txt"))

    @staticmethod
    def _resolve_qlib(pool: str, qlib_data_dir: Path | None, max_tickers: int | None) -> Universe:
        if not qlib_data_dir:
            raise ValueError("--qlib-data-dir is required for a qlib:<pool> universe")
        path = Path(qlib_data_dir).expanduser() / "instruments" / f"{pool}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Qlib pool not found: {path}")
        tickers: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ticker = line.split("\t", 1)[0].strip().upper()
            if ticker not in tickers:
                tickers.append(ticker)
        if max_tickers is not None:
            tickers = tickers[: int(max_tickers)]
        if len(tickers) < 3:
            raise ValueError(f"Qlib pool '{pool}' contains fewer than three usable instruments")
        return Universe(
            name=f"qlib:{pool}",
            tickers=tickers,
            source="qlib-pool-file",
            description=f"Installed Qlib stock pool '{pool}'",
            metadata={"path": str(path)},
        )
