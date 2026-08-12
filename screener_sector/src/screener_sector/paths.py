"""Filesystem path resolution.

This is the ONLY module in the package permitted to build filesystem paths.
Every path derives from a single DATA_DIR root, which is what makes the data
directory relocatable to any machine at any path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = "data"
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,15}$")


@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the pipeline, rooted at DATA_DIR."""

    root: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Paths:
        source = os.environ if env is None else env
        return cls(root=Path(source.get("DATA_DIR") or DEFAULT_DATA_DIR))

    @property
    def manifest_file(self) -> Path:
        return self.root / "manifest.json"

    @property
    def universe_csv(self) -> Path:
        return self.root / "universe.csv"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def symbols_parquet(self) -> Path:
        return self.meta_dir / "symbols.parquet"

    @property
    def info_parquet(self) -> Path:
        return self.meta_dir / "info.parquet"

    @property
    def failures_csv(self) -> Path:
        return self.meta_dir / "failures.csv"

    @property
    def prices_dir(self) -> Path:
        return self.root / "prices"

    def price_file(self, ticker: str) -> Path:
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"invalid ticker for filename: {ticker!r}")
        return self.prices_dir / f"{ticker}.parquet"

    def derived_dir(self, profile: str) -> Path:
        if not profile.isidentifier():
            raise ValueError(f"invalid profile name: {profile!r}")
        return self.root / "derived" / profile

    def ensure(self) -> None:
        for directory in (self.root, self.meta_dir, self.prices_dir):
            directory.mkdir(parents=True, exist_ok=True)
