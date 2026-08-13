"""Typed configuration with profile resolution.

Profiles change parameters only, never code paths, so a dev run exercises
exactly the code a prod run will execute.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Windows:
    short: int
    mid: int
    corr: int


@dataclass(frozen=True)
class TrendWeights:
    slope: float
    r2: float
    adx: float
    ma_stack: float


@dataclass(frozen=True)
class ReboundWeights:
    breadth: float
    stretch: float
    oscillator: float
    volume: float
    confirmation: float


@dataclass(frozen=True)
class UniverseFilters:
    min_price: float
    min_dollar_volume: float
    min_history_days: int


@dataclass(frozen=True)
class BacktestParams:
    warmup_years: int
    label_k: int
    label_forward_days: int
    label_min_return: float
    initial_fit_years: int
    step_years: int
    horizons: tuple[int, ...]


@dataclass(frozen=True)
class NetworkParams:
    enrich_pause_seconds: float
    rate_limit_backoff_seconds: tuple[float, ...]


@dataclass(frozen=True)
class Config:
    profile: str
    start: date
    end: date | None
    universe_mode: str
    static_tickers: tuple[str, ...]
    benchmark: str
    windows: Windows
    trend_weights: TrendWeights
    rebound_weights: ReboundWeights
    corr_threshold: float
    min_cluster_size: int
    filters: UniverseFilters
    backtest: BacktestParams
    network: NetworkParams

    @property
    def fetch_start(self) -> date:
        """Data floor, earlier than the backtest floor so the first fold's fit
        window has history to fit on."""
        target_year = self.start.year - self.backtest.warmup_years
        try:
            return self.start.replace(year=target_year)
        except ValueError:
            # Raised when start is Feb 29 and target_year is not a leap year.
            # Clamp to Feb 28 instead.
            return self.start.replace(year=target_year, day=28)

    @classmethod
    def load(cls, config_dir: Path, profile: str) -> Config:
        raw = yaml.safe_load((config_dir / "params.yaml").read_text())
        if profile not in raw["profiles"]:
            raise KeyError(f"unknown profile: {profile!r}")

        merged = _deep_merge(deepcopy(raw["defaults"]), raw["profiles"][profile])

        static: tuple[str, ...] = ()
        if merged["universe_mode"] == "static":
            static_file = config_dir / f"universe.{profile}.yaml"
            static = tuple(yaml.safe_load(static_file.read_text())["tickers"])

        end_raw = merged.get("end")
        return cls(
            profile=profile,
            start=date.fromisoformat(merged["start"]),
            end=date.fromisoformat(end_raw) if end_raw else None,
            universe_mode=merged["universe_mode"],
            static_tickers=static,
            benchmark=merged["benchmark"],
            windows=Windows(**merged["windows"]),
            trend_weights=TrendWeights(**merged["trend_weights"]),
            rebound_weights=ReboundWeights(**merged["rebound_weights"]),
            corr_threshold=float(merged["corr_threshold"]),
            min_cluster_size=int(merged["min_cluster_size"]),
            filters=UniverseFilters(**merged["filters"]),
            backtest=BacktestParams(
                **{**merged["backtest"], "horizons": tuple(merged["backtest"]["horizons"])}
            ),
            network=NetworkParams(
                enrich_pause_seconds=float(merged["network"]["enrich_pause_seconds"]),
                rate_limit_backoff_seconds=tuple(merged["network"]["rate_limit_backoff_seconds"]),
            ),
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay `override` onto `base`, mutating and returning base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
