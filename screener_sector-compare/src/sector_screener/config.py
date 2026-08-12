from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    config_dir: Path

    @property
    def stage(self) -> str:
        return str(self.raw["stage"])

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw["analysis"]

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw["universe"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def cache_dir(self) -> Path:
        return Path(os.getenv("SCREENER_CACHE_DIR", self.data["cache_dir"])).resolve()

    @property
    def output_dir(self) -> Path:
        return Path(os.getenv("SCREENER_OUTPUT_DIR", self.data["output_dir"])).resolve()


def load_settings(stage: str, config_dir: Path | None = None) -> Settings:
    if stage not in {"poc", "prod"}:
        raise ValueError("stage must be 'poc' or 'prod'")
    directory = Path(
        config_dir or os.getenv("SCREENER_CONFIG_DIR", Path.cwd() / "config")
    ).resolve()
    merged = _deep_merge(
        _read_yaml(directory / "base.yaml"), _read_yaml(directory / f"{stage}.yaml")
    )
    merged.setdefault("universe", {})
    return Settings(raw=merged, config_dir=directory)
