from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class UniverseConfig(BaseModel):
    source: Literal["nasdaq_screener"] = "nasdaq_screener"
    industry: str
    sector: str | None = None
    export_path: str | None = None
    manifest_path: str
    refresh_on_run: bool = True
    min_market_cap: float = Field(default=0.0, ge=0.0)
    max_symbols: int | None = Field(default=None, ge=2)
    min_snapshot_retention: float = Field(default=0.80, gt=0.0, le=1.0)
    countries: list[str] = Field(default_factory=list)
    exclude_name_contains: list[str] = Field(
        default_factory=lambda: ["Warrant", "Right", "Unit", "Preferred"]
    )
    initial_effective_from: str | None = None


class DataConfig(BaseModel):
    provider: Literal["csv", "yahoo", "qlib"] = "csv"
    csv_path: str | None = None
    qlib_provider_uri: str | None = None
    tickers: list[str] = Field(default_factory=list)
    membership_manifest_path: str | None = None
    benchmark: str
    start: str
    end: str | None = None

    @model_validator(mode="after")
    def validate_provider_settings(self) -> "DataConfig":
        self.tickers = [ticker.upper() for ticker in self.tickers]
        self.benchmark = self.benchmark.upper()
        if self.tickers and len(set(self.tickers)) < 2:
            raise ValueError("At least two distinct group tickers are required")
        if self.benchmark in self.tickers:
            raise ValueError("benchmark must not also be a group ticker")
        if self.provider == "csv" and not self.csv_path:
            raise ValueError("csv_path is required when provider=csv")
        if self.provider == "qlib" and not self.qlib_provider_uri:
            raise ValueError("qlib_provider_uri is required when provider=qlib")
        return self


class FeatureConfig(BaseModel):
    beta_lookback: int = Field(default=60, ge=20)
    breadth_ma_windows: list[int] = [20, 50]
    trend_lookbacks: list[int] = [5, 20, 60]
    correlation_lookback: int = Field(default=60, ge=20)
    volatility_lookback: int = Field(default=20, ge=5)
    pca_lookback: int = Field(default=252, ge=60)
    min_constituent_coverage: float = Field(default=0.80, gt=0.0, le=1.0)


class HMMConfig(BaseModel):
    states: int = Field(default=3, ge=2, le=5)
    min_train_size: int = Field(default=504, ge=126)
    refit_every: int = Field(default=21, ge=1)
    n_iter: int = Field(default=300, ge=50)
    seeds: list[int] = [7, 17, 29]


class XGBoostConfig(BaseModel):
    horizon: int = Field(default=20, ge=1)
    min_train_size: int = Field(default=504, ge=126)
    refit_every: int = Field(default=21, ge=1)
    n_estimators: int = Field(default=400, ge=10)
    max_depth: int = Field(default=3, ge=1, le=10)
    learning_rate: float = Field(default=0.03, gt=0.0, le=1.0)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    random_state: int = 42


class SignalConfig(BaseModel):
    bullish_probability: float = Field(default=0.60, gt=0.5, lt=1.0)
    bearish_probability: float = Field(default=0.40, gt=0.0, lt=0.5)
    hmm_probability: float = Field(default=0.50, gt=0.0, lt=1.0)
    bullish_breadth: float = Field(default=0.60, gt=0.5, le=1.0)
    bearish_breadth: float = Field(default=0.40, ge=0.0, lt=0.5)


class PipelineConfig(BaseModel):
    data: DataConfig
    universe: UniverseConfig | None = None
    features: FeatureConfig = FeatureConfig()
    hmm: HMMConfig = HMMConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    signal: SignalConfig = SignalConfig()

    @model_validator(mode="after")
    def validate_universe_or_tickers(self) -> "PipelineConfig":
        if not self.universe and len(set(self.data.tickers)) < 2:
            raise ValueError("Configure either universe or at least two distinct tickers")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return PipelineConfig.model_validate(raw)
