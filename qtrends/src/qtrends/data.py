from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qtrends.config import DataConfig


@dataclass(frozen=True)
class MarketData:
    close: pd.DataFrame
    volume: pd.DataFrame
    membership: pd.DataFrame | None = None

    def validate(self, config: DataConfig) -> "MarketData":
        required = set(config.tickers + [config.benchmark])
        missing = required.difference(self.close.columns)
        if missing:
            raise ValueError(f"Missing close data for: {sorted(missing)}")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("Market data dates must be sorted")
        if self.close.index.has_duplicates:
            raise ValueError("Market data contains duplicate dates")
        if len(self.close) < 200:
            raise ValueError("At least 200 daily observations are required")
        return self


def load_market_data(config: DataConfig) -> MarketData:
    if config.provider == "csv":
        result = _load_csv(Path(config.csv_path or ""))
    elif config.provider == "yahoo":
        result = _load_yahoo(config)
    elif config.provider == "qlib":
        result = _load_qlib(config)
    else:  # pragma: no cover - guarded by pydantic
        raise ValueError(f"Unsupported provider: {config.provider}")

    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end) if config.end else None
    close = result.close.loc[start:end].sort_index()
    volume = result.volume.reindex(close.index).sort_index()
    close.columns = [str(column).upper() for column in close.columns]
    volume.columns = [str(column).upper() for column in volume.columns]
    membership = _membership_matrix(close.index, config)
    if membership is not None:
        for ticker in config.tickers:
            if ticker in close and ticker in membership:
                close.loc[~membership[ticker], ticker] = np.nan
                volume.loc[~membership[ticker], ticker] = np.nan
    return MarketData(close=close, volume=volume, membership=membership).validate(config)


def _membership_matrix(index: pd.DatetimeIndex, config: DataConfig) -> pd.DataFrame | None:
    if not config.membership_manifest_path:
        return None
    manifest_path = Path(config.membership_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Universe membership manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"symbol": str})
    manifest["symbol"] = manifest["symbol"].astype(str).str.upper()
    manifest = manifest.set_index("symbol")
    membership = pd.DataFrame(False, index=index, columns=config.tickers, dtype=bool)
    for ticker in config.tickers:
        if ticker not in manifest.index:
            raise ValueError(f"Ticker {ticker} is absent from membership manifest")
        effective_from = pd.Timestamp(manifest.loc[ticker, "effective_from"])
        membership[ticker] = index >= effective_from
    return membership


def _load_csv(path: Path) -> MarketData:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV data not found at {path}. Run `qtrends generate-sample` or use provider=yahoo/qlib."
        )
    raw = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "close", "volume"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    raw["ticker"] = raw["ticker"].astype(str).str.upper()
    close = raw.pivot(index="date", columns="ticker", values="close")
    volume = raw.pivot(index="date", columns="ticker", values="volume")
    return MarketData(close=close, volume=volume)


def _load_yahoo(config: DataConfig) -> MarketData:
    import yfinance as yf

    symbols = config.tickers + [config.benchmark]
    raw = yf.download(
        tickers=symbols,
        start=config.start,
        end=config.end,
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("Yahoo returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": symbols[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    return MarketData(close=close, volume=volume)


def _load_qlib(config: DataConfig) -> MarketData:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    provider_uri = str(Path(config.qlib_provider_uri or "").expanduser().resolve())
    qlib.init(provider_uri=provider_uri, region=REG_US)
    instruments = config.tickers + [config.benchmark]
    frame = D.features(
        instruments=instruments,
        fields=["$close", "$volume"],
        start_time=config.start,
        end_time=config.end,
        freq="day",
    )
    if frame.empty:
        raise ValueError(f"Qlib provider at {provider_uri} returned no data")
    normalized = frame.reset_index()
    normalized["instrument"] = normalized["instrument"].astype(str).str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"])
    close = normalized.pivot(index="datetime", columns="instrument", values="$close")
    volume = normalized.pivot(index="datetime", columns="instrument", values="$volume")
    return MarketData(close=close, volume=volume)


def generate_synthetic_csv(
    path: str | Path,
    tickers: list[str] | None = None,
    benchmark: str = "MARKET",
    periods: int = 1250,
    seed: int = 42,
) -> Path:
    """Create deterministic, regime-changing OHLCV-like input for demos and tests."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = tickers or ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=periods)

    thirds = np.array_split(np.arange(periods), 3)
    group_drift = np.zeros(periods)
    group_vol = np.zeros(periods)
    group_drift[thirds[0]], group_vol[thirds[0]] = 0.00045, 0.007
    group_drift[thirds[1]], group_vol[thirds[1]] = -0.00035, 0.013
    group_drift[thirds[2]], group_vol[thirds[2]] = 0.00075, 0.009

    market_returns = rng.normal(0.00025, 0.009, periods)
    common = group_drift + rng.normal(0.0, group_vol)
    rows: list[pd.DataFrame] = []

    market_price = 100.0 * np.exp(np.cumsum(market_returns))
    market_volume = rng.lognormal(mean=15.5, sigma=0.20, size=periods)
    rows.append(
        pd.DataFrame(
            {"date": dates, "ticker": benchmark, "close": market_price, "volume": market_volume}
        )
    )

    for position, ticker in enumerate(names):
        beta = 0.75 + 0.12 * position
        idiosyncratic = rng.normal(0.0, 0.006 + 0.0008 * position, periods)
        returns = beta * market_returns + common + idiosyncratic
        price = (40.0 + position * 15.0) * np.exp(np.cumsum(returns))
        volume = rng.lognormal(mean=14.8 + 0.1 * position, sigma=0.25, size=periods)
        rows.append(
            pd.DataFrame({"date": dates, "ticker": ticker, "close": price, "volume": volume})
        )

    pd.concat(rows, ignore_index=True).to_csv(output, index=False)
    return output
