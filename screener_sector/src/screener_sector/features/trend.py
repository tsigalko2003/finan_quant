"""Trend strength and trend quality.

The R-squared term is what separates a clean advance from a drift with the
same net move: two names can have identical 60-day returns while one trends
smoothly and the other whipsaws. Only the first is tradeable as a trend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener_sector.config import TrendWeights, Windows

TRADING_DAYS_PER_YEAR = 252


def log_slope_r2(close: pd.Series, window: int) -> tuple[float, float]:
    """Annualized slope of log price and the fit's R-squared over the last
    `window` bars. Returns (0.0, 0.0) when the series is too short or flat."""
    tail = close.dropna().tail(window)
    if len(tail) < 3:
        return 0.0, 0.0
    y = np.log(tail.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)
    if np.allclose(y, y[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope) * TRADING_DAYS_PER_YEAR, float(max(r2, 0.0))


def adx(ohlcv: pd.DataFrame, window: int = 14) -> float:
    """Average Directional Index over the final `window` bars, 0-100."""
    frame = ohlcv.dropna(subset=["high", "low", "close"])
    if len(frame) < window * 2 + 1:
        return 0.0

    high, low, close = frame["high"], frame["low"], frame["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr

    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    value = dx.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    return 0.0 if pd.isna(value) else float(value)


def _vote(left: float, right: float, tolerance: float = 1e-9) -> int:
    """+1 if left exceeds right, -1 if below, 0 when indistinguishable."""
    if pd.isna(left) or pd.isna(right):
        return 0
    difference = float(left) - float(right)
    scale = max(abs(float(right)), 1.0)
    if abs(difference) <= tolerance * scale:
        return 0
    return 1 if difference > 0 else -1


def ma_stack_score(close: pd.Series, short: int, mid: int) -> float:
    """+1 when price > short MA > mid MA and both are rising; -1 when fully
    inverted; 0 when flat or mixed. Each check votes +1, -1, or 0, so a
    trendless series scores neutral rather than bearish."""
    series = close.dropna()
    if len(series) < mid + 5:
        return 0.0
    short_ma = series.rolling(short).mean()
    mid_ma = series.rolling(mid).mean()
    votes = [
        _vote(series.iloc[-1], short_ma.iloc[-1]),
        _vote(short_ma.iloc[-1], mid_ma.iloc[-1]),
        _vote(short_ma.iloc[-1], short_ma.iloc[-5]),
        _vote(mid_ma.iloc[-1], mid_ma.iloc[-5]),
    ]
    return float(sum(votes) / len(votes))


@dataclass(frozen=True)
class TrendResult:
    slope: float
    r2: float
    adx: float
    ma_stack: float
    score: float


def trend_score(
    ohlcv: pd.DataFrame, window: int, weights: TrendWeights
) -> TrendResult:
    close = ohlcv["close"]
    slope, r2 = log_slope_r2(close, window)

    returns = np.log(close).diff().dropna().tail(window)
    volatility = float(returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    normalized = 0.0 if volatility == 0 else slope / volatility
    slope_component = float(np.tanh(normalized))

    direction = np.sign(slope_component)
    adx_value = adx(ohlcv)
    stack = ma_stack_score(close, max(window // 3, 2), window)

    score = 100.0 * (
        weights.slope * slope_component
        + weights.r2 * r2 * direction
        + weights.adx * min(adx_value / 50.0, 1.0) * direction
        + weights.ma_stack * stack
    )
    return TrendResult(
        slope=slope,
        r2=r2,
        adx=adx_value,
        ma_stack=stack,
        score=float(np.clip(score, -100.0, 100.0)),
    )


def trend_table(
    frames: dict[str, pd.DataFrame], windows: Windows, weights: TrendWeights
) -> pd.DataFrame:
    rows = []
    for ticker, frame in frames.items():
        if len(frame.dropna(subset=["close"])) < windows.mid + 5:
            continue
        short = trend_score(frame, windows.short, weights)
        mid = trend_score(frame, windows.mid, weights)
        rows.append(
            {
                "ticker": ticker,
                "short_score": short.score,
                "mid_score": mid.score,
                "short_r2": short.r2,
                "mid_r2": mid.r2,
                "adx": mid.adx,
                "ma_stack": mid.ma_stack,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker", "short_score", "mid_score", "short_r2", "mid_r2",
            "adx", "ma_stack",
        ],
    )
