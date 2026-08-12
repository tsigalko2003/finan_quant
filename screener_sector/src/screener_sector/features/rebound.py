"""Rebound alarm: group washout first, individual confirmation second.

Oversold oscillators fire constantly and mean little on their own. Requiring
the whole correlated group to be washed out, then requiring a confirmation bar
on the individual name, is what makes the signal selective enough to act on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from screener_sector.config import ReboundWeights, Windows
from screener_sector.features.correlation import Cluster

REBOUND_COLUMNS = [
    "ticker",
    "cluster",
    "alarm",
    "washout",
    "stretch_z",
    "rsi",
    "volume",
    "divergence",
    "confirmed",
    "fired",
]

WASHOUT_GATE = 0.5
ALARM_GATE = 60.0


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def williams_r(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    highest = ohlcv["high"].rolling(window).max()
    lowest = ohlcv["low"].rolling(window).min()
    span = (highest - lowest).replace(0.0, np.nan)
    return -100.0 * (highest - ohlcv["close"]) / span


def stretch_z(close: pd.Series, window: int) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ((close - mean) / std.replace(0.0, np.nan)).fillna(0.0)


def volume_signal(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
    """Capitulation spike followed by dry-up, scored 0-1.

    A high reading means volume blew out recently and has since gone quiet,
    which is the classic seller-exhaustion pattern.
    """
    volume = ohlcv["volume"]
    baseline = volume.rolling(window).median()
    relative = volume / baseline.replace(0.0, np.nan)
    recent_spike = relative.rolling(10).max()
    current_dryness = 1.0 / relative.replace(0.0, np.nan)
    spike_component = ((recent_spike - 2.0) / 3.0).clip(0.0, 1.0)
    dry_component = ((current_dryness - 1.0) / 1.5).clip(0.0, 1.0)
    return (0.6 * spike_component + 0.4 * dry_component).fillna(0.0)


def bullish_divergence(
    close: pd.Series, oscillator: pd.Series, lookback: int = 20
) -> pd.Series:
    """Price makes a lower low over `lookback` while the oscillator does not."""
    price_low = close.rolling(lookback).min()
    prior_price_low = price_low.shift(lookback)
    osc_at_low = oscillator.rolling(lookback).min()
    prior_osc_low = osc_at_low.shift(lookback)
    lower_price = price_low < prior_price_low
    higher_osc = osc_at_low > prior_osc_low
    return (lower_price & higher_osc).fillna(False)


def confirmation(ohlcv: pd.DataFrame, short_window: int) -> pd.Series:
    """Close above the prior bar's high, or a reclaim of the short MA."""
    above_prior_high = ohlcv["close"] > ohlcv["high"].shift(1)
    short_ma = ohlcv["close"].rolling(short_window).mean()
    reclaim = (ohlcv["close"] > short_ma) & (
        ohlcv["close"].shift(1) <= short_ma.shift(1)
    )
    return (above_prior_high | reclaim).fillna(False)


def cluster_washout(
    panel: pd.DataFrame, members: Sequence[str], window: int
) -> pd.Series:
    """Fraction of the group that is both oversold and below its mid-window
    mean, on each date."""
    present = [m for m in members if m in panel.columns]
    if not present:
        return pd.Series(0.0, index=panel.index)
    flags = []
    for ticker in present:
        close = panel[ticker]
        oversold = rsi(close) < 35.0
        below_mean = close < close.rolling(window).mean()
        flags.append((oversold & below_mean).astype(float))
    return pd.concat(flags, axis=1).mean(axis=1).fillna(0.0)


def ticker_alarm(
    ohlcv: pd.DataFrame,
    washout: pd.Series,
    weights: ReboundWeights,
    windows: Windows,
) -> pd.Series:
    close = ohlcv["close"]
    oscillator = rsi(close)

    breadth_component = washout.reindex(close.index).fillna(0.0).clip(0.0, 1.0)
    # stretch: -2 sigma or lower scores 1.0, at or above the mean scores 0.0
    stretch_component = (-stretch_z(close, windows.mid) / 2.0).clip(0.0, 1.0)
    # oscillator: RSI 20 or lower scores 1.0, RSI 50 or higher scores 0.0
    oscillator_component = ((50.0 - oscillator) / 30.0).clip(0.0, 1.0).fillna(0.0)
    volume_component = volume_signal(ohlcv)
    divergence_flag = bullish_divergence(close, oscillator, windows.short)
    confirm_flag = confirmation(ohlcv, windows.short)
    confirmation_component = (
        0.5 * confirm_flag.astype(float) + 0.5 * divergence_flag.astype(float)
    )

    score = 100.0 * (
        weights.breadth * breadth_component
        + weights.stretch * stretch_component
        + weights.oscillator * oscillator_component
        + weights.volume * volume_component
        + weights.confirmation * confirmation_component
    )
    return score.clip(0.0, 100.0)


def rebound_table(
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    clusters: Sequence[Cluster],
    weights: ReboundWeights,
    windows: Windows,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for cluster in clusters:
        members = [m for m in cluster.members if m in frames]
        if not members:
            continue
        washout = cluster_washout(panel, members, windows.mid)

        for ticker in members:
            frame = frames[ticker]
            alarm = ticker_alarm(frame, washout, weights, windows)
            close = frame["close"]
            oscillator = rsi(close)
            divergence = bullish_divergence(close, oscillator, windows.short)
            confirmed = confirmation(frame, windows.short)
            stretch = stretch_z(close, windows.mid)
            volume = volume_signal(frame)

            stamp = as_of if as_of is not None else frame.index[-1]
            if stamp not in frame.index:
                continue

            washout_value = float(washout.reindex(frame.index).fillna(0.0).loc[stamp])
            alarm_value = float(alarm.loc[stamp])
            confirmed_value = bool(confirmed.loc[stamp])

            rows.append(
                {
                    "ticker": ticker,
                    "cluster": cluster.label,
                    "alarm": alarm_value,
                    "washout": washout_value,
                    "stretch_z": float(stretch.loc[stamp]),
                    "rsi": float(oscillator.loc[stamp]),
                    "volume": float(volume.loc[stamp]),
                    "divergence": bool(divergence.loc[stamp]),
                    "confirmed": confirmed_value,
                    "fired": (
                        washout_value > WASHOUT_GATE
                        and alarm_value > ALARM_GATE
                        and confirmed_value
                    ),
                }
            )

    return pd.DataFrame(rows, columns=REBOUND_COLUMNS)
