"""Ground-truth bottom labels.

This is the ONLY module permitted to read data after the evaluation date.
Labels answer 'was this actually a bottom?', which is unknowable in real time
and is exactly why it belongs here and not in features/.
"""

from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def label_bottoms(
    ohlcv: pd.DataFrame, k: int, forward_days: int, min_return: float
) -> pd.Series:
    """True where the low is the minimum of a +/-k window AND the forward
    return over `forward_days` clears `min_return`.

    The second condition matters: a local minimum that goes nowhere is not a
    tradeable bottom, and labeling it as one would teach the evaluator that
    noise counts as success.
    """
    low = ohlcv["low"]
    close = ohlcv["close"]
    window = 2 * k + 1
    rolling_min = low.rolling(window, center=True, min_periods=window).min()
    is_local_min = low <= rolling_min

    forward = forward_return(close, forward_days)
    labels = is_local_min & (forward >= min_return)
    return labels.fillna(False).astype(bool)
