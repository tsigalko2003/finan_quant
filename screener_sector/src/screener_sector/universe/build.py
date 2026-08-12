"""Universe assembly: theme scope plus liquidity filters.

Rejected tickers are retained with a reason rather than dropped, so the
question 'why isn't X in the screen?' always has an answer in the artifact.
"""

from __future__ import annotations

import pandas as pd

from screener_sector.config import UniverseFilters
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths
from screener_sector.universe.classify import ThemeRules, is_in_scope, match_themes

UNIVERSE_COLUMNS = [
    "ticker",
    "name",
    "industry",
    "themes",
    "exchange",
    "median_dollar_volume",
    "last_close",
    "history_days",
    "included",
    "reason",
]


def liquidity_stats(ohlcv: pd.DataFrame, window: int = 60) -> tuple[float, float, int]:
    if ohlcv.empty:
        return 0.0, 0.0, 0
    dollar_volume = (ohlcv["close"] * ohlcv["volume"]).tail(window)
    return (
        float(dollar_volume.median()),
        float(ohlcv["close"].iloc[-1]),
        int(len(ohlcv)),
    )


def build_universe(
    paths: Paths,
    symbols: pd.DataFrame,
    info: pd.DataFrame,
    store: PriceStore,
    rules: ThemeRules,
    filters: UniverseFilters,
) -> pd.DataFrame:
    merged = symbols.merge(info, on="ticker", how="left").fillna("")
    rows: list[dict[str, object]] = []

    for record in merged.to_dict("records"):
        ticker = str(record["ticker"])
        name = str(record.get("long_name") or record.get("name") or "")
        industry = str(record.get("industry") or "")
        summary = str(record.get("summary") or "")

        themes = match_themes(name, summary, rules)
        in_scope = is_in_scope(industry, name, summary, rules)

        if store.has(ticker):
            median_dv, last_close, history_days = liquidity_stats(store.load(ticker))
        else:
            median_dv, last_close, history_days = 0.0, 0.0, 0

        reasons: list[str] = []
        if not in_scope:
            reasons.append("off theme")
        if history_days < filters.min_history_days:
            reasons.append(f"insufficient history ({history_days}d)")
        if last_close < filters.min_price:
            reasons.append(f"price below floor ({last_close:.2f})")
        if median_dv < filters.min_dollar_volume:
            reasons.append(f"dollar_volume below floor ({median_dv:.0f})")

        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "industry": industry,
                "themes": "|".join(themes),
                "exchange": str(record.get("exchange") or ""),
                "median_dollar_volume": median_dv,
                "last_close": last_close,
                "history_days": history_days,
                "included": not reasons,
                "reason": "; ".join(reasons),
            }
        )

    return pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)


def save_universe(paths: Paths, df: pd.DataFrame) -> None:
    paths.ensure()
    df[UNIVERSE_COLUMNS].to_csv(paths.universe_csv, index=False)


def load_universe(paths: Paths, included_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(paths.universe_csv)
    df["themes"] = df["themes"].fillna("")
    df["reason"] = df["reason"].fillna("")
    if included_only:
        df = df[df["included"]].reset_index(drop=True)
    return df
