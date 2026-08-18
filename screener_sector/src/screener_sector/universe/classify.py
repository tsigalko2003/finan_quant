"""Theme classification from company name and business summary.

A ticker enters the universe if its Yahoo industry is on the allow-list OR its
name/summary matches a theme keyword. The industry check alone misses optical
and AI-adjacent names; the keyword check alone pulls in false positives from
marketing language. Requiring either, not both, is the deliberate trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class ThemeRules:
    industry_allow_list: frozenset[str]
    theme_keywords: dict[str, tuple[str, ...]]
    seed_etfs: tuple[str, ...]
    seed_tickers: frozenset[str]
    exchanges: frozenset[str]

    @classmethod
    def load(cls, config_dir: Path) -> ThemeRules:
        raw = yaml.safe_load((config_dir / "universe.yaml").read_text())
        return cls(
            industry_allow_list=frozenset(raw["industry_allow_list"]),
            theme_keywords={
                theme: tuple(words) for theme, words in raw["theme_keywords"].items()
            },
            seed_etfs=tuple(raw["seed_etfs"]),
            seed_tickers=frozenset(str(t) for t in raw.get("seed_tickers", [])),
            exchanges=frozenset(raw["exchanges"]),
        )


def _pattern(keyword: str) -> re.Pattern[str]:
    # Word-boundary at both ends, with an optional trailing plural 's'.
    return re.compile(rf"\b{re.escape(keyword)}s?\b", re.IGNORECASE)


def match_themes(name: str, summary: str, rules: ThemeRules) -> tuple[str, ...]:
    haystack = f"{name or ''} {summary or ''}"
    matched = [
        theme
        for theme, keywords in rules.theme_keywords.items()
        if any(_pattern(word).search(haystack) for word in keywords)
    ]
    return tuple(matched)


def is_in_scope(industry: str, name: str, summary: str, rules: ThemeRules) -> bool:
    if industry in rules.industry_allow_list:
        return True
    return bool(match_themes(name, summary, rules))


def enrichment_candidates(symbols: pd.DataFrame, rules: ThemeRules) -> list[str]:
    """Symbols worth spending a Yahoo profile request on.

    Enriching all ~8000 US-listed symbols triggers rate limiting long before it
    finishes. Security names come free with the NASDAQ Trader files, and most
    in-scope companies name themselves ("... Semiconductor", "... Photonics").
    The seed lists cover the ones that do not: Coherent, Lumentum, Credo and
    similar reveal nothing in their title.

    Note this matches on NAME ONLY. The business summary does not exist yet at
    this stage - obtaining it is precisely what the enrichment pass is for.
    """
    if symbols.empty:
        return []

    seeds = set(rules.seed_tickers) | set(rules.seed_etfs)
    keep: set[str] = set()

    for ticker, name in zip(symbols["ticker"], symbols["name"], strict=False):
        symbol = str(ticker)
        if symbol in seeds or match_themes(str(name), "", rules):
            keep.add(symbol)

    return sorted(keep)
