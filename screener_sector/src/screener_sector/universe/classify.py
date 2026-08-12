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

import yaml


@dataclass(frozen=True)
class ThemeRules:
    industry_allow_list: frozenset[str]
    theme_keywords: dict[str, tuple[str, ...]]
    seed_etfs: tuple[str, ...]
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
