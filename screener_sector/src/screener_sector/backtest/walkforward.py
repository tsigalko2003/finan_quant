"""Expanding-window walk-forward splits.

Parameters are always fit on data strictly before the period they are tested
on. Fold-to-fold parameter stability is a first-class diagnostic: thresholds
that thrash between folds indicate overfitting more reliably than any single
aggregate score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fold:
    index: int
    fit_start: date
    fit_end: date
    test_start: date
    test_end: date
    partial: bool


def expanding_folds(
    start: date, end: date, initial_fit_years: int, step_years: int
) -> tuple[Fold, ...]:
    folds: list[Fold] = []
    first_test_year = start.year + initial_fit_years
    index = 0

    for test_year in range(first_test_year, end.year + 1, step_years):
        test_start = date(test_year, 1, 1)
        if test_start > end:
            break
        natural_end = date(test_year + step_years - 1, 12, 31)
        test_end = min(natural_end, end)
        folds.append(
            Fold(
                index=index,
                fit_start=start,
                fit_end=date(test_year - 1, 12, 31),
                test_start=test_start,
                test_end=test_end,
                partial=test_end < natural_end,
            )
        )
        index += 1

    return tuple(folds)
