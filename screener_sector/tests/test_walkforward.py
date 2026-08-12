from datetime import date

import pytest

from screener_sector.backtest.walkforward import expanding_folds


def test_prod_shape_produces_expected_fold_count():
    folds = expanding_folds(
        date(2010, 1, 1), date(2026, 8, 12), initial_fit_years=5, step_years=1
    )
    assert len(folds) == 12
    assert folds[0].test_start == date(2015, 1, 1)
    assert folds[-1].test_start == date(2026, 1, 1)


def test_fit_window_is_expanding_not_rolling():
    folds = expanding_folds(
        date(2010, 1, 1), date(2020, 12, 31), initial_fit_years=5, step_years=1
    )
    assert all(f.fit_start == date(2010, 1, 1) for f in folds)
    assert folds[1].fit_end > folds[0].fit_end


def test_fit_window_never_overlaps_its_test_window():
    folds = expanding_folds(
        date(2010, 1, 1), date(2026, 8, 12), initial_fit_years=5, step_years=1
    )
    assert all(f.fit_end < f.test_start for f in folds)


def test_final_partial_year_is_flagged():
    folds = expanding_folds(
        date(2010, 1, 1), date(2026, 8, 12), initial_fit_years=5, step_years=1
    )
    assert folds[-1].partial is True
    assert all(not f.partial for f in folds[:-1])


def test_complete_final_year_is_not_flagged_partial():
    folds = expanding_folds(
        date(2010, 1, 1), date(2025, 12, 31), initial_fit_years=5, step_years=1
    )
    assert all(not f.partial for f in folds)


def test_dev_shape_produces_two_folds():
    folds = expanding_folds(
        date(2022, 1, 1), date(2026, 8, 12), initial_fit_years=2, step_years=1
    )
    assert [f.test_start.year for f in folds] == [2024, 2025, 2026]


def test_folds_are_indexed_sequentially():
    folds = expanding_folds(
        date(2010, 1, 1), date(2020, 12, 31), initial_fit_years=5, step_years=1
    )
    assert [f.index for f in folds] == list(range(len(folds)))


def test_insufficient_span_yields_no_folds():
    assert expanding_folds(
        date(2024, 1, 1), date(2024, 12, 31), initial_fit_years=5, step_years=1
    ) == ()
