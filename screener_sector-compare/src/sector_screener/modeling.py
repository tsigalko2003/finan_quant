from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier

    XGB_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001 - native loader failures vary by platform
    XGBClassifier = None  # type: ignore[assignment]
    XGB_IMPORT_ERROR = str(exc)


@dataclass
class ReboundModelResult:
    probability: pd.Series
    validation: dict
    feature_importance: dict[str, float]


def make_rebound_labels(
    sector_index: pd.Series, features: pd.DataFrame, params: dict
) -> tuple[pd.Series, pd.Series]:
    horizon = int(params["rebound_horizon"])
    target = float(params["rebound_return"])
    adverse = float(params["max_adverse_return"])
    correction = float(params["correction_drawdown"])
    eligible = (features["drawdown_60d"] <= correction) | (features["hmm_bear_probability"] >= 0.60)
    labels = pd.Series(np.nan, index=features.index, dtype=float)
    for position in range(len(sector_index) - horizon):
        if not bool(eligible.iloc[position]):
            continue
        base = sector_index.iloc[position]
        path = sector_index.iloc[position + 1 : position + horizon + 1] / base - 1.0
        positive_hits = np.flatnonzero(path.to_numpy() >= target)
        adverse_hits = np.flatnonzero(path.to_numpy() <= adverse)
        first_positive = positive_hits[0] if positive_hits.size else math.inf
        first_adverse = adverse_hits[0] if adverse_hits.size else math.inf
        labels.iloc[position] = float(first_positive < first_adverse)
    return labels, eligible


def _classifier(params: dict, seed: int, positive_weight: float):
    if XGBClassifier is None:
        raise RuntimeError(f"XGBoost runtime unavailable: {XGB_IMPORT_ERROR}")
    return XGBClassifier(
        n_estimators=int(params["xgb_estimators"]),
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=max(1.0, positive_weight),
        random_state=seed,
        n_jobs=1,
    )


def fit_rebound_model(
    features: pd.DataFrame, sector_index: pd.Series, params: dict, seed: int
) -> ReboundModelResult:
    labels, eligible = make_rebound_labels(sector_index, features, params)
    numeric = features.select_dtypes(include=[np.number]).copy()
    dataset = numeric.loc[eligible & labels.notna()].replace([np.inf, -np.inf], np.nan).dropna()
    y = labels.loc[dataset.index].astype(int)
    probabilities = pd.Series(np.nan, index=features.index, dtype=float)
    validation: dict = {
        "method": "purged expanding walk-forward",
        "samples": len(dataset),
        "positive_events": int(y.sum()) if len(y) else 0,
        "status": "insufficient_data",
    }
    importance: dict[str, float] = {}
    horizon = int(params["rebound_horizon"])

    if XGBClassifier is None:
        heuristic = (
            -features["drawdown_60d"].fillna(0) * 4
            + features["breadth_thrust_5d"].fillna(0) * 2
            + features["hmm_bull_probability"].fillna(0)
        )
        probabilities.loc[:] = 1 / (1 + np.exp(-heuristic.clip(-10, 10)))
        validation["note"] = f"XGBoost runtime unavailable; heuristic shown: {XGB_IMPORT_ERROR}"
        return ReboundModelResult(probabilities, validation, importance)

    if len(dataset) < 60 or y.nunique() < 2:
        heuristic = (
            -features["drawdown_60d"].fillna(0) * 4
            + features["breadth_thrust_5d"].fillna(0) * 2
            + features["hmm_bull_probability"].fillna(0)
        )
        probabilities.loc[:] = 1 / (1 + np.exp(-heuristic.clip(-10, 10)))
        validation["note"] = "XGBoost not fit: too few eligible labeled events; heuristic shown"
        return ReboundModelResult(probabilities, validation, importance)

    negatives = max(1, int((y == 0).sum()))
    positives = max(1, int((y == 1).sum()))
    weight = negatives / positives
    requested_splits = int(params["validation_splits"])
    splits = min(requested_splits, max(2, len(dataset) // 30))
    splitter = TimeSeriesSplit(n_splits=splits, gap=horizon)
    out_of_sample = pd.Series(np.nan, index=dataset.index, dtype=float)
    for train_index, test_index in splitter.split(dataset):
        if y.iloc[train_index].nunique() < 2:
            continue
        model = _classifier(params, seed, weight)
        model.fit(dataset.iloc[train_index], y.iloc[train_index])
        out_of_sample.iloc[test_index] = model.predict_proba(dataset.iloc[test_index])[:, 1]

    scored = out_of_sample.dropna()
    probabilities.loc[scored.index] = scored
    if len(scored) and y.loc[scored.index].nunique() == 2:
        truth = y.loc[scored.index]
        threshold = float(params["rebound_probability"])
        predicted = (scored >= threshold).astype(int)
        validation.update(
            {
                "status": "ok",
                "oos_samples": len(scored),
                "roc_auc": float(roc_auc_score(truth, scored)),
                "pr_auc": float(average_precision_score(truth, scored)),
                "brier_score": float(brier_score_loss(truth, scored)),
                "precision_at_threshold": float(precision_score(truth, predicted, zero_division=0)),
                "recall_at_threshold": float(recall_score(truth, predicted, zero_division=0)),
                "threshold": threshold,
            }
        )

    final_model = _classifier(params, seed, weight)
    final_model.fit(dataset, y)
    current = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if not current.empty:
        current_date = current.index[-1]
        probabilities.loc[current_date] = float(
            final_model.predict_proba(current.loc[[current_date], dataset.columns])[:, 1][0]
        )
    importance = {
        column: float(value)
        for column, value in sorted(
            zip(dataset.columns, final_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    return ReboundModelResult(probabilities, validation, importance)
