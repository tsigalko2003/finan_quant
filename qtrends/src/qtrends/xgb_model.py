from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from qtrends.config import XGBoostConfig


MODEL_FEATURES = [
    "relative_return_5",
    "relative_return_20",
    "relative_return_60",
    "breadth_positive_1d",
    "breadth_above_ma20",
    "breadth_above_ma50",
    "breadth_new_high_5",
    "advancing_volume_breadth",
    "median_volume_ratio",
    "cross_sectional_dispersion",
    "realized_volatility",
    "average_pairwise_correlation",
    "pca_factor",
    "pca_explained_variance",
    "hmm_bear_probability",
    "hmm_neutral_probability",
    "hmm_bull_probability",
]


@dataclass
class XGBoostResult:
    oos_probability: pd.Series
    live_probability: pd.Series
    target: pd.Series
    future_return: pd.Series
    model: XGBClassifier
    feature_importance: pd.Series


def _new_model(config: XGBoostConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=1,
        tree_method="hist",
        reg_lambda=1.0,
        reg_alpha=0.05,
    )


def walk_forward_xgboost(
    features: pd.DataFrame,
    future_return: pd.Series,
    config: XGBoostConfig,
) -> XGBoostResult:
    missing = [column for column in MODEL_FEATURES if column not in features]
    if missing:
        raise ValueError(f"Missing XGBoost features: {missing}")

    matrix = features[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    target = (future_return > 0.0).astype(float).where(future_return.notna())
    complete_features = matrix.dropna()
    labeled_index = complete_features.index.intersection(target.dropna().index)
    labeled_x = complete_features.loc[labeled_index]
    labeled_y = target.loc[labeled_index].astype(int)

    initial_prediction = config.min_train_size + config.horizon
    if len(labeled_x) <= initial_prediction:
        raise ValueError(
            f"Need more than {initial_prediction} labeled feature rows for XGBoost; got {len(labeled_x)}"
        )

    oos = pd.Series(np.nan, index=features.index, name="xgb_probability_oos")
    for start in range(initial_prediction, len(labeled_x), config.refit_every):
        end = min(start + config.refit_every, len(labeled_x))
        train_end = start - config.horizon
        train_x = labeled_x.iloc[:train_end]
        train_y = labeled_y.iloc[:train_end]
        if train_y.nunique() < 2:
            continue
        model = _new_model(config)
        model.fit(train_x, train_y)
        test_x = labeled_x.iloc[start:end]
        oos.loc[test_x.index] = model.predict_proba(test_x)[:, 1]

    final_model = _new_model(config)
    final_model.fit(labeled_x, labeled_y)
    live = pd.Series(np.nan, index=features.index, name="xgb_probability")
    live.loc[complete_features.index] = final_model.predict_proba(complete_features)[:, 1]
    importance = pd.Series(
        final_model.feature_importances_, index=MODEL_FEATURES, name="importance"
    ).sort_values(ascending=False)
    return XGBoostResult(oos, live, target, future_return, final_model, importance)

