from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from qtrends.config import HMMConfig


HMM_FEATURES = [
    "group_excess_return",
    "realized_volatility",
    "breadth_positive_1d",
    "breadth_above_ma20",
    "pca_factor",
    "pca_explained_variance",
]


@dataclass
class HMMResult:
    probabilities: pd.DataFrame
    last_model: GaussianHMM
    last_scaler: StandardScaler
    state_labels: dict[int, str]


def _fit_best_hmm(values: np.ndarray, config: HMMConfig) -> GaussianHMM:
    best_model: GaussianHMM | None = None
    best_score = -np.inf
    for seed in config.seeds:
        model = GaussianHMM(
            n_components=config.states,
            covariance_type="full",
            n_iter=config.n_iter,
            tol=1e-4,
            random_state=seed,
            min_covar=1e-5,
        )
        hmm_logger = logging.getLogger("hmmlearn.base")
        previous_level = hmm_logger.level
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                hmm_logger.setLevel(logging.ERROR)
                model.fit(values)
                score = float(model.score(values))
            except (ValueError, FloatingPointError):
                continue
            finally:
                hmm_logger.setLevel(previous_level)
        if np.isfinite(score) and score > best_score:
            best_model, best_score = model, score
    if best_model is None:
        raise RuntimeError("All HMM fits failed")
    return best_model


def _emission_log_probability(model: GaussianHMM, observation: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            multivariate_normal.logpdf(
                observation,
                mean=model.means_[state],
                cov=model.covars_[state],
                allow_singular=True,
            )
            for state in range(model.n_components)
        ],
        dtype=float,
    )


def _filter_sequence(
    model: GaussianHMM,
    values: np.ndarray,
    prior_log_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    log_alpha = prior_log_alpha
    log_transition = np.log(np.clip(model.transmat_, 1e-300, None))
    log_start = np.log(np.clip(model.startprob_, 1e-300, None))
    for observation in values:
        emission = _emission_log_probability(model, observation)
        if log_alpha is None:
            current = log_start + emission
        else:
            current = logsumexp(log_alpha[:, None] + log_transition, axis=0) + emission
        current -= logsumexp(current)
        log_alpha = current
        rows.append(np.exp(current))
    return np.asarray(rows), np.asarray(log_alpha)


def _label_states(
    model: GaussianHMM,
) -> dict[int, str]:
    # The first standardized HMM input is group excess return. Sorting that
    # state's emission mean is deterministic and also labels rarely visited states.
    ordered = list(np.argsort(model.means_[:, 0]))
    labels: dict[int, str] = {}
    labels[int(ordered[0])] = "bear"
    labels[int(ordered[-1])] = "bull"
    for state in ordered[1:-1]:
        labels[int(state)] = "neutral"
    return labels


def walk_forward_hmm(features: pd.DataFrame, config: HMMConfig) -> HMMResult:
    missing = [column for column in HMM_FEATURES if column not in features]
    if missing:
        raise ValueError(f"Missing HMM features: {missing}")
    usable = features[HMM_FEATURES].dropna()
    if len(usable) <= config.min_train_size:
        raise ValueError(
            f"Need more than {config.min_train_size} complete feature rows for HMM; got {len(usable)}"
        )

    output = pd.DataFrame(
        index=features.index,
        columns=["hmm_bear_probability", "hmm_neutral_probability", "hmm_bull_probability"],
        dtype=float,
    )
    last_model: GaussianHMM | None = None
    last_scaler: StandardScaler | None = None
    last_labels: dict[int, str] = {}

    for start in range(config.min_train_size, len(usable), config.refit_every):
        end = min(start + config.refit_every, len(usable))
        train = usable.iloc[:start]
        test = usable.iloc[start:end]
        scaler = StandardScaler().fit(train)
        scaled_train = scaler.transform(train)
        model = _fit_best_hmm(scaled_train, config)
        labels = _label_states(model)

        _, prior = _filter_sequence(model, scaled_train)
        filtered, _ = _filter_sequence(model, scaler.transform(test), prior)
        for row_position, date in enumerate(test.index):
            for state in range(config.states):
                label = labels.get(state, "neutral")
                column = f"hmm_{label}_probability"
                current = output.loc[date, column]
                output.loc[date, column] = (
                    0.0 if pd.isna(current) else float(current)
                ) + filtered[row_position, state]

        last_model, last_scaler, last_labels = model, scaler, labels

    if last_model is None or last_scaler is None:
        raise RuntimeError("HMM walk-forward loop produced no model")
    probability_columns = [
        "hmm_bear_probability",
        "hmm_neutral_probability",
        "hmm_bull_probability",
    ]
    output["hmm_regime"] = pd.Series(index=output.index, dtype="object")
    valid = output[probability_columns].notna().any(axis=1)
    regimes = (
        output.loc[valid, probability_columns]
        .idxmax(axis=1)
        .str.removeprefix("hmm_")
        .str.removesuffix("_probability")
    )
    output.loc[valid, "hmm_regime"] = regimes
    return HMMResult(output, last_model, last_scaler, last_labels)
