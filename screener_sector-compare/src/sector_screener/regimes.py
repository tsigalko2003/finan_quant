from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

HMM_COLUMNS = [
    "sector_return_5d",
    "sector_return_20d",
    "realized_vol_20d",
    "drawdown_60d",
    "breadth_positive_20d",
    "breadth_thrust_5d",
    "median_pairwise_correlation",
    "pc1_explained_variance",
    "pc1_score",
]


def causal_hmm_features(features: pd.DataFrame, params: dict, seed: int) -> pd.DataFrame:
    usable = features[HMM_COLUMNS].dropna()
    output = pd.DataFrame(
        index=features.index,
        columns=["hmm_state", "hmm_bear_probability", "hmm_bull_probability"],
        dtype=float,
    )
    min_train = int(params["hmm_min_train"])
    refit_every = int(params["hmm_refit_every"])
    states = int(params["hmm_states"])
    if len(usable) <= min_train:
        return output

    model: GaussianHMM | None = None
    scaler: StandardScaler | None = None
    bear_state = bull_state = 0
    for position in range(min_train, len(usable)):
        history = usable.iloc[:position]
        if model is None or (position - min_train) % refit_every == 0:
            scaler = StandardScaler().fit(history)
            model = GaussianHMM(
                n_components=states,
                covariance_type="diag",
                n_iter=150,
                min_covar=1e-4,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(scaler.transform(history))
            assigned = model.predict(scaler.transform(history))
            state_return = {
                state: history.loc[assigned == state, "sector_return_20d"].mean()
                for state in range(states)
            }
            bear_state = min(state_return, key=lambda key: np.nan_to_num(state_return[key], nan=0))
            bull_state = max(state_return, key=lambda key: np.nan_to_num(state_return[key], nan=0))
        assert model is not None and scaler is not None
        context = usable.iloc[max(0, position - 30) : position + 1]
        probabilities = model.predict_proba(scaler.transform(context))[-1]
        date = usable.index[position]
        output.loc[date, "hmm_state"] = int(np.argmax(probabilities))
        output.loc[date, "hmm_bear_probability"] = float(probabilities[bear_state])
        output.loc[date, "hmm_bull_probability"] = float(probabilities[bull_state])
    return output
