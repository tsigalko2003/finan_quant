from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from qtrends.config import PipelineConfig
from qtrends.data import load_market_data
from qtrends.features import build_features, forward_compound_return
from qtrends.hmm_model import walk_forward_hmm
from qtrends.reporting import evaluate_predictions, write_json, write_summary
from qtrends.xgb_model import walk_forward_xgboost


@dataclass
class RunArtifacts:
    output_dir: Path
    latest_signal: dict[str, Any]
    metrics: dict[str, Any]


def _assign_signals(frame: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    ma_column = f"breadth_above_ma{config.features.breadth_ma_windows[0]}"
    trend_column = f"relative_return_{config.features.trend_lookbacks[1]}"
    bullish = (
        (frame["xgb_probability"] >= config.signal.bullish_probability)
        & (frame["hmm_bull_probability"] >= config.signal.hmm_probability)
        & (frame[ma_column] >= config.signal.bullish_breadth)
        & (frame[trend_column] > 0.0)
    )
    bearish = (
        (frame["xgb_probability"] <= config.signal.bearish_probability)
        & (frame["hmm_bear_probability"] >= config.signal.hmm_probability)
        & (frame[ma_column] <= config.signal.bearish_breadth)
        & (frame[trend_column] < 0.0)
    )
    result = pd.Series("neutral", index=frame.index, name="signal")
    result.loc[bullish] = "bullish"
    result.loc[bearish] = "bearish"
    result.loc[frame["xgb_probability"].isna()] = "unavailable"
    return result


def _latest_payload(frame: pd.DataFrame) -> dict[str, Any]:
    available = frame[frame["signal"] != "unavailable"]
    if available.empty:
        raise RuntimeError("No live signal was produced")
    date = available.index[-1]
    row = available.loc[date]
    keys = [
        "signal",
        "xgb_probability",
        "hmm_regime",
        "hmm_bear_probability",
        "hmm_neutral_probability",
        "hmm_bull_probability",
        "pca_factor",
        "pca_explained_variance",
        "breadth_positive_1d",
        "breadth_above_ma20",
        "breadth_above_ma50",
        "relative_return_20",
        "realized_volatility",
        "average_pairwise_correlation",
        "constituent_coverage",
    ]
    payload: dict[str, Any] = {"date": date.strftime("%Y-%m-%d")}
    for key in keys:
        value = row.get(key)
        if isinstance(value, (np.floating, float)):
            payload[key] = float(value) if np.isfinite(value) else None
        else:
            payload[key] = value
    return payload


def run_pipeline(config: PipelineConfig, output_dir: str | Path) -> RunArtifacts:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    market = load_market_data(config.data)
    features, _ = build_features(
        market,
        tickers=config.data.tickers,
        benchmark=config.data.benchmark,
        config=config.features,
    )
    future_return = forward_compound_return(
        features["group_excess_return"], config.xgboost.horizon
    ).rename("future_excess_return")

    hmm = walk_forward_hmm(features, config.hmm)
    combined = features.join(hmm.probabilities)
    xgb = walk_forward_xgboost(combined, future_return, config.xgboost)
    combined = combined.join(xgb.oos_probability).join(xgb.live_probability).join(future_return)
    combined["signal"] = _assign_signals(combined, config)

    metrics = evaluate_predictions(xgb.oos_probability, xgb.target, xgb.future_return)
    metrics.update(
        {
            "start_date": str(combined.index.min().date()),
            "end_date": str(combined.index.max().date()),
            "price_observations": int(len(combined)),
            "group_size": len(config.data.tickers),
            "forecast_horizon_days": config.xgboost.horizon,
        }
    )
    latest = _latest_payload(combined)

    combined.to_csv(destination / "signals.csv", index_label="date")
    xgb.feature_importance.to_csv(destination / "feature_importance.csv", header=True)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "latest_signal.json", latest)
    write_json(destination / "resolved_config.json", config.model_dump(mode="json"))
    write_summary(
        destination / "summary.md",
        latest,
        metrics,
        config.data.tickers,
        config.data.benchmark,
    )
    joblib.dump(xgb.model, destination / "xgboost_model.joblib")
    joblib.dump(
        {
            "model": hmm.last_model,
            "scaler": hmm.last_scaler,
            "state_labels": hmm.state_labels,
        },
        destination / "hmm_model.joblib",
    )
    return RunArtifacts(destination, latest, metrics)
