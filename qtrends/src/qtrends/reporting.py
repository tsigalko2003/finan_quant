from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


def evaluate_predictions(
    probability: pd.Series,
    target: pd.Series,
    future_return: pd.Series,
) -> dict[str, Any]:
    frame = pd.concat([probability, target.rename("target"), future_return.rename("future_return")], axis=1)
    frame = frame.dropna()
    metrics: dict[str, Any] = {"oos_observations": int(len(frame))}
    if frame.empty:
        return metrics
    metrics["accuracy_at_0_5"] = float(accuracy_score(frame["target"], frame[probability.name] >= 0.5))
    metrics["brier_score"] = float(brier_score_loss(frame["target"], frame[probability.name]))
    if frame["target"].nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(frame["target"], frame[probability.name]))
    bullish = frame[frame[probability.name] >= 0.60]
    bearish = frame[frame[probability.name] <= 0.40]
    metrics["bullish_observations"] = int(len(bullish))
    metrics["bearish_observations"] = int(len(bearish))
    metrics["mean_forward_return_when_bullish"] = _finite_or_none(bullish["future_return"].mean())
    metrics["mean_forward_return_when_bearish"] = _finite_or_none(bearish["future_return"].mean())
    metrics["bullish_hit_rate"] = _finite_or_none((bullish["future_return"] > 0).mean())
    metrics["bearish_hit_rate"] = _finite_or_none((bearish["future_return"] < 0).mean())
    return metrics


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    latest: dict[str, Any],
    metrics: dict[str, Any],
    tickers: list[str],
    benchmark: str,
) -> None:
    lines = [
        "# Qtrends run summary",
        "",
        f"- Group: {', '.join(tickers)}",
        f"- Benchmark: {benchmark}",
        f"- As of: {latest.get('date')}",
        f"- Signal: **{str(latest.get('signal', 'unknown')).upper()}**",
        f"- XGBoost probability: {latest.get('xgb_probability')}",
        f"- HMM regime: {latest.get('hmm_regime')}",
        f"- PCA explained variance: {latest.get('pca_explained_variance')}",
        f"- Breadth above MA20: {latest.get('breadth_above_ma20')}",
        "",
        "## Walk-forward evaluation",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in metrics.items())
    lines.extend(
        [
            "",
            "> Research output only. It is not an investment recommendation. Transaction costs,",
            "> liquidity, membership history, and execution assumptions require separate validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

