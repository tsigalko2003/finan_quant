from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MANIFEST = Path("data/universes/example_semiconductors.csv")
OUTPUT = Path("outputs/example_semiconductors")
EXPECTED_ARTIFACTS = {
    "feature_importance.csv",
    "hmm_model.joblib",
    "latest_signal.json",
    "metrics.json",
    "resolved_config.json",
    "signals.csv",
    "summary.md",
    "xgboost_model.joblib",
}


def main() -> None:
    missing = sorted(name for name in EXPECTED_ARTIFACTS if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing example artifacts: {missing}")

    manifest = pd.read_csv(MANIFEST, dtype={"symbol": str}).set_index("symbol")
    active = sorted(
        manifest[manifest["active"].astype(str).str.casefold().isin({"true", "1", "yes"})].index
    )
    if active != ["ALPHA", "BETA", "DELTA", "EPSILON", "GAMMA"]:
        raise RuntimeError(f"Unexpected active universe: {active}")
    if manifest.loc["EPSILON", "effective_from"] != "2021-07-01":
        raise RuntimeError("EPSILON was not assigned its incremental discovery date")
    if "EPSW" in manifest.index or "OTHER" in manifest.index:
        raise RuntimeError("Industry or security-name filtering did not work")

    config = json.loads((OUTPUT / "resolved_config.json").read_text(encoding="utf-8"))
    signal = json.loads((OUTPUT / "latest_signal.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    signals = pd.read_csv(OUTPUT / "signals.csv", parse_dates=["date"]).set_index("date")
    if sorted(config["data"]["tickers"]) != active:
        raise RuntimeError("Resolved pipeline tickers do not match the manifest")
    if metrics["group_size"] != len(active):
        raise RuntimeError("Reported group size does not match the manifest")
    before_addition = signals.loc[signals.index < "2021-07-01", "active_constituents"]
    after_addition = signals.loc[signals.index >= "2021-07-01", "active_constituents"]
    if before_addition.empty or before_addition.max() != 4:
        raise RuntimeError("Expected four active constituents before EPSILON's discovery")
    if after_addition.empty or after_addition.max() != 5:
        raise RuntimeError("Expected five active constituents after EPSILON's discovery")

    print("\nValidated end-to-end example")
    print(f"  Active universe: {', '.join(active)}")
    print("  Incremental addition: EPSILON (effective 2021-07-01)")
    print("  Membership masking: 4 constituents before discovery, 5 afterward")
    print(f"  Latest observation: {signal['date']}")
    print(f"  Composite signal: {signal['signal']}")
    print(f"  XGBoost probability: {signal['xgb_probability']:.4f}")
    print(f"  HMM regime: {signal['hmm_regime']}")
    print(f"  Breadth above MA20: {signal['breadth_above_ma20']:.4f}")
    print(f"  Walk-forward observations: {metrics['oos_observations']}")
    print(f"  Walk-forward ROC AUC: {metrics.get('roc_auc')}")
    print(f"  First-read result: {OUTPUT / 'summary.md'}")
    print("  Research demonstration only; not an investment recommendation.")


if __name__ == "__main__":
    main()
