from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import MarketDataCache
from .config import Settings
from .features import build_sector_features, rank_tickers
from .modeling import fit_rebound_model
from .providers import YahooFinanceProvider
from .regimes import causal_hmm_features
from .report import render_report
from .universe import Universe, UniverseCatalog


def resolve_dates(
    settings: Settings, start: str | None, end: str | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    exclusive_end = (
        pd.Timestamp(end).normalize() if end else pd.Timestamp.now().normalize() + pd.offsets.Day(1)
    )
    begin = (
        pd.Timestamp(start).normalize()
        if start
        else exclusive_end - pd.offsets.Day(int(settings.data["lookback_days"]))
    )
    if begin >= exclusive_end:
        raise ValueError("start must be earlier than end (end is exclusive)")
    return begin, exclusive_end


def resolve_universe(
    settings: Settings,
    industry: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    qlib_data_dir: Path | None = None,
) -> Universe:
    catalog = UniverseCatalog(settings.config_dir / "industries.yaml")
    return catalog.resolve(
        industry,
        max_tickers=settings.universe.get("max_tickers"),
        include=include,
        exclude=exclude,
        qlib_data_dir=qlib_data_dir,
        nasdaq_cache_dir=settings.cache_dir,
    )


def make_cache(settings: Settings) -> MarketDataCache:
    provider_name = str(settings.data["provider"]).lower()
    if provider_name == "google":
        raise NotImplementedError(
            "Google Finance has no supported official bulk historical-price API. "
            "Use provider=yahoo or implement a licensed provider adapter."
        )
    if provider_name != "yahoo":
        raise ValueError(f"Unsupported provider: {provider_name}")
    return MarketDataCache(
        root=settings.cache_dir,
        provider=YahooFinanceProvider(),
        interval=str(settings.data["interval"]),
        auto_adjust=bool(settings.data["auto_adjust"]),
    )


def download_stage(
    settings: Settings,
    universe: Universe,
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh_tail_days: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    cache = make_cache(settings)
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}

    def fetch(ticker: str):
        return cache.fetch(ticker, start, end, refresh_tail_days=refresh_tail_days, force=force)

    with ThreadPoolExecutor(max_workers=int(settings.data["max_workers"])) as executor:
        futures = {executor.submit(fetch, ticker): ticker for ticker in universe.tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                results[ticker] = {
                    "rows": len(result.frame),
                    "cache_hit": result.cache_hit,
                    "remote_ranges": [
                        {"start": a.date().isoformat(), "end": b.date().isoformat()}
                        for a, b in result.requested_ranges
                    ],
                }
            except Exception as exc:  # noqa: BLE001 - retain all per-ticker diagnostics
                failures[ticker] = str(exc)
    minimum = max(3, int(np.ceil(len(universe.tickers) * float(settings.data["min_coverage"]))))
    if len(results) < minimum:
        raise RuntimeError(
            f"Download coverage gate failed: {len(results)}/{len(universe.tickers)} succeeded; "
            f"need {minimum}. Failures: {failures}"
        )
    summary = {
        "stage": settings.stage,
        "universe": asdict(universe),
        "start": start.date().isoformat(),
        "end_exclusive": end.date().isoformat(),
        "cache_hits": sum(int(item["cache_hit"]) for item in results.values()),
        "remote_fetches": sum(int(not item["cache_hit"]) for item in results.values()),
        "tickers": results,
        "failures": failures,
    }
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    (
        settings.cache_dir
        / f"last_download_{settings.stage}_{universe.name.replace(':', '_')}.json"
    ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_panels(
    settings: Settings, universe: Universe, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    cache = make_cache(settings)
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    rejected: dict[str, str] = {}
    for ticker in universe.tickers:
        try:
            frame = cache.load(ticker, start, end)
            if len(frame) < int(settings.data["min_rows"]):
                raise ValueError(f"only {len(frame)} rows; need {settings.data['min_rows']}")
            closes[ticker] = frame["close"]
            volumes[ticker] = frame["volume"]
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not hide others
            rejected[ticker] = str(exc)
    minimum = max(3, int(np.ceil(len(universe.tickers) * float(settings.data["min_coverage"]))))
    if len(closes) < minimum:
        raise RuntimeError(
            f"Analysis coverage gate failed: {len(closes)}/{len(universe.tickers)} usable; "
            f"need {minimum}. Run download or inspect: {rejected}"
        )
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(close.index)
    return close, volume, rejected


def analyze_stage(
    settings: Settings,
    universe: Universe,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    close, volume, rejected = _load_panels(settings, universe, start, end)
    features, sector_index = build_sector_features(close, volume, settings.analysis)
    hmm = causal_hmm_features(features, settings.analysis, settings.seed)
    features = features.join(hmm)
    model = fit_rebound_model(features, sector_index, settings.analysis, settings.seed)
    features["rebound_probability"] = model.probability
    usable = features.dropna(subset=["median_pairwise_correlation", "pc1_explained_variance"])
    if usable.empty:
        raise RuntimeError("No analyzable feature rows after correlation/PCA warm-up")
    latest = usable.iloc[-1]
    ranking = rank_tickers(
        close.loc[: latest.name], features, int(settings.analysis["capture_window"])
    )

    watch = bool(
        latest["drawdown_60d"] <= float(settings.analysis["correction_drawdown"])
        and latest["median_pairwise_correlation"] >= float(settings.analysis["minimum_correlation"])
    )
    triggered = bool(
        watch
        and latest.get("rebound_probability", 0) >= float(settings.analysis["rebound_probability"])
        and latest["breadth_thrust_5d"] >= float(settings.analysis["minimum_breadth_thrust"])
    )
    reason = (
        f"drawdown={latest['drawdown_60d']:.1%}, correlation={latest['median_pairwise_correlation']:.3f}, "
        f"breadth thrust={latest['breadth_thrust_5d']:.1%}, "
        f"rebound probability={latest.get('rebound_probability', float('nan')):.1%}"
    )
    alert = {
        "industry": universe.name,
        "stage": settings.stage,
        "data_through": latest.name.date().isoformat(),
        "watch": watch,
        "triggered": triggered,
        "reason": reason,
        "actionable": "next_session",
        "top_fall_resistant": ranking.nsmallest(3, "fall_resistance_rank").index.tolist(),
        "top_rise_strength": ranking.nsmallest(3, "rise_strength_rank").index.tolist(),
        "thresholds": {
            "correction_drawdown": settings.analysis["correction_drawdown"],
            "minimum_correlation": settings.analysis["minimum_correlation"],
            "rebound_probability": settings.analysis["rebound_probability"],
            "minimum_breadth_thrust": settings.analysis["minimum_breadth_thrust"],
        },
    }

    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        settings.output_dir / f"{settings.stage}_{universe.name.replace(':', '_')}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    features.to_csv(run_dir / "sector_features.csv", index_label="date")
    ranking.to_csv(run_dir / "ticker_rankings.csv", index_label="ticker")
    (run_dir / "validation.json").write_text(
        json.dumps(model.validation, indent=2), encoding="utf-8"
    )
    (run_dir / "alert.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")
    manifest = {
        "stage": settings.stage,
        "universe": asdict(universe),
        "requested_range": {"start": str(start.date()), "end_exclusive": str(end.date())},
        "actual_range": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
        },
        "rejected_tickers": rejected,
        "feature_importance": model.feature_importance,
        "alert": alert,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    render_report(
        run_dir / "report.html",
        universe.name,
        settings.stage,
        latest,
        ranking,
        model.validation,
        alert,
        len(universe.tickers),
        universe.source,
    )
    latest_pointer = {
        "run_dir": str(run_dir),
        "report": str(run_dir / "report.html"),
        "alert": alert,
    }
    (
        settings.output_dir / f"latest_{settings.stage}_{universe.name.replace(':', '_')}.json"
    ).write_text(json.dumps(latest_pointer, indent=2), encoding="utf-8")
    return latest_pointer
