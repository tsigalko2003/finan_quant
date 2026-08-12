from pathlib import Path

import pandas as pd
import pytest

from qtrends.config import PipelineConfig
from qtrends.universe import active_symbols_from_manifest, sync_universe


def _write_export(path: Path, symbols: list[tuple[str, str, str]]) -> None:
    rows = []
    for symbol, name, ipo_year in symbols:
        rows.append(
            {
                "Symbol": symbol,
                "Name": name,
                "Market Cap": "10000000000",
                "Country": "United States",
                "IPO Year": ipo_year,
                "Sector": "Technology",
                "Industry": "Semiconductors",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _config(export: Path, manifest: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "data": {
                "provider": "yahoo",
                "tickers": [],
                "benchmark": "SOXX",
                "start": "2018-01-01",
            },
            "universe": {
                "source": "nasdaq_screener",
                "industry": "Semiconductors",
                "sector": "Technology",
                "export_path": str(export),
                "manifest_path": str(manifest),
                "min_market_cap": 1000000000,
                "min_snapshot_retention": 0.50,
            },
        }
    )


def test_incremental_universe_manifest_preserves_membership_history(tmp_path: Path) -> None:
    export = tmp_path / "nasdaq.csv"
    manifest = tmp_path / "manifest.csv"
    _write_export(export, [("AAA", "Alpha Common Stock", "2020"), ("BBB", "Beta Common Stock", "2010")])
    config = _config(export, manifest)

    first = sync_universe(config, snapshot_date="2024-01-02")
    assert first.added_symbols == ["AAA", "BBB"]
    initial = pd.read_csv(manifest, dtype={"symbol": str}).set_index("symbol")
    assert initial.loc["AAA", "effective_from"] == "2020-01-01"
    assert initial.loc["BBB", "effective_from"] == "2018-01-01"

    _write_export(export, [("AAA", "Alpha Common Stock", "2020"), ("CCC", "Gamma Common Stock", "2024")])
    second = sync_universe(config, snapshot_date="2024-02-01")
    assert second.added_symbols == ["CCC"]
    assert second.deactivated_symbols == ["BBB"]
    assert active_symbols_from_manifest(manifest) == ["AAA", "CCC"]

    updated = pd.read_csv(manifest, dtype={"symbol": str}).set_index("symbol")
    assert updated.loc["AAA", "effective_from"] == "2020-01-01"
    assert updated.loc["CCC", "effective_from"] == "2024-02-01"
    assert str(updated.loc["BBB", "active"]).lower() == "false"


def test_security_name_exclusions_apply_to_export(tmp_path: Path) -> None:
    export = tmp_path / "nasdaq.csv"
    manifest = tmp_path / "manifest.csv"
    _write_export(
        export,
        [
            ("AAA", "Alpha Common Stock", "2020"),
            ("AAAW", "Alpha Warrant", "2020"),
        ],
    )
    config = _config(export, manifest)
    config.universe.min_market_cap = 0
    result = sync_universe(config, snapshot_date="2024-01-02")
    assert result.active_symbols == ["AAA"]


def test_incomplete_snapshot_does_not_modify_manifest(tmp_path: Path) -> None:
    export = tmp_path / "nasdaq.csv"
    manifest = tmp_path / "manifest.csv"
    initial_symbols = [
        ("AAA", "Alpha Common Stock", "2020"),
        ("BBB", "Beta Common Stock", "2020"),
        ("CCC", "Gamma Common Stock", "2020"),
        ("DDD", "Delta Common Stock", "2020"),
    ]
    _write_export(export, initial_symbols)
    config = _config(export, manifest)
    config.universe.min_snapshot_retention = 0.80
    sync_universe(config, snapshot_date="2024-01-02")
    before = manifest.read_text()

    _write_export(export, [initial_symbols[0]])
    with pytest.raises(RuntimeError, match="manifest was not modified"):
        sync_universe(config, snapshot_date="2024-02-01")
    assert manifest.read_text() == before
