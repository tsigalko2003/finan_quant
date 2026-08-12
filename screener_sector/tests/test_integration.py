"""End-to-end guarantees: relocatability and no lookahead.

Per-module tests check individual functions. These check the properties that
only break when the pieces are wired together.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from screener_sector.config import Config
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.manifest import SCHEMA_VERSION, SchemaVersionError, load_manifest
from screener_sector.paths import Paths
from screener_sector.pipeline import run_screen, save_screen

CONFIG_DIR = Path("/app/config")


def build_frames(seed: int = 21) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 500
    idx = pd.bdate_range("2021-01-04", periods=n)
    driver = rng.normal(0.0004, 0.014, n)
    frames = {}
    for i in range(6):
        noise = rng.normal(0, 0.004, n)
        close = pd.Series(100.0 * np.exp(np.cumsum(driver + noise)), index=idx)
        frames[f"T{i}"] = make_ohlcv(close)
    frames["SOXX"] = make_ohlcv(
        pd.Series(100.0 * np.exp(np.cumsum(driver)), index=idx)
    )
    return frames


def populate(root: Path) -> tuple[Paths, PriceStore, list[str]]:
    paths = Paths.from_env({"DATA_DIR": str(root)})
    paths.ensure()
    frames = build_frames()
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2020, 1, 1))
    return paths, store, [t for t in frames if t != "SOXX"]


def test_relocated_data_dir_produces_identical_output(tmp_path):
    """Copy data/ to a different absolute path; results must be identical."""
    import shutil

    first_root = tmp_path / "original"
    paths_a, store_a, tickers = populate(first_root)
    cfg = Config.load(CONFIG_DIR, "dev")
    as_of = store_a.load("T0").index[-1].date()
    output_a = run_screen(store_a, tickers, cfg, as_of)

    second_root = tmp_path / "deeply" / "nested" / "elsewhere"
    second_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(first_root, second_root)

    paths_b = Paths.from_env({"DATA_DIR": str(second_root)})
    store_b = PriceStore(paths_b, FakeFetcher({}))
    output_b = run_screen(store_b, tickers, cfg, as_of)

    pd.testing.assert_frame_equal(output_a.trend, output_b.trend)
    pd.testing.assert_frame_equal(output_a.strength, output_b.strength)
    pd.testing.assert_frame_equal(output_a.rebound, output_b.rebound)


def test_no_artifact_contains_an_absolute_path(tmp_path):
    paths, store, tickers = populate(tmp_path / "root")
    cfg = Config.load(CONFIG_DIR, "dev")
    as_of = store.load("T0").index[-1].date()
    save_screen(paths, run_screen(store, tickers, cfg, as_of), "dev")

    for path in paths.root.rglob("*.csv"):
        assert str(paths.root) not in path.read_text()
    if paths.manifest_file.exists():
        assert str(paths.root) not in paths.manifest_file.read_text()


def test_future_data_cannot_influence_the_screen(tmp_path):
    """The lookahead test. Replacing every bar after as_of with garbage must
    not change a single computed value."""
    paths, store, tickers = populate(tmp_path / "root")
    cfg = Config.load(CONFIG_DIR, "dev")
    as_of = store.load("T0").index[300].date()
    before = run_screen(store, tickers, cfg, as_of)

    rng = np.random.default_rng(99)
    for ticker in tickers + ["SOXX"]:
        frame = store.load(ticker)
        mask = frame.index > pd.Timestamp(as_of)
        replacement = rng.uniform(1.0, 5000.0, int(mask.sum()))
        for column in ("open", "high", "low", "close"):
            frame.loc[mask, column] = replacement
        frame.loc[mask, "volume"] = rng.uniform(1, 1e9, int(mask.sum()))
        frame.to_parquet(paths.price_file(ticker))

    after = run_screen(store, tickers, cfg, as_of)
    pd.testing.assert_frame_equal(before.trend, after.trend)
    pd.testing.assert_frame_equal(before.strength, after.strength)
    pd.testing.assert_frame_equal(before.rebound, after.rebound)
    pd.testing.assert_frame_equal(before.clusters.raw_corr, after.clusters.raw_corr)


def test_incompatible_schema_version_is_refused(tmp_path):
    paths, _, _ = populate(tmp_path / "root")
    paths.manifest_file.write_text(
        f'{{"schema_version": {SCHEMA_VERSION + 99}, "stages": {{}}, "profiles": {{}}}}'
    )
    with pytest.raises(SchemaVersionError):
        load_manifest(paths)


def test_dev_cache_is_a_valid_subset_for_prod(tmp_path):
    """Switching profiles must not invalidate or truncate the cache."""
    paths, store, tickers = populate(tmp_path / "root")
    dev = Config.load(CONFIG_DIR, "dev")
    prod = Config.load(CONFIG_DIR, "prod")
    as_of = store.load("T0").index[-1].date()

    rows_before = len(store.load("T0"))
    run_screen(store, tickers, dev, as_of)
    run_screen(store, tickers, prod, as_of)
    assert len(store.load("T0")) == rows_before
