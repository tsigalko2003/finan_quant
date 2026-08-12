from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import exponential_trend, make_ohlcv
from screener_sector.cli import app, resolve_tickers
from screener_sector.config import Config
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths

CONFIG_DIR = Path("/app/config")
runner = CliRunner()


def test_resolve_tickers_uses_static_list_in_dev(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    cfg = Config.load(CONFIG_DIR, "dev")
    tickers = resolve_tickers(paths, cfg)
    assert "NVDA" in tickers
    assert len(tickers) == len(cfg.static_tickers)


def test_resolve_tickers_reads_universe_in_prod(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    pd.DataFrame(
        {
            "ticker": ["NVDA", "REJECT"],
            "name": ["NVIDIA", "Reject Co"],
            "industry": ["Semiconductors"] * 2,
            "themes": ["semiconductor"] * 2,
            "exchange": ["NASDAQ"] * 2,
            "median_dollar_volume": [1e9, 1.0],
            "last_close": [100.0, 1.0],
            "history_days": [500, 500],
            "included": [True, False],
            "reason": ["", "price below floor"],
        }
    ).to_csv(paths.universe_csv, index=False)
    cfg = Config.load(CONFIG_DIR, "prod")
    assert resolve_tickers(paths, cfg) == ["NVDA"]


def test_info_command_reports_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["info", "--profile", "dev"])
    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout


def test_build_universe_is_skipped_in_static_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["build-universe", "--profile", "dev"])
    assert result.exit_code == 0
    assert "static" in result.stdout.lower()


def test_screen_command_writes_derived_output(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    cfg = Config.load(CONFIG_DIR, "dev")

    frames = {
        ticker: make_ohlcv(exponential_trend(400, 0.001, noise=0.02, seed=i))
        for i, ticker in enumerate(list(cfg.static_tickers))
    }
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2020, 1, 1))

    as_of = list(frames.values())[0].index[-1].date().isoformat()
    result = runner.invoke(app, ["screen", "--profile", "dev", "--as-of", as_of])
    assert result.exit_code == 0, result.stdout
    assert (paths.derived_dir("dev") / as_of / "trend.csv").exists()


def test_report_command_writes_html(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    cfg = Config.load(CONFIG_DIR, "dev")
    frames = {
        ticker: make_ohlcv(exponential_trend(400, 0.001, noise=0.02, seed=i))
        for i, ticker in enumerate(list(cfg.static_tickers))
    }
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2020, 1, 1))

    as_of = list(frames.values())[0].index[-1].date().isoformat()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["report", "--profile", "dev", "--as-of", as_of, "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "dev" / f"{as_of}.html").exists()


def test_unknown_profile_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["info", "--profile", "bogus"])
    assert result.exit_code != 0


def test_readonly_commands_never_touch_the_network(tmp_path, monkeypatch):
    """screen/backtest/report must read the cache only. Only fetch and
    build-universe may reach out. A regression here would silently re-download
    data the user already has on disk."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    cfg = Config.load(CONFIG_DIR, "dev")

    frames = {
        ticker: make_ohlcv(exponential_trend(400, 0.001, noise=0.02, seed=i))
        for i, ticker in enumerate(list(cfg.static_tickers))
    }
    PriceStore(paths, FakeFetcher(frames)).refresh(list(frames), date(2020, 1, 1))

    calls: list[str] = []

    class ExplodingFetcher:
        def history(self, ticker, start, end):
            calls.append(ticker)
            raise AssertionError(f"network access attempted for {ticker}")

    monkeypatch.setattr(
        "screener_sector.cli.YFinanceFetcher", lambda *a, **k: ExplodingFetcher()
    )

    as_of = list(frames.values())[0].index[-1].date().isoformat()
    out_dir = tmp_path / "out"

    invocations = [
        ["screen", "--profile", "dev", "--as-of", as_of],
        ["backtest", "--profile", "dev", "--as-of", as_of, "--out", str(out_dir)],
        ["report", "--profile", "dev", "--as-of", as_of, "--out", str(out_dir)],
    ]
    for argv in invocations:
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"{argv[0]} failed: {result.stdout}"
        assert calls == [], f"{argv[0]} attempted network access for {calls}"
