"""Command-line entrypoint.

Every command takes --profile and --as-of, so the same code path serves both a
historical backtest and 'what does the screen say today'.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import typer

from screener_sector.backtest.runner import run_backtest
from screener_sector.config import Config
from screener_sector.data.fetcher import YFinanceFetcher
from screener_sector.data.store import PriceStore
from screener_sector.manifest import load_manifest, record_stage
from screener_sector.paths import Paths
from screener_sector.pipeline import run_screen, save_screen
from screener_sector.report.render import render_report
from screener_sector.universe.build import (
    build_universe,
    load_universe,
    save_universe,
)
from screener_sector.universe.classify import ThemeRules
from screener_sector.universe.enrich import YFinanceInfoSource, enrich
from screener_sector.universe.symbols import (
    HttpTextSource,
    fetch_symbols,
    save_symbols,
)

app = typer.Typer(add_completion=False, help="Semiconductor sector screener.")

ProfileOption = typer.Option(None, "--profile", help="dev or prod")
AsOfOption = typer.Option(None, "--as-of", help="YYYY-MM-DD, defaults to today")
ConfigOption = typer.Option("/app/config", "--config-dir")
OutOption = typer.Option("/app/out", "--out")


def _resolve(profile: str | None, config_dir: str) -> tuple[Paths, Config]:
    name = profile or os.environ.get("PROFILE") or "dev"
    try:
        config = Config.load(Path(config_dir), name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return Paths.from_env(), config


def _as_of(value: str | None) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else date.today()


def resolve_tickers(paths: Paths, config: Config) -> list[str]:
    if config.universe_mode == "static":
        return list(config.static_tickers)
    return list(load_universe(paths, included_only=True)["ticker"])


@app.command()
def info(profile: str = ProfileOption, config_dir: str = ConfigOption) -> None:
    """Print where the data lives and what has been computed."""
    paths, config = _resolve(profile, config_dir)
    manifest = load_manifest(paths)
    typer.echo(f"DATA_DIR:       {paths.root}")
    typer.echo(f"profile:        {config.profile}")
    typer.echo(f"schema_version: {manifest.schema_version}")
    typer.echo(f"range:          {config.start} .. {config.end or 'today'}")
    typer.echo(f"universe_mode:  {config.universe_mode}")
    for stage, when in sorted(manifest.stages.items()):
        typer.echo(f"  stage {stage}: {when}")
    if paths.universe_csv.exists():
        typer.echo(f"universe rows:  {len(load_universe(paths, False))}")


@app.command("build-universe")
def build_universe_command(
    profile: str = ProfileOption, config_dir: str = ConfigOption
) -> None:
    """Discover the themed universe. No-op for static profiles."""
    paths, config = _resolve(profile, config_dir)
    if config.universe_mode == "static":
        typer.echo(
            f"profile {config.profile} uses a static ticker list "
            f"({len(config.static_tickers)} names); discovery skipped."
        )
        return

    rules = ThemeRules.load(Path(config_dir))
    symbols = fetch_symbols(HttpTextSource())
    save_symbols(paths, symbols)
    typer.echo(f"symbols: {len(symbols)}")

    def progress_callback(completed: int, total: int) -> None:
        typer.echo(f"enriched {completed}/{total}")

    info_frame = enrich(
        paths,
        list(symbols["ticker"]),
        YFinanceInfoSource(
            pause=config.network.enrich_pause_seconds,
            rate_limit_backoff_seconds=config.network.rate_limit_backoff_seconds,
        ),
        now=datetime.now().isoformat(timespec="seconds"),
        on_progress=progress_callback,
    )
    typer.echo(f"enriched: {len(info_frame)}")

    store = PriceStore(
        paths,
        YFinanceFetcher(
            rate_limit_backoff_seconds=config.network.rate_limit_backoff_seconds
        ),
    )
    candidates = [
        row["ticker"]
        for row in info_frame.to_dict("records")
        if _candidate(row, rules)
    ]
    store.refresh(candidates, config.fetch_start, config.end)

    universe = build_universe(
        paths, symbols, info_frame, store, rules, config.filters
    )
    save_universe(paths, universe)
    record_stage(paths, "universe", datetime.now().isoformat(timespec="seconds"))
    typer.echo(f"universe: {int(universe['included'].sum())} included "
               f"of {len(universe)} evaluated")


def _candidate(row: dict, rules: ThemeRules) -> bool:
    from screener_sector.universe.classify import is_in_scope

    return is_in_scope(
        str(row.get("industry") or ""),
        str(row.get("long_name") or ""),
        str(row.get("summary") or ""),
        rules,
    )


@app.command()
def fetch(profile: str = ProfileOption, config_dir: str = ConfigOption) -> None:
    """Refresh the price cache for the resolved universe."""
    paths, config = _resolve(profile, config_dir)
    store = PriceStore(
        paths,
        YFinanceFetcher(
            rate_limit_backoff_seconds=config.network.rate_limit_backoff_seconds
        ),
    )
    tickers = list(dict.fromkeys(resolve_tickers(paths, config) + [config.benchmark]))
    result = store.refresh(tickers, config.fetch_start, config.end)
    record_stage(paths, "fetch", datetime.now().isoformat(timespec="seconds"))
    typer.echo(
        f"fetched {len(result.fetched)}, skipped {len(result.skipped)}, "
        f"failed {len(result.failed)}"
    )


@app.command()
def screen(
    profile: str = ProfileOption,
    as_of: str = AsOfOption,
    config_dir: str = ConfigOption,
) -> None:
    """Run the screen and save derived artifacts."""
    paths, config = _resolve(profile, config_dir)
    store = PriceStore(paths, YFinanceFetcher())
    output = run_screen(store, resolve_tickers(paths, config), config, _as_of(as_of))
    directory = save_screen(paths, output, config.profile)
    record_stage(paths, "screen", datetime.now().isoformat(timespec="seconds"))

    typer.echo(f"clusters: {len(output.clusters.clusters)}")
    if not output.rebound.empty:
        fired = output.rebound[output.rebound["fired"]]
        typer.echo(f"alarms fired: {len(fired)}")
        for row in fired.to_dict("records"):
            typer.echo(
                f"  {row['ticker']:<6} cluster={row['cluster']} "
                f"alarm={row['alarm']:.1f} washout={row['washout']:.2f}"
            )
    typer.echo(f"written: {directory}")


@app.command()
def backtest(
    profile: str = ProfileOption,
    as_of: str = AsOfOption,
    config_dir: str = ConfigOption,
    out: str = OutOption,
) -> None:
    """Walk-forward validation of the rebound alarm."""
    paths, config = _resolve(profile, config_dir)
    store = PriceStore(paths, YFinanceFetcher())
    result = run_backtest(
        store, resolve_tickers(paths, config), config, _as_of(as_of)
    )

    directory = Path(out) / config.profile
    directory.mkdir(parents=True, exist_ok=True)
    result.per_fold.to_csv(directory / "per_fold.csv", index=False)
    result.economics.to_csv(directory / "economics.csv", index=False)
    result.baseline.to_csv(directory / "baseline.csv", index=False)
    result.edges.to_csv(directory / "edges.csv", index=False)
    result.fitted_gates.to_csv(directory / "fitted_gates.csv", index=False)
    record_stage(paths, "backtest", datetime.now().isoformat(timespec="seconds"))

    typer.echo(result.per_fold.to_string(index=False))
    typer.echo("")
    typer.echo("Edge over random entry:")
    typer.echo(result.edges.to_string(index=False))
    if config.profile == "dev":
        typer.echo("")
        typer.echo(
            "DEV PROFILE: short window, for debugging only. Not evidence "
            "that the method works."
        )


@app.command()
def report(
    profile: str = ProfileOption,
    as_of: str = AsOfOption,
    config_dir: str = ConfigOption,
    out: str = OutOption,
) -> None:
    """Render the HTML report for a given as-of date."""
    paths, config = _resolve(profile, config_dir)
    store = PriceStore(paths, YFinanceFetcher())
    output = run_screen(store, resolve_tickers(paths, config), config, _as_of(as_of))
    destination = render_report(output, config, Path(out))
    typer.echo(f"written: {destination}")


if __name__ == "__main__":
    app()
