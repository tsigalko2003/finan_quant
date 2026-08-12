from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import load_settings
from .nasdaq_universe import NasdaqUniverseCache
from .pipeline import analyze_stage, download_stage, resolve_dates, resolve_universe
from .universe import UniverseCatalog

app = typer.Typer(no_args_is_help=True, help="Cached sector correction and rebound screener")


def _split(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


@app.command("list-industries")
def list_industries(
    config_dir: Annotated[Path | None, typer.Option(help="Configuration directory")] = None,
    qlib_data_dir: Annotated[
        Path | None, typer.Option(help="Optional installed Qlib data directory")
    ] = None,
    include_nasdaq: Annotated[
        bool, typer.Option(help="Include cached Nasdaq export industries")
    ] = True,
) -> None:
    settings = load_settings("poc", config_dir)
    catalog = UniverseCatalog(settings.config_dir / "industries.yaml")
    typer.echo("Project industry catalog:")
    for item in catalog.describe():
        typer.echo(f"  {item['name']:<28} {item['tickers']:>3}  {item['description']}")
    pools = catalog.qlib_pools(qlib_data_dir)
    if qlib_data_dir:
        typer.echo("\nInstalled Qlib pools (select as qlib:<name>):")
        for pool in pools:
            typer.echo(f"  qlib:{pool}")
        if not pools:
            typer.echo("  (none found)")
    if include_nasdaq:
        nasdaq = NasdaqUniverseCache(settings.cache_dir).describe()
        typer.echo("\nCached Nasdaq export industries (select as nasdaq:<industry>):")
        for item in nasdaq:
            typer.echo(f"  {item['name']:<48} {item['tickers']:>4}")
        if not nasdaq:
            typer.echo("  (cache missing; run refresh-universe --source nasdaq)")


@app.command("refresh-universe")
def refresh_universe(
    source: Annotated[str, typer.Option(help="Universe source; currently nasdaq")] = "nasdaq",
    force: Annotated[bool, typer.Option(help="Refresh even when the snapshot is fresh")] = False,
    config_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Network stage: cache a validated full-market universe snapshot."""
    if source.lower() != "nasdaq":
        raise typer.BadParameter("source must be 'nasdaq'")
    settings = load_settings("poc", config_dir)
    snapshot = NasdaqUniverseCache(settings.cache_dir).ensure(refresh=force, force=force)
    typer.echo(
        json.dumps(
            {
                key: snapshot[key]
                for key in (
                    "source",
                    "snapshot_id",
                    "retrieved_at",
                    "source_rows",
                    "normalized_rows",
                    "membership_sha256",
                    "cache_hit",
                    "stale_cache_used",
                )
            },
            indent=2,
        )
    )


@app.command("resolve")
def resolve(
    industry: Annotated[str, typer.Option(help="Catalog name or qlib:<pool>")] = "semiconductor",
    stage: Annotated[str, typer.Option(help="poc or prod")] = "poc",
    include: Annotated[str, typer.Option(help="Comma-separated additions")] = "",
    exclude: Annotated[str, typer.Option(help="Comma-separated exclusions")] = "",
    config_dir: Annotated[Path | None, typer.Option()] = None,
    qlib_data_dir: Annotated[Path | None, typer.Option()] = None,
    refresh_nasdaq: Annotated[
        bool, typer.Option(help="Refresh Nasdaq export before resolving")
    ] = False,
) -> None:
    settings = load_settings(stage, config_dir)
    if industry.startswith("nasdaq:") and refresh_nasdaq:
        NasdaqUniverseCache(settings.cache_dir).ensure(refresh=True)
    universe = resolve_universe(settings, industry, _split(include), _split(exclude), qlib_data_dir)
    typer.echo(
        json.dumps(
            {"name": universe.name, "source": universe.source, "tickers": universe.tickers},
            indent=2,
        )
    )


@app.command("download")
def download(
    industry: Annotated[str, typer.Option()] = "semiconductor",
    stage: Annotated[str, typer.Option()] = "poc",
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD override")] = None,
    end: Annotated[str | None, typer.Option(help="Exclusive YYYY-MM-DD override")] = None,
    refresh_tail: Annotated[
        int, typer.Option(help="Deliberately redownload the latest N calendar days")
    ] = 0,
    force: Annotated[bool, typer.Option(help="Replace cache coverage for requested range")] = False,
    include: Annotated[str, typer.Option()] = "",
    exclude: Annotated[str, typer.Option()] = "",
    config_dir: Annotated[Path | None, typer.Option()] = None,
    qlib_data_dir: Annotated[Path | None, typer.Option()] = None,
    refresh_nasdaq: Annotated[
        bool, typer.Option(help="Refresh Nasdaq export before price download")
    ] = False,
) -> None:
    """Network stage: incrementally download only cache gaps."""
    settings = load_settings(stage, config_dir)
    if industry.startswith("nasdaq:"):
        NasdaqUniverseCache(settings.cache_dir).ensure(refresh=refresh_nasdaq)
    dates = resolve_dates(settings, start, end)
    universe = resolve_universe(settings, industry, _split(include), _split(exclude), qlib_data_dir)
    typer.echo(
        json.dumps(download_stage(settings, universe, *dates, refresh_tail, force), indent=2)
    )


@app.command("analyze")
def analyze(
    industry: Annotated[str, typer.Option()] = "semiconductor",
    stage: Annotated[str, typer.Option()] = "poc",
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option(help="Exclusive YYYY-MM-DD override")] = None,
    include: Annotated[str, typer.Option()] = "",
    exclude: Annotated[str, typer.Option()] = "",
    config_dir: Annotated[Path | None, typer.Option()] = None,
    qlib_data_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Offline stage: analyze cached data only; no implicit download."""
    settings = load_settings(stage, config_dir)
    dates = resolve_dates(settings, start, end)
    universe = resolve_universe(settings, industry, _split(include), _split(exclude), qlib_data_dir)
    typer.echo(json.dumps(analyze_stage(settings, universe, *dates), indent=2))


@app.command("run")
def run(
    industry: Annotated[str, typer.Option()] = "semiconductor",
    stage: Annotated[str, typer.Option()] = "poc",
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option(help="Exclusive YYYY-MM-DD override")] = None,
    config_dir: Annotated[Path | None, typer.Option()] = None,
    qlib_data_dir: Annotated[Path | None, typer.Option()] = None,
    refresh_nasdaq: Annotated[
        bool, typer.Option(help="Refresh Nasdaq export before running")
    ] = False,
) -> None:
    """Convenience command that explicitly runs download, then analysis."""
    settings = load_settings(stage, config_dir)
    if industry.startswith("nasdaq:"):
        NasdaqUniverseCache(settings.cache_dir).ensure(refresh=refresh_nasdaq)
    dates = resolve_dates(settings, start, end)
    universe = resolve_universe(settings, industry, qlib_data_dir=qlib_data_dir)
    download_stage(settings, universe, *dates)
    typer.echo(json.dumps(analyze_stage(settings, universe, *dates), indent=2))


if __name__ == "__main__":
    app()
