from __future__ import annotations

import argparse
import json
from pathlib import Path

from qtrends.config import load_config
from qtrends.data import generate_synthetic_csv
from qtrends.pipeline import run_pipeline
from qtrends.universe import resolve_universe, sync_universe


def _run(config_path: str, output_dir: str, refresh_universe: bool | None = None) -> int:
    config = load_config(config_path)
    config, universe_result = resolve_universe(config, refresh=refresh_universe)
    if universe_result:
        print(
            "Universe sync: "
            f"active={len(universe_result.active_symbols)} "
            f"added={len(universe_result.added_symbols)} "
            f"deactivated={len(universe_result.deactivated_symbols)} "
            f"manifest={universe_result.manifest_path}"
        )
    artifacts = run_pipeline(config, output_dir)
    print(json.dumps(artifacts.latest_signal, indent=2, default=str))
    print(f"Artifacts written to {artifacts.output_dir.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qtrends")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the configured pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--no-universe-refresh",
        action="store_true",
        help="Use the existing manifest without contacting Nasdaq",
    )

    demo = subparsers.add_parser("demo", help="Generate deterministic data and run the sample pipeline")
    demo.add_argument("--config", default="configs/sample.yaml")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--periods", type=int, default=1250)

    sample = subparsers.add_parser("generate-sample", help="Generate deterministic sample market data")
    sample.add_argument("--output", default="data/sample_prices.csv")
    sample.add_argument("--periods", type=int, default=1250)
    sample.add_argument("--seed", type=int, default=42)

    validate = subparsers.add_parser("validate-config", help="Validate and print a configuration")
    validate.add_argument("--config", required=True)

    universe = subparsers.add_parser(
        "sync-universe", help="Refresh and persist an industry universe from Nasdaq Screener"
    )
    universe.add_argument("--config", required=True)
    universe.add_argument("--snapshot-date", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        refresh = False if args.no_universe_refresh else None
        return _run(args.config, args.output_dir, refresh_universe=refresh)
    if args.command == "demo":
        config = load_config(args.config)
        csv_path = Path(config.data.csv_path or "data/sample_prices.csv")
        generate_synthetic_csv(
            csv_path,
            tickers=config.data.tickers,
            benchmark=config.data.benchmark,
            periods=args.periods,
        )
        return _run(args.config, args.output_dir)
    if args.command == "generate-sample":
        path = generate_synthetic_csv(args.output, periods=args.periods, seed=args.seed)
        print(path.resolve())
        return 0
    if args.command == "validate-config":
        config = load_config(args.config)
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "sync-universe":
        result = sync_universe(load_config(args.config), snapshot_date=args.snapshot_date)
        print(
            json.dumps(
                {
                    "snapshot_date": result.snapshot_date,
                    "manifest_path": str(result.manifest_path),
                    "active_count": len(result.active_symbols),
                    "added": result.added_symbols,
                    "deactivated": result.deactivated_symbols,
                },
                indent=2,
            )
        )
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
