"""HTML and CSV report rendering.

Every report carries the survivorship caveat, and dev reports carry a louder
warning still. A dev run proves the code works; it does not prove the method
works, and the artifact has to say so or someone will cite it later as if it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Template

from screener_sector.config import Config
from screener_sector.features.correlation import ClusterResult

DEV_WARNING = (
    "DEV PROFILE — this run covers a short window chosen for fast iteration. "
    "It is sufficient to verify the code behaves as intended and NOT sufficient "
    "to conclude the method works. Do not cite these numbers as evidence."
)


@dataclass(frozen=True)
class ScreenOutput:
    as_of: date
    trend: pd.DataFrame
    clusters: ClusterResult
    strength: pd.DataFrame
    rebound: pd.DataFrame


def clusters_frame(result: ClusterResult) -> pd.DataFrame:
    rows = [
        {
            "cluster": cluster.label,
            "ticker": ticker,
            "mean_correlation": cluster.mean_correlation,
        }
        for cluster in result.clusters
        for ticker in cluster.members
    ]
    return pd.DataFrame(rows, columns=["cluster", "ticker", "mean_correlation"])


def _html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '<p class="empty">No rows.</p>'
    return df.to_html(index=False, float_format=lambda v: f"{v:.3f}")


def write_csvs(output: ScreenOutput, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    output.trend.to_csv(directory / "trend.csv", index=False)
    output.strength.to_csv(directory / "strength.csv", index=False)
    output.rebound.to_csv(directory / "rebound.csv", index=False)
    clusters_frame(output.clusters).to_csv(directory / "clusters.csv", index=False)


def render_report(output: ScreenOutput, config: Config, out_dir: Path) -> Path:
    directory = out_dir / config.profile
    write_csvs(output, directory)

    template_path = Path(__file__).with_name("template.html")
    template = Template(template_path.read_text())
    html = template.render(
        as_of=output.as_of.isoformat(),
        profile=config.profile,
        start=config.start.isoformat(),
        benchmark=config.benchmark,
        dev_warning=DEV_WARNING if config.profile == "dev" else "",
        rebound_table=_html_table(output.rebound),
        clusters_table=_html_table(clusters_frame(output.clusters)),
        strength_table=_html_table(output.strength),
        trend_table=_html_table(output.trend),
    )

    destination = directory / f"{output.as_of.isoformat()}.html"
    destination.write_text(html)
    return destination
