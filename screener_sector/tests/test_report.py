from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from screener_sector.config import Config
from screener_sector.features.correlation import Cluster, ClusterResult
from screener_sector.report.render import DEV_WARNING, ScreenOutput, render_report

CONFIG_DIR = Path("/app/config")


@pytest.fixture
def output():
    corr = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]], index=["NVDA", "AMD"], columns=["NVDA", "AMD"]
    )
    return ScreenOutput(
        as_of=date(2026, 8, 12),
        trend=pd.DataFrame(
            {
                "ticker": ["NVDA", "AMD"],
                "short_score": [72.0, 65.0],
                "mid_score": [80.0, 55.0],
                "short_r2": [0.9, 0.8],
                "mid_r2": [0.95, 0.7],
                "adx": [30.0, 22.0],
                "ma_stack": [1.0, 0.5],
            }
        ),
        clusters=ClusterResult(
            clusters=(Cluster(0, ("NVDA", "AMD"), 0.82),),
            assignments=pd.Series({"NVDA": 0, "AMD": 0}),
            raw_corr=corr,
            residual_corr=corr * 0.5,
        ),
        strength=pd.DataFrame(
            {
                "ticker": ["NVDA", "AMD"],
                "cluster": [0, 0],
                "up_capture": [1.3, 0.9],
                "down_capture": [0.7, 1.2],
                "capture_spread": [0.6, -0.3],
                "max_drawdown": [-0.2, -0.35],
                "recovery_days": [15, 40],
                "rank_in_cluster": [1, 2],
            }
        ),
        rebound=pd.DataFrame(
            {
                "ticker": ["NVDA"],
                "cluster": [0],
                "alarm": [72.0],
                "washout": [0.8],
                "stretch_z": [-2.1],
                "rsi": [24.0],
                "volume": [0.7],
                "divergence": [True],
                "confirmed": [True],
                "fired": [True],
            }
        ),
    )


def test_render_writes_html_at_expected_path(output, tmp_path):
    cfg = Config.load(CONFIG_DIR, "dev")
    path = render_report(output, cfg, tmp_path)
    assert path == tmp_path / "dev" / "2026-08-12.html"
    assert path.exists()


def test_report_is_profile_namespaced(output, tmp_path):
    dev_path = render_report(output, Config.load(CONFIG_DIR, "dev"), tmp_path)
    prod_path = render_report(output, Config.load(CONFIG_DIR, "prod"), tmp_path)
    assert dev_path.parent.name == "dev"
    assert prod_path.parent.name == "prod"


def test_dev_report_contains_the_dev_warning(output, tmp_path):
    path = render_report(output, Config.load(CONFIG_DIR, "dev"), tmp_path)
    assert DEV_WARNING in path.read_text()


def test_prod_report_omits_the_dev_warning(output, tmp_path):
    path = render_report(output, Config.load(CONFIG_DIR, "prod"), tmp_path)
    assert DEV_WARNING not in path.read_text()


def test_every_report_contains_survivorship_caveat(output, tmp_path):
    for profile in ("dev", "prod"):
        path = render_report(output, Config.load(CONFIG_DIR, profile), tmp_path)
        assert "survivorship" in path.read_text().lower()


def test_every_report_discloses_universe_selection_bias(output, tmp_path):
    for profile in ("dev", "prod"):
        path = render_report(output, Config.load(CONFIG_DIR, profile), tmp_path)
        assert "current liquidity" in path.read_text().lower()


def test_report_contains_ticker_and_cluster_data(output, tmp_path):
    text = render_report(output, Config.load(CONFIG_DIR, "dev"), tmp_path).read_text()
    assert "NVDA" in text
    assert "AMD" in text


def test_render_writes_sibling_csvs(output, tmp_path):
    path = render_report(output, Config.load(CONFIG_DIR, "dev"), tmp_path)
    for name in ("trend.csv", "strength.csv", "rebound.csv", "clusters.csv"):
        assert (path.parent / name).exists()


def test_clusters_csv_has_one_row_per_member(output, tmp_path):
    path = render_report(output, Config.load(CONFIG_DIR, "dev"), tmp_path)
    clusters = pd.read_csv(path.parent / "clusters.csv")
    assert set(clusters.columns) == {"cluster", "ticker", "mean_correlation"}
    assert len(clusters) == 2


def test_empty_frames_render_without_error(tmp_path):
    empty = ScreenOutput(
        as_of=date(2026, 8, 12),
        trend=pd.DataFrame(),
        clusters=ClusterResult((), pd.Series(dtype=int), pd.DataFrame(), pd.DataFrame()),
        strength=pd.DataFrame(),
        rebound=pd.DataFrame(),
    )
    path = render_report(empty, Config.load(CONFIG_DIR, "dev"), tmp_path)
    assert path.exists()
