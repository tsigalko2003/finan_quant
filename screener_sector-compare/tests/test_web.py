from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sector_screener import web
from sector_screener.web import JobRequest, app, settings_from_request


@pytest.fixture
def web_environment(tmp_path: Path, monkeypatch):
    config_dir = Path(__file__).parents[1] / "config"
    monkeypatch.setenv("SCREENER_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("SCREENER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SCREENER_OUTPUT_DIR", str(tmp_path / "outputs"))
    web.manager = None
    yield tmp_path
    if web.manager is not None:
        web.manager._executor.shutdown(wait=True, cancel_futures=True)
    web.manager = None


def test_parameter_guardrails():
    with pytest.raises(ValueError, match="short < mid"):
        JobRequest(short_window=60, mid_window=20)
    with pytest.raises(ValueError, match="adverse target"):
        JobRequest(rebound_return_pct=2, max_adverse_pct=-3)
    with pytest.raises(ValueError, match="both include and exclude"):
        JobRequest(include=["nvda"], exclude=["NVDA"])
    with pytest.raises(ValueError, match="later than tomorrow"):
        JobRequest(end=datetime.now(UTC).date() + timedelta(days=2))

    assert JobRequest(end=datetime.now(UTC).date() + timedelta(days=1)).end

    settings = settings_from_request(
        JobRequest(correction_drawdown_pct=-10, rebound_probability=0.8, min_coverage_pct=90)
    )
    assert settings.analysis["correction_drawdown"] == -0.10
    assert settings.analysis["rebound_probability"] == 0.8
    assert settings.data["min_coverage"] == 0.9


def test_options_health_and_assets(web_environment):
    client = TestClient(app)
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200
    options = client.get("/api/options").json()
    assert any(item["name"] == "semiconductor" for item in options["industries"])
    assert client.get("/").status_code == 200
    assert "Sector Screener Lab" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200


def test_report_listing_and_path_traversal_protection(web_environment):
    output = web_environment / "outputs"
    run = output / "poc_semiconductor_20260101T000000Z"
    run.mkdir(parents=True)
    (run / "report.html").write_text("<html><body>safe report</body></html>", encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "poc",
                "universe": {"name": "semiconductor"},
                "actual_range": {"start": "2024-01-01", "end": "2025-01-01"},
                "alert": {"watch": False, "triggered": False},
            }
        ),
        encoding="utf-8",
    )
    (run / "validation.json").write_text('{"status":"ok"}', encoding="utf-8")
    (run / "alert.json").write_text('{"watch":false,"triggered":false}', encoding="utf-8")

    client = TestClient(app)
    runs = client.get("/api/runs").json()
    assert [item["id"] for item in runs] == [run.name]
    report = client.get(f"/api/runs/{run.name}/report")
    assert report.status_code == 200
    assert "default-src 'none'" in report.headers["content-security-policy"]
    assert "safe report" in report.text
    assert client.get("/api/runs/..%2Fconfig/report").status_code in {400, 404}
    assert client.get("/api/runs/not-a-run/report").status_code == 404


def test_identical_active_jobs_are_deduplicated(web_environment, monkeypatch):
    release = web.threading.Event()

    def blocked_download(*args, **kwargs):
        release.wait(timeout=3)
        return {"cache_hits": 6, "remote_fetches": 0}

    monkeypatch.setattr(web, "download_stage", blocked_download)
    client = TestClient(app)
    payload = {"stage": "poc", "industry": "semiconductor"}
    first = client.post("/api/jobs/download", json=payload)
    second = client.post("/api/jobs/download", json=payload)
    assert first.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    release.set()


def test_nasdaq_universe_job_is_exposed_without_traceback(web_environment, monkeypatch):
    class FakeNasdaqCache:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def ensure(self, refresh=False):
            return {
                "source": "nasdaq-stock-screener-export",
                "snapshot_id": "snapshot-test",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "source_rows": 7_000,
                "normalized_rows": 6_990,
                "membership_sha256": "abc",
                "cache_hit": True,
                "stale_cache_used": False,
            }

        def describe(self):
            return []

    monkeypatch.setattr(web, "NasdaqUniverseCache", FakeNasdaqCache)
    client = TestClient(app)
    submitted = client.post("/api/jobs/universe", json={})
    assert submitted.status_code == 202
    job_id = submitted.json()["id"]
    result = None
    for _ in range(100):
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert result["result"]["universe"]["source_rows"] == 7_000
    assert "trace" not in (result.get("error") or {})
