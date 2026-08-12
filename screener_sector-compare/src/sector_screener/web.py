from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from .config import Settings, load_settings
from .nasdaq_universe import NasdaqUniverseCache
from .pipeline import analyze_stage, download_stage, resolve_dates, resolve_universe
from .universe import UniverseCatalog

ASSET_DIR = Path(__file__).parent / "web_assets"
RUN_PREFIXES = ("poc_", "prod_")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class JobRequest(BaseModel):
    stage: Literal["poc", "prod"] = "poc"
    industry: str = Field(default="semiconductor", pattern=r"^[A-Za-z0-9_:.-]+$")
    start: date | None = None
    end: date | None = None
    include: list[str] = Field(default_factory=list, max_length=50)
    exclude: list[str] = Field(default_factory=list, max_length=50)
    refresh_tail_days: int = Field(default=0, ge=0, le=90)
    force_download: bool = False
    refresh_universe: bool = False
    max_tickers: int | None = Field(default=None, ge=3, le=250)
    short_window: int = Field(default=20, ge=5, le=63)
    mid_window: int = Field(default=60, ge=20, le=126)
    long_window: int = Field(default=200, ge=60, le=252)
    correlation_window: int = Field(default=60, ge=20, le=252)
    pca_window: int = Field(default=60, ge=40, le=252)
    hmm_states: int = Field(default=3, ge=2, le=6)
    capture_window: int = Field(default=60, ge=20, le=252)
    rebound_horizon: int = Field(default=10, ge=5, le=40)
    rebound_return_pct: float = Field(default=5.0, ge=2.0, le=15.0)
    max_adverse_pct: float = Field(default=-3.0, ge=-10.0, le=-1.0)
    correction_drawdown_pct: float = Field(default=-8.0, ge=-30.0, le=-3.0)
    minimum_correlation: float = Field(default=0.55, ge=0.0, le=0.95)
    rebound_probability: float = Field(default=0.70, ge=0.50, le=0.95)
    minimum_breadth_thrust_pct: float = Field(default=10.0, ge=5.0, le=40.0)
    min_coverage_pct: float = Field(default=80.0, ge=70.0, le=100.0)

    @model_validator(mode="after")
    def validate_relationships(self) -> JobRequest:
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be earlier than the exclusive end date")
        latest_exclusive_end = datetime.now(UTC).date() + timedelta(days=1)
        if self.end and self.end > latest_exclusive_end:
            raise ValueError("exclusive end date cannot be later than tomorrow")
        if not self.short_window < self.mid_window <= self.long_window:
            raise ValueError("trend windows must satisfy short < mid <= long")
        if self.rebound_return_pct <= abs(self.max_adverse_pct):
            raise ValueError("rebound target must exceed the absolute adverse target")
        normalized_include = [item.strip().upper() for item in self.include if item.strip()]
        normalized_exclude = [item.strip().upper() for item in self.exclude if item.strip()]
        if set(normalized_include) & set(normalized_exclude):
            raise ValueError("a ticker cannot be in both include and exclude")
        self.include = normalized_include
        self.exclude = normalized_exclude
        return self


def settings_from_request(request: JobRequest) -> Settings:
    base = load_settings(request.stage)
    raw = deepcopy(base.raw)
    analysis = raw["analysis"]
    raw["data"]["min_coverage"] = request.min_coverage_pct / 100.0
    if request.max_tickers is not None:
        raw["universe"]["max_tickers"] = request.max_tickers
    analysis.update(
        {
            "short_window": request.short_window,
            "mid_window": request.mid_window,
            "long_window": request.long_window,
            "correlation_window": request.correlation_window,
            "pca_window": request.pca_window,
            "hmm_states": request.hmm_states,
            "capture_window": request.capture_window,
            "rebound_horizon": request.rebound_horizon,
            "rebound_return": request.rebound_return_pct / 100.0,
            "max_adverse_return": request.max_adverse_pct / 100.0,
            "correction_drawdown": request.correction_drawdown_pct / 100.0,
            "minimum_correlation": request.minimum_correlation,
            "rebound_probability": request.rebound_probability,
            "minimum_breadth_thrust": request.minimum_breadth_thrust_pct / 100.0,
        }
    )
    minimum_hmm_rows = request.hmm_states * 40
    if request.stage == "prod":
        minimum_hmm_rows = request.hmm_states * 100
    analysis["hmm_min_train"] = max(int(analysis["hmm_min_train"]), minimum_hmm_rows)
    return Settings(raw=raw, config_dir=base.config_dir)


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screener-web")
        self._jobs: dict[str, dict[str, Any]] = {}
        settings = load_settings("poc")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._store = settings.output_dir / "web_jobs.json"
        self._load()

    def _load(self) -> None:
        if not self._store.exists():
            return
        try:
            jobs = json.loads(self._store.read_text(encoding="utf-8"))
            for job in jobs:
                if job.get("status") in {
                    "queued",
                    "refreshing_universe",
                    "downloading",
                    "analyzing",
                }:
                    job["status"] = "failed"
                    job["message"] = "Web service restarted before the job completed"
                    job["finished_at"] = _utc_now()
                self._jobs[job["id"]] = job
        except (json.JSONDecodeError, KeyError, TypeError):
            self._jobs = {}

    def _persist(self) -> None:
        temporary = self._store.with_suffix(".tmp.json")
        payload = sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self._store)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)
            self._jobs[job_id]["updated_at"] = _utc_now()
            self._persist()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                (deepcopy(job) for job in self._jobs.values()),
                key=lambda item: item["created_at"],
                reverse=True,
            )[:100]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def submit(
        self, action: Literal["universe", "download", "analyze", "run"], request: JobRequest
    ) -> dict:
        request_payload = request.model_dump(mode="json")
        fingerprint = hashlib.sha256(
            json.dumps({"action": action, "request": request_payload}, sort_keys=True).encode()
        ).hexdigest()
        with self._lock:
            for job in self._jobs.values():
                if job["fingerprint"] == fingerprint and job["status"] in {
                    "queued",
                    "refreshing_universe",
                    "downloading",
                    "analyzing",
                }:
                    return deepcopy(job)
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "action": action,
                "status": "queued",
                "message": "Waiting for the single-user worker",
                "progress": 0,
                "fingerprint": fingerprint,
                "request": request_payload,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._persist()
        self._executor.submit(self._execute, job_id, action, request)
        return deepcopy(job)

    def _execute(self, job_id: str, action: str, request: JobRequest) -> None:
        try:
            settings = settings_from_request(request)
            if action == "universe":
                self._update(
                    job_id,
                    status="refreshing_universe",
                    started_at=_utc_now(),
                    progress=20,
                    message="Refreshing and validating the full Nasdaq export",
                )
                snapshot = NasdaqUniverseCache(settings.cache_dir).ensure(
                    refresh=request.refresh_universe
                )
                universe_result = {
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
                }
                self._update(
                    job_id,
                    status="completed",
                    progress=100,
                    message=f"Nasdaq universe ready: {snapshot['normalized_rows']} symbols",
                    result={"universe": universe_result},
                    finished_at=_utc_now(),
                )
                return
            if request.industry.startswith("nasdaq:") and action in {"download", "run"}:
                self._update(
                    job_id,
                    status="refreshing_universe",
                    started_at=_utc_now(),
                    progress=5,
                    message="Checking the cached Nasdaq universe snapshot",
                )
                NasdaqUniverseCache(settings.cache_dir).ensure(refresh=request.refresh_universe)
            start, end = resolve_dates(
                settings,
                request.start.isoformat() if request.start else None,
                request.end.isoformat() if request.end else None,
            )
            universe = resolve_universe(
                settings,
                request.industry,
                request.include,
                request.exclude,
            )
            self._update(job_id, started_at=_utc_now(), message="Inputs validated")
            download_result = None
            analysis_result = None
            if action in {"download", "run"}:
                self._update(
                    job_id,
                    status="downloading",
                    progress=15,
                    message="Filling only missing cache ranges",
                )
                download_result = download_stage(
                    settings,
                    universe,
                    start,
                    end,
                    request.refresh_tail_days,
                    request.force_download,
                )
            if action in {"analyze", "run"}:
                self._update(
                    job_id,
                    status="analyzing",
                    progress=55,
                    message="Running cached breadth, PCA, HMM, and XGBoost analysis",
                )
                analysis_result = analyze_stage(settings, universe, start, end)
            result = {"download": download_result, "analysis": analysis_result}
            self._update(
                job_id,
                status="completed",
                progress=100,
                message="Job completed",
                result=result,
                finished_at=_utc_now(),
            )
        except Exception as exc:  # noqa: BLE001 - persist a safe job failure
            self._update(
                job_id,
                status="failed",
                message=str(exc),
                error={
                    "type": type(exc).__name__,
                    "detail": str(exc),
                },
                finished_at=_utc_now(),
            )


manager: JobManager | None = None
app = FastAPI(title="Sector Screener Lab", version="0.2.0", docs_url="/api/docs")


def get_manager() -> JobManager:
    global manager
    if manager is None:
        manager = JobManager()
    return manager


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> JSONResponse:
    try:
        settings = load_settings("poc")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        return JSONResponse({"status": "ready"})
    except OSError as exc:
        return JSONResponse({"status": "not_ready", "error": str(exc)}, status_code=503)


@app.get("/api/options")
def options() -> dict[str, Any]:
    settings = load_settings("poc")
    catalog = UniverseCatalog(settings.config_dir / "industries.yaml")
    nasdaq = NasdaqUniverseCache(settings.cache_dir)
    nasdaq_industries = nasdaq.describe()
    nasdaq_status: dict[str, Any] = {"cached": bool(nasdaq_industries)}
    if nasdaq_industries:
        nasdaq_status.update(
            {
                "snapshot_id": nasdaq_industries[0]["snapshot_id"],
                "retrieved_at": nasdaq_industries[0]["retrieved_at"],
                "industries": len(nasdaq_industries),
            }
        )
    return {
        "industries": [*catalog.describe(), *nasdaq_industries],
        "nasdaq": nasdaq_status,
        "stages": ["poc", "prod"],
        "defaults": JobRequest().model_dump(mode="json"),
        "provider": "yahoo",
        "disclaimer": "Research screen, not an investment recommendation.",
    }


@app.get("/api/jobs")
def jobs() -> list[dict[str, Any]]:
    return get_manager().list()


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    try:
        return get_manager().get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


def _submit(
    action: Literal["universe", "download", "analyze", "run"], request: JobRequest
) -> JSONResponse:
    job_record = get_manager().submit(action, request)
    return JSONResponse(job_record, status_code=202)


@app.post("/api/jobs/download")
def submit_download(request: JobRequest) -> JSONResponse:
    return _submit("download", request)


@app.post("/api/jobs/universe")
def submit_universe(request: JobRequest) -> JSONResponse:
    return _submit("universe", request)


@app.post("/api/jobs/analyze")
def submit_analyze(request: JobRequest) -> JSONResponse:
    return _submit("analyze", request)


@app.post("/api/jobs/run")
def submit_run(request: JobRequest) -> JSONResponse:
    return _submit("run", request)


def _run_root(run_id: str) -> Path:
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in run_id
    ):
        raise HTTPException(status_code=400, detail="Invalid run identifier")
    output_root = load_settings("poc").output_dir.resolve()
    candidate = (output_root / run_id).resolve()
    if candidate.parent != output_root or not candidate.is_dir() or candidate.is_symlink():
        raise HTTPException(status_code=404, detail="Run not found")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    output_root = load_settings("poc").output_dir
    result: list[dict[str, Any]] = []
    for directory in sorted(output_root.iterdir(), reverse=True):
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or not directory.name.startswith(RUN_PREFIXES)
        ):
            continue
        report = directory / "report.html"
        manifest = directory / "run_manifest.json"
        if not report.is_file() or not manifest.is_file():
            continue
        details = _read_json(manifest)
        result.append(
            {
                "id": directory.name,
                "stage": details.get("stage"),
                "industry": details.get("universe", {}).get("name"),
                "actual_range": details.get("actual_range"),
                "alert": details.get("alert"),
                "validation": _read_json(directory / "validation.json"),
                "report_url": f"/api/runs/{directory.name}/report",
            }
        )
    return result[:100]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    root = _run_root(run_id)
    return {
        "id": run_id,
        "manifest": _read_json(root / "run_manifest.json"),
        "validation": _read_json(root / "validation.json"),
        "alert": _read_json(root / "alert.json"),
        "report_url": f"/api/runs/{run_id}/report",
    }


@app.get("/api/runs/{run_id}/report", response_class=FileResponse)
def report(run_id: str) -> FileResponse:
    path = _run_root(run_id) / "report.html"
    if not path.is_file() or path.is_symlink():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        path,
        media_type="text/html",
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'self'",
            "Cache-Control": "no-store",
        },
    )


@app.get("/assets/{asset_name}", response_class=FileResponse)
def asset(asset_name: str) -> FileResponse:
    if asset_name not in {"app.js", "styles.css"}:
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = "application/javascript" if asset_name.endswith(".js") else "text/css"
    return FileResponse(ASSET_DIR / asset_name, media_type=media_type)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(ASSET_DIR / "index.html", media_type="text/html")
