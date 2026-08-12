# Semiconductor Correlation & Rebound Screener — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a containerized screener that discovers a semiconductor/AI/optical equity universe, scores trends, clusters correlated names, ranks relative strength, raises rebound alarms, and validates the alarms with walk-forward backtests.

**Architecture:** A Python package with a staged CLI pipeline. Each stage reads and writes typed artifacts inside a single relocatable `data/` directory. All network access is behind injectable protocol interfaces so every test runs offline. Two config profiles (`dev`, `prod`) change parameters only, never code paths.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, scikit-learn (clustering only), yfinance, pyarrow, typer, jinja2, pytest.

## Global Constraints

These apply to every task. Do not violate them even when a task's steps do not restate them.

- **Python 3.12.** Container base image `python:3.12-slim`. Do not use syntax newer than 3.12.
- **All commands run in Docker.** Never run `pip install` or `pytest` on the host. Every command in this plan is prefixed `docker compose run --rm screener`.
- **Tests never touch the network.** Any module performing I/O takes a protocol-typed collaborator injected via constructor. A test that would hit Yahoo or NASDAQ is a plan violation.
- **All filesystem paths resolve through `paths.py` from `DATA_DIR`.** No module may construct a path from `__file__`, `os.getcwd()`, or a hardcoded absolute path. This is what makes `data/` relocatable.
- **Point-in-time correctness.** Feature functions may only read data at or before the evaluation timestamp. Only label functions in `backtest/labels.py` may look forward.
- **Profile namespacing.** Derived artifacts go to `data/derived/<profile>/`, reports to `out/<profile>/`. Never write derived output to an unnamespaced path.
- **Price files always store maximum available history**, regardless of the active profile's date range. Profiles narrow analysis, never the cache.
- **Typed returns.** Public functions return `pandas` objects or frozen dataclasses, never bare tuples of more than two elements or untyped dicts.
- **Commit after every task.** Conventional commit messages (`feat:`, `test:`, `chore:`).

## File Structure

| File | Responsibility |
|---|---|
| `Dockerfile` | Python 3.12 image, pinned deps |
| `docker-compose.yml` | `screener` service, mounts `DATA_DIR`→`/data`, `./out`, `./config` |
| `requirements.txt` | pinned dependencies |
| `config/params.yaml` | windows, weights, thresholds, per profile |
| `config/universe.yaml` | theme keywords, industry allow-list, liquidity filters |
| `config/universe.dev.yaml` | ~30 static tickers for the dev profile |
| `src/screener_sector/paths.py` | the only module resolving filesystem paths |
| `src/screener_sector/config.py` | typed config + profile resolution |
| `src/screener_sector/manifest.py` | `manifest.json` read/write, schema version guard |
| `src/screener_sector/data/fetcher.py` | `PriceFetcher` protocol + yfinance adapter |
| `src/screener_sector/data/store.py` | parquet price cache, incremental refresh |
| `src/screener_sector/universe/symbols.py` | NASDAQ Trader symbol files |
| `src/screener_sector/universe/enrich.py` | Yahoo profile field cache, resumable |
| `src/screener_sector/universe/classify.py` | theme keyword matching |
| `src/screener_sector/universe/build.py` | filters → `universe.csv` |
| `src/screener_sector/features/trend.py` | trend composite score |
| `src/screener_sector/features/correlation.py` | returns correlation, residualization, clustering |
| `src/screener_sector/features/strength.py` | up/down capture ratios |
| `src/screener_sector/features/rebound.py` | oscillators, breadth, alarm score |
| `src/screener_sector/backtest/labels.py` | forward-return bottom labeling |
| `src/screener_sector/backtest/walkforward.py` | expanding-window fold generation |
| `src/screener_sector/backtest/evaluate.py` | classification + economic metrics, baseline |
| `src/screener_sector/report/render.py` | HTML + CSV output |
| `src/screener_sector/cli.py` | typer subcommands |
| `tests/conftest.py` | shared fixtures, synthetic series builders |

---

### Task 1: Container, package skeleton, and path resolution

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `.env.example`, `.gitignore`, `pyproject.toml`
- Create: `src/screener_sector/__init__.py`, `src/screener_sector/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Paths` frozen dataclass with `Paths.from_env(env: Mapping[str, str] | None = None) -> Paths`; properties `root`, `manifest_file`, `universe_csv`, `meta_dir`, `symbols_parquet`, `info_parquet`, `failures_csv`, `prices_dir`; methods `price_file(ticker: str) -> Path`, `derived_dir(profile: str) -> Path`, `ensure()` (creates all directories).

- [ ] **Step 1: Create `requirements.txt`**

```
pandas==2.2.3
numpy==2.1.3
scipy==1.14.1
scikit-learn==1.5.2
yfinance==0.2.50
pyarrow==18.1.0
typer==0.15.1
jinja2==3.1.4
pyyaml==6.0.2
requests==2.32.3
pytest==8.3.4
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "screener_sector.cli"]
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "screener-sector"
version = "0.1.0"
requires-python = ">=3.12"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Create `docker-compose.yml`**

```yaml
services:
  screener:
    build: .
    env_file:
      - .env
    volumes:
      - ${DATA_DIR:-./data}:/data
      - ./out:/app/out
      - ./config:/app/config
      - ./tests:/app/tests
      - ./src:/app/src
    environment:
      DATA_DIR: /data
      PROFILE: ${PROFILE:-dev}
```

- [ ] **Step 5: Create `.env.example` and `.gitignore`**

`.env.example`:
```
DATA_DIR=./data
PROFILE=dev
```

`.gitignore`:
```
data/
out/
.env
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 6: Copy `.env.example` to `.env` and build the image**

```bash
cp .env.example .env && docker compose build
```

Expected: image builds successfully.

- [ ] **Step 7: Write the failing test**

Create `tests/test_paths.py`:

```python
from pathlib import Path

from screener_sector.paths import Paths


def test_from_env_uses_data_dir():
    paths = Paths.from_env({"DATA_DIR": "/tmp/somewhere"})
    assert paths.root == Path("/tmp/somewhere")


def test_from_env_defaults_to_local_data():
    paths = Paths.from_env({})
    assert paths.root == Path("data")


def test_all_paths_are_under_root(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    for candidate in [
        paths.manifest_file,
        paths.universe_csv,
        paths.meta_dir,
        paths.symbols_parquet,
        paths.info_parquet,
        paths.failures_csv,
        paths.prices_dir,
        paths.price_file("NVDA"),
        paths.derived_dir("dev"),
    ]:
        assert candidate.is_relative_to(tmp_path)


def test_price_file_is_ticker_named(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    assert paths.price_file("NVDA").name == "NVDA.parquet"


def test_price_file_rejects_path_traversal(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    for bad in ["../etc", "a/b", ""]:
        try:
            paths.price_file(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_derived_dir_is_profile_namespaced(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    assert paths.derived_dir("dev") != paths.derived_dir("prod")
    assert paths.derived_dir("dev").name == "dev"


def test_ensure_creates_directories(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path / "fresh")})
    paths.ensure()
    assert paths.meta_dir.is_dir()
    assert paths.prices_dir.is_dir()
```

- [ ] **Step 8: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_paths.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.paths'`.

- [ ] **Step 9: Implement `src/screener_sector/paths.py`**

```python
"""Filesystem path resolution.

This is the ONLY module in the package permitted to build filesystem paths.
Every path derives from a single DATA_DIR root, which is what makes the data
directory relocatable to any machine at any path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = "data"
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,15}$")


@dataclass(frozen=True)
class Paths:
    """All filesystem locations used by the pipeline, rooted at DATA_DIR."""

    root: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Paths:
        source = os.environ if env is None else env
        return cls(root=Path(source.get("DATA_DIR") or DEFAULT_DATA_DIR))

    @property
    def manifest_file(self) -> Path:
        return self.root / "manifest.json"

    @property
    def universe_csv(self) -> Path:
        return self.root / "universe.csv"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    @property
    def symbols_parquet(self) -> Path:
        return self.meta_dir / "symbols.parquet"

    @property
    def info_parquet(self) -> Path:
        return self.meta_dir / "info.parquet"

    @property
    def failures_csv(self) -> Path:
        return self.meta_dir / "failures.csv"

    @property
    def prices_dir(self) -> Path:
        return self.root / "prices"

    def price_file(self, ticker: str) -> Path:
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"invalid ticker for filename: {ticker!r}")
        return self.prices_dir / f"{ticker}.parquet"

    def derived_dir(self, profile: str) -> Path:
        if not profile.isidentifier():
            raise ValueError(f"invalid profile name: {profile!r}")
        return self.root / "derived" / profile

    def ensure(self) -> None:
        for directory in (self.root, self.meta_dir, self.prices_dir):
            directory.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 10: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_paths.py -v
```

Expected: 7 passed.

- [ ] **Step 11: Commit**

```bash
git add -A && git commit -m "feat: container skeleton and DATA_DIR path resolution"
```

---

### Task 2: Manifest with schema version guard

**Files:**
- Create: `src/screener_sector/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `Paths` from Task 1.
- Produces: `SCHEMA_VERSION: int`; `Manifest` frozen dataclass with fields `schema_version: int`, `stages: dict[str, str]` (stage name → ISO timestamp), `profiles: dict[str, str]` (profile → last as-of date); `load_manifest(paths: Paths) -> Manifest` (returns a fresh manifest if absent, raises `SchemaVersionError` on mismatch); `save_manifest(paths: Paths, manifest: Manifest) -> None`; `record_stage(paths: Paths, stage: str, when: str) -> Manifest`; exception `SchemaVersionError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest.py`:

```python
import json

import pytest

from screener_sector.manifest import (
    SCHEMA_VERSION,
    Manifest,
    SchemaVersionError,
    load_manifest,
    record_stage,
    save_manifest,
)
from screener_sector.paths import Paths


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_env({"DATA_DIR": str(tmp_path)})
    p.ensure()
    return p


def test_load_missing_manifest_returns_fresh(paths):
    manifest = load_manifest(paths)
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.stages == {}
    assert manifest.profiles == {}


def test_save_then_load_roundtrips(paths):
    original = Manifest(
        schema_version=SCHEMA_VERSION,
        stages={"fetch": "2026-08-12T10:00:00"},
        profiles={"dev": "2026-08-11"},
    )
    save_manifest(paths, original)
    assert load_manifest(paths) == original


def test_incompatible_schema_version_raises(paths):
    paths.manifest_file.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION + 1, "stages": {}, "profiles": {}})
    )
    with pytest.raises(SchemaVersionError):
        load_manifest(paths)


def test_record_stage_appends_without_losing_others(paths):
    save_manifest(
        paths,
        Manifest(SCHEMA_VERSION, {"fetch": "2026-08-01T00:00:00"}, {}),
    )
    updated = record_stage(paths, "trend", "2026-08-12T11:00:00")
    assert updated.stages["fetch"] == "2026-08-01T00:00:00"
    assert updated.stages["trend"] == "2026-08-12T11:00:00"
    assert load_manifest(paths).stages["trend"] == "2026-08-12T11:00:00"


def test_manifest_contains_no_absolute_paths(paths):
    save_manifest(paths, Manifest(SCHEMA_VERSION, {"fetch": "x"}, {"dev": "y"}))
    text = paths.manifest_file.read_text()
    assert str(paths.root) not in text
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.manifest'`.

- [ ] **Step 3: Implement `src/screener_sector/manifest.py`**

```python
"""The data directory's self-description.

manifest.json records what has been computed and under which schema. The
version guard refuses to write into a directory produced by an incompatible
build rather than corrupting it silently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from screener_sector.paths import Paths

SCHEMA_VERSION = 1


class SchemaVersionError(RuntimeError):
    """Raised when the data directory was written by an incompatible version."""


@dataclass(frozen=True)
class Manifest:
    schema_version: int = SCHEMA_VERSION
    stages: dict[str, str] = field(default_factory=dict)
    profiles: dict[str, str] = field(default_factory=dict)


def load_manifest(paths: Paths) -> Manifest:
    if not paths.manifest_file.exists():
        return Manifest()
    raw = json.loads(paths.manifest_file.read_text())
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"data directory {paths.root} has schema_version {version}, "
            f"this build requires {SCHEMA_VERSION}"
        )
    return Manifest(
        schema_version=version,
        stages=dict(raw.get("stages", {})),
        profiles=dict(raw.get("profiles", {})),
    )


def save_manifest(paths: Paths, manifest: Manifest) -> None:
    paths.ensure()
    tmp = paths.manifest_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    tmp.replace(paths.manifest_file)


def record_stage(paths: Paths, stage: str, when: str) -> Manifest:
    current = load_manifest(paths)
    updated = Manifest(
        schema_version=current.schema_version,
        stages={**current.stages, stage: when},
        profiles=dict(current.profiles),
    )
    save_manifest(paths, updated)
    return updated
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_manifest.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: manifest with schema version guard"
```

---

### Task 3: Typed configuration and profiles

**Files:**
- Create: `config/params.yaml`, `config/universe.yaml`, `config/universe.dev.yaml`
- Create: `src/screener_sector/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: frozen dataclasses `Windows(short: int, mid: int, corr: int)`, `TrendWeights(slope: float, r2: float, adx: float, ma_stack: float)`, `ReboundWeights(breadth: float, stretch: float, oscillator: float, volume: float, confirmation: float)`, `UniverseFilters(min_price: float, min_dollar_volume: float, min_history_days: int)`, `BacktestParams(label_k: int, label_forward_days: int, label_min_return: float, initial_fit_years: int, step_years: int, horizons: tuple[int, ...])`, `Config(profile: str, start: date, end: date | None, universe_mode: str, static_tickers: tuple[str, ...], benchmark: str, windows: Windows, trend_weights: TrendWeights, rebound_weights: ReboundWeights, corr_threshold: float, min_cluster_size: int, filters: UniverseFilters, backtest: BacktestParams)`; classmethod `Config.load(config_dir: Path, profile: str) -> Config`.

- [ ] **Step 1: Create `config/params.yaml`**

```yaml
defaults:
  benchmark: SOXX
  windows:
    short: 20
    mid: 60
    corr: 120
  trend_weights:
    slope: 0.40
    r2: 0.30
    adx: 0.15
    ma_stack: 0.15
  rebound_weights:
    breadth: 0.25
    stretch: 0.20
    oscillator: 0.25
    volume: 0.15
    confirmation: 0.15
  corr_threshold: 0.60
  min_cluster_size: 3
  filters:
    min_price: 2.0
    min_dollar_volume: 5000000.0
    min_history_days: 250
  backtest:
    label_k: 10
    label_forward_days: 20
    label_min_return: 0.08
    initial_fit_years: 5
    step_years: 1
    horizons: [5, 10, 20]

profiles:
  dev:
    start: "2022-01-01"
    end: null
    universe_mode: static
    backtest:
      initial_fit_years: 2
  prod:
    start: "2006-01-01"
    end: null
    universe_mode: discover
```

- [ ] **Step 2: Create `config/universe.dev.yaml`**

```yaml
tickers:
  - NVDA
  - AMD
  - AVGO
  - TSM
  - ASML
  - AMAT
  - LRCX
  - KLAC
  - MU
  - INTC
  - QCOM
  - TXN
  - ADI
  - NXPI
  - ON
  - MCHP
  - MRVL
  - SWKS
  - QRVO
  - MPWR
  - TER
  - ENTG
  - COHR
  - LITE
  - AAOI
  - POET
  - CRDO
  - ALAB
  - SNPS
  - CDNS
  - SOXX
```

- [ ] **Step 3: Create `config/universe.yaml`**

```yaml
industry_allow_list:
  - Semiconductors
  - Semiconductor Equipment & Materials
  - Communication Equipment
  - Electronic Components
  - Computer Hardware
  - Scientific & Technical Instruments

theme_keywords:
  semiconductor:
    - semiconductor
    - integrated circuit
    - wafer
    - foundry
    - fabless
    - lithography
    - advanced packaging
    - chiplet
    - analog chip
    - microcontroller
  ai_compute:
    - gpu
    - accelerator
    - ai chip
    - neural processing
    - hbm
    - high bandwidth memory
    - asic
    - inference chip
    - data center interconnect
  optical:
    - photonic
    - silicon photonics
    - optical transceiver
    - optical component
    - laser diode
    - fiber optic
    - co-packaged optics
  design_tools:
    - electronic design automation
    - eda
    - semiconductor ip
    - chip design software

seed_etfs:
  - SOXX
  - SMH
  - BOTZ
  - AIQ

exchanges:
  - NASDAQ
  - NYSE
  - NYSE MKT
  - NYSE ARCA
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_config.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from screener_sector.config import Config

CONFIG_DIR = Path("/app/config")


def test_dev_profile_loads():
    cfg = Config.load(CONFIG_DIR, "dev")
    assert cfg.profile == "dev"
    assert cfg.start == date(2022, 1, 1)
    assert cfg.end is None
    assert cfg.universe_mode == "static"


def test_prod_profile_loads():
    cfg = Config.load(CONFIG_DIR, "prod")
    assert cfg.start == date(2006, 1, 1)
    assert cfg.universe_mode == "discover"


def test_defaults_are_shared_across_profiles():
    dev = Config.load(CONFIG_DIR, "dev")
    prod = Config.load(CONFIG_DIR, "prod")
    assert dev.windows == prod.windows
    assert dev.benchmark == prod.benchmark == "SOXX"


def test_profile_overrides_nested_default():
    dev = Config.load(CONFIG_DIR, "dev")
    prod = Config.load(CONFIG_DIR, "prod")
    assert dev.backtest.initial_fit_years == 2
    assert prod.backtest.initial_fit_years == 5
    # unrelated backtest values still come from defaults
    assert dev.backtest.label_k == prod.backtest.label_k == 10


def test_dev_profile_loads_static_tickers():
    cfg = Config.load(CONFIG_DIR, "dev")
    assert "NVDA" in cfg.static_tickers
    assert len(cfg.static_tickers) >= 25


def test_prod_profile_has_no_static_tickers():
    cfg = Config.load(CONFIG_DIR, "prod")
    assert cfg.static_tickers == ()


def test_trend_weights_sum_to_one():
    cfg = Config.load(CONFIG_DIR, "dev")
    w = cfg.trend_weights
    assert abs(w.slope + w.r2 + w.adx + w.ma_stack - 1.0) < 1e-9


def test_rebound_weights_sum_to_one():
    w = Config.load(CONFIG_DIR, "dev").rebound_weights
    total = w.breadth + w.stretch + w.oscillator + w.volume + w.confirmation
    assert abs(total - 1.0) < 1e-9


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        Config.load(CONFIG_DIR, "nope")


def test_config_is_frozen():
    cfg = Config.load(CONFIG_DIR, "dev")
    with pytest.raises(Exception):
        cfg.profile = "prod"
```

- [ ] **Step 5: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.config'`.

- [ ] **Step 6: Implement `src/screener_sector/config.py`**

```python
"""Typed configuration with profile resolution.

Profiles change parameters only, never code paths, so a dev run exercises
exactly the code a prod run will execute.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Windows:
    short: int
    mid: int
    corr: int


@dataclass(frozen=True)
class TrendWeights:
    slope: float
    r2: float
    adx: float
    ma_stack: float


@dataclass(frozen=True)
class ReboundWeights:
    breadth: float
    stretch: float
    oscillator: float
    volume: float
    confirmation: float


@dataclass(frozen=True)
class UniverseFilters:
    min_price: float
    min_dollar_volume: float
    min_history_days: int


@dataclass(frozen=True)
class BacktestParams:
    label_k: int
    label_forward_days: int
    label_min_return: float
    initial_fit_years: int
    step_years: int
    horizons: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    profile: str
    start: date
    end: date | None
    universe_mode: str
    static_tickers: tuple[str, ...]
    benchmark: str
    windows: Windows
    trend_weights: TrendWeights
    rebound_weights: ReboundWeights
    corr_threshold: float
    min_cluster_size: int
    filters: UniverseFilters
    backtest: BacktestParams

    @classmethod
    def load(cls, config_dir: Path, profile: str) -> Config:
        raw = yaml.safe_load((config_dir / "params.yaml").read_text())
        if profile not in raw["profiles"]:
            raise KeyError(f"unknown profile: {profile!r}")

        merged = _deep_merge(deepcopy(raw["defaults"]), raw["profiles"][profile])

        static: tuple[str, ...] = ()
        if merged["universe_mode"] == "static":
            static_file = config_dir / f"universe.{profile}.yaml"
            static = tuple(yaml.safe_load(static_file.read_text())["tickers"])

        end_raw = merged.get("end")
        return cls(
            profile=profile,
            start=date.fromisoformat(merged["start"]),
            end=date.fromisoformat(end_raw) if end_raw else None,
            universe_mode=merged["universe_mode"],
            static_tickers=static,
            benchmark=merged["benchmark"],
            windows=Windows(**merged["windows"]),
            trend_weights=TrendWeights(**merged["trend_weights"]),
            rebound_weights=ReboundWeights(**merged["rebound_weights"]),
            corr_threshold=float(merged["corr_threshold"]),
            min_cluster_size=int(merged["min_cluster_size"]),
            filters=UniverseFilters(**merged["filters"]),
            backtest=BacktestParams(
                **{**merged["backtest"], "horizons": tuple(merged["backtest"]["horizons"])}
            ),
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay `override` onto `base`, mutating and returning base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
```

- [ ] **Step 7: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_config.py -v
```

Expected: 10 passed.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: typed config with dev and prod profiles"
```

---

### Task 4: Test fixtures — synthetic series builders

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pytest fixtures and helpers importable by every later test:
  `trading_days(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex`;
  `make_ohlcv(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame` (columns `open, high, low, close, volume`);
  `exponential_trend(n: int, daily_rate: float, noise: float = 0.0, seed: int = 0) -> pd.Series`;
  `v_bottom(n_down: int, n_up: int, depth: float = 0.30) -> pd.Series`;
  `correlated_returns(n: int, rho: float, seed: int = 0) -> tuple[pd.Series, pd.Series]`;
  `flat_series(n: int, level: float = 100.0) -> pd.Series`.

These are plain functions defined in `conftest.py` and re-exported, not fixtures, so they can be called with arguments. Import them in tests as `from conftest import make_ohlcv` (pytest puts the rootdir on `sys.path` via the `pythonpath` setting from Task 1).

- [ ] **Step 1: Write the failing test**

Create `tests/test_conftest_fixtures.py`:

```python
import numpy as np
import pandas as pd

from conftest import (
    correlated_returns,
    exponential_trend,
    flat_series,
    make_ohlcv,
    trading_days,
    v_bottom,
)


def test_trading_days_excludes_weekends():
    idx = trading_days(10)
    assert len(idx) == 10
    assert all(d.weekday() < 5 for d in idx)


def test_exponential_trend_is_monotonic_without_noise():
    s = exponential_trend(50, daily_rate=0.002)
    assert s.is_monotonic_increasing
    assert len(s) == 50


def test_exponential_trend_is_reproducible_with_seed():
    a = exponential_trend(50, daily_rate=0.002, noise=0.01, seed=7)
    b = exponential_trend(50, daily_rate=0.002, noise=0.01, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_v_bottom_has_minimum_in_the_middle():
    s = v_bottom(30, 30, depth=0.30)
    assert s.idxmin() == s.index[29]
    assert s.iloc[29] < s.iloc[0] * 0.75


def test_correlated_returns_recovers_rho():
    a, b = correlated_returns(4000, rho=0.8, seed=1)
    assert abs(a.corr(b) - 0.8) < 0.05


def test_flat_series_has_zero_variance():
    assert flat_series(40).std() == 0.0


def test_make_ohlcv_has_expected_columns_and_bounds():
    close = exponential_trend(30, 0.001, noise=0.02, seed=3)
    df = make_ohlcv(close)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()
    assert isinstance(df.index, pd.DatetimeIndex)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_conftest_fixtures.py -v
```

Expected: FAIL with `ImportError: cannot import name 'correlated_returns' from 'conftest'`.

- [ ] **Step 3: Implement `tests/conftest.py`**

```python
"""Synthetic data builders shared by every test.

Tests never touch the network. Series here have known analytic properties so
assertions can be exact rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_days(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def exponential_trend(
    n: int, daily_rate: float, noise: float = 0.0, seed: int = 0
) -> pd.Series:
    """Price series compounding at `daily_rate` with optional lognormal noise."""
    idx = trading_days(n)
    drift = np.exp(np.arange(n) * daily_rate)
    if noise:
        rng = np.random.default_rng(seed)
        drift = drift * np.exp(rng.normal(0.0, noise, n))
    return pd.Series(100.0 * drift, index=idx, name="close")


def v_bottom(n_down: int, n_up: int, depth: float = 0.30) -> pd.Series:
    """Linear decline to a trough at index n_down-1, then a linear recovery."""
    idx = trading_days(n_down + n_up)
    trough = 100.0 * (1.0 - depth)
    down = np.linspace(100.0, trough, n_down)
    up = np.linspace(trough, 100.0, n_up + 1)[1:]
    return pd.Series(np.concatenate([down, up]), index=idx, name="close")


def correlated_returns(n: int, rho: float, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Two return series with population correlation `rho`."""
    rng = np.random.default_rng(seed)
    idx = trading_days(n)
    z1 = rng.normal(0.0, 0.01, n)
    z2 = rng.normal(0.0, 0.01, n)
    mixed = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2
    return pd.Series(z1, index=idx, name="a"), pd.Series(mixed, index=idx, name="b")


def flat_series(n: int, level: float = 100.0) -> pd.Series:
    return pd.Series(np.full(n, level), index=trading_days(n), name="close")


def make_ohlcv(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Wrap a close series into a full OHLCV frame with consistent bounds."""
    prev = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([close, prev], axis=1).max(axis=1) * 1.01
    low = pd.concat([close, prev], axis=1).min(axis=1) * 0.99
    if volume is None:
        volume = pd.Series(1_000_000.0, index=close.index)
    return pd.DataFrame(
        {
            "open": prev.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
        },
        index=close.index,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_conftest_fixtures.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test: synthetic series builders for offline tests"
```

---

### Task 5: Price fetcher protocol and yfinance adapter

**Files:**
- Create: `src/screener_sector/data/__init__.py`, `src/screener_sector/data/fetcher.py`
- Test: `tests/test_fetcher.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PriceFetcher` protocol with `history(ticker: str, start: date, end: date | None) -> pd.DataFrame` returning a DatetimeIndex frame with columns `open, high, low, close, volume`; `FetchError` exception; `YFinanceFetcher(sleep: Callable[[float], None] = time.sleep, max_retries: int = 3)` implementing it; `FakeFetcher(data: dict[str, pd.DataFrame], fail: set[str] | None = None)` for tests, defined in the same module so both production and test code share one contract; `normalize_frame(df: pd.DataFrame) -> pd.DataFrame` which lowercases and validates columns.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetcher.py`:

```python
from datetime import date

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.data.fetcher import (
    FakeFetcher,
    FetchError,
    YFinanceFetcher,
    normalize_frame,
)


def test_normalize_lowercases_yahoo_columns():
    raw = make_ohlcv(exponential_trend(10, 0.001))
    raw.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = normalize_frame(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_rejects_missing_columns():
    df = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
    with pytest.raises(FetchError):
        normalize_frame(df)


def test_normalize_drops_rows_with_null_close():
    df = make_ohlcv(exponential_trend(5, 0.001))
    df.iloc[2, df.columns.get_loc("close")] = None
    assert len(normalize_frame(df)) == 4


def test_fake_fetcher_returns_configured_data():
    df = make_ohlcv(exponential_trend(10, 0.001))
    fetcher = FakeFetcher({"NVDA": df})
    out = fetcher.history("NVDA", date(2020, 1, 1), None)
    assert len(out) == 10


def test_fake_fetcher_raises_for_configured_failures():
    fetcher = FakeFetcher({}, fail={"BAD"})
    with pytest.raises(FetchError):
        fetcher.history("BAD", date(2020, 1, 1), None)


def test_fake_fetcher_slices_by_start_date():
    df = make_ohlcv(exponential_trend(20, 0.001))
    fetcher = FakeFetcher({"NVDA": df})
    cutoff = df.index[10].date()
    out = fetcher.history("NVDA", cutoff, None)
    assert out.index.min().date() >= cutoff


def test_yfinance_fetcher_retries_then_raises():
    calls = []

    class AlwaysFails:
        def history(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("boom")

    fetcher = YFinanceFetcher(
        sleep=lambda _: None,
        max_retries=3,
        ticker_factory=lambda symbol: AlwaysFails(),
    )
    with pytest.raises(FetchError):
        fetcher.history("NVDA", date(2020, 1, 1), None)
    assert len(calls) == 3


def test_yfinance_fetcher_succeeds_on_second_attempt():
    frame = make_ohlcv(exponential_trend(10, 0.001))
    frame.columns = ["Open", "High", "Low", "Close", "Volume"]
    attempts = {"n": 0}

    class FlakyOnce:
        def history(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("rate limited")
            return frame

    fetcher = YFinanceFetcher(
        sleep=lambda _: None,
        max_retries=3,
        ticker_factory=lambda symbol: FlakyOnce(),
    )
    out = fetcher.history("NVDA", date(2020, 1, 1), None)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert attempts["n"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_fetcher.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.data'`.

- [ ] **Step 3: Create `src/screener_sector/data/__init__.py`**

Empty file.

- [ ] **Step 4: Implement `src/screener_sector/data/fetcher.py`**

```python
"""Price retrieval behind a protocol.

Yahoo's endpoint is unofficial and rate-limits aggressively, so every call is
retried with backoff and failures are raised as FetchError for the caller to
quarantine. Tests use FakeFetcher and never open a socket.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Protocol

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class FetchError(RuntimeError):
    """A ticker could not be retrieved or its data was unusable."""


class PriceFetcher(Protocol):
    def history(
        self, ticker: str, start: date, end: date | None
    ) -> pd.DataFrame: ...


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, verify the schema, and drop unusable rows."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise FetchError(f"missing columns: {missing}")
    out = out[REQUIRED_COLUMNS].astype(float)
    out = out.dropna(subset=["close"])
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out.sort_index()


class YFinanceFetcher:
    """Adapter over yfinance with bounded retries."""

    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        ticker_factory: Callable[[str], object] | None = None,
    ) -> None:
        self._sleep = sleep
        self._max_retries = max_retries
        self._ticker_factory = ticker_factory or _default_ticker_factory

    def history(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                handle = self._ticker_factory(ticker)
                raw = handle.history(
                    start=start.isoformat(),
                    end=end.isoformat() if end else None,
                    interval="1d",
                    auto_adjust=True,
                )
                if raw is None or len(raw) == 0:
                    raise FetchError(f"empty history for {ticker}")
                return normalize_frame(raw)
            except Exception as exc:  # noqa: BLE001 - deliberate: retry anything
                last_error = exc
                if attempt < self._max_retries - 1:
                    self._sleep(2.0**attempt)
        raise FetchError(f"failed to fetch {ticker}: {last_error}") from last_error


def _default_ticker_factory(symbol: str):
    import yfinance

    return yfinance.Ticker(symbol)


class FakeFetcher:
    """In-memory PriceFetcher for tests."""

    def __init__(
        self, data: dict[str, pd.DataFrame], fail: set[str] | None = None
    ) -> None:
        self._data = data
        self._fail = fail or set()
        self.calls: list[tuple[str, date, date | None]] = []

    def history(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        self.calls.append((ticker, start, end))
        if ticker in self._fail:
            raise FetchError(f"configured failure for {ticker}")
        if ticker not in self._data:
            raise FetchError(f"no data for {ticker}")
        frame = self._data[ticker]
        sliced = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            sliced = sliced.loc[sliced.index <= pd.Timestamp(end)]
        if sliced.empty:
            raise FetchError(f"empty slice for {ticker}")
        return sliced.copy()
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_fetcher.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: price fetcher protocol with yfinance adapter"
```

---

### Task 6: Parquet price store with incremental refresh

**Files:**
- Create: `src/screener_sector/data/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Paths` (Task 1), `PriceFetcher`/`FetchError`/`FakeFetcher` (Task 5).
- Produces: `RefreshResult` frozen dataclass with `fetched: tuple[str, ...]`, `skipped: tuple[str, ...]`, `failed: dict[str, str]`; `PriceStore(paths: Paths, fetcher: PriceFetcher)` with methods `refresh(tickers: Sequence[str], start: date, end: date | None = None) -> RefreshResult`, `load(ticker: str) -> pd.DataFrame`, `has(ticker: str) -> bool`, `close_panel(tickers: Sequence[str], start: date | None = None, end: date | None = None) -> pd.DataFrame` (wide frame, one column per ticker, union of dates, no forward fill), `available(tickers: Sequence[str], min_days: int) -> list[str]`.

**Critical behavior:** `refresh` always requests from the earliest date it does not already have, and it never truncates existing history when the requested `start` is later than what is cached. This is the "price files always store maximum history" constraint from the design.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
from datetime import date

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_env({"DATA_DIR": str(tmp_path)})
    p.ensure()
    return p


@pytest.fixture
def sample():
    return make_ohlcv(exponential_trend(200, 0.001, noise=0.01, seed=2))


def test_refresh_writes_parquet(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    result = store.refresh(["NVDA"], date(2020, 1, 1))
    assert result.fetched == ("NVDA",)
    assert paths.price_file("NVDA").exists()


def test_load_roundtrips_values(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    store.refresh(["NVDA"], date(2020, 1, 1))
    loaded = store.load("NVDA")
    pd.testing.assert_series_equal(
        loaded["close"], sample["close"], check_freq=False
    )


def test_failed_ticker_is_quarantined_not_raised(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}, fail={"BAD"}))
    result = store.refresh(["NVDA", "BAD"], date(2020, 1, 1))
    assert result.fetched == ("NVDA",)
    assert "BAD" in result.failed
    assert paths.failures_csv.exists()
    assert "BAD" in paths.failures_csv.read_text()


def test_refresh_is_incremental(paths, sample):
    first_half = sample.iloc[:100]
    fetcher = FakeFetcher({"NVDA": first_half})
    store = PriceStore(paths, fetcher)
    store.refresh(["NVDA"], date(2020, 1, 1))

    fetcher_full = FakeFetcher({"NVDA": sample})
    store2 = PriceStore(paths, fetcher_full)
    store2.refresh(["NVDA"], date(2020, 1, 1))

    # second call asks only for dates after what is cached
    requested_start = fetcher_full.calls[0][1]
    assert requested_start > date(2020, 1, 1)
    assert len(store2.load("NVDA")) == 200


def test_refresh_never_truncates_existing_history(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    store.refresh(["NVDA"], date(2020, 1, 1))
    before = len(store.load("NVDA"))

    later = sample.index[150].date()
    store.refresh(["NVDA"], later)
    assert len(store.load("NVDA")) == before


def test_close_panel_aligns_on_union_of_dates(paths):
    a = make_ohlcv(exponential_trend(50, 0.001, seed=1))
    b = make_ohlcv(exponential_trend(30, 0.002, seed=2))
    store = PriceStore(paths, FakeFetcher({"A": a, "B": b}))
    store.refresh(["A", "B"], date(2019, 1, 1))
    panel = store.close_panel(["A", "B"])
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == 50
    assert panel["B"].isna().sum() == 20


def test_close_panel_does_not_forward_fill(paths):
    a = make_ohlcv(exponential_trend(50, 0.001, seed=1))
    b = make_ohlcv(exponential_trend(30, 0.002, seed=2))
    store = PriceStore(paths, FakeFetcher({"A": a, "B": b}))
    store.refresh(["A", "B"], date(2019, 1, 1))
    panel = store.close_panel(["A", "B"])
    # B has only 30 of the 50 bars; the trailing 20 must stay NaN rather than
    # carry a forward-filled price that downstream code would treat as observed.
    assert panel["B"].head(30).notna().all()
    assert panel["B"].tail(20).isna().all()


def test_available_filters_by_min_history(paths):
    a = make_ohlcv(exponential_trend(300, 0.001, seed=1))
    b = make_ohlcv(exponential_trend(100, 0.002, seed=2))
    store = PriceStore(paths, FakeFetcher({"A": a, "B": b}))
    store.refresh(["A", "B"], date(2019, 1, 1))
    assert store.available(["A", "B"], min_days=250) == ["A"]


def test_corrupt_parquet_is_refetched(paths, sample):
    store = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    store.refresh(["NVDA"], date(2020, 1, 1))
    paths.price_file("NVDA").write_bytes(b"not parquet")

    store2 = PriceStore(paths, FakeFetcher({"NVDA": sample}))
    result = store2.refresh(["NVDA"], date(2020, 1, 1))
    assert result.fetched == ("NVDA",)
    assert len(store2.load("NVDA")) == 200
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_store.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.data.store'`.

- [ ] **Step 3: Implement `src/screener_sector/data/store.py`**

```python
"""Parquet price cache.

One file per ticker holding maximum available history, independent of the
active profile. Refresh is incremental: only bars after the cached maximum are
requested, and an existing file is never truncated by a later start date.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from screener_sector.data.fetcher import FetchError, PriceFetcher
from screener_sector.paths import Paths


@dataclass(frozen=True)
class RefreshResult:
    fetched: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: dict[str, str]


class PriceStore:
    def __init__(self, paths: Paths, fetcher: PriceFetcher) -> None:
        self._paths = paths
        self._fetcher = fetcher
        self._paths.ensure()

    def has(self, ticker: str) -> bool:
        return self._paths.price_file(ticker).exists()

    def load(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self._paths.price_file(ticker))

    def refresh(
        self, tickers: Sequence[str], start: date, end: date | None = None
    ) -> RefreshResult:
        fetched: list[str] = []
        skipped: list[str] = []
        failed: dict[str, str] = {}

        for ticker in tickers:
            existing = self._read_existing(ticker)
            request_start = start
            if existing is not None and not existing.empty:
                cached_max = existing.index.max().date()
                if end is not None and cached_max >= end:
                    skipped.append(ticker)
                    continue
                # Ask only for bars we do not already have. Never re-request
                # history, and never let a later `start` truncate the cache.
                request_start = cached_max + timedelta(days=1)

            try:
                incoming = self._fetcher.history(ticker, request_start, end)
            except FetchError as exc:
                if existing is not None and not existing.empty:
                    skipped.append(ticker)
                else:
                    failed[ticker] = str(exc)
                continue

            combined = self._merge(existing, incoming)
            combined.to_parquet(self._paths.price_file(ticker))
            fetched.append(ticker)

        if failed:
            self._record_failures(failed)
        return RefreshResult(tuple(fetched), tuple(skipped), failed)

    def close_panel(
        self,
        tickers: Sequence[str],
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Wide close-price frame. No forward fill: gaps stay NaN so downstream
        code cannot silently treat a stale price as a real observation."""
        columns: dict[str, pd.Series] = {}
        for ticker in tickers:
            if not self.has(ticker):
                continue
            frame = self.load(ticker)
            columns[ticker] = frame["close"]
        if not columns:
            return pd.DataFrame()
        panel = pd.DataFrame(columns).sort_index()
        if start is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start)]
        if end is not None:
            panel = panel.loc[panel.index <= pd.Timestamp(end)]
        return panel

    def available(self, tickers: Sequence[str], min_days: int) -> list[str]:
        out = []
        for ticker in tickers:
            if self.has(ticker) and len(self.load(ticker)) >= min_days:
                out.append(ticker)
        return out

    def _read_existing(self, ticker: str) -> pd.DataFrame | None:
        path = self._paths.price_file(ticker)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt file: treat as absent
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _merge(
        existing: pd.DataFrame | None, incoming: pd.DataFrame
    ) -> pd.DataFrame:
        if existing is None or existing.empty:
            return incoming.sort_index()
        combined = pd.concat([existing, incoming])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def _record_failures(self, failed: dict[str, str]) -> None:
        path = self._paths.failures_csv
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(["ticker", "reason"])
            for ticker, reason in failed.items():
                writer.writerow([ticker, reason])
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_store.py -v
```

Expected: 9 passed. If `test_refresh_is_incremental` fails, the `request_start` branch in `refresh` is the culprit — it must resolve to `cached_max + 1 day` whenever a cache exists, and to `start` only when it does not.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: parquet price store with incremental refresh"
```

---

### Task 7: NASDAQ Trader symbol source

**Files:**
- Create: `src/screener_sector/universe/__init__.py`, `src/screener_sector/universe/symbols.py`
- Test: `tests/test_symbols.py`

**Interfaces:**
- Consumes: `Paths` (Task 1).
- Produces: `TextSource` protocol with `get(url: str) -> str`; `HttpTextSource(timeout: int = 30)`; `FakeTextSource(pages: dict[str, str])`; `NASDAQ_LISTED_URL`, `OTHER_LISTED_URL` constants; `parse_nasdaq_listed(text: str) -> pd.DataFrame`; `parse_other_listed(text: str) -> pd.DataFrame`; `fetch_symbols(source: TextSource) -> pd.DataFrame` with columns `ticker, name, exchange, etf`; `save_symbols(paths, df)` / `load_symbols(paths) -> pd.DataFrame`.

The NASDAQ Trader files are pipe-delimited with a trailing `File Creation Time` line that must be dropped. `nasdaqlisted.txt` columns: `Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares`. `otherlisted.txt` columns: `ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol`. Exchange codes in `otherlisted.txt`: `A`=NYSE MKT, `N`=NYSE, `P`=NYSE ARCA, `Z`=BATS, `V`=IEX.

- [ ] **Step 1: Write the failing test**

Create `tests/test_symbols.py`:

```python
import pytest

from screener_sector.universe.symbols import (
    FakeTextSource,
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    fetch_symbols,
    load_symbols,
    parse_nasdaq_listed,
    parse_other_listed,
    save_symbols,
)
from screener_sector.paths import Paths

NASDAQ_TEXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
NVDA|NVIDIA Corporation - Common Stock|Q|N|N|100|N|N
AMAT|Applied Materials Inc. - Common Stock|Q|N|N|100|N|N
ZTEST|Test Issue Corp - Common Stock|Q|Y|N|100|N|N
SOXX|iShares Semiconductor ETF|G|N|N|100|Y|N
File Creation Time: 0812202617:30|||||||
"""

OTHER_TEXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
TSM|Taiwan Semiconductor Manufacturing Company Ltd.|N|TSM|N|100|N|TSM
SMH|VanEck Semiconductor ETF|P|SMH|Y|100|N|SMH
File Creation Time: 0812202617:30||||||||
"""


def test_parse_nasdaq_drops_footer_and_test_issues():
    df = parse_nasdaq_listed(NASDAQ_TEXT)
    assert set(df["ticker"]) == {"NVDA", "AMAT", "SOXX"}


def test_parse_nasdaq_flags_etfs():
    df = parse_nasdaq_listed(NASDAQ_TEXT).set_index("ticker")
    assert bool(df.loc["SOXX", "etf"]) is True
    assert bool(df.loc["NVDA", "etf"]) is False


def test_parse_nasdaq_sets_exchange():
    df = parse_nasdaq_listed(NASDAQ_TEXT)
    assert set(df["exchange"]) == {"NASDAQ"}


def test_parse_other_maps_exchange_codes():
    df = parse_other_listed(OTHER_TEXT).set_index("ticker")
    assert df.loc["TSM", "exchange"] == "NYSE"
    assert df.loc["SMH", "exchange"] == "NYSE ARCA"


def test_fetch_symbols_combines_both_files():
    source = FakeTextSource(
        {NASDAQ_LISTED_URL: NASDAQ_TEXT, OTHER_LISTED_URL: OTHER_TEXT}
    )
    df = fetch_symbols(source)
    assert set(df["ticker"]) == {"NVDA", "AMAT", "SOXX", "TSM", "SMH"}
    assert list(df.columns) == ["ticker", "name", "exchange", "etf"]


def test_fetch_symbols_deduplicates():
    source = FakeTextSource(
        {NASDAQ_LISTED_URL: NASDAQ_TEXT, OTHER_LISTED_URL: OTHER_TEXT}
    )
    df = fetch_symbols(source)
    assert df["ticker"].is_unique


def test_save_and_load_symbols_roundtrip(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    source = FakeTextSource(
        {NASDAQ_LISTED_URL: NASDAQ_TEXT, OTHER_LISTED_URL: OTHER_TEXT}
    )
    df = fetch_symbols(source)
    save_symbols(paths, df)
    assert len(load_symbols(paths)) == len(df)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_symbols.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.universe'`.

- [ ] **Step 3: Create `src/screener_sector/universe/__init__.py`**

Empty file.

- [ ] **Step 4: Implement `src/screener_sector/universe/symbols.py`**

```python
"""US-listed symbol universe from the NASDAQ Trader public files.

These files are the seed for discovery. They are free, stable, and require no
API key. Both end with a 'File Creation Time' footer line that is not data.
"""

from __future__ import annotations

import io
from typing import Protocol

import pandas as pd

from screener_sector.paths import Paths

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_EXCHANGE_CODES = {
    "A": "NYSE MKT",
    "N": "NYSE",
    "P": "NYSE ARCA",
    "Z": "BATS",
    "V": "IEX",
}

COLUMNS = ["ticker", "name", "exchange", "etf"]


class TextSource(Protocol):
    def get(self, url: str) -> str: ...


class HttpTextSource:
    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def get(self, url: str) -> str:
        import requests

        response = requests.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.text


class FakeTextSource:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def get(self, url: str) -> str:
        return self._pages[url]


def _read_pipe_table(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str).fillna("")
    return df[~df.iloc[:, 0].str.startswith("File Creation Time")]


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    df = _read_pipe_table(text)
    df = df[df["Test Issue"] == "N"]
    return pd.DataFrame(
        {
            "ticker": df["Symbol"].str.strip(),
            "name": df["Security Name"].str.strip(),
            "exchange": "NASDAQ",
            "etf": df["ETF"].str.strip() == "Y",
        }
    ).reset_index(drop=True)


def parse_other_listed(text: str) -> pd.DataFrame:
    df = _read_pipe_table(text)
    df = df[df["Test Issue"] == "N"]
    return pd.DataFrame(
        {
            "ticker": df["ACT Symbol"].str.strip(),
            "name": df["Security Name"].str.strip(),
            "exchange": df["Exchange"].str.strip().map(_EXCHANGE_CODES).fillna("OTHER"),
            "etf": df["ETF"].str.strip() == "Y",
        }
    ).reset_index(drop=True)


def fetch_symbols(source: TextSource) -> pd.DataFrame:
    nasdaq = parse_nasdaq_listed(source.get(NASDAQ_LISTED_URL))
    other = parse_other_listed(source.get(OTHER_LISTED_URL))
    combined = pd.concat([nasdaq, other], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    return combined[COLUMNS].sort_values("ticker").reset_index(drop=True)


def save_symbols(paths: Paths, df: pd.DataFrame) -> None:
    paths.ensure()
    df.to_parquet(paths.symbols_parquet)


def load_symbols(paths: Paths) -> pd.DataFrame:
    return pd.read_parquet(paths.symbols_parquet)
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_symbols.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: NASDAQ Trader symbol source"
```

---

### Task 8: Theme classification

**Files:**
- Create: `src/screener_sector/universe/classify.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: `config/universe.yaml` (Task 3).
- Produces: `ThemeRules` frozen dataclass with `industry_allow_list: frozenset[str]`, `theme_keywords: dict[str, tuple[str, ...]]`, `seed_etfs: tuple[str, ...]`, `exchanges: frozenset[str]`, and classmethod `ThemeRules.load(config_dir: Path) -> ThemeRules`; `match_themes(name: str, summary: str, rules: ThemeRules) -> tuple[str, ...]`; `is_in_scope(industry: str, name: str, summary: str, rules: ThemeRules) -> bool`.

Keyword matching is case-insensitive and word-boundary aware, so `eda` does not match "Ceda" and `gpu` does not match "gpus"… actually `gpu` **should** match "GPUs", so the boundary applies to the start of the token and allows a trailing `s`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_classify.py`:

```python
from pathlib import Path

from screener_sector.universe.classify import ThemeRules, is_in_scope, match_themes

CONFIG_DIR = Path("/app/config")


def rules():
    return ThemeRules.load(CONFIG_DIR)


def test_rules_load_from_yaml():
    r = rules()
    assert "Semiconductors" in r.industry_allow_list
    assert "semiconductor" in r.theme_keywords
    assert "SOXX" in r.seed_etfs


def test_matches_semiconductor_theme():
    themes = match_themes(
        "Applied Materials Inc.",
        "Provides wafer fabrication equipment for the semiconductor industry.",
        rules(),
    )
    assert "semiconductor" in themes


def test_matches_optical_theme():
    themes = match_themes(
        "Applied Optoelectronics",
        "Designs optical transceiver modules and laser diode products.",
        rules(),
    )
    assert "optical" in themes


def test_matches_ai_compute_theme():
    themes = match_themes(
        "NVIDIA Corporation",
        "Designs GPUs and accelerator platforms for data center AI workloads.",
        rules(),
    )
    assert "ai_compute" in themes


def test_matches_multiple_themes():
    themes = match_themes(
        "Broadcom Inc.",
        "Semiconductor supplier of ASIC accelerators and co-packaged optics.",
        rules(),
    )
    assert set(themes) >= {"semiconductor", "ai_compute", "optical"}


def test_no_match_for_unrelated_company():
    themes = match_themes(
        "Coca-Cola Company",
        "Manufactures and distributes non-alcoholic beverages worldwide.",
        rules(),
    )
    assert themes == ()


def test_keyword_matching_is_word_boundary_aware():
    themes = match_themes(
        "Ceda Holdings", "A general holding company with no chip exposure.", rules()
    )
    assert "design_tools" not in themes


def test_plural_keyword_still_matches():
    themes = match_themes(
        "Some Corp", "We build GPUs for rendering.", rules()
    )
    assert "ai_compute" in themes


def test_in_scope_via_industry_even_without_keywords():
    assert is_in_scope("Semiconductors", "Mystery Corp", "No description.", rules())


def test_in_scope_via_keywords_even_with_odd_industry():
    assert is_in_scope(
        "Specialty Business Services",
        "Photonics Co",
        "Builds silicon photonics engines.",
        rules(),
    )


def test_out_of_scope_when_neither_matches():
    assert not is_in_scope(
        "Beverages", "Coca-Cola", "Sells soft drinks.", rules()
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_classify.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/universe/classify.py`**

```python
"""Theme classification from company name and business summary.

A ticker enters the universe if its Yahoo industry is on the allow-list OR its
name/summary matches a theme keyword. The industry check alone misses optical
and AI-adjacent names; the keyword check alone pulls in false positives from
marketing language. Requiring either, not both, is the deliberate trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ThemeRules:
    industry_allow_list: frozenset[str]
    theme_keywords: dict[str, tuple[str, ...]]
    seed_etfs: tuple[str, ...]
    exchanges: frozenset[str]

    @classmethod
    def load(cls, config_dir: Path) -> ThemeRules:
        raw = yaml.safe_load((config_dir / "universe.yaml").read_text())
        return cls(
            industry_allow_list=frozenset(raw["industry_allow_list"]),
            theme_keywords={
                theme: tuple(words) for theme, words in raw["theme_keywords"].items()
            },
            seed_etfs=tuple(raw["seed_etfs"]),
            exchanges=frozenset(raw["exchanges"]),
        )


def _pattern(keyword: str) -> re.Pattern[str]:
    # Word-boundary at both ends, with an optional trailing plural 's'.
    return re.compile(rf"\b{re.escape(keyword)}s?\b", re.IGNORECASE)


def match_themes(name: str, summary: str, rules: ThemeRules) -> tuple[str, ...]:
    haystack = f"{name or ''} {summary or ''}"
    matched = [
        theme
        for theme, keywords in rules.theme_keywords.items()
        if any(_pattern(word).search(haystack) for word in keywords)
    ]
    return tuple(matched)


def is_in_scope(industry: str, name: str, summary: str, rules: ThemeRules) -> bool:
    if industry in rules.industry_allow_list:
        return True
    return bool(match_themes(name, summary, rules))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_classify.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: theme keyword classification"
```

---

### Task 9: Resumable profile enrichment cache

**Files:**
- Create: `src/screener_sector/universe/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: `Paths` (Task 1).
- Produces: `InfoSource` protocol with `info(ticker: str) -> dict[str, object]`; `YFinanceInfoSource(sleep, max_retries, ticker_factory)`; `FakeInfoSource(data: dict[str, dict], fail: set[str] | None = None)`; `INFO_COLUMNS = ["ticker", "long_name", "sector", "industry", "summary", "quote_type", "fetched_at"]`; `enrich(paths, tickers, source, now: str, batch_flush: int = 50) -> pd.DataFrame`; `load_info(paths) -> pd.DataFrame`.

**Critical behavior:** enrichment is the slow prod step (thousands of throttled requests). It must be resumable — already-cached tickers are never re-requested — and it must flush partial results to disk every `batch_flush` tickers so an interrupted run loses at most that many.

- [ ] **Step 1: Write the failing test**

Create `tests/test_enrich.py`:

```python
import pytest

from screener_sector.paths import Paths
from screener_sector.universe.enrich import (
    FakeInfoSource,
    enrich,
    load_info,
)


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_env({"DATA_DIR": str(tmp_path)})
    p.ensure()
    return p


def info_for(name, industry, summary, quote_type="EQUITY"):
    return {
        "longName": name,
        "sector": "Technology",
        "industry": industry,
        "longBusinessSummary": summary,
        "quoteType": quote_type,
    }


def test_enrich_writes_expected_columns(paths):
    source = FakeInfoSource({"NVDA": info_for("NVIDIA", "Semiconductors", "GPUs.")})
    df = enrich(paths, ["NVDA"], source, now="2026-08-12T00:00:00")
    assert list(df.columns) == [
        "ticker", "long_name", "sector", "industry", "summary",
        "quote_type", "fetched_at",
    ]
    assert df.iloc[0]["long_name"] == "NVIDIA"


def test_enrich_is_resumable(paths):
    source = FakeInfoSource(
        {
            "NVDA": info_for("NVIDIA", "Semiconductors", "GPUs."),
            "AMD": info_for("AMD", "Semiconductors", "CPUs."),
        }
    )
    enrich(paths, ["NVDA"], source, now="2026-08-12T00:00:00")
    source.calls.clear()
    enrich(paths, ["NVDA", "AMD"], source, now="2026-08-12T00:00:00")
    assert source.calls == ["AMD"]
    assert len(load_info(paths)) == 2


def test_enrich_records_failures_without_stopping(paths):
    source = FakeInfoSource(
        {"NVDA": info_for("NVIDIA", "Semiconductors", "GPUs.")}, fail={"BAD"}
    )
    df = enrich(paths, ["BAD", "NVDA"], source, now="2026-08-12T00:00:00")
    assert set(df["ticker"]) == {"NVDA"}
    assert "BAD" in paths.failures_csv.read_text()


def test_enrich_flushes_partially_on_interruption(paths):
    payloads = {f"T{i}": info_for(f"T{i}", "Semiconductors", "chips.") for i in range(5)}

    class ExplodesAtIndexThree(FakeInfoSource):
        def info(self, ticker):
            if ticker == "T3":
                raise KeyboardInterrupt
            return super().info(ticker)

    source = ExplodesAtIndexThree(payloads)
    with pytest.raises(KeyboardInterrupt):
        enrich(
            paths,
            [f"T{i}" for i in range(5)],
            source,
            now="2026-08-12T00:00:00",
            batch_flush=2,
        )
    assert len(load_info(paths)) == 2


def test_missing_fields_become_empty_strings(paths):
    source = FakeInfoSource({"XYZ": {"longName": "XYZ Corp"}})
    df = enrich(paths, ["XYZ"], source, now="2026-08-12T00:00:00")
    row = df.iloc[0]
    assert row["industry"] == ""
    assert row["summary"] == ""
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_enrich.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/universe/enrich.py`**

```python
"""Company profile enrichment, cached permanently and resumable.

For prod this is thousands of throttled requests taking hours. It writes
partial results every `batch_flush` tickers so an interrupted run resumes
almost where it stopped rather than starting over.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import pandas as pd

from screener_sector.paths import Paths

INFO_COLUMNS = [
    "ticker",
    "long_name",
    "sector",
    "industry",
    "summary",
    "quote_type",
    "fetched_at",
]


class InfoLookupError(RuntimeError):
    """Profile fields for a ticker could not be retrieved."""


class InfoSource(Protocol):
    def info(self, ticker: str) -> dict[str, object]: ...


class YFinanceInfoSource:
    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        pause: float = 0.3,
        ticker_factory: Callable[[str], object] | None = None,
    ) -> None:
        self._sleep = sleep
        self._max_retries = max_retries
        self._pause = pause
        self._ticker_factory = ticker_factory or _default_ticker_factory

    def info(self, ticker: str) -> dict[str, object]:
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                payload = self._ticker_factory(ticker).info
                if not payload:
                    raise InfoLookupError(f"empty info for {ticker}")
                self._sleep(self._pause)
                return dict(payload)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - retry anything transient
                last = exc
                if attempt < self._max_retries - 1:
                    self._sleep(2.0**attempt)
        raise InfoLookupError(f"failed info for {ticker}: {last}") from last


def _default_ticker_factory(symbol: str):
    import yfinance

    return yfinance.Ticker(symbol)


class FakeInfoSource:
    def __init__(
        self, data: dict[str, dict[str, object]], fail: set[str] | None = None
    ) -> None:
        self._data = data
        self._fail = fail or set()
        self.calls: list[str] = []

    def info(self, ticker: str) -> dict[str, object]:
        self.calls.append(ticker)
        if ticker in self._fail or ticker not in self._data:
            raise InfoLookupError(f"no info for {ticker}")
        return dict(self._data[ticker])


def load_info(paths: Paths) -> pd.DataFrame:
    if not paths.info_parquet.exists():
        return pd.DataFrame(columns=INFO_COLUMNS)
    return pd.read_parquet(paths.info_parquet)


def _save_info(paths: Paths, df: pd.DataFrame) -> None:
    paths.ensure()
    df[INFO_COLUMNS].to_parquet(paths.info_parquet, index=False)


def _row(ticker: str, payload: dict[str, object], now: str) -> dict[str, object]:
    def text(key: str) -> str:
        value = payload.get(key)
        return "" if value is None else str(value)

    return {
        "ticker": ticker,
        "long_name": text("longName"),
        "sector": text("sector"),
        "industry": text("industry"),
        "summary": text("longBusinessSummary"),
        "quote_type": text("quoteType"),
        "fetched_at": now,
    }


def enrich(
    paths: Paths,
    tickers: Sequence[str],
    source: InfoSource,
    now: str,
    batch_flush: int = 50,
) -> pd.DataFrame:
    cached = load_info(paths)
    known = set(cached["ticker"]) if not cached.empty else set()
    pending = [t for t in tickers if t not in known]

    rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}

    def flush() -> pd.DataFrame:
        nonlocal cached, rows
        if rows:
            cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            rows = []
            _save_info(paths, cached)
        return cached

    try:
        for index, ticker in enumerate(pending, start=1):
            try:
                rows.append(_row(ticker, source.info(ticker), now))
            except InfoLookupError as exc:
                failures[ticker] = str(exc)
            if index % batch_flush == 0:
                flush()
    finally:
        flush()
        if failures:
            _record_failures(paths, failures)

    return cached


def _record_failures(paths: Paths, failures: dict[str, str]) -> None:
    path = paths.failures_csv
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["ticker", "reason"])
        for ticker, reason in failures.items():
            writer.writerow([ticker, reason])
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_enrich.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: resumable company profile enrichment"
```

---

### Task 10: Universe assembly with liquidity filters

**Files:**
- Create: `src/screener_sector/universe/build.py`
- Test: `tests/test_universe_build.py`

**Interfaces:**
- Consumes: `Paths`, `Config`/`UniverseFilters` (Task 3), `ThemeRules`/`match_themes`/`is_in_scope` (Task 8), `PriceStore` (Task 6).
- Produces: `UNIVERSE_COLUMNS = ["ticker", "name", "industry", "themes", "exchange", "median_dollar_volume", "last_close", "history_days", "included", "reason"]`; `liquidity_stats(ohlcv: pd.DataFrame, window: int = 60) -> tuple[float, float, int]` returning `(median_dollar_volume, last_close, history_days)`; `build_universe(paths, symbols: pd.DataFrame, info: pd.DataFrame, store: PriceStore, rules: ThemeRules, filters: UniverseFilters) -> pd.DataFrame`; `save_universe(paths, df)` / `load_universe(paths, included_only: bool = True) -> pd.DataFrame`.

**Critical behavior:** rejected rows are **kept** with `included=False` and a populated `reason`, never dropped. The design requires visibility into why a name is absent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe_build.py`:

```python
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv
from screener_sector.config import UniverseFilters
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths
from screener_sector.universe.build import (
    build_universe,
    liquidity_stats,
    load_universe,
    save_universe,
)
from screener_sector.universe.classify import ThemeRules

CONFIG_DIR = Path("/app/config")
FILTERS = UniverseFilters(
    min_price=2.0, min_dollar_volume=5_000_000.0, min_history_days=250
)


def ohlcv(n=300, price_scale=1.0, volume=1_000_000.0):
    close = exponential_trend(n, 0.0005, noise=0.01, seed=4) * price_scale
    vol = pd.Series(volume, index=close.index)
    return make_ohlcv(close, vol)


@pytest.fixture
def env(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    frames = {
        "NVDA": ohlcv(),
        "PENNY": ohlcv(price_scale=0.01),
        "THIN": ohlcv(volume=100.0),
        "NEW": ohlcv(n=100),
        "KO": ohlcv(),
    }
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2015, 1, 1))
    symbols = pd.DataFrame(
        {
            "ticker": list(frames),
            "name": ["NVIDIA", "Penny Chips", "Thin Optics", "New Silicon", "Coca-Cola"],
            "exchange": ["NASDAQ"] * 5,
            "etf": [False] * 5,
        }
    )
    info = pd.DataFrame(
        {
            "ticker": list(frames),
            "long_name": symbols["name"],
            "sector": ["Technology"] * 4 + ["Consumer Defensive"],
            "industry": ["Semiconductors"] * 4 + ["Beverages"],
            "summary": [
                "Designs GPUs.",
                "Makes semiconductor parts.",
                "Optical transceiver maker.",
                "Wafer processing.",
                "Sells soft drinks.",
            ],
            "quote_type": ["EQUITY"] * 5,
            "fetched_at": ["2026-08-12"] * 5,
        }
    )
    return paths, store, symbols, info


def test_liquidity_stats_computes_median_dollar_volume():
    frame = ohlcv(n=100, volume=1_000_000.0)
    median_dv, last_close, days = liquidity_stats(frame, window=60)
    assert days == 100
    assert last_close == pytest.approx(frame["close"].iloc[-1])
    assert median_dv > 0


def test_included_ticker_passes_all_filters(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["NVDA", "included"]) is True
    assert df.loc["NVDA", "reason"] == ""


def test_low_price_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["PENNY", "included"]) is False
    assert "price" in df.loc["PENNY", "reason"]


def test_illiquid_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["THIN", "included"]) is False
    assert "dollar_volume" in df.loc["THIN", "reason"]


def test_short_history_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["NEW", "included"]) is False
    assert "history" in df.loc["NEW", "reason"]


def test_off_theme_rejected_with_reason(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert bool(df.loc["KO", "included"]) is False
    assert "theme" in df.loc["KO", "reason"]


def test_rejected_rows_are_retained_not_dropped(env):
    paths, store, symbols, info = env
    df = build_universe(paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS)
    assert len(df) == 5


def test_themes_are_recorded_as_pipe_delimited(env):
    paths, store, symbols, info = env
    df = build_universe(
        paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS
    ).set_index("ticker")
    assert "semiconductor" in df.loc["NVDA", "themes"]


def test_save_and_load_included_only(env):
    paths, store, symbols, info = env
    df = build_universe(paths, symbols, info, store, ThemeRules.load(CONFIG_DIR), FILTERS)
    save_universe(paths, df)
    assert len(load_universe(paths, included_only=True)) == 1
    assert len(load_universe(paths, included_only=False)) == 5
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_universe_build.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/universe/build.py`**

```python
"""Universe assembly: theme scope plus liquidity filters.

Rejected tickers are retained with a reason rather than dropped, so the
question 'why isn't X in the screen?' always has an answer in the artifact.
"""

from __future__ import annotations

import pandas as pd

from screener_sector.config import UniverseFilters
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths
from screener_sector.universe.classify import ThemeRules, is_in_scope, match_themes

UNIVERSE_COLUMNS = [
    "ticker",
    "name",
    "industry",
    "themes",
    "exchange",
    "median_dollar_volume",
    "last_close",
    "history_days",
    "included",
    "reason",
]


def liquidity_stats(ohlcv: pd.DataFrame, window: int = 60) -> tuple[float, float, int]:
    if ohlcv.empty:
        return 0.0, 0.0, 0
    dollar_volume = (ohlcv["close"] * ohlcv["volume"]).tail(window)
    return (
        float(dollar_volume.median()),
        float(ohlcv["close"].iloc[-1]),
        int(len(ohlcv)),
    )


def build_universe(
    paths: Paths,
    symbols: pd.DataFrame,
    info: pd.DataFrame,
    store: PriceStore,
    rules: ThemeRules,
    filters: UniverseFilters,
) -> pd.DataFrame:
    merged = symbols.merge(info, on="ticker", how="left").fillna("")
    rows: list[dict[str, object]] = []

    for record in merged.to_dict("records"):
        ticker = str(record["ticker"])
        name = str(record.get("long_name") or record.get("name") or "")
        industry = str(record.get("industry") or "")
        summary = str(record.get("summary") or "")

        themes = match_themes(name, summary, rules)
        in_scope = is_in_scope(industry, name, summary, rules)

        if store.has(ticker):
            median_dv, last_close, history_days = liquidity_stats(store.load(ticker))
        else:
            median_dv, last_close, history_days = 0.0, 0.0, 0

        reasons: list[str] = []
        if not in_scope:
            reasons.append("off theme")
        if history_days < filters.min_history_days:
            reasons.append(f"insufficient history ({history_days}d)")
        if last_close < filters.min_price:
            reasons.append(f"price below floor ({last_close:.2f})")
        if median_dv < filters.min_dollar_volume:
            reasons.append(f"dollar_volume below floor ({median_dv:.0f})")

        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "industry": industry,
                "themes": "|".join(themes),
                "exchange": str(record.get("exchange") or ""),
                "median_dollar_volume": median_dv,
                "last_close": last_close,
                "history_days": history_days,
                "included": not reasons,
                "reason": "; ".join(reasons),
            }
        )

    return pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)


def save_universe(paths: Paths, df: pd.DataFrame) -> None:
    paths.ensure()
    df[UNIVERSE_COLUMNS].to_csv(paths.universe_csv, index=False)


def load_universe(paths: Paths, included_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(paths.universe_csv)
    df["themes"] = df["themes"].fillna("")
    df["reason"] = df["reason"].fillna("")
    if included_only:
        df = df[df["included"]].reset_index(drop=True)
    return df
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_universe_build.py -v
```

Expected: 9 passed. Note `test_off_theme_rejected_with_reason` asserts the substring `theme`, which `"off theme"` satisfies.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: universe assembly with liquidity filters"
```

---

### Task 11: Trend composite score

**Files:**
- Create: `src/screener_sector/features/__init__.py`, `src/screener_sector/features/trend.py`
- Test: `tests/test_trend.py`

**Interfaces:**
- Consumes: `Windows`, `TrendWeights` (Task 3).
- Produces: `log_slope_r2(close: pd.Series, window: int) -> tuple[float, float]` (annualized log slope, R²); `adx(ohlcv: pd.DataFrame, window: int = 14) -> float`; `ma_stack_score(close: pd.Series, short: int, mid: int) -> float` in `[-1, 1]`; `TrendResult` frozen dataclass with `slope: float`, `r2: float`, `adx: float`, `ma_stack: float`, `score: float`; `trend_score(ohlcv: pd.DataFrame, window: int, weights: TrendWeights) -> TrendResult`; `trend_table(frames: dict[str, pd.DataFrame], windows: Windows, weights: TrendWeights) -> pd.DataFrame` with columns `ticker, short_score, mid_score, short_r2, mid_r2, adx, ma_stack`.

**Score convention:** `score` is signed and bounded to `[-100, 100]`. Slope is normalized by realized volatility, then squashed with `tanh` so a single explosive move cannot dominate. R², ADX, and MA-stack contributions are multiplied by the slope's sign, so a clean *downtrend* scores strongly negative rather than strongly positive.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trend.py`:

```python
import numpy as np
import pytest

from conftest import exponential_trend, flat_series, make_ohlcv, v_bottom
from screener_sector.config import TrendWeights, Windows
from screener_sector.features.trend import (
    adx,
    log_slope_r2,
    ma_stack_score,
    trend_score,
    trend_table,
)

WEIGHTS = TrendWeights(slope=0.40, r2=0.30, adx=0.15, ma_stack=0.15)
WINDOWS = Windows(short=20, mid=60, corr=120)


def test_pure_exponential_trend_has_r2_near_one():
    close = exponential_trend(60, daily_rate=0.002)
    slope, r2 = log_slope_r2(close, window=60)
    assert r2 > 0.999
    assert slope > 0


def test_noisy_trend_has_lower_r2_than_clean_trend():
    clean = exponential_trend(60, 0.002)
    noisy = exponential_trend(60, 0.002, noise=0.05, seed=5)
    _, clean_r2 = log_slope_r2(clean, 60)
    _, noisy_r2 = log_slope_r2(noisy, 60)
    assert clean_r2 > noisy_r2


def test_downtrend_has_negative_slope():
    close = exponential_trend(60, daily_rate=-0.002)
    slope, r2 = log_slope_r2(close, 60)
    assert slope < 0
    assert r2 > 0.999


def test_flat_series_has_zero_slope():
    slope, r2 = log_slope_r2(flat_series(60), 60)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_slope_uses_only_the_last_window_bars():
    close = exponential_trend(200, 0.002)
    full = log_slope_r2(close, 60)
    tail = log_slope_r2(close.tail(60), 60)
    assert full == pytest.approx(tail)


def test_adx_is_higher_for_trending_than_choppy():
    trending = make_ohlcv(exponential_trend(100, 0.004))
    choppy = make_ohlcv(flat_series(100) + np.tile([1.0, -1.0], 50))
    assert adx(trending) > adx(choppy)


def test_ma_stack_positive_when_price_above_rising_mas():
    close = exponential_trend(150, 0.003)
    assert ma_stack_score(close, 20, 60) > 0.5


def test_ma_stack_negative_in_downtrend():
    close = exponential_trend(150, -0.003)
    assert ma_stack_score(close, 20, 60) < -0.5


def test_uptrend_scores_strongly_positive():
    frame = make_ohlcv(exponential_trend(150, 0.003))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert result.score > 50
    assert result.r2 > 0.9


def test_downtrend_scores_strongly_negative():
    frame = make_ohlcv(exponential_trend(150, -0.003))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert result.score < -50


def test_flat_series_scores_near_zero():
    frame = make_ohlcv(flat_series(150))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert abs(result.score) < 15


def test_score_is_bounded():
    frame = make_ohlcv(exponential_trend(150, 0.05))
    result = trend_score(frame, window=60, weights=WEIGHTS)
    assert -100.0 <= result.score <= 100.0


def test_trend_table_reports_both_windows():
    frames = {
        "UP": make_ohlcv(exponential_trend(150, 0.003)),
        "DOWN": make_ohlcv(exponential_trend(150, -0.003)),
    }
    table = trend_table(frames, WINDOWS, WEIGHTS).set_index("ticker")
    assert list(table.columns) == [
        "short_score", "mid_score", "short_r2", "mid_r2", "adx", "ma_stack",
    ]
    assert table.loc["UP", "mid_score"] > 0
    assert table.loc["DOWN", "mid_score"] < 0


def test_trend_table_skips_tickers_with_insufficient_bars():
    frames = {
        "OK": make_ohlcv(exponential_trend(150, 0.003)),
        "SHORT": make_ohlcv(exponential_trend(10, 0.003)),
    }
    table = trend_table(frames, WINDOWS, WEIGHTS)
    assert list(table["ticker"]) == ["OK"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_trend.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.features'`.

- [ ] **Step 3: Create `src/screener_sector/features/__init__.py`**

Empty file.

- [ ] **Step 4: Implement `src/screener_sector/features/trend.py`**

```python
"""Trend strength and trend quality.

The R-squared term is what separates a clean advance from a drift with the
same net move: two names can have identical 60-day returns while one trends
smoothly and the other whipsaws. Only the first is tradeable as a trend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener_sector.config import TrendWeights, Windows

TRADING_DAYS_PER_YEAR = 252


def log_slope_r2(close: pd.Series, window: int) -> tuple[float, float]:
    """Annualized slope of log price and the fit's R-squared over the last
    `window` bars. Returns (0.0, 0.0) when the series is too short or flat."""
    tail = close.dropna().tail(window)
    if len(tail) < 3:
        return 0.0, 0.0
    y = np.log(tail.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)
    if np.allclose(y, y[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope) * TRADING_DAYS_PER_YEAR, float(max(r2, 0.0))


def adx(ohlcv: pd.DataFrame, window: int = 14) -> float:
    """Average Directional Index over the final `window` bars, 0-100."""
    frame = ohlcv.dropna(subset=["high", "low", "close"])
    if len(frame) < window * 2 + 1:
        return 0.0

    high, low, close = frame["high"], frame["low"], frame["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr

    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    value = dx.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    return 0.0 if pd.isna(value) else float(value)


def ma_stack_score(close: pd.Series, short: int, mid: int) -> float:
    """+1 when price > short MA > mid MA and both are rising; -1 when fully
    inverted. Intermediate configurations land in between."""
    series = close.dropna()
    if len(series) < mid + 5:
        return 0.0
    short_ma = series.rolling(short).mean()
    mid_ma = series.rolling(mid).mean()
    checks = [
        series.iloc[-1] > short_ma.iloc[-1],
        short_ma.iloc[-1] > mid_ma.iloc[-1],
        short_ma.iloc[-1] > short_ma.iloc[-5],
        mid_ma.iloc[-1] > mid_ma.iloc[-5],
    ]
    positives = sum(1 for check in checks if bool(check))
    return (positives / len(checks)) * 2.0 - 1.0


@dataclass(frozen=True)
class TrendResult:
    slope: float
    r2: float
    adx: float
    ma_stack: float
    score: float


def trend_score(
    ohlcv: pd.DataFrame, window: int, weights: TrendWeights
) -> TrendResult:
    close = ohlcv["close"]
    slope, r2 = log_slope_r2(close, window)

    returns = np.log(close).diff().dropna().tail(window)
    volatility = float(returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    normalized = 0.0 if volatility == 0 else slope / volatility
    slope_component = float(np.tanh(normalized))

    direction = np.sign(slope_component)
    adx_value = adx(ohlcv)
    stack = ma_stack_score(close, max(window // 3, 2), window)

    score = 100.0 * (
        weights.slope * slope_component
        + weights.r2 * r2 * direction
        + weights.adx * min(adx_value / 50.0, 1.0) * direction
        + weights.ma_stack * stack
    )
    return TrendResult(
        slope=slope,
        r2=r2,
        adx=adx_value,
        ma_stack=stack,
        score=float(np.clip(score, -100.0, 100.0)),
    )


def trend_table(
    frames: dict[str, pd.DataFrame], windows: Windows, weights: TrendWeights
) -> pd.DataFrame:
    rows = []
    for ticker, frame in frames.items():
        if len(frame.dropna(subset=["close"])) < windows.mid + 5:
            continue
        short = trend_score(frame, windows.short, weights)
        mid = trend_score(frame, windows.mid, weights)
        rows.append(
            {
                "ticker": ticker,
                "short_score": short.score,
                "mid_score": mid.score,
                "short_r2": short.r2,
                "mid_r2": mid.r2,
                "adx": mid.adx,
                "ma_stack": mid.ma_stack,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker", "short_score", "mid_score", "short_r2", "mid_r2",
            "adx", "ma_stack",
        ],
    )
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_trend.py -v
```

Expected: 14 passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: trend composite score with R-squared quality term"
```

---

### Task 12: Correlation, residualization, and clustering

**Files:**
- Create: `src/screener_sector/features/correlation.py`
- Test: `tests/test_correlation.py`

**Interfaces:**
- Consumes: `Windows` (Task 3).
- Produces: `log_returns(panel: pd.DataFrame) -> pd.DataFrame`; `correlation_matrix(returns: pd.DataFrame, min_overlap: int = 30) -> pd.DataFrame`; `residualize(returns: pd.DataFrame, factor: pd.Series) -> pd.DataFrame`; `correlation_distance(corr: pd.DataFrame) -> np.ndarray` (condensed form); `Cluster` frozen dataclass with `label: int`, `members: tuple[str, ...]`, `mean_correlation: float`; `ClusterResult` frozen dataclass with `clusters: tuple[Cluster, ...]`, `assignments: pd.Series` (ticker → label, `-1` for unclustered), `raw_corr: pd.DataFrame`, `residual_corr: pd.DataFrame`; `cluster_universe(panel, benchmark: pd.Series | None, threshold: float, min_size: int, window: int) -> ClusterResult`.

**Why returns, not prices:** correlating price levels while everything trends together produces near-1.0 correlations that carry no information. Correlating log returns measures co-movement. Residualizing against the benchmark then separates "moves with the sector" from "moves with *this* group specifically."

- [ ] **Step 1: Write the failing test**

Create `tests/test_correlation.py`:

```python
import numpy as np
import pandas as pd
import pytest

from conftest import correlated_returns, exponential_trend, trading_days
from screener_sector.features.correlation import (
    cluster_universe,
    correlation_distance,
    correlation_matrix,
    log_returns,
    residualize,
)


def panel_from_returns(columns: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    return 100.0 * np.exp(frame.cumsum())


def test_log_returns_shape_and_first_row_dropped():
    prices = pd.DataFrame(
        {"A": exponential_trend(50, 0.001), "B": exponential_trend(50, 0.002)}
    )
    returns = log_returns(prices)
    assert len(returns) == 49
    assert list(returns.columns) == ["A", "B"]


def test_correlation_recovers_known_rho():
    a, b = correlated_returns(3000, rho=0.75, seed=11)
    corr = correlation_matrix(pd.DataFrame({"A": a, "B": b}))
    assert corr.loc["A", "B"] == pytest.approx(0.75, abs=0.05)


def test_correlation_of_prices_would_differ_from_returns():
    """Two independent uptrends have near-zero return correlation but very
    high price correlation. This is why the pipeline uses returns."""
    a = exponential_trend(500, 0.001, noise=0.02, seed=1)
    b = exponential_trend(500, 0.001, noise=0.02, seed=2)
    prices = pd.DataFrame({"A": a, "B": b})
    price_corr = prices.corr().loc["A", "B"]
    return_corr = correlation_matrix(log_returns(prices)).loc["A", "B"]
    assert price_corr > 0.9
    assert abs(return_corr) < 0.2


def test_correlation_requires_minimum_overlap():
    idx = trading_days(100)
    a = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
    b = a.copy()
    b.iloc[:90] = np.nan
    corr = correlation_matrix(pd.DataFrame({"A": a, "B": b}), min_overlap=30)
    assert np.isnan(corr.loc["A", "B"])


def test_residualize_removes_common_factor():
    rng = np.random.default_rng(3)
    idx = trading_days(1000)
    factor = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    a = 1.2 * factor + pd.Series(rng.normal(0, 0.002, 1000), index=idx)
    b = 0.8 * factor + pd.Series(rng.normal(0, 0.002, 1000), index=idx)
    returns = pd.DataFrame({"A": a, "B": b})

    before = correlation_matrix(returns).loc["A", "B"]
    after = correlation_matrix(residualize(returns, factor)).loc["A", "B"]
    assert before > 0.9
    assert abs(after) < 0.2


def test_correlation_distance_is_zero_for_perfect_correlation():
    corr = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert correlation_distance(corr)[0] == pytest.approx(0.0)


def test_correlation_distance_is_two_for_perfect_anticorrelation():
    corr = pd.DataFrame(
        [[1.0, -1.0], [-1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    assert correlation_distance(corr)[0] == pytest.approx(2.0)


def test_clustering_recovers_three_synthetic_blocks():
    rng = np.random.default_rng(9)
    idx = trading_days(600)
    columns: dict[str, pd.Series] = {}
    for block in range(3):
        driver = rng.normal(0, 0.012, 600)
        for member in range(4):
            noise = rng.normal(0, 0.003, 600)
            columns[f"B{block}_{member}"] = pd.Series(driver + noise, index=idx)

    result = cluster_universe(
        panel_from_returns(columns),
        benchmark=None,
        threshold=0.6,
        min_size=3,
        window=600,
    )
    assert len(result.clusters) == 3
    for cluster in result.clusters:
        prefixes = {name.split("_")[0] for name in cluster.members}
        assert len(prefixes) == 1
        assert cluster.mean_correlation > 0.6


def test_clustering_drops_groups_below_min_size():
    rng = np.random.default_rng(4)
    idx = trading_days(600)
    driver = rng.normal(0, 0.012, 600)
    columns = {
        "A": pd.Series(driver + rng.normal(0, 0.002, 600), index=idx),
        "B": pd.Series(driver + rng.normal(0, 0.002, 600), index=idx),
        "LONER": pd.Series(rng.normal(0, 0.012, 600), index=idx),
    }
    result = cluster_universe(
        panel_from_returns(columns), None, threshold=0.6, min_size=3, window=600
    )
    assert result.clusters == ()
    assert set(result.assignments) == {-1}


def test_cluster_result_exposes_both_matrices():
    rng = np.random.default_rng(5)
    idx = trading_days(400)
    factor = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    columns = {
        f"T{i}": pd.Series(factor + rng.normal(0, 0.003, 400), index=idx)
        for i in range(4)
    }
    panel = panel_from_returns(columns)
    benchmark = 100.0 * np.exp(factor.cumsum())
    result = cluster_universe(panel, benchmark, threshold=0.6, min_size=3, window=400)
    assert result.raw_corr.shape == (4, 4)
    assert result.residual_corr.shape == (4, 4)
    raw_mean = result.raw_corr.to_numpy()[np.triu_indices(4, 1)].mean()
    residual_mean = result.residual_corr.to_numpy()[np.triu_indices(4, 1)].mean()
    assert raw_mean > residual_mean
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_correlation.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/features/correlation.py`**

```python
"""Return correlation, benchmark residualization, and hierarchical clustering.

Correlating price levels is meaningless when every name is trending: the
result is near 1.0 for unrelated stocks. Everything here works on log returns.
Residualizing against the sector benchmark then answers the sharper question:
which names move together for reasons beyond shared sector beta?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log(panel.astype(float)).diff().iloc[1:]


def correlation_matrix(returns: pd.DataFrame, min_overlap: int = 30) -> pd.DataFrame:
    corr = returns.corr(min_periods=min_overlap)
    return corr.reindex(index=returns.columns, columns=returns.columns)


def residualize(returns: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Strip the benchmark factor from each column via OLS, keeping residuals."""
    aligned_factor = factor.reindex(returns.index)
    out: dict[str, pd.Series] = {}
    for column in returns.columns:
        pair = pd.concat([returns[column], aligned_factor], axis=1).dropna()
        if len(pair) < 30 or pair.iloc[:, 1].std() == 0:
            out[column] = returns[column]
            continue
        y = pair.iloc[:, 0].to_numpy(dtype=float)
        x = pair.iloc[:, 1].to_numpy(dtype=float)
        beta, alpha = np.polyfit(x, y, 1)
        residual = pd.Series(y - (beta * x + alpha), index=pair.index)
        out[column] = residual.reindex(returns.index)
    return pd.DataFrame(out, index=returns.index)


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Condensed distance vector from a correlation matrix.

    d = sqrt(2 * (1 - rho)): 0 for perfectly correlated, 2 for perfectly
    anticorrelated. NaN correlations become maximum distance so that names
    without enough overlap never merge into a cluster.
    """
    filled = corr.to_numpy(dtype=float).copy()
    filled[np.isnan(filled)] = -1.0
    np.fill_diagonal(filled, 1.0)
    distance = np.sqrt(2.0 * (1.0 - np.clip(filled, -1.0, 1.0)))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return squareform(distance, checks=False)


@dataclass(frozen=True)
class Cluster:
    label: int
    members: tuple[str, ...]
    mean_correlation: float


@dataclass(frozen=True)
class ClusterResult:
    clusters: tuple[Cluster, ...]
    assignments: pd.Series
    raw_corr: pd.DataFrame
    residual_corr: pd.DataFrame


def _mean_pairwise(corr: pd.DataFrame, members: list[str]) -> float:
    block = corr.loc[members, members].to_numpy(dtype=float)
    upper = block[np.triu_indices(len(members), 1)]
    upper = upper[~np.isnan(upper)]
    return float(upper.mean()) if upper.size else float("nan")


def cluster_universe(
    panel: pd.DataFrame,
    benchmark: pd.Series | None,
    threshold: float,
    min_size: int,
    window: int,
) -> ClusterResult:
    returns = log_returns(panel).tail(window)
    raw_corr = correlation_matrix(returns)

    if benchmark is not None:
        factor = log_returns(benchmark.to_frame("bm")).tail(window)["bm"]
        residual_returns = residualize(returns, factor)
    else:
        residual_returns = returns
    residual_corr = correlation_matrix(residual_returns)

    tickers = list(returns.columns)
    assignments = pd.Series(-1, index=tickers, dtype=int)

    if len(tickers) < min_size:
        return ClusterResult((), assignments, raw_corr, residual_corr)

    distance = correlation_distance(residual_corr)
    tree = linkage(distance, method="average")
    cut = np.sqrt(2.0 * (1.0 - threshold))
    labels = fcluster(tree, t=cut, criterion="distance")

    clusters: list[Cluster] = []
    next_label = 0
    for raw_label in sorted(set(labels)):
        members = [t for t, lab in zip(tickers, labels) if lab == raw_label]
        if len(members) < min_size:
            continue
        mean_corr = _mean_pairwise(residual_corr, members)
        if not np.isfinite(mean_corr) or mean_corr < threshold:
            continue
        clusters.append(Cluster(next_label, tuple(members), mean_corr))
        assignments.loc[members] = next_label
        next_label += 1

    return ClusterResult(tuple(clusters), assignments, raw_corr, residual_corr)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_correlation.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: return correlation with residualization and clustering"
```

---

### Task 13: Relative strength via up/down capture

**Files:**
- Create: `src/screener_sector/features/strength.py`
- Test: `tests/test_strength.py`

**Interfaces:**
- Consumes: `Cluster`, `log_returns` (Task 12).
- Produces: `group_return(returns: pd.DataFrame, members: Sequence[str]) -> pd.Series` (equal-weight daily mean); `Capture` frozen dataclass with `up: float`, `down: float`, `up_days: int`, `down_days: int`; `capture_ratios(ticker_returns: pd.Series, group_returns: pd.Series) -> Capture`; `max_drawdown(close: pd.Series) -> float` (negative fraction); `recovery_days(close: pd.Series) -> int` (bars from the trough back to the prior peak, `-1` if never recovered); `strength_table(panel: pd.DataFrame, clusters: Sequence[Cluster], window: int) -> pd.DataFrame` with columns `ticker, cluster, up_capture, down_capture, capture_spread, max_drawdown, recovery_days, rank_in_cluster`.

**Reading the output:** `down_capture` below 1.0 means the name falls less than its group; `up_capture` above 1.0 means it rises more. `capture_spread = up_capture - down_capture` is the single ranking number, and `rank_in_cluster` is 1 for the leader.

- [ ] **Step 1: Write the failing test**

Create `tests/test_strength.py`:

```python
import numpy as np
import pandas as pd
import pytest

from conftest import trading_days
from screener_sector.features.correlation import Cluster
from screener_sector.features.strength import (
    capture_ratios,
    group_return,
    max_drawdown,
    recovery_days,
    strength_table,
)


def panel_from_returns(columns: dict[str, pd.Series]) -> pd.DataFrame:
    return 100.0 * np.exp(pd.DataFrame(columns).cumsum())


def test_group_return_is_equal_weight_mean():
    idx = trading_days(5)
    returns = pd.DataFrame(
        {"A": pd.Series([0.02] * 5, index=idx), "B": pd.Series([0.00] * 5, index=idx)}
    )
    assert group_return(returns, ["A", "B"]).iloc[0] == pytest.approx(0.01)


def test_group_return_ignores_missing_members():
    idx = trading_days(3)
    returns = pd.DataFrame(
        {"A": pd.Series([0.02, 0.02, 0.02], index=idx),
         "B": pd.Series([np.nan, 0.00, 0.00], index=idx)}
    )
    assert group_return(returns, ["A", "B"]).iloc[0] == pytest.approx(0.02)


def test_defensive_name_has_low_down_capture():
    idx = trading_days(200)
    rng = np.random.default_rng(1)
    group = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
    # falls half as much on down days, matches on up days
    ticker = group.where(group > 0, group * 0.5)
    capture = capture_ratios(ticker, group)
    assert capture.down < 0.7
    assert capture.up == pytest.approx(1.0, abs=0.05)


def test_aggressive_name_has_high_up_capture():
    idx = trading_days(200)
    rng = np.random.default_rng(2)
    group = pd.Series(rng.normal(0.0, 0.02, 200), index=idx)
    ticker = group.where(group < 0, group * 1.8)
    capture = capture_ratios(ticker, group)
    assert capture.up > 1.5
    assert capture.down == pytest.approx(1.0, abs=0.05)


def test_capture_counts_up_and_down_days():
    idx = trading_days(10)
    group = pd.Series([0.01] * 6 + [-0.01] * 4, index=idx)
    capture = capture_ratios(group.copy(), group)
    assert capture.up_days == 6
    assert capture.down_days == 4


def test_capture_is_nan_when_no_down_days():
    idx = trading_days(5)
    group = pd.Series([0.01] * 5, index=idx)
    assert np.isnan(capture_ratios(group.copy(), group).down)


def test_max_drawdown_matches_hand_computation():
    close = pd.Series([100.0, 120.0, 60.0, 90.0], index=trading_days(4))
    assert max_drawdown(close) == pytest.approx(-0.5)


def test_recovery_days_counts_bars_back_to_prior_peak():
    close = pd.Series([100.0, 80.0, 90.0, 101.0, 105.0], index=trading_days(5))
    assert recovery_days(close) == 2


def test_recovery_days_is_minus_one_when_never_recovered():
    close = pd.Series([100.0, 80.0, 85.0], index=trading_days(3))
    assert recovery_days(close) == -1


def test_strength_table_ranks_leader_first():
    idx = trading_days(250)
    rng = np.random.default_rng(6)
    driver = pd.Series(rng.normal(0.0, 0.02, 250), index=idx)
    columns = {
        "LEADER": driver.where(driver < 0, driver * 1.5).where(driver > 0, driver * 0.5),
        "LAGGARD": driver.where(driver > 0, driver * 1.5).where(driver < 0, driver * 0.5),
        "MIDDLE": driver.copy(),
    }
    clusters = [Cluster(0, ("LEADER", "LAGGARD", "MIDDLE"), 0.9)]
    table = strength_table(panel_from_returns(columns), clusters, window=250)
    table = table.set_index("ticker")
    assert list(table.columns) == [
        "cluster", "up_capture", "down_capture", "capture_spread",
        "max_drawdown", "recovery_days", "rank_in_cluster",
    ]
    assert table.loc["LEADER", "rank_in_cluster"] == 1
    assert table.loc["LAGGARD", "rank_in_cluster"] == 3
    assert table.loc["LEADER", "down_capture"] < table.loc["LAGGARD", "down_capture"]


def test_strength_table_is_empty_without_clusters():
    idx = trading_days(100)
    columns = {"A": pd.Series(np.zeros(100), index=idx)}
    table = strength_table(panel_from_returns(columns), [], window=100)
    assert table.empty
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_strength.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/features/strength.py`**

```python
"""Relative strength within a correlated group.

Splitting days by the group's own direction answers the question directly:
which member falls least when the group falls, and rises most when it rises?
A single blended relative-strength number cannot separate those two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener_sector.features.correlation import Cluster, log_returns


def group_return(returns: pd.DataFrame, members: Sequence[str]) -> pd.Series:
    present = [m for m in members if m in returns.columns]
    return returns[present].mean(axis=1, skipna=True)


@dataclass(frozen=True)
class Capture:
    up: float
    down: float
    up_days: int
    down_days: int


def capture_ratios(
    ticker_returns: pd.Series, group_returns: pd.Series
) -> Capture:
    pair = pd.concat(
        [ticker_returns.rename("t"), group_returns.rename("g")], axis=1
    ).dropna()
    up_mask = pair["g"] > 0
    down_mask = pair["g"] < 0

    def ratio(mask: pd.Series) -> float:
        if not mask.any():
            return float("nan")
        denominator = pair.loc[mask, "g"].mean()
        if denominator == 0:
            return float("nan")
        return float(pair.loc[mask, "t"].mean() / denominator)

    return Capture(
        up=ratio(up_mask),
        down=ratio(down_mask),
        up_days=int(up_mask.sum()),
        down_days=int(down_mask.sum()),
    )


def max_drawdown(close: pd.Series) -> float:
    series = close.dropna()
    if series.empty:
        return 0.0
    running_peak = series.cummax()
    return float((series / running_peak - 1.0).min())


def recovery_days(close: pd.Series) -> int:
    """Bars from the deepest trough back to the peak that preceded it."""
    series = close.dropna()
    if series.empty:
        return -1
    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    trough_position = int(drawdown.to_numpy().argmin())
    peak_level = float(running_peak.iloc[trough_position])
    after = series.iloc[trough_position:]
    recovered = after[after >= peak_level]
    if recovered.empty:
        return -1
    return int(series.index.get_loc(recovered.index[0]) - trough_position)


STRENGTH_COLUMNS = [
    "ticker",
    "cluster",
    "up_capture",
    "down_capture",
    "capture_spread",
    "max_drawdown",
    "recovery_days",
    "rank_in_cluster",
]


def strength_table(
    panel: pd.DataFrame, clusters: Sequence[Cluster], window: int
) -> pd.DataFrame:
    if not clusters:
        return pd.DataFrame(columns=STRENGTH_COLUMNS)

    returns = log_returns(panel).tail(window)
    prices = panel.tail(window)
    frames: list[pd.DataFrame] = []

    for cluster in clusters:
        members = [m for m in cluster.members if m in returns.columns]
        if not members:
            continue
        benchmark = group_return(returns, members)
        rows = []
        for ticker in members:
            capture = capture_ratios(returns[ticker], benchmark)
            spread = capture.up - capture.down
            rows.append(
                {
                    "ticker": ticker,
                    "cluster": cluster.label,
                    "up_capture": capture.up,
                    "down_capture": capture.down,
                    "capture_spread": spread,
                    "max_drawdown": max_drawdown(prices[ticker]),
                    "recovery_days": recovery_days(prices[ticker]),
                }
            )
        block = pd.DataFrame(rows)
        block["rank_in_cluster"] = (
            block["capture_spread"].rank(ascending=False, method="min").astype(int)
        )
        frames.append(block)

    if not frames:
        return pd.DataFrame(columns=STRENGTH_COLUMNS)
    return pd.concat(frames, ignore_index=True)[STRENGTH_COLUMNS]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_strength.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: relative strength via up/down capture ratios"
```

---

### Task 14: Rebound alarm

**Files:**
- Create: `src/screener_sector/features/rebound.py`
- Test: `tests/test_rebound.py`

**Interfaces:**
- Consumes: `ReboundWeights` (Task 3), `Cluster`, `log_returns` (Task 12).
- Produces: `rsi(close: pd.Series, window: int = 14) -> pd.Series`; `williams_r(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series`; `stretch_z(close: pd.Series, window: int) -> pd.Series` (negative = below mean); `volume_signal(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series` in `[0, 1]`; `bullish_divergence(close: pd.Series, oscillator: pd.Series, lookback: int = 20) -> pd.Series` (bool); `confirmation(ohlcv: pd.DataFrame, short_window: int) -> pd.Series` (bool); `cluster_washout(panel: pd.DataFrame, members: Sequence[str], window: int) -> pd.Series` in `[0, 1]`; `ticker_alarm(ohlcv, washout: pd.Series, weights: ReboundWeights, windows: Windows) -> pd.Series` in `[0, 100]`; `rebound_table(panel, frames: dict[str, pd.DataFrame], clusters, weights, windows, as_of: pd.Timestamp | None = None) -> pd.DataFrame` with columns `ticker, cluster, alarm, washout, stretch_z, rsi, volume, divergence, confirmed, fired`.

**Gating rule:** `fired` is True only when the *cluster* washout on that date exceeds 0.5 **and** the ticker's alarm exceeds 60 **and** `confirmed` is True. This is what turns forty uncorrelated pings into one group-level call.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rebound.py`:

```python
import numpy as np
import pandas as pd
import pytest

from conftest import (
    exponential_trend,
    flat_series,
    make_ohlcv,
    trading_days,
    v_bottom,
)
from screener_sector.config import ReboundWeights, Windows
from screener_sector.features.correlation import Cluster
from screener_sector.features.rebound import (
    bullish_divergence,
    cluster_washout,
    confirmation,
    rebound_table,
    rsi,
    stretch_z,
    ticker_alarm,
    volume_signal,
    williams_r,
)

WEIGHTS = ReboundWeights(
    breadth=0.25, stretch=0.20, oscillator=0.25, volume=0.15, confirmation=0.15
)
WINDOWS = Windows(short=20, mid=60, corr=120)


def test_rsi_is_bounded():
    values = rsi(exponential_trend(200, 0.001, noise=0.03, seed=1)).dropna()
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_rsi_is_high_in_uptrend_and_low_in_downtrend():
    up = rsi(exponential_trend(100, 0.004)).iloc[-1]
    down = rsi(exponential_trend(100, -0.004)).iloc[-1]
    assert up > 70
    assert down < 30


def test_williams_r_is_negative_bounded():
    values = williams_r(make_ohlcv(exponential_trend(100, 0.002))).dropna()
    assert values.min() >= -100.0
    assert values.max() <= 0.0


def test_stretch_z_is_negative_below_mean():
    close = pd.concat([flat_series(80), pd.Series([70.0] * 5)], ignore_index=True)
    close.index = trading_days(85)
    assert stretch_z(close, 60).iloc[-1] < 0


def test_stretch_z_is_zero_for_flat_series():
    assert stretch_z(flat_series(100), 60).iloc[-1] == pytest.approx(0.0)


def test_volume_signal_rewards_spike_then_dryup():
    idx = trading_days(60)
    volume = pd.Series([1_000_000.0] * 60, index=idx)
    volume.iloc[-6] = 6_000_000.0           # capitulation spike
    volume.iloc[-5:] = 400_000.0            # dry-up
    frame = make_ohlcv(flat_series(60), volume)
    assert volume_signal(frame).iloc[-1] > 0.6


def test_volume_signal_is_low_for_constant_volume():
    frame = make_ohlcv(flat_series(60))
    assert volume_signal(frame).iloc[-1] < 0.4


def test_bullish_divergence_detects_lower_low_with_higher_oscillator():
    idx = trading_days(60)
    close = pd.Series(
        np.concatenate([np.linspace(100, 80, 30), np.linspace(80, 78, 30)]), index=idx
    )
    oscillator = pd.Series(
        np.concatenate([np.linspace(50, 20, 30), np.linspace(20, 35, 30)]), index=idx
    )
    assert bool(bullish_divergence(close, oscillator, lookback=20).iloc[-1])


def test_no_divergence_in_clean_downtrend():
    idx = trading_days(60)
    close = pd.Series(np.linspace(100, 60, 60), index=idx)
    oscillator = pd.Series(np.linspace(60, 15, 60), index=idx)
    assert not bool(bullish_divergence(close, oscillator, lookback=20).iloc[-1])


def test_confirmation_fires_on_close_above_prior_high():
    close = pd.Series(
        list(np.linspace(100, 80, 40)) + [88.0], index=trading_days(41)
    )
    frame = make_ohlcv(close)
    assert bool(confirmation(frame, short_window=20).iloc[-1])


def test_confirmation_does_not_fire_mid_decline():
    frame = make_ohlcv(pd.Series(np.linspace(100, 80, 41), index=trading_days(41)))
    assert not bool(confirmation(frame, short_window=20).iloc[-1])


def test_cluster_washout_is_high_when_all_members_oversold():
    idx = trading_days(120)
    declining = pd.Series(np.linspace(100, 60, 120), index=idx)
    panel = pd.DataFrame({f"T{i}": declining * (1 + 0.01 * i) for i in range(4)})
    washout = cluster_washout(panel, [f"T{i}" for i in range(4)], window=60)
    assert washout.iloc[-1] > 0.8


def test_cluster_washout_is_low_when_members_are_strong():
    idx = trading_days(120)
    rising = pd.Series(np.linspace(60, 100, 120), index=idx)
    panel = pd.DataFrame({f"T{i}": rising * (1 + 0.01 * i) for i in range(4)})
    washout = cluster_washout(panel, [f"T{i}" for i in range(4)], window=60)
    assert washout.iloc[-1] < 0.2


def test_alarm_fires_near_a_v_bottom():
    close = v_bottom(80, 40, depth=0.35)
    frame = make_ohlcv(close)
    washout = pd.Series(1.0, index=close.index)
    alarm = ticker_alarm(frame, washout, WEIGHTS, WINDOWS)
    trough = close.idxmin()
    trough_position = close.index.get_loc(trough)
    nearby = alarm.iloc[trough_position : trough_position + 10]
    assert nearby.max() > 60


def test_alarm_stays_low_on_flat_series():
    frame = make_ohlcv(flat_series(200))
    washout = pd.Series(0.0, index=frame.index)
    alarm = ticker_alarm(frame, washout, WEIGHTS, WINDOWS)
    assert alarm.dropna().max() < 40


def test_alarm_stays_low_in_steady_uptrend():
    frame = make_ohlcv(exponential_trend(200, 0.002))
    washout = pd.Series(0.0, index=frame.index)
    alarm = ticker_alarm(frame, washout, WEIGHTS, WINDOWS)
    assert alarm.dropna().max() < 50


def test_rebound_table_gates_on_cluster_washout():
    close = v_bottom(80, 40, depth=0.35)
    frames = {f"T{i}": make_ohlcv(close * (1 + 0.01 * i)) for i in range(3)}
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    clusters = [Cluster(0, tuple(frames), 0.95)]

    as_of = close.index[80]  # one bar past the trough
    table = rebound_table(panel, frames, clusters, WEIGHTS, WINDOWS, as_of=as_of)
    assert list(table.columns) == [
        "ticker", "cluster", "alarm", "washout", "stretch_z", "rsi",
        "volume", "divergence", "confirmed", "fired",
    ]
    assert table["washout"].max() > 0.5


def test_rebound_table_fires_nothing_in_uptrend():
    frames = {
        f"T{i}": make_ohlcv(exponential_trend(200, 0.002) * (1 + 0.01 * i))
        for i in range(3)
    }
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    clusters = [Cluster(0, tuple(frames), 0.95)]
    table = rebound_table(panel, frames, clusters, WEIGHTS, WINDOWS)
    assert not table["fired"].any()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_rebound.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/features/rebound.py`**

```python
"""Rebound alarm: group washout first, individual confirmation second.

Oversold oscillators fire constantly and mean little on their own. Requiring
the whole correlated group to be washed out, then requiring a confirmation bar
on the individual name, is what makes the signal selective enough to act on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from screener_sector.config import ReboundWeights, Windows
from screener_sector.features.correlation import Cluster

REBOUND_COLUMNS = [
    "ticker",
    "cluster",
    "alarm",
    "washout",
    "stretch_z",
    "rsi",
    "volume",
    "divergence",
    "confirmed",
    "fired",
]

WASHOUT_GATE = 0.5
ALARM_GATE = 60.0


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def williams_r(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    highest = ohlcv["high"].rolling(window).max()
    lowest = ohlcv["low"].rolling(window).min()
    span = (highest - lowest).replace(0.0, np.nan)
    return -100.0 * (highest - ohlcv["close"]) / span


def stretch_z(close: pd.Series, window: int) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ((close - mean) / std.replace(0.0, np.nan)).fillna(0.0)


def volume_signal(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
    """Capitulation spike followed by dry-up, scored 0-1.

    A high reading means volume blew out recently and has since gone quiet,
    which is the classic seller-exhaustion pattern.
    """
    volume = ohlcv["volume"]
    baseline = volume.rolling(window).median()
    relative = volume / baseline.replace(0.0, np.nan)
    recent_spike = relative.rolling(10).max()
    current_dryness = 1.0 / relative.replace(0.0, np.nan)
    spike_component = ((recent_spike - 2.0) / 3.0).clip(0.0, 1.0)
    dry_component = ((current_dryness - 1.0) / 1.5).clip(0.0, 1.0)
    return (0.6 * spike_component + 0.4 * dry_component).fillna(0.0)


def bullish_divergence(
    close: pd.Series, oscillator: pd.Series, lookback: int = 20
) -> pd.Series:
    """Price makes a lower low over `lookback` while the oscillator does not."""
    price_low = close.rolling(lookback).min()
    prior_price_low = price_low.shift(lookback)
    osc_at_low = oscillator.rolling(lookback).min()
    prior_osc_low = osc_at_low.shift(lookback)
    lower_price = price_low < prior_price_low
    higher_osc = osc_at_low > prior_osc_low
    return (lower_price & higher_osc).fillna(False)


def confirmation(ohlcv: pd.DataFrame, short_window: int) -> pd.Series:
    """Close above the prior bar's high, or a reclaim of the short MA."""
    above_prior_high = ohlcv["close"] > ohlcv["high"].shift(1)
    short_ma = ohlcv["close"].rolling(short_window).mean()
    reclaim = (ohlcv["close"] > short_ma) & (
        ohlcv["close"].shift(1) <= short_ma.shift(1)
    )
    return (above_prior_high | reclaim).fillna(False)


def cluster_washout(
    panel: pd.DataFrame, members: Sequence[str], window: int
) -> pd.Series:
    """Fraction of the group that is both oversold and below its mid-window
    mean, on each date."""
    present = [m for m in members if m in panel.columns]
    if not present:
        return pd.Series(0.0, index=panel.index)
    flags = []
    for ticker in present:
        close = panel[ticker]
        oversold = rsi(close) < 35.0
        below_mean = close < close.rolling(window).mean()
        flags.append((oversold & below_mean).astype(float))
    return pd.concat(flags, axis=1).mean(axis=1).fillna(0.0)


def ticker_alarm(
    ohlcv: pd.DataFrame,
    washout: pd.Series,
    weights: ReboundWeights,
    windows: Windows,
) -> pd.Series:
    close = ohlcv["close"]
    oscillator = rsi(close)

    breadth_component = washout.reindex(close.index).fillna(0.0).clip(0.0, 1.0)
    # stretch: -2 sigma or lower scores 1.0, at or above the mean scores 0.0
    stretch_component = (-stretch_z(close, windows.mid) / 2.0).clip(0.0, 1.0)
    # oscillator: RSI 20 or lower scores 1.0, RSI 50 or higher scores 0.0
    oscillator_component = ((50.0 - oscillator) / 30.0).clip(0.0, 1.0).fillna(0.0)
    volume_component = volume_signal(ohlcv)
    divergence_flag = bullish_divergence(close, oscillator, windows.short)
    confirm_flag = confirmation(ohlcv, windows.short)
    confirmation_component = (
        0.5 * confirm_flag.astype(float) + 0.5 * divergence_flag.astype(float)
    )

    score = 100.0 * (
        weights.breadth * breadth_component
        + weights.stretch * stretch_component
        + weights.oscillator * oscillator_component
        + weights.volume * volume_component
        + weights.confirmation * confirmation_component
    )
    return score.clip(0.0, 100.0)


def rebound_table(
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    clusters: Sequence[Cluster],
    weights: ReboundWeights,
    windows: Windows,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for cluster in clusters:
        members = [m for m in cluster.members if m in frames]
        if not members:
            continue
        washout = cluster_washout(panel, members, windows.mid)

        for ticker in members:
            frame = frames[ticker]
            alarm = ticker_alarm(frame, washout, weights, windows)
            close = frame["close"]
            oscillator = rsi(close)
            divergence = bullish_divergence(close, oscillator, windows.short)
            confirmed = confirmation(frame, windows.short)
            stretch = stretch_z(close, windows.mid)
            volume = volume_signal(frame)

            stamp = as_of if as_of is not None else frame.index[-1]
            if stamp not in frame.index:
                continue

            washout_value = float(washout.reindex(frame.index).fillna(0.0).loc[stamp])
            alarm_value = float(alarm.loc[stamp])
            confirmed_value = bool(confirmed.loc[stamp])

            rows.append(
                {
                    "ticker": ticker,
                    "cluster": cluster.label,
                    "alarm": alarm_value,
                    "washout": washout_value,
                    "stretch_z": float(stretch.loc[stamp]),
                    "rsi": float(oscillator.loc[stamp]),
                    "volume": float(volume.loc[stamp]),
                    "divergence": bool(divergence.loc[stamp]),
                    "confirmed": confirmed_value,
                    "fired": (
                        washout_value > WASHOUT_GATE
                        and alarm_value > ALARM_GATE
                        and confirmed_value
                    ),
                }
            )

    return pd.DataFrame(rows, columns=REBOUND_COLUMNS)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_rebound.py -v
```

Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: rebound alarm gated on cluster washout"
```

---

### Task 15: Bottom labels and walk-forward folds

**Files:**
- Create: `src/screener_sector/backtest/__init__.py`, `src/screener_sector/backtest/labels.py`, `src/screener_sector/backtest/walkforward.py`
- Test: `tests/test_labels.py`, `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `BacktestParams` (Task 3).
- Produces:
  - `label_bottoms(ohlcv: pd.DataFrame, k: int, forward_days: int, min_return: float) -> pd.Series` (bool, indexed like `ohlcv`);
  - `forward_return(close: pd.Series, horizon: int) -> pd.Series`;
  - `Fold` frozen dataclass with `index: int`, `fit_start: date`, `fit_end: date`, `test_start: date`, `test_end: date`, `partial: bool`;
  - `expanding_folds(start: date, end: date, initial_fit_years: int, step_years: int) -> tuple[Fold, ...]`.

**This is the only module allowed to look forward.** `labels.py` exists precisely to encode future information for evaluation; nothing in `features/` may import it.

- [ ] **Step 1: Write the failing test for labels**

Create `tests/test_labels.py`:

```python
import numpy as np
import pandas as pd
import pytest

from conftest import exponential_trend, flat_series, make_ohlcv, trading_days, v_bottom
from screener_sector.backtest.labels import forward_return, label_bottoms


def test_v_bottom_is_labeled_at_the_trough():
    close = v_bottom(40, 40, depth=0.40)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert bool(labels.loc[close.idxmin()])


def test_only_the_trough_region_is_labeled():
    close = v_bottom(40, 40, depth=0.40)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert labels.sum() <= 3


def test_uptrend_has_no_bottom_labels():
    labels = label_bottoms(
        make_ohlcv(exponential_trend(200, 0.002)), k=10, forward_days=20, min_return=0.10
    )
    assert not labels.any()


def test_flat_series_has_no_bottom_labels():
    labels = label_bottoms(
        make_ohlcv(flat_series(200)), k=10, forward_days=20, min_return=0.10
    )
    assert not labels.any()


def test_shallow_bounce_below_threshold_is_not_labeled():
    close = v_bottom(40, 40, depth=0.03)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert not labels.any()


def test_no_label_within_forward_window_of_the_end():
    close = v_bottom(40, 10, depth=0.40)
    labels = label_bottoms(make_ohlcv(close), k=10, forward_days=20, min_return=0.10)
    assert not labels.tail(20).any()


def test_forward_return_matches_hand_computation():
    close = pd.Series([100.0, 110.0, 121.0], index=trading_days(3))
    assert forward_return(close, 2).iloc[0] == pytest.approx(0.21)


def test_forward_return_is_nan_at_the_tail():
    close = pd.Series([100.0, 110.0, 121.0], index=trading_days(3))
    assert np.isnan(forward_return(close, 2).iloc[-1])
```

- [ ] **Step 2: Run it to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_labels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.backtest'`.

- [ ] **Step 3: Create `src/screener_sector/backtest/__init__.py`**

Empty file.

- [ ] **Step 4: Implement `src/screener_sector/backtest/labels.py`**

```python
"""Ground-truth bottom labels.

This is the ONLY module permitted to read data after the evaluation date.
Labels answer 'was this actually a bottom?', which is unknowable in real time
and is exactly why it belongs here and not in features/.
"""

from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def label_bottoms(
    ohlcv: pd.DataFrame, k: int, forward_days: int, min_return: float
) -> pd.Series:
    """True where the low is the minimum of a +/-k window AND the forward
    return over `forward_days` clears `min_return`.

    The second condition matters: a local minimum that goes nowhere is not a
    tradeable bottom, and labeling it as one would teach the evaluator that
    noise counts as success.
    """
    low = ohlcv["low"]
    close = ohlcv["close"]
    window = 2 * k + 1
    rolling_min = low.rolling(window, center=True, min_periods=window).min()
    is_local_min = low <= rolling_min

    forward = forward_return(close, forward_days)
    labels = is_local_min & (forward >= min_return)
    return labels.fillna(False).astype(bool)
```

- [ ] **Step 5: Run the labels test**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_labels.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Write the failing test for folds**

Create `tests/test_walkforward.py`:

```python
from datetime import date

import pytest

from screener_sector.backtest.walkforward import expanding_folds


def test_prod_shape_produces_expected_fold_count():
    folds = expanding_folds(
        date(2010, 1, 1), date(2026, 8, 12), initial_fit_years=5, step_years=1
    )
    assert len(folds) == 12
    assert folds[0].test_start == date(2015, 1, 1)
    assert folds[-1].test_start == date(2026, 1, 1)


def test_fit_window_is_expanding_not_rolling():
    folds = expanding_folds(
        date(2010, 1, 1), date(2020, 12, 31), initial_fit_years=5, step_years=1
    )
    assert all(f.fit_start == date(2010, 1, 1) for f in folds)
    assert folds[1].fit_end > folds[0].fit_end


def test_fit_window_never_overlaps_its_test_window():
    folds = expanding_folds(
        date(2010, 1, 1), date(2026, 8, 12), initial_fit_years=5, step_years=1
    )
    assert all(f.fit_end < f.test_start for f in folds)


def test_final_partial_year_is_flagged():
    folds = expanding_folds(
        date(2010, 1, 1), date(2026, 8, 12), initial_fit_years=5, step_years=1
    )
    assert folds[-1].partial is True
    assert all(not f.partial for f in folds[:-1])


def test_complete_final_year_is_not_flagged_partial():
    folds = expanding_folds(
        date(2010, 1, 1), date(2025, 12, 31), initial_fit_years=5, step_years=1
    )
    assert all(not f.partial for f in folds)


def test_dev_shape_produces_two_folds():
    folds = expanding_folds(
        date(2022, 1, 1), date(2026, 8, 12), initial_fit_years=2, step_years=1
    )
    assert [f.test_start.year for f in folds] == [2024, 2025, 2026]


def test_folds_are_indexed_sequentially():
    folds = expanding_folds(
        date(2010, 1, 1), date(2020, 12, 31), initial_fit_years=5, step_years=1
    )
    assert [f.index for f in folds] == list(range(len(folds)))


def test_insufficient_span_yields_no_folds():
    assert expanding_folds(
        date(2024, 1, 1), date(2024, 12, 31), initial_fit_years=5, step_years=1
    ) == ()
```

- [ ] **Step 7: Run it to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_walkforward.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 8: Implement `src/screener_sector/backtest/walkforward.py`**

```python
"""Expanding-window walk-forward splits.

Parameters are always fit on data strictly before the period they are tested
on. Fold-to-fold parameter stability is a first-class diagnostic: thresholds
that thrash between folds indicate overfitting more reliably than any single
aggregate score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fold:
    index: int
    fit_start: date
    fit_end: date
    test_start: date
    test_end: date
    partial: bool


def expanding_folds(
    start: date, end: date, initial_fit_years: int, step_years: int
) -> tuple[Fold, ...]:
    folds: list[Fold] = []
    first_test_year = start.year + initial_fit_years
    index = 0

    for test_year in range(first_test_year, end.year + 1, step_years):
        test_start = date(test_year, 1, 1)
        if test_start > end:
            break
        natural_end = date(test_year + step_years - 1, 12, 31)
        test_end = min(natural_end, end)
        folds.append(
            Fold(
                index=index,
                fit_start=start,
                fit_end=date(test_year - 1, 12, 31),
                test_start=test_start,
                test_end=test_end,
                partial=test_end < natural_end,
            )
        )
        index += 1

    return tuple(folds)
```

- [ ] **Step 9: Run the folds test**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_walkforward.py -v
```

Expected: 8 passed.

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat: bottom labels and expanding walk-forward folds"
```

---

### Task 16: Evaluation metrics and random baseline

**Files:**
- Create: `src/screener_sector/backtest/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `forward_return` (Task 15), `BacktestParams` (Task 3).
- Produces: `ClassificationMetrics` frozen dataclass with `precision: float`, `recall: float`, `f1: float`, `signals: int`, `labels: int`, `mean_lead_days: float`; `classification_metrics(signals: pd.Series, labels: pd.Series, tolerance_days: int) -> ClassificationMetrics`; `EconomicMetrics` frozen dataclass with `horizon: int`, `n: int`, `mean_return: float`, `median_return: float`, `hit_rate: float`; `economic_metrics(close: pd.Series, signal_dates: Sequence[pd.Timestamp], horizons: Sequence[int]) -> pd.DataFrame` with columns `horizon, n, mean_return, median_return, hit_rate`; `random_baseline(close: pd.Series, n_signals: int, horizons: Sequence[int], seed: int, draws: int = 200) -> pd.DataFrame` with the same columns; `edge_table(signal_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame) -> pd.DataFrame` adding `mean_edge` and `hit_rate_edge` columns.

**Tolerance matching:** a signal counts as a true positive if a label falls within `tolerance_days` bars of it. Requiring an exact-day match would score a signal one day early as a total miss, which is not how the alarm would be used.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from conftest import exponential_trend, trading_days, v_bottom
from screener_sector.backtest.evaluate import (
    classification_metrics,
    economic_metrics,
    edge_table,
    random_baseline,
)


def flags(index, positions):
    series = pd.Series(False, index=index)
    series.iloc[list(positions)] = True
    return series


def test_perfect_signal_scores_one():
    idx = trading_days(100)
    labels = flags(idx, [30, 60])
    metrics = classification_metrics(labels.copy(), labels, tolerance_days=3)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_signal_within_tolerance_counts_as_hit():
    idx = trading_days(100)
    labels = flags(idx, [30])
    signals = flags(idx, [28])
    metrics = classification_metrics(signals, labels, tolerance_days=3)
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0


def test_signal_outside_tolerance_is_a_miss():
    idx = trading_days(100)
    metrics = classification_metrics(
        flags(idx, [10]), flags(idx, [30]), tolerance_days=3
    )
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_extra_signals_reduce_precision_not_recall():
    idx = trading_days(100)
    labels = flags(idx, [30])
    signals = flags(idx, [30, 50, 70, 90])
    metrics = classification_metrics(signals, labels, tolerance_days=3)
    assert metrics.recall == 1.0
    assert metrics.precision == pytest.approx(0.25)


def test_no_signals_gives_zero_not_nan():
    idx = trading_days(100)
    metrics = classification_metrics(
        flags(idx, []), flags(idx, [30]), tolerance_days=3
    )
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_mean_lead_days_is_positive_when_early():
    idx = trading_days(100)
    metrics = classification_metrics(
        flags(idx, [28]), flags(idx, [30]), tolerance_days=5
    )
    assert metrics.mean_lead_days == pytest.approx(2.0)


def test_economic_metrics_computes_forward_returns():
    close = exponential_trend(100, 0.01)  # ~1% per day, always positive
    dates = [close.index[10], close.index[20]]
    table = economic_metrics(close, dates, horizons=[5, 10])
    assert list(table.columns) == [
        "horizon", "n", "mean_return", "median_return", "hit_rate"
    ]
    assert table.set_index("horizon").loc[5, "hit_rate"] == 1.0
    assert table.set_index("horizon").loc[10, "mean_return"] > 0


def test_economic_metrics_drops_signals_without_full_horizon():
    close = exponential_trend(100, 0.01)
    dates = [close.index[95]]
    table = economic_metrics(close, dates, horizons=[20]).set_index("horizon")
    assert table.loc[20, "n"] == 0


def test_economic_metrics_on_bottom_entries_beats_downtrend_entries():
    close = v_bottom(60, 60, depth=0.40)
    bottom_entry = economic_metrics(close, [close.index[59]], [20])
    early_entry = economic_metrics(close, [close.index[20]], [20])
    assert bottom_entry["mean_return"].iloc[0] > early_entry["mean_return"].iloc[0]


def test_random_baseline_is_reproducible_with_seed():
    close = exponential_trend(300, 0.001, noise=0.02, seed=1)
    a = random_baseline(close, n_signals=10, horizons=[10], seed=42, draws=20)
    b = random_baseline(close, n_signals=10, horizons=[10], seed=42, draws=20)
    pd.testing.assert_frame_equal(a, b)


def test_random_baseline_approximates_unconditional_return():
    close = exponential_trend(1000, 0.001)
    baseline = random_baseline(close, n_signals=50, horizons=[10], seed=7, draws=100)
    expected = np.exp(0.001 * 10) - 1.0
    assert baseline["mean_return"].iloc[0] == pytest.approx(expected, rel=0.15)


def test_edge_table_subtracts_baseline():
    signal = pd.DataFrame(
        {"horizon": [10], "n": [5], "mean_return": [0.08],
         "median_return": [0.07], "hit_rate": [0.8]}
    )
    baseline = pd.DataFrame(
        {"horizon": [10], "n": [100], "mean_return": [0.02],
         "median_return": [0.01], "hit_rate": [0.55]}
    )
    table = edge_table(signal, baseline).set_index("horizon")
    assert table.loc[10, "mean_edge"] == pytest.approx(0.06)
    assert table.loc[10, "hit_rate_edge"] == pytest.approx(0.25)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_evaluate.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/backtest/evaluate.py`**

```python
"""Scoring the alarm as a classifier and as an entry rule.

Absolute hit rates are uninformative in a market that rose over the sample:
buying at random would also have 'worked'. The random baseline is what turns a
number into evidence of edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener_sector.backtest.labels import forward_return

ECONOMIC_COLUMNS = ["horizon", "n", "mean_return", "median_return", "hit_rate"]


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    signals: int
    labels: int
    mean_lead_days: float


def classification_metrics(
    signals: pd.Series, labels: pd.Series, tolerance_days: int
) -> ClassificationMetrics:
    """Match signals to labels within +/- tolerance_days bars.

    Exact-day matching would score a signal one bar early as a complete miss,
    which does not reflect how the alarm is used.
    """
    signal_positions = np.flatnonzero(signals.to_numpy(dtype=bool))
    label_positions = np.flatnonzero(labels.to_numpy(dtype=bool))

    if signal_positions.size == 0 or label_positions.size == 0:
        return ClassificationMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            signals=int(signal_positions.size),
            labels=int(label_positions.size),
            mean_lead_days=float("nan"),
        )

    matched_labels: set[int] = set()
    true_positives = 0
    leads: list[float] = []

    for position in signal_positions:
        distances = np.abs(label_positions - position)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tolerance_days:
            true_positives += 1
            matched_labels.add(int(label_positions[nearest]))
            leads.append(float(label_positions[nearest] - position))

    precision = true_positives / len(signal_positions)
    recall = len(matched_labels) / len(label_positions)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        signals=int(signal_positions.size),
        labels=int(label_positions.size),
        mean_lead_days=float(np.mean(leads)) if leads else float("nan"),
    )


def economic_metrics(
    close: pd.Series, signal_dates: Sequence[pd.Timestamp], horizons: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        forward = forward_return(close, horizon)
        values = forward.reindex(pd.DatetimeIndex(signal_dates)).dropna()
        rows.append(
            {
                "horizon": horizon,
                "n": int(len(values)),
                "mean_return": float(values.mean()) if len(values) else float("nan"),
                "median_return": float(values.median()) if len(values) else float("nan"),
                "hit_rate": float((values > 0).mean()) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=ECONOMIC_COLUMNS)


def random_baseline(
    close: pd.Series,
    n_signals: int,
    horizons: Sequence[int],
    seed: int,
    draws: int = 200,
) -> pd.DataFrame:
    """Average economic metrics over `draws` random entry sets of the same size."""
    rng = np.random.default_rng(seed)
    accumulated: list[pd.DataFrame] = []

    for _ in range(draws):
        eligible = close.index[: max(len(close) - max(horizons), 1)]
        if len(eligible) == 0:
            break
        size = min(n_signals, len(eligible))
        picks = rng.choice(len(eligible), size=size, replace=False)
        dates = [eligible[int(p)] for p in picks]
        accumulated.append(economic_metrics(close, dates, horizons))

    if not accumulated:
        return pd.DataFrame(columns=ECONOMIC_COLUMNS)

    stacked = pd.concat(accumulated)
    return (
        stacked.groupby("horizon", as_index=False)
        .mean(numeric_only=True)[ECONOMIC_COLUMNS]
        .reset_index(drop=True)
    )


def edge_table(
    signal_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame
) -> pd.DataFrame:
    merged = signal_metrics.merge(
        baseline_metrics, on="horizon", suffixes=("", "_baseline")
    )
    merged["mean_edge"] = merged["mean_return"] - merged["mean_return_baseline"]
    merged["hit_rate_edge"] = merged["hit_rate"] - merged["hit_rate_baseline"]
    return merged
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_evaluate.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: classification and economic metrics with random baseline"
```

---

### Task 17: HTML and CSV reporting

**Files:**
- Create: `src/screener_sector/report/__init__.py`, `src/screener_sector/report/render.py`, `src/screener_sector/report/template.html`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Config` (Task 3), `ClusterResult` (Task 12).
- Produces: `ScreenOutput` frozen dataclass with `as_of: date`, `trend: pd.DataFrame`, `clusters: ClusterResult`, `strength: pd.DataFrame`, `rebound: pd.DataFrame`; `DEV_WARNING: str`; `render_report(output: ScreenOutput, config: Config, out_dir: Path) -> Path` writing `out_dir/<profile>/<as_of>.html` plus sibling CSVs `trend.csv`, `strength.csv`, `rebound.csv`, `clusters.csv`; `write_csvs(output, out_dir_for_profile) -> None`.

**Required header content:** every report embeds the survivorship-bias caveat, and dev-profile reports additionally embed `DEV_WARNING`. The design makes this non-optional — a dev report must not be mistakable for evidence later.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_report.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/screener_sector/report/__init__.py`**

Empty file.

- [ ] **Step 4: Create `src/screener_sector/report/template.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Semiconductor Screen — {{ as_of }} ({{ profile }})</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; }
  table { border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
  th, td { border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f5f5f5; }
  .caveat { background: #fff8e1; border-left: 4px solid #ffb300; padding: 0.8rem; margin: 1rem 0; }
  .dev { background: #ffebee; border-left: 4px solid #c62828; padding: 0.8rem; margin: 1rem 0; }
  .empty { color: #888; font-style: italic; }
</style>
</head>
<body>
<h1>Semiconductor Screen — {{ as_of }}</h1>
<p>Profile: <strong>{{ profile }}</strong> · Window: {{ start }} to {{ as_of }} ·
   Benchmark: {{ benchmark }}</p>

{% if dev_warning %}<div class="dev">{{ dev_warning }}</div>{% endif %}

<div class="caveat">
  The universe is built from currently listed securities, so this screen and any
  backtest derived from it carry <strong>survivorship bias</strong>: companies
  that were delisted are absent, which inflates historical results. Yahoo also
  restates adjusted prices over time, so results are not bit-reproducible across
  re-fetches.
</div>

<h2>Rebound alarms</h2>
{{ rebound_table }}

<h2>Clusters</h2>
{{ clusters_table }}

<h2>Relative strength</h2>
{{ strength_table }}

<h2>Trend scores</h2>
{{ trend_table }}
</body>
</html>
```

- [ ] **Step 5: Implement `src/screener_sector/report/render.py`**

```python
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
```

Note the one deliberate exception to the "no paths from `__file__`" rule: `template.html` is package *source*, not data, and ships inside the image. The rule protects `DATA_DIR` portability, which this does not touch.

- [ ] **Step 6: Add the template to the package build**

Add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"screener_sector.report" = ["*.html"]
```

- [ ] **Step 7: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_report.py -v
```

Expected: 9 passed.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: HTML and CSV reporting with mandatory caveats"
```

---

### Task 18: Screen pipeline orchestration

**Files:**
- Create: `src/screener_sector/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Config`, `PriceStore`, `trend_table`, `cluster_universe`, `strength_table`, `rebound_table`, `ScreenOutput`.
- Produces: `run_screen(store: PriceStore, tickers: Sequence[str], config: Config, as_of: date) -> ScreenOutput`; `save_screen(paths: Paths, output: ScreenOutput, profile: str) -> Path` (writes to `derived_dir(profile)/<as_of>/`); `load_frames(store, tickers, as_of) -> dict[str, pd.DataFrame]`.

**The single most important property of this function:** it truncates every frame at `as_of` *before* any feature is computed. This is the enforcement point for point-in-time correctness — every downstream feature is then structurally incapable of seeing the future, no matter what it does internally.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import exponential_trend, make_ohlcv, trading_days
from screener_sector.config import Config
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths
from screener_sector.pipeline import load_frames, run_screen, save_screen

CONFIG_DIR = Path("/app/config")


@pytest.fixture
def env(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    rng = np.random.default_rng(12)
    idx = trading_days(400)
    driver = rng.normal(0.0005, 0.015, 400)
    frames = {}
    for i in range(5):
        noise = rng.normal(0, 0.004, 400)
        close = pd.Series(100.0 * np.exp(np.cumsum(driver + noise)), index=idx)
        frames[f"T{i}"] = make_ohlcv(close)
    frames["SOXX"] = make_ohlcv(
        pd.Series(100.0 * np.exp(np.cumsum(driver)), index=idx)
    )
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2015, 1, 1))
    return paths, store, list(frames), idx


def test_load_frames_truncates_at_as_of(env):
    paths, store, tickers, idx = env
    as_of = idx[300].date()
    frames = load_frames(store, tickers, as_of)
    for frame in frames.values():
        assert frame.index.max().date() <= as_of


def test_run_screen_produces_all_sections(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers, cfg, idx[350].date())
    assert not output.trend.empty
    assert output.clusters.raw_corr.shape[0] > 0
    assert output.as_of == idx[350].date()


def test_run_screen_excludes_benchmark_from_the_screen(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers, cfg, idx[350].date())
    assert "SOXX" not in set(output.trend["ticker"])


def test_run_screen_is_point_in_time(env):
    """Mutating data after as_of must not change any computed value."""
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    as_of = idx[300].date()
    before = run_screen(store, tickers, cfg, as_of)

    for ticker in tickers:
        frame = store.load(ticker)
        mask = frame.index > pd.Timestamp(as_of)
        frame.loc[mask, ["open", "high", "low", "close"]] *= 3.0
        frame.to_parquet(paths.price_file(ticker))

    after = run_screen(store, tickers, cfg, as_of)
    pd.testing.assert_frame_equal(before.trend, after.trend)
    pd.testing.assert_frame_equal(before.strength, after.strength)
    pd.testing.assert_frame_equal(before.rebound, after.rebound)


def test_run_screen_ignores_tickers_with_no_data(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers + ["MISSING"], cfg, idx[350].date())
    assert "MISSING" not in set(output.trend["ticker"])


def test_save_screen_writes_under_profile_namespace(env):
    paths, store, tickers, idx = env
    cfg = Config.load(CONFIG_DIR, "dev")
    output = run_screen(store, tickers, cfg, idx[350].date())
    directory = save_screen(paths, output, "dev")
    assert directory.is_relative_to(paths.derived_dir("dev"))
    assert (directory / "trend.csv").exists()
    assert (directory / "rebound.csv").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/pipeline.py`**

```python
"""Screen orchestration.

Truncation at `as_of` happens here, once, before any feature runs. That makes
point-in-time correctness a structural property of the pipeline rather than a
discipline each feature function has to remember.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from screener_sector.config import Config
from screener_sector.data.store import PriceStore
from screener_sector.features.correlation import cluster_universe
from screener_sector.features.rebound import rebound_table
from screener_sector.features.strength import strength_table
from screener_sector.features.trend import trend_table
from screener_sector.paths import Paths
from screener_sector.report.render import ScreenOutput, write_csvs


def load_frames(
    store: PriceStore, tickers: Sequence[str], as_of: date
) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(as_of)
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if not store.has(ticker):
            continue
        frame = store.load(ticker)
        truncated = frame.loc[frame.index <= cutoff]
        if truncated.empty:
            continue
        frames[ticker] = truncated
    return frames


def run_screen(
    store: PriceStore, tickers: Sequence[str], config: Config, as_of: date
) -> ScreenOutput:
    frames = load_frames(store, list(tickers) + [config.benchmark], as_of)

    benchmark_frame = frames.pop(config.benchmark, None)
    benchmark = benchmark_frame["close"] if benchmark_frame is not None else None

    if not frames:
        empty_clusters = cluster_universe(
            pd.DataFrame(), None, config.corr_threshold, config.min_cluster_size, 1
        )
        return ScreenOutput(as_of, pd.DataFrame(), empty_clusters, pd.DataFrame(), pd.DataFrame())

    trend = trend_table(frames, config.windows, config.trend_weights)

    panel = pd.DataFrame({t: f["close"] for t, f in frames.items()}).sort_index()
    clusters = cluster_universe(
        panel,
        benchmark,
        config.corr_threshold,
        config.min_cluster_size,
        config.windows.corr,
    )

    strength = strength_table(panel, clusters.clusters, config.windows.corr)
    rebound = rebound_table(
        panel,
        frames,
        clusters.clusters,
        config.rebound_weights,
        config.windows,
        as_of=pd.Timestamp(as_of) if pd.Timestamp(as_of) in panel.index else None,
    )

    return ScreenOutput(
        as_of=as_of,
        trend=trend,
        clusters=clusters,
        strength=strength,
        rebound=rebound,
    )


def save_screen(paths: Paths, output: ScreenOutput, profile: str) -> Path:
    directory = paths.derived_dir(profile) / output.as_of.isoformat()
    write_csvs(output, directory)
    return directory
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_pipeline.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: screen pipeline with point-in-time truncation"
```

---

### Task 19: Walk-forward backtest runner

**Files:**
- Create: `src/screener_sector/backtest/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Config`, `PriceStore`, `expanding_folds`, `Fold` (Task 15), `label_bottoms`, `classification_metrics`, `economic_metrics`, `random_baseline`, `edge_table` (Task 16), `rebound_table`/`ticker_alarm`/`cluster_washout` (Task 14), `cluster_universe` (Task 12).
- Produces: `alarm_series(store, tickers, config, start: date, end: date, alarm_gate: float) -> dict[str, pd.Series]` (bool signal series per ticker over the window, computed with a single clustering fitted on data up to `start`); `fit_alarm_gate(store, tickers, config, fold: Fold, candidates: Sequence[float]) -> float` (picks the gate maximizing F1 on the fit window); `run_fold(store, tickers, config, fold) -> pd.DataFrame`; `run_backtest(store, tickers, config, end: date) -> BacktestResult`; `BacktestResult` frozen dataclass with `per_fold: pd.DataFrame`, `economics: pd.DataFrame`, `baseline: pd.DataFrame`, `edges: pd.DataFrame`, `fitted_gates: pd.DataFrame`.

`per_fold` columns: `fold, partial, test_start, test_end, tickers, gate, precision, recall, f1, signals, labels, mean_lead_days`.
`fitted_gates` columns: `fold, gate` — the parameter-stability diagnostic.

**Fitting rule:** the only parameter fitted per fold is `alarm_gate`. Clustering, weights, and windows stay fixed at their config values. Keeping the fitted surface to one dimension is what makes fold-to-fold stability readable; a dozen jointly fitted parameters would produce a stability table nobody can interpret.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from screener_sector.backtest.runner import (
    alarm_series,
    fit_alarm_gate,
    run_backtest,
    run_fold,
)
from screener_sector.backtest.walkforward import Fold
from screener_sector.config import Config
from screener_sector.data.fetcher import FakeFetcher
from screener_sector.data.store import PriceStore
from screener_sector.paths import Paths

CONFIG_DIR = Path("/app/config")


def cyclical_panel(n_years: int = 6, seed: int = 3) -> dict[str, pd.DataFrame]:
    """A correlated group with repeated drawdown-and-recovery cycles, so the
    backtest has bottoms to find."""
    rng = np.random.default_rng(seed)
    n = 252 * n_years
    idx = pd.bdate_range("2018-01-01", periods=n)
    cycle = np.sin(np.arange(n) * 2 * np.pi / 252) * 0.004
    driver = cycle + rng.normal(0.0003, 0.010, n)
    frames = {}
    for i in range(5):
        noise = rng.normal(0, 0.003, n)
        close = pd.Series(100.0 * np.exp(np.cumsum(driver + noise)), index=idx)
        frames[f"T{i}"] = make_ohlcv(close)
    frames["SOXX"] = make_ohlcv(
        pd.Series(100.0 * np.exp(np.cumsum(driver)), index=idx)
    )
    return frames


@pytest.fixture
def env(tmp_path):
    paths = Paths.from_env({"DATA_DIR": str(tmp_path)})
    paths.ensure()
    frames = cyclical_panel()
    store = PriceStore(paths, FakeFetcher(frames))
    store.refresh(list(frames), date(2017, 1, 1))
    return paths, store, [t for t in frames if t != "SOXX"]


def test_alarm_series_returns_boolean_series_per_ticker(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    series = alarm_series(
        store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), alarm_gate=60.0
    )
    assert set(series) <= set(tickers)
    for value in series.values():
        assert value.dtype == bool


def test_alarm_series_is_confined_to_the_requested_window(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    series = alarm_series(
        store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), alarm_gate=60.0
    )
    for value in series.values():
        assert value.index.min() >= pd.Timestamp("2021-01-01")
        assert value.index.max() <= pd.Timestamp("2021-12-31")


def test_lower_gate_produces_at_least_as_many_signals(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    loose = alarm_series(store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), 40.0)
    tight = alarm_series(store, tickers, cfg, date(2021, 1, 1), date(2021, 12, 31), 80.0)
    assert sum(s.sum() for s in loose.values()) >= sum(s.sum() for s in tight.values())


def test_fit_alarm_gate_returns_a_candidate(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    fold = Fold(0, date(2018, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31), False)
    candidates = [40.0, 50.0, 60.0, 70.0]
    assert fit_alarm_gate(store, tickers, cfg, fold, candidates) in candidates


def test_run_fold_reports_expected_columns(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    fold = Fold(0, date(2018, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31), False)
    row = run_fold(store, tickers, cfg, fold)
    assert list(row.columns) == [
        "fold", "partial", "test_start", "test_end", "tickers", "gate",
        "precision", "recall", "f1", "signals", "labels", "mean_lead_days",
    ]
    assert row["fold"].iloc[0] == 0


def test_run_fold_never_fits_on_test_data(env):
    """Gate fitted for a fold must not change when test-window prices change."""
    paths, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    fold = Fold(0, date(2018, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31), False)
    candidates = [40.0, 50.0, 60.0, 70.0]
    before = fit_alarm_gate(store, tickers, cfg, fold, candidates)

    for ticker in tickers:
        frame = store.load(ticker)
        mask = frame.index >= pd.Timestamp("2021-01-01")
        frame.loc[mask, ["open", "high", "low", "close"]] *= 0.5
        frame.to_parquet(paths.price_file(ticker))

    assert fit_alarm_gate(store, tickers, cfg, fold, candidates) == before


def test_run_backtest_produces_all_result_frames(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    result = run_backtest(store, tickers, cfg, end=date(2023, 12, 31))
    assert not result.per_fold.empty
    assert list(result.fitted_gates.columns) == ["fold", "gate"]
    assert "mean_edge" in result.edges.columns


def test_run_backtest_marks_partial_final_fold(env):
    _, store, tickers = env
    cfg = Config.load(CONFIG_DIR, "dev")
    result = run_backtest(store, tickers, cfg, end=date(2023, 6, 30))
    assert bool(result.per_fold["partial"].iloc[-1]) is True
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/screener_sector/backtest/runner.py`**

```python
"""Walk-forward backtest.

Exactly one parameter is fitted per fold: the alarm gate. Everything else is
held at its configured value. A single fitted dimension is what makes the
fold-to-fold stability table readable — a dozen jointly-fitted parameters
would produce a diagnostic nobody can interpret.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from screener_sector.backtest.evaluate import (
    classification_metrics,
    economic_metrics,
    edge_table,
    random_baseline,
)
from screener_sector.backtest.labels import label_bottoms
from screener_sector.backtest.walkforward import Fold, expanding_folds
from screener_sector.config import Config
from screener_sector.data.store import PriceStore
from screener_sector.features.correlation import cluster_universe
from screener_sector.features.rebound import (
    WASHOUT_GATE,
    cluster_washout,
    confirmation,
    ticker_alarm,
)
from screener_sector.pipeline import load_frames

PER_FOLD_COLUMNS = [
    "fold",
    "partial",
    "test_start",
    "test_end",
    "tickers",
    "gate",
    "precision",
    "recall",
    "f1",
    "signals",
    "labels",
    "mean_lead_days",
]


@dataclass(frozen=True)
class BacktestResult:
    per_fold: pd.DataFrame
    economics: pd.DataFrame
    baseline: pd.DataFrame
    edges: pd.DataFrame
    fitted_gates: pd.DataFrame


def alarm_series(
    store: PriceStore,
    tickers: Sequence[str],
    config: Config,
    start: date,
    end: date,
    alarm_gate: float,
) -> dict[str, pd.Series]:
    """Boolean alarm signals per ticker across [start, end].

    Clusters are fitted once on data up to `start` and then held fixed for the
    window, so no signal inside the window depends on data from later in it.
    """
    fit_frames = load_frames(store, list(tickers) + [config.benchmark], start)
    benchmark_fit = fit_frames.pop(config.benchmark, None)
    if not fit_frames:
        return {}

    fit_panel = pd.DataFrame(
        {t: f["close"] for t, f in fit_frames.items()}
    ).sort_index()
    clusters = cluster_universe(
        fit_panel,
        benchmark_fit["close"] if benchmark_fit is not None else None,
        config.corr_threshold,
        config.min_cluster_size,
        config.windows.corr,
    )
    if not clusters.clusters:
        return {}

    full_frames = load_frames(store, list(tickers) + [config.benchmark], end)
    full_frames.pop(config.benchmark, None)
    full_panel = pd.DataFrame(
        {t: f["close"] for t, f in full_frames.items()}
    ).sort_index()

    out: dict[str, pd.Series] = {}
    window = (full_panel.index >= pd.Timestamp(start)) & (
        full_panel.index <= pd.Timestamp(end)
    )

    for cluster in clusters.clusters:
        members = [m for m in cluster.members if m in full_frames]
        if not members:
            continue
        washout = cluster_washout(full_panel, members, config.windows.mid)
        for ticker in members:
            frame = full_frames[ticker]
            alarm = ticker_alarm(frame, washout, config.rebound_weights, config.windows)
            confirmed = confirmation(frame, config.windows.short)
            fired = (
                (washout.reindex(frame.index).fillna(0.0) > WASHOUT_GATE)
                & (alarm > alarm_gate)
                & confirmed
            )
            out[ticker] = fired.reindex(full_panel.index).fillna(False)[window]
    return out


def _labels_for(
    store: PriceStore, ticker: str, config: Config, start: date, end: date
) -> pd.Series:
    frame = store.load(ticker)
    labels = label_bottoms(
        frame,
        config.backtest.label_k,
        config.backtest.label_forward_days,
        config.backtest.label_min_return,
    )
    mask = (labels.index >= pd.Timestamp(start)) & (labels.index <= pd.Timestamp(end))
    return labels[mask]


def _score(
    store: PriceStore,
    tickers: Sequence[str],
    config: Config,
    start: date,
    end: date,
    gate: float,
) -> tuple[float, float, float, int, int, float]:
    signals = alarm_series(store, tickers, config, start, end, gate)
    precisions, recalls, f1s, leads = [], [], [], []
    total_signals = total_labels = 0

    for ticker, series in signals.items():
        labels = _labels_for(store, ticker, config, start, end).reindex(
            series.index, fill_value=False
        )
        metrics = classification_metrics(
            series, labels, tolerance_days=config.backtest.label_k
        )
        precisions.append(metrics.precision)
        recalls.append(metrics.recall)
        f1s.append(metrics.f1)
        total_signals += metrics.signals
        total_labels += metrics.labels
        if metrics.mean_lead_days == metrics.mean_lead_days:  # not NaN
            leads.append(metrics.mean_lead_days)

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return (
        mean(precisions),
        mean(recalls),
        mean(f1s),
        total_signals,
        total_labels,
        mean(leads) if leads else float("nan"),
    )


def fit_alarm_gate(
    store: PriceStore,
    tickers: Sequence[str],
    config: Config,
    fold: Fold,
    candidates: Sequence[float],
) -> float:
    best_gate = float(candidates[0])
    best_f1 = -1.0
    for gate in candidates:
        _, _, f1, _, _, _ = _score(
            store, tickers, config, fold.fit_start, fold.fit_end, float(gate)
        )
        if f1 > best_f1:
            best_f1, best_gate = f1, float(gate)
    return best_gate


DEFAULT_GATES = (40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0)


def run_fold(
    store: PriceStore, tickers: Sequence[str], config: Config, fold: Fold
) -> pd.DataFrame:
    gate = fit_alarm_gate(store, tickers, config, fold, DEFAULT_GATES)
    precision, recall, f1, signals, labels, lead = _score(
        store, tickers, config, fold.test_start, fold.test_end, gate
    )
    return pd.DataFrame(
        [
            {
                "fold": fold.index,
                "partial": fold.partial,
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "tickers": len(tickers),
                "gate": gate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "signals": signals,
                "labels": labels,
                "mean_lead_days": lead,
            }
        ],
        columns=PER_FOLD_COLUMNS,
    )


def run_backtest(
    store: PriceStore, tickers: Sequence[str], config: Config, end: date
) -> BacktestResult:
    folds = expanding_folds(
        config.start,
        end,
        config.backtest.initial_fit_years,
        config.backtest.step_years,
    )

    fold_rows: list[pd.DataFrame] = []
    signal_economics: list[pd.DataFrame] = []
    baseline_economics: list[pd.DataFrame] = []

    for fold in folds:
        row = run_fold(store, tickers, config, fold)
        fold_rows.append(row)

        gate = float(row["gate"].iloc[0])
        signals = alarm_series(
            store, tickers, config, fold.test_start, fold.test_end, gate
        )
        for ticker, series in signals.items():
            dates = list(series[series].index)
            if not dates:
                continue
            close = store.load(ticker)["close"]
            signal_economics.append(
                economic_metrics(close, dates, config.backtest.horizons)
            )
            baseline_economics.append(
                random_baseline(
                    close,
                    n_signals=len(dates),
                    horizons=config.backtest.horizons,
                    seed=fold.index,
                    draws=50,
                )
            )

    per_fold = (
        pd.concat(fold_rows, ignore_index=True)
        if fold_rows
        else pd.DataFrame(columns=PER_FOLD_COLUMNS)
    )
    economics = _aggregate(signal_economics)
    baseline = _aggregate(baseline_economics)
    edges = (
        edge_table(economics, baseline)
        if not economics.empty and not baseline.empty
        else pd.DataFrame(columns=["horizon", "mean_edge", "hit_rate_edge"])
    )
    fitted_gates = (
        per_fold[["fold", "gate"]].copy()
        if not per_fold.empty
        else pd.DataFrame(columns=["fold", "gate"])
    )
    return BacktestResult(per_fold, economics, baseline, edges, fitted_gates)


def _aggregate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=["horizon", "n", "mean_return", "median_return", "hit_rate"]
        )
    stacked = pd.concat(frames, ignore_index=True)
    return stacked.groupby("horizon", as_index=False).mean(numeric_only=True)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_runner.py -v
```

Expected: 8 passed. This suite is the slowest in the project (it clusters and scores repeatedly); if it exceeds ~90 seconds, reduce `cyclical_panel`'s `n_years` to 4 in the fixture rather than reducing coverage elsewhere.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: walk-forward backtest runner with per-fold gate fitting"
```

---

### Task 20: CLI

**Files:**
- Create: `src/screener_sector/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a typer app with subcommands `build-universe`, `fetch`, `screen`, `backtest`, `report`, `info`. Shared options: `--profile` (default from `PROFILE` env, else `dev`), `--as-of` (default today), `--config-dir` (default `/app/config`), `--out` (default `/app/out`). Also `resolve_tickers(paths, config) -> list[str]` which returns `config.static_tickers` for the `static` universe mode and the included rows of `universe.csv` for `discover`.

**Command behavior:**
- `build-universe` — symbols → enrich → fetch → filter → `universe.csv`. Skipped entirely in `static` mode with a clear message, since the dev profile ships its list.
- `fetch` — refresh prices for the resolved tickers plus the benchmark, from `config.start`, always requesting maximum history.
- `screen` — `run_screen` at `--as-of`, save derived artifacts, print the fired alarms.
- `backtest` — `run_backtest` through `--as-of`, write `per_fold.csv`, `economics.csv`, `baseline.csv`, `edges.csv`, `fitted_gates.csv` under the profile's out dir, print the per-fold table.
- `report` — render HTML from the screen output.
- `info` — print `DATA_DIR`, schema version, stage timestamps, universe size. The command to run first when something looks wrong.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'screener_sector.cli'`.

- [ ] **Step 3: Implement `src/screener_sector/cli.py`**

```python
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

    info_frame = enrich(
        paths,
        list(symbols["ticker"]),
        YFinanceInfoSource(),
        now=datetime.now().isoformat(timespec="seconds"),
    )
    typer.echo(f"enriched: {len(info_frame)}")

    store = PriceStore(paths, YFinanceFetcher())
    candidates = [
        row["ticker"]
        for row in info_frame.to_dict("records")
        if _candidate(row, rules)
    ]
    store.refresh(candidates, config.start, config.end)

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
    store = PriceStore(paths, YFinanceFetcher())
    tickers = resolve_tickers(paths, config) + [config.benchmark]
    result = store.refresh(tickers, config.start, config.end)
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_cli.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Verify the container entrypoint works**

```bash
docker compose run --rm screener info --profile dev
```

Expected: prints `DATA_DIR: /data`, `profile: dev`, `schema_version: 1`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: CLI with screen, backtest, and report commands"
```

---

### Task 21: Portability and end-to-end integration tests

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`
- Modify: `docker-compose.yml` (add the `DATA_DIR` relocation note)

**Interfaces:**
- Consumes: everything.
- Produces: no new production interfaces. This task proves the two structural guarantees the design rests on — relocatability and absence of lookahead — end to end rather than per-module.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_integration.py`:

```python
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
```

- [ ] **Step 2: Run it**

```bash
docker compose run --rm --entrypoint pytest screener tests/test_integration.py -v
```

Expected: 5 passed. If `test_future_data_cannot_influence_the_screen` fails, a feature is reading past `as_of` — the cause is almost always a `center=True` rolling window or a negative `shift()` outside `backtest/labels.py`. Fix the feature; do not weaken the test.

- [ ] **Step 3: Run the full suite**

```bash
docker compose run --rm --entrypoint pytest screener tests/ -v
```

Expected: all tests pass, roughly 160 of them.

- [ ] **Step 4: Write `README.md`**

````markdown
# screener_sector

Semiconductor / AI / optical equity screener: trend scoring, correlation
clustering, relative strength, and rebound alarms, validated walk-forward.

## Quick start

```bash
cp .env.example .env
docker compose build
docker compose run --rm screener info
```

## Commands

All commands take `--profile dev|prod` and `--as-of YYYY-MM-DD`.

```bash
docker compose run --rm screener build-universe --profile prod
docker compose run --rm screener fetch --profile dev
docker compose run --rm screener screen --profile dev
docker compose run --rm screener backtest --profile dev
docker compose run --rm screener report --profile dev
```

## Profiles

`dev` (default) runs 2022→now over ~30 checked-in tickers in about two minutes.
It exists to debug the methodology. **Dev results are not evidence that the
method works** — the window covers roughly one and a half drawdown cycles.

`prod` runs 2006→now over the full discovered universe with 12 walk-forward
folds. `build-universe` in prod takes hours; it is resumable and cached.

## Relocating the data directory

`data/` is self-contained: no absolute paths, no external references. To move it:

```bash
cp -r data /Volumes/external/screener-data
# then in .env:
DATA_DIR=/Volumes/external/screener-data
```

Price files always hold maximum available history regardless of profile, so a
cache built under `dev` is a valid starting point for `prod`.

## Known limitations

- **Survivorship bias.** The universe comes from currently listed securities;
  delisted companies are absent, which inflates backtest results, more so the
  further back the test runs.
- **Restated history.** Yahoo revises adjusted prices, so backtests are not
  bit-reproducible across re-fetches.
- **Unofficial data source.** `yfinance` uses an unsupported endpoint that can
  change or rate-limit without notice.
- Not investment advice.
````

- [ ] **Step 5: Add the relocation note to `docker-compose.yml`**

Add above the `services:` line:

```yaml
# DATA_DIR (from .env) is the single knob for where persistent data lives.
# Point it anywhere - external drive, NAS, synced folder - and nothing else changes.
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "test: end-to-end portability and lookahead guarantees"
```

---

## Execution Notes

**Suggested order for a first prod run**, once all 21 tasks are complete:

```bash
docker compose run --rm screener build-universe --profile prod   # hours, resumable
docker compose run --rm screener fetch --profile prod            # ~20 min
docker compose run --rm screener backtest --profile prod         # long
docker compose run --rm screener report --profile prod
```

**Reading the backtest output.** Look at `fitted_gates.csv` before anything
else. If the fitted gate is stable across folds, the remaining metrics are
worth reading. If it swings between 40 and 80 from year to year, the alarm is
fitting noise and the `edges.csv` numbers should not be trusted regardless of
how good they look.

**Expected honest outcome.** A rebound alarm with a small positive edge over
random entry is a good result. If `mean_edge` comes out large and positive on
the first run, suspect a bug before celebrating — most commonly a lookahead
leak that `test_future_data_cannot_influence_the_screen` did not catch because
it entered through the universe (which is built from today's listings) rather
than through the price data.
