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
