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
