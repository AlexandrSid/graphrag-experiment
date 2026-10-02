from __future__ import annotations

from pathlib import Path
from typing import Any

from graphrag_lab.util import dump_json, load_json


def manifest_path(index_dir: Path) -> Path:
    return index_dir / "manifest.json"


def write_manifest(index_dir: Path, data: dict[str, Any]) -> None:
    dump_json(manifest_path(index_dir), data)


def read_manifest(index_dir: Path) -> dict[str, Any]:
    path = manifest_path(index_dir)
    if not path.exists():
        return {}
    return load_json(path)
