from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_CONFIG = ROOT / "config" / "index.yaml"
DEFAULT_QUERY_CONFIG = ROOT / "config" / "query.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def ollama_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = dict(cfg.get("ollama") or {})
    section["host"] = os.environ.get("OLLAMA_HOST", section.get("host", "http://localhost:11490"))
    section["chat_model"] = os.environ.get("OLLAMA_CHAT_MODEL", section.get("chat_model", "gemma3:4b-it-q4_K_M"))
    section["embed_model"] = os.environ.get("OLLAMA_EMBED_MODEL", section.get("embed_model", "mxbai-embed-large"))
    return section


def load_index_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_INDEX_CONFIG)


def load_query_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_QUERY_CONFIG)
