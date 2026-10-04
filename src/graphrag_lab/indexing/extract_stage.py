from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from graphrag_lab.agents.extract import extract_chunk
from graphrag_lab.config import ollama_settings
from graphrag_lab.ollama_client import OllamaClient, OllamaError
from graphrag_lab.indexing.cancel import check
from graphrag_lab.indexing.progress import log
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import append_jsonl, stable_id, write_jsonl


def run_extract(store: IndexStore, cfg: dict, log_path) -> dict:
    settings = ollama_settings(cfg)
    client = OllamaClient(settings["host"], timeout_s=float(settings.get("timeout_s", 300)))
    client.require_models(settings["chat_model"])
    retries = int(settings.get("extract_retries", 2))
    pending = store.pending_extract_chunks()
    done = 0
    failed = 0
    total = len(pending)
    already = store.count("extract_jobs", "status IN ('done','failed')")
    log(f"extract start: {total} pending, {already} already processed")
    for index, unit in enumerate(pending, start=1):
        check()
        log(
            f"extract {index}/{total} {unit['id']} pos={unit['position']} "
            f"tokens={unit['token_count']} {unit['chapter']}"
        )
        try:
            payload = extract_chunk(
                client,
                settings["chat_model"],
                unit["id"],
                unit["chapter"],
                unit["text"],
                retries=retries,
            )
            rels = _unique_rels(unit["id"], payload.relationships)
            store.replace_raw_for_chunk(
                unit["id"],
                [e.model_dump() for e in payload.entities],
                rels,
            )
            store.save_extract_job(unit["id"], "done", payload.model_dump(), None)
            append_jsonl(
                log_path,
                {"ts": datetime.now(timezone.utc).isoformat(), "stage": "extract", "chunk_id": unit["id"], "ok": True},
            )
            done += 1
            log(
                f"extract {index}/{total} {unit['id']} ok "
                f"entities={len(payload.entities)} rels={len(rels)}"
            )
        except (OllamaError, sqlite3.IntegrityError, ValueError) as exc:
            store.save_extract_job(unit["id"], "failed", None, str(exc))
            append_jsonl(
                log_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "extract",
                    "chunk_id": unit["id"],
                    "ok": False,
                    "error": str(exc),
                },
            )
            failed += 1
            log(f"extract {index}/{total} {unit['id']} FAIL {exc}")
    _export(store)
    total = store.count("text_units")
    finished = store.count("extract_jobs", "status = 'done'")
    failed_n = store.count("extract_jobs", "status = 'failed'")
    log(f"extract finished: done={finished} failed={failed_n}")
    return {
        "chunks_total": total,
        "extract_done": finished,
        "extract_failed": failed_n,
        "this_run_done": done,
        "this_run_failed": failed,
        "raw_entities": store.count("raw_entities"),
        "raw_relationships": store.count("raw_relationships"),
    }


def _unique_rels(chunk_id: str, relationships) -> list[dict]:
    used: set[str] = set()
    rels: list[dict] = []
    for index, rel in enumerate(relationships):
        rid = stable_id(chunk_id, rel.source, rel.target, rel.type, prefix="r")
        if rid in used:
            rid = stable_id(chunk_id, rel.source, rel.target, rel.type, rel.description, str(index), prefix="r")
        used.add(rid)
        rels.append(
            {
                "id": rid,
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "description": rel.description,
                "quote": rel.quote,
                "confidence": rel.confidence,
            }
        )
    return rels


def _export(store: IndexStore) -> None:
    entities = [dict(r) for r in store.raw_entities()]
    rels = [dict(r) for r in store.raw_relationships()]
    write_jsonl(store.index_dir / "exports" / "extract" / "entities.jsonl", entities)
    write_jsonl(store.index_dir / "exports" / "extract" / "relationships.jsonl", rels)
    failed_rows = store.conn.execute("SELECT * FROM extract_jobs WHERE status = 'failed'").fetchall()
    write_jsonl(store.index_dir / "exports" / "extract" / "failed.jsonl", [dict(r) for r in failed_rows])
