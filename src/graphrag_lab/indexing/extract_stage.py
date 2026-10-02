from __future__ import annotations

from datetime import datetime, timezone

from graphrag_lab.agents.extract import extract_chunk
from graphrag_lab.config import ollama_settings
from graphrag_lab.ollama_client import OllamaClient, OllamaError
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
    for unit in pending:
        try:
            payload = extract_chunk(
                client,
                settings["chat_model"],
                unit["id"],
                unit["chapter"],
                unit["text"],
                retries=retries,
            )
            rels = []
            for rel in payload.relationships:
                rels.append(
                    {
                        "id": stable_id(unit["id"], rel.source, rel.target, rel.type, prefix="r"),
                        "source": rel.source,
                        "target": rel.target,
                        "type": rel.type,
                        "description": rel.description,
                        "quote": rel.quote,
                        "confidence": rel.confidence,
                    }
                )
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
        except OllamaError as exc:
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
    _export(store)
    total = store.count("text_units")
    finished = store.count("extract_jobs", "status = 'done'")
    failed_n = store.count("extract_jobs", "status = 'failed'")
    return {
        "chunks_total": total,
        "extract_done": finished,
        "extract_failed": failed_n,
        "this_run_done": done,
        "this_run_failed": failed,
        "raw_entities": store.count("raw_entities"),
        "raw_relationships": store.count("raw_relationships"),
    }


def _export(store: IndexStore) -> None:
    entities = [dict(r) for r in store.raw_entities()]
    rels = [dict(r) for r in store.raw_relationships()]
    write_jsonl(store.index_dir / "exports" / "extract" / "entities.jsonl", entities)
    write_jsonl(store.index_dir / "exports" / "extract" / "relationships.jsonl", rels)
    failed_rows = store.conn.execute("SELECT * FROM extract_jobs WHERE status = 'failed'").fetchall()
    write_jsonl(store.index_dir / "exports" / "extract" / "failed.jsonl", [dict(r) for r in failed_rows])
