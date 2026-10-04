from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from graphrag_lab.indexing.cancel import check
from graphrag_lab.indexing.progress import log

from graphrag_lab.agents.verify import verify_chunk
from graphrag_lab.config import ollama_settings
from graphrag_lab.ollama_client import OllamaClient, OllamaError
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import append_jsonl, write_jsonl


def run_verify(store: IndexStore, cfg: dict, log_path) -> dict:
    settings = ollama_settings(cfg)
    client = OllamaClient(settings["host"], timeout_s=float(settings.get("timeout_s", 300)))
    client.require_models(settings["chat_model"])
    retries = int(settings.get("extract_retries", 2))
    pending = store.pending_verify()
    by_chunk: dict[str, list] = defaultdict(list)
    for row in pending:
        by_chunk[row["chunk_id"]].append(dict(row))

    this_ok = 0
    this_fail = 0
    total = len(by_chunk)
    log(f"verify start: {total} chunks, {len(pending)} relationships")
    for index, (chunk_id, rels) in enumerate(by_chunk.items(), start=1):
        check()
        unit = store.text_unit(chunk_id)
        if unit is None:
            for rel in rels:
                store.save_verify(rel["id"], False, "missing chunk", "")
            log(f"verify {index}/{total} {chunk_id} missing chunk, rejected {len(rels)}")
            continue
        log(f"verify {index}/{total} {chunk_id} checking {len(rels)} relationships...")
        try:
            payload = verify_chunk(client, settings["chat_model"], unit["text"], rels, retries=retries)
            verdicts = {v.relationship_key: v for v in payload.verdicts}
            accepted_n = 0
            rejected_n = 0
            for rel in rels:
                verdict = verdicts.get(rel["id"])
                if verdict is None:
                    store.save_verify(rel["id"], False, "no verdict from verifier", rel.get("quote", ""))
                    rejected_n += 1
                else:
                    store.save_verify(rel["id"], verdict.accepted, verdict.reason, verdict.quote or rel.get("quote", ""))
                    if verdict.accepted:
                        accepted_n += 1
                    else:
                        rejected_n += 1
            append_jsonl(
                log_path,
                {"ts": datetime.now(timezone.utc).isoformat(), "stage": "verify", "chunk_id": chunk_id, "ok": True},
            )
            this_ok += 1
            log(f"verify {index}/{total} {chunk_id} ok accepted={accepted_n} rejected={rejected_n}")
        except OllamaError as exc:
            append_jsonl(
                log_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "verify",
                    "chunk_id": chunk_id,
                    "ok": False,
                    "error": str(exc),
                },
            )
            this_fail += 1
            log(f"verify {index}/{total} {chunk_id} FAIL {exc}")
            for rel in rels:
                if store.conn.execute(
                    "SELECT 1 FROM verify_jobs WHERE relationship_id = ?", (rel["id"],)
                ).fetchone() is None:
                    store.conn.execute(
                        "INSERT OR IGNORE INTO verify_jobs(relationship_id, status) VALUES (?, 'pending')",
                        (rel["id"],),
                    )
            store.conn.commit()

    accepted = store.count("raw_relationships", "status = 'accepted'")
    rejected = store.count("raw_relationships", "status = 'rejected'")
    pending_left = store.count("raw_relationships", "status = 'raw'")
    log(f"verify finished: accepted={accepted} rejected={rejected} pending={pending_left}")
    write_jsonl(
        store.index_dir / "exports" / "verify" / "relationships.jsonl",
        [dict(r) for r in store.raw_relationships()],
    )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "chunks_verified_this_run": this_ok,
        "chunks_failed_this_run": this_fail,
        "pending": store.count("raw_relationships", "status = 'raw'"),
    }
