from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from graphrag_lab.config import load_index_config, ollama_settings
from graphrag_lab.indexing.chunking import chunk_book
from graphrag_lab.indexing.communities import run_leiden
from graphrag_lab.indexing.embed import run_embed
from graphrag_lab.indexing.extract_stage import run_extract
from graphrag_lab.indexing.graph import run_graph
from graphrag_lab.indexing.report_stage import ingest_report, start_report
from graphrag_lab.indexing.resolve_stage import ingest_resolve, start_resolve
from graphrag_lab.indexing.verify_stage import run_verify
from graphrag_lab.mailbox import Mailbox
from graphrag_lab.models import STAGE_ORDER
from graphrag_lab.storage.manifest import read_manifest, write_manifest
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import file_sha256, write_jsonl

PREDECESSOR = {
    "chunk": None,
    "extract": "chunk",
    "verify": "extract",
    "resolve": "verify",
    "graph": "resolve",
    "leiden": "graph",
    "report": "leiden",
    "embed": "report",
}


def open_store(index_dir: Path) -> IndexStore:
    return IndexStore(index_dir)


def ensure_ready(store: IndexStore, stage: str, force: bool) -> None:
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stage}")
    prev = PREDECESSOR[stage]
    if prev is None:
        return
    status = store.stage_status(prev)
    if status != "done":
        raise RuntimeError(f"Stage '{stage}' requires '{prev}' to be done (now {status})")
    current = store.stage_status(stage)
    if current == "done" and not force:
        raise RuntimeError(f"Stage '{stage}' is already done. Re-run with --force")


def _reset_stage_data(store: IndexStore, stage: str) -> None:
    if stage == "chunk":
        store.conn.execute("DELETE FROM text_units")
    elif stage == "extract":
        store.conn.executescript(
            "DELETE FROM extract_jobs; DELETE FROM raw_entities; DELETE FROM raw_relationships;"
        )
    elif stage == "verify":
        store.conn.execute("DELETE FROM verify_jobs")
        store.conn.execute("UPDATE raw_relationships SET status = 'raw'")
    elif stage == "resolve":
        store.conn.executescript(
            "DELETE FROM entities; DELETE FROM entity_mentions; DELETE FROM relationships; "
            "DELETE FROM relationship_evidence; DELETE FROM merges;"
        )
    elif stage == "leiden":
        store.conn.execute("DELETE FROM communities")
    elif stage == "report":
        store.conn.execute("DELETE FROM community_reports")
    store.conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_stage(
    index_dir: Path,
    stage: str,
    input_path: Path | None = None,
    force: bool = False,
    config_path: Path | None = None,
) -> dict:
    cfg = load_index_config(config_path)
    store = open_store(index_dir)
    mailbox = Mailbox(index_dir)
    try:
        ensure_ready(store, stage, force)
        if force:
            store.mark_stale_after(stage)
            _reset_stage_data(store, stage)
        store.set_stage(stage, "running", started_at=_now())
        log_path = index_dir / "logs" / "index.jsonl"
        stats: dict
        if stage == "chunk":
            if input_path is None:
                raise ValueError("chunk requires --input")
            chunk_cfg = cfg.get("chunking") or {}
            units = chunk_book(
                input_path,
                target_tokens=int(chunk_cfg.get("target_tokens", 650)),
                min_tokens=int(chunk_cfg.get("min_tokens", 500)),
                max_tokens=int(chunk_cfg.get("max_tokens", 800)),
                overlap_tokens=int(chunk_cfg.get("overlap_tokens", 100)),
            )
            store.replace_text_units(units)
            write_jsonl(
                index_dir / "exports" / "chunk" / "text_units.jsonl",
                [
                    {
                        "id": u.id,
                        "chapter": u.chapter,
                        "chapter_num": u.chapter_num,
                        "position": u.position,
                        "token_count": u.token_count,
                    }
                    for u in units
                ],
            )
            ollama = ollama_settings(cfg)
            write_manifest(
                index_dir,
                {
                    **read_manifest(index_dir),
                    "book_path": str(input_path),
                    "book_sha256": file_sha256(input_path),
                    "prompt_version": (cfg.get("prompts") or {}).get("version", "1"),
                    "chat_model": ollama["chat_model"],
                    "embed_model": ollama["embed_model"],
                    "chunking": chunk_cfg,
                },
            )
            tokens = [u.token_count for u in units]
            stats = {
                "chunks": len(units),
                "chapters": len({u.chapter_num for u in units}),
                "token_min": min(tokens) if tokens else 0,
                "token_max": max(tokens) if tokens else 0,
                "token_median": sorted(tokens)[len(tokens) // 2] if tokens else 0,
            }
            store.set_stage(stage, "done", stats=stats, finished_at=_now())
            return stats
        if stage == "extract":
            stats = run_extract(store, cfg, log_path)
            pending = store.count("text_units") - store.count("extract_jobs", "status IN ('done','failed')")
            store.set_stage(stage, "done" if pending == 0 else "running", stats=stats, finished_at=_now() if pending == 0 else None)
            return stats
        if stage == "verify":
            stats = run_verify(store, cfg, log_path)
            pending = store.count("raw_relationships", "status = 'raw'")
            store.set_stage(stage, "done" if pending == 0 else "running", stats=stats, finished_at=_now() if pending == 0 else None)
            return stats
        if stage == "resolve":
            stats = start_resolve(store, mailbox, cfg)
            store.set_stage(stage, "waiting_llm" if stats.get("waiting") else "done", stats=stats, finished_at=None if stats.get("waiting") else _now())
            return stats
        if stage == "graph":
            stats = run_graph(store)
            store.set_stage(stage, "done", stats=stats, finished_at=_now())
            return stats
        if stage == "leiden":
            stats = run_leiden(store, cfg)
            store.set_stage(stage, "done", stats=stats, finished_at=_now())
            return stats
        if stage == "report":
            stats = start_report(store, mailbox, cfg)
            store.set_stage(stage, "waiting_llm" if stats.get("waiting") else "done", stats=stats, finished_at=None if stats.get("waiting") else _now())
            return stats
        if stage == "embed":
            stats = run_embed(store, cfg)
            store.set_stage(stage, "done", stats=stats, finished_at=_now())
            return stats
        raise ValueError(stage)
    except Exception:
        store.set_stage(stage, "failed", finished_at=_now())
        raise
    finally:
        store.close()


def ingest_mailbox(index_dir: Path, response_path: Path, config_path: Path | None = None) -> dict:
    cfg = load_index_config(config_path)
    store = open_store(index_dir)
    mailbox = Mailbox(index_dir)
    job = mailbox.pending()
    if job is None:
        raise RuntimeError("No pending mailbox job")
    try:
        response = mailbox.load_response(response_path)
        if job.stage == "resolve":
            stats = ingest_resolve(store, mailbox, response, cfg)
        elif job.stage == "report":
            stats = ingest_report(store, mailbox, response, cfg)
        else:
            raise RuntimeError(f"Unsupported mailbox stage: {job.stage}")
        if stats.get("waiting"):
            store.set_stage(job.stage, "waiting_llm", stats=stats)
        else:
            store.set_stage(job.stage, "done", stats=stats, finished_at=_now())
        return {"stage": job.stage, **stats}
    except ValueError:
        # invalid response: keep waiting
        store.set_stage(job.stage, "waiting_llm")
        raise
    finally:
        store.close()


def run_range(index_dir: Path, start: str, end: str, input_path: Path | None, force: bool, config_path: Path | None) -> list[dict]:
    if start not in STAGE_ORDER or end not in STAGE_ORDER:
        raise ValueError("Unknown stage in range")
    i, j = STAGE_ORDER.index(start), STAGE_ORDER.index(end)
    if j < i:
        raise ValueError("--to must be after --from")
    results = []
    for name in STAGE_ORDER[i : j + 1]:
        if name in {"resolve", "report"}:
            raise RuntimeError(f"Refusing to auto-run mailbox stage '{name}'. Run it alone.")
        results.append({"stage": name, **run_stage(index_dir, name, input_path=input_path, force=force, config_path=config_path)})
    return results
