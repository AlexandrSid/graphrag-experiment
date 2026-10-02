from __future__ import annotations

import json
from pathlib import Path

from graphrag_lab.config import load_query_config, ollama_settings
from graphrag_lab.models import Citation, RetrievalPacket
from graphrag_lab.ollama_client import OllamaClient
from graphrag_lab.storage.manifest import read_manifest
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.storage.vectors import VectorStore
from graphrag_lab.util import estimate_tokens


def _require_ready(index_dir: Path, store: IndexStore, cfg: dict) -> None:
    if store.stage_status("embed") != "done":
        raise RuntimeError("Index is not ready: stage embed is not done")
    manifest = read_manifest(index_dir)
    settings = ollama_settings(cfg)
    if manifest.get("embed_model") and manifest["embed_model"] != settings["embed_model"]:
        raise RuntimeError(
            f"Embedding model mismatch: index={manifest['embed_model']} local={settings['embed_model']}"
        )


def retrieve(index_dir: Path, question: str, mode: str) -> RetrievalPacket:
    cfg = load_query_config()
    store = IndexStore(index_dir)
    try:
        _require_ready(index_dir, store, cfg)
        settings = ollama_settings(cfg)
        retr = cfg.get("retrieval") or {}
        client = OllamaClient(settings["host"], timeout_s=float(settings.get("timeout_s", 180)))
        client.require_models(settings["embed_model"], settings["chat_model"])
        qvec = client.embed(settings["embed_model"], question)
        vectors = VectorStore(index_dir)
        if mode == "vector":
            return _vector(store, vectors, qvec, question, retr)
        if mode == "local":
            return _local(store, vectors, qvec, question, retr)
        if mode == "global":
            return _global(store, vectors, qvec, question, retr)
        raise ValueError(f"Unknown mode: {mode}")
    finally:
        store.close()


def _trim(store: IndexStore, question: str, mode: str, chunk_ids: list[str], extra_lines: list[str], entity_ids=None, rel_ids=None, comm_ids=None) -> RetrievalPacket:
    retr = load_query_config().get("retrieval") or {}
    budget = int(retr.get("max_packet_tokens", 1800))
    max_chunks = int(retr.get("max_chunks", 10))
    citations: list[Citation] = []
    parts = list(extra_lines)
    used = []
    tokens = estimate_tokens("\n".join(parts))
    for cid in chunk_ids[:max_chunks]:
        unit = store.text_unit(cid)
        if unit is None:
            continue
        block = f"[chunk {unit['id']} / {unit['chapter']}]\n{unit['text']}"
        need = estimate_tokens(block)
        if used and tokens + need > budget:
            break
        parts.append(block)
        used.append(cid)
        tokens += need
        quote = unit["text"].strip().split("\n")[0][:240]
        citations.append(Citation(chunk_id=cid, quote=quote, chapter=unit["chapter"]))
    return RetrievalPacket(
        mode=mode,
        question=question,
        chunk_ids=used,
        entity_ids=entity_ids or [],
        relationship_ids=rel_ids or [],
        community_ids=comm_ids or [],
        context="\n\n".join(parts),
        citations=citations,
    )


def _vector(store: IndexStore, vectors: VectorStore, qvec, question: str, retr: dict) -> RetrievalPacket:
    hits = vectors.search("chunks", qvec, int(retr.get("vector_top_k", 8)))
    return _trim(store, question, "vector", [h["id"] for h in hits], [])


def _local(store: IndexStore, vectors: VectorStore, qvec, question: str, retr: dict) -> RetrievalPacket:
    hits = vectors.search("entities", qvec, int(retr.get("entity_top_k", 5)))
    entity_ids = [h["id"] for h in hits]
    rel_ids: list[str] = []
    chunk_ids: list[str] = []
    lines = ["СУЩНОСТИ И СВЯЗИ:"]
    seen_e = set(entity_ids)
    for eid in list(entity_ids):
        ent = store.conn.execute("SELECT * FROM entities WHERE id = ?", (eid,)).fetchone()
        if not ent:
            continue
        lines.append(f"- {ent['name']} ({ent['type']}): {ent['description']}")
        rels = store.conn.execute(
            "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
            (eid, eid),
        ).fetchall()
        for rel in rels:
            rel_ids.append(rel["id"])
            other = rel["target_id"] if rel["source_id"] == eid else rel["source_id"]
            if other not in seen_e:
                seen_e.add(other)
                entity_ids.append(other)
            other_row = store.conn.execute("SELECT name FROM entities WHERE id = ?", (other,)).fetchone()
            other_name = other_row["name"] if other_row else other
            lines.append(f"  * {ent['name']} --{rel['rel_type']}--> {other_name}: {rel['description']}")
            for ev in store.evidence_for(rel["id"]):
                chunk_ids.append(ev["chunk_id"])
        for mention in store.conn.execute(
            "SELECT chunk_id FROM entity_mentions WHERE entity_id = ?", (eid,)
        ).fetchall():
            chunk_ids.append(mention["chunk_id"])
    # unique preserve order
    chunk_ids = list(dict.fromkeys(chunk_ids))
    return _trim(store, question, "local", chunk_ids, lines, entity_ids=entity_ids, rel_ids=list(dict.fromkeys(rel_ids)))


def _global(store: IndexStore, vectors: VectorStore, qvec, question: str, retr: dict) -> RetrievalPacket:
    hits = vectors.search("reports", qvec, int(retr.get("community_top_k", 6)))
    lines = ["ОТЧЁТЫ СООБЩЕСТВ:"]
    comm_ids = []
    chunk_ids: list[str] = []
    for hit in hits:
        comm_ids.append(hit["id"])
        level = hit.get("level")
        community = hit.get("community")
        report = None
        if level is not None and community is not None:
            report = store.conn.execute(
                "SELECT * FROM community_reports WHERE community = ? AND level = ?",
                (community, level),
            ).fetchone()
        if report:
            findings = json.loads(report["findings_json"] or "[]")
            lines.append(f"- [{hit['id']}] {report['title']}: {report['summary']}")
            for finding in findings:
                lines.append(f"  * {finding}")
            chunk_ids.extend(json.loads(report["evidence_json"] or "[]"))
        else:
            lines.append(f"- {hit.get('text', '')}")
    return _trim(store, question, "global", list(dict.fromkeys(chunk_ids)), lines, comm_ids=comm_ids)
