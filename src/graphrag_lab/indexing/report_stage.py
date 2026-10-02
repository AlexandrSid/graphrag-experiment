from __future__ import annotations

import json

from graphrag_lab.agents.report import build_report_prompt, report_schema
from graphrag_lab.mailbox import Mailbox, MailboxJob, parse_payload
from graphrag_lab.models import ReportPayload
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import estimate_tokens, stable_id, write_jsonl


def _community_payload(store: IndexStore, row) -> dict:
    entity_ids = json.loads(row["entity_ids_json"] or "[]")
    rel_ids = json.loads(row["relationship_ids_json"] or "[]")
    entities = []
    for eid in entity_ids:
        ent = store.conn.execute("SELECT * FROM entities WHERE id = ?", (eid,)).fetchone()
        if ent:
            entities.append({"id": ent["id"], "name": ent["name"], "type": ent["type"], "description": ent["description"]})
    relationships = []
    evidence_ids: list[str] = []
    for rid in rel_ids:
        rel = store.conn.execute("SELECT * FROM relationships WHERE id = ?", (rid,)).fetchone()
        if not rel:
            continue
        ev = [dict(x) for x in store.evidence_for(rid)]
        evidence_ids.extend(item["chunk_id"] for item in ev)
        relationships.append(
            {
                "id": rel["id"],
                "source": rel["source_id"],
                "target": rel["target_id"],
                "type": rel["rel_type"],
                "description": rel["description"],
                "evidence": ev[:3],
            }
        )
    return {
        "community": row["community"],
        "level": row["level"],
        "parent": row["parent"],
        "size": row["size"],
        "entity_ids": entity_ids,
        "entities": entities,
        "relationships": relationships,
        "evidence_chunk_ids": sorted(set(evidence_ids)),
    }


def start_report(store: IndexStore, mailbox: Mailbox, cfg: dict) -> dict:
    levels = store.community_levels()
    store.set_graph_stat("report_levels", levels)
    store.set_graph_stat("report_done_levels", [])
    store.set_graph_stat("report_queue", [])
    return _queue_next(store, mailbox, cfg)


def _queue_next(store: IndexStore, mailbox: Mailbox, cfg: dict) -> dict:
    levels = store.graph_stat("report_levels", store.community_levels())
    done = set(store.graph_stat("report_done_levels", []))
    remaining = [lvl for lvl in levels if lvl not in done]
    if not remaining:
        mailbox.clear_current()
        return {"waiting": False, "remaining_levels": []}

    level = remaining[0]
    queue = store.graph_stat("report_queue", [])
    communities = [_community_payload(store, row) for row in store.communities(level)]
    if not queue:
        # split into token-budget batches of community ids
        budget = int((cfg.get("mailbox") or {}).get("max_prompt_tokens", 200000))
        batches: list[list[int]] = []
        current: list[dict] = []
        tokens = 0
        child_reports = [dict(r) for r in store.reports(level - 1)] if level > 0 else []
        for comm in communities:
            piece = build_report_prompt(level, [comm], child_reports)
            need = estimate_tokens(piece)
            if current and tokens + need > budget:
                batches.append([c["community"] for c in current])
                current = [comm]
                tokens = need
            else:
                current.append(comm)
                tokens += need
        if current:
            batches.append([c["community"] for c in current])
        queue = batches
        store.set_graph_stat("report_queue", queue)

    if not queue:
        done.add(level)
        store.set_graph_stat("report_done_levels", sorted(done))
        store.set_graph_stat("report_queue", [])
        return _queue_next(store, mailbox, cfg)

    batch_ids = queue[0]
    selected = [c for c in communities if c["community"] in batch_ids]
    child_reports = []
    if level > 0:
        for row in store.communities(level - 1):
            if row["parent"] in batch_ids:
                report = store.conn.execute(
                    "SELECT * FROM community_reports WHERE community = ? AND level = ?",
                    (row["community"], level - 1),
                ).fetchone()
                if report:
                    child_reports.append(dict(report))
    prompt = build_report_prompt(level, selected, child_reports)
    job = MailboxJob(
        job_id=stable_id("report", str(level), *map(str, batch_ids), prefix="report-"),
        stage="report",
        kind="report",
        level=level,
        schema_name="ReportPayload",
        input_ids=[str(i) for i in batch_ids],
        extra={"level": level, "batch": batch_ids},
    )
    mailbox.write_job(job, prompt, report_schema())
    return {"waiting": True, "level": level, "batch": batch_ids, "job_id": job.job_id, "remaining_levels": remaining}


def ingest_report(store: IndexStore, mailbox: Mailbox, response, cfg: dict) -> dict:
    payload = parse_payload(ReportPayload, response)
    assert isinstance(payload, ReportPayload)
    job = mailbox.pending()
    if job is None:
        raise ValueError("No pending mailbox job")
    level = int(job.level if job.level is not None else (job.extra or {}).get("level", 0))
    reports = []
    for item in payload.reports:
        reports.append(
            {
                "community": item.community,
                "level": level,
                "title": item.title,
                "summary": item.summary,
                "findings": item.findings,
                "evidence_chunk_ids": item.evidence_chunk_ids,
            }
        )
    store.upsert_reports(reports)
    write_jsonl(
        store.index_dir / "exports" / "community_reports.jsonl",
        [dict(r) for r in store.reports()],
    )
    mailbox.archive_current(payload.model_dump())
    queue = list(store.graph_stat("report_queue", []))
    if queue:
        queue = queue[1:]
    store.set_graph_stat("report_queue", queue)
    if not queue:
        done = set(store.graph_stat("report_done_levels", []))
        done.add(level)
        store.set_graph_stat("report_done_levels", sorted(done))
    return _queue_next(store, mailbox, cfg)
