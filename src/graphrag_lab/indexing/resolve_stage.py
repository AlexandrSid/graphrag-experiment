from __future__ import annotations

from collections import defaultdict

from graphrag_lab.agents.resolve import build_resolve_prompt, resolve_schema
from graphrag_lab.mailbox import Mailbox, MailboxJob, parse_payload
from graphrag_lab.models import ResolvePayload
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import estimate_tokens, slugify, stable_id, write_jsonl


def _components(store: IndexStore) -> list[dict]:
    rels = [dict(r) for r in store.accepted_raw()]
    entities = [dict(r) for r in store.raw_entities()]
    parent: dict[str, str] = {}

    def find(name: str) -> str:
        key = name.strip().lower()
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            nxt = parent[key]
            parent[key] = root
            key = nxt
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    names = set()
    for ent in entities:
        names.add(ent["name"].strip())
    for rel in rels:
        names.add(rel["source"].strip())
        names.add(rel["target"].strip())
    for name in names:
        if name:
            find(name)
    for rel in rels:
        union(rel["source"], rel["target"])
    groups: dict[str, set[str]] = defaultdict(set)
    for name in names:
        if name:
            groups[find(name)].add(name)
    components = []
    for root, members in groups.items():
        member_l = {m.lower() for m in members}
        crels = [r for r in rels if r["source"].strip().lower() in member_l and r["target"].strip().lower() in member_l]
        cents = [e for e in entities if e["name"].strip() in members]
        components.append({"id": root, "names": sorted(members), "entities": cents, "relationships": crels})
    components.sort(key=lambda c: (-len(c["relationships"]), -len(c["names"])))
    return components


def start_resolve(store: IndexStore, mailbox: Mailbox, cfg: dict) -> dict:
    components = _components(store)
    store.set_graph_stat("resolve_components", [{"id": c["id"], "size": len(c["names"])} for c in components])
    store.set_graph_stat("resolve_queue", [c["id"] for c in components])
    store.set_graph_stat("resolve_done", [])
    return _queue_next(store, mailbox, cfg)


def _queue_next(store: IndexStore, mailbox: Mailbox, cfg: dict) -> dict:
    queue = list(store.graph_stat("resolve_queue", []))
    done = set(store.graph_stat("resolve_done", []))
    pending_ids = [cid for cid in queue if cid not in done]
    if not pending_ids:
        mailbox.clear_current()
        return {"waiting": False, "remaining": 0}

    budget = int((cfg.get("mailbox") or {}).get("max_prompt_tokens", 200000))
    components = {c["id"]: c for c in _components(store)}
    batch: list[dict] = []
    tokens = 0
    for cid in pending_ids:
        comp = components.get(cid)
        if not comp:
            done.add(cid)
            continue
        prompt = build_resolve_prompt(comp["entities"], comp["relationships"])
        need = estimate_tokens(prompt)
        if batch and tokens + need > budget:
            break
        batch.append(comp)
        tokens += need
        if tokens > budget and len(batch) == 1:
            break
    if not batch:
        mailbox.clear_current()
        return {"waiting": False, "remaining": 0}

    entities = []
    rels = []
    for comp in batch:
        entities.extend(comp["entities"])
        rels.extend(comp["relationships"])
    prompt = build_resolve_prompt(entities, rels)
    job = MailboxJob(
        job_id=stable_id(*[c["id"] for c in batch], prefix="resolve-"),
        stage="resolve",
        kind="resolve",
        schema_name="ResolvePayload",
        input_ids=[c["id"] for c in batch],
    )
    mailbox.write_job(job, prompt, resolve_schema())
    store.set_graph_stat("resolve_current_ids", [c["id"] for c in batch])
    return {"waiting": True, "remaining": len(pending_ids), "job_id": job.job_id, "batch": len(batch)}


def ingest_resolve(store: IndexStore, mailbox: Mailbox, response, cfg: dict) -> dict:
    payload = parse_payload(ResolvePayload, response)
    assert isinstance(payload, ResolvePayload)
    current_ids = set(store.graph_stat("resolve_current_ids", []))
    done = set(store.graph_stat("resolve_done", []))
    done.update(current_ids)
    store.set_graph_stat("resolve_done", sorted(done))

    accepted = [dict(r) for r in store.accepted_raw()]
    raw_entities = [dict(r) for r in store.raw_entities()]
    alias_to_canon: dict[str, str] = {}
    entities: list[dict] = []
    existing = {row["id"]: dict(row) for row in store.entities()}
    for item in payload.entities:
        aliases = list(dict.fromkeys([item.canonical_name, *item.aliases]))
        eid = existing_id_for(existing, item.canonical_name) or slugify(item.canonical_name)
        if eid in existing:
            prev = existing[eid]
            aliases = list(dict.fromkeys(aliases + (prev.get("aliases") or [])))
        entity = {
            "id": eid,
            "name": item.canonical_name,
            "type": item.type,
            "aliases": aliases,
            "description": item.description,
        }
        entities.append(entity)
        existing[eid] = entity
        for alias in aliases:
            alias_to_canon[alias.strip().lower()] = eid

    # keep previously resolved entities not in this batch
    kept = [dict(row) for row in store.entities()]
    kept_ids = {e["id"] for e in entities}
    for row in kept:
        if row["id"] not in kept_ids:
            aliases = _as_list(row.get("aliases_json"))
            entities.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    "aliases": aliases,
                    "description": row["description"],
                }
            )
            for alias in [row["name"], *aliases]:
                alias_to_canon.setdefault(str(alias).strip().lower(), row["id"])

    def canon(name: str) -> str | None:
        return alias_to_canon.get(name.strip().lower())

    # fallback: unmapped accepted names become their own entities
    for row in raw_entities:
        key = row["name"].strip().lower()
        if key and key not in alias_to_canon:
            eid = slugify(row["name"])
            alias_to_canon[key] = eid
            if eid not in {e["id"] for e in entities}:
                entities.append(
                    {
                        "id": eid,
                        "name": row["name"],
                        "type": row.get("type") or "concept",
                        "aliases": [row["name"]],
                        "description": row.get("description") or "",
                    }
                )

    mentions = []
    for row in raw_entities:
        eid = canon(row["name"])
        if eid:
            mentions.append({"entity_id": eid, "chunk_id": row["chunk_id"], "surface": row["name"]})

    rels: dict[str, dict] = {row["id"]: dict(row) for row in store.relationships()}
    evidence = [
        {"relationship_id": e["relationship_id"], "chunk_id": e["chunk_id"], "quote": e["quote"]}
        for e in store.conn.execute("SELECT * FROM relationship_evidence").fetchall()
    ]
    for row in accepted:
        sid = canon(row["source"])
        tid = canon(row["target"])
        if not sid or not tid or sid == tid:
            continue
        rid = stable_id(sid, tid, row["rel_type"], prefix="e")
        if rid not in rels:
            rels[rid] = {
                "id": rid,
                "source_id": sid,
                "target_id": tid,
                "rel_type": row["rel_type"],
                "description": row["description"],
                "weight": 1.0,
                "confidence": row["confidence"],
            }
        else:
            rels[rid]["weight"] = float(rels[rid]["weight"]) + 1.0
        evidence.append({"relationship_id": rid, "chunk_id": row["chunk_id"], "quote": row["quote"]})

    merges = [(alias, cid) for alias, cid in alias_to_canon.items()]
    store.replace_canonical(entities, mentions, list(rels.values()), evidence, merges)
    write_jsonl(
        store.index_dir / "exports" / "resolve" / "merges.jsonl",
        [{"alias": a, "canonical_id": c} for a, c in merges],
    )
    write_jsonl(store.index_dir / "exports" / "entities.jsonl", entities)
    write_jsonl(store.index_dir / "exports" / "relationships.jsonl", list(rels.values()))
    mailbox.archive_current(payload.model_dump())
    nxt = _queue_next(store, mailbox, cfg)
    nxt.update({"entities": len(entities), "relationships": len(rels), "rejected_merges": len(payload.rejected_merges)})
    return nxt


def existing_id_for(existing: dict, name: str) -> str | None:
    key = name.strip().lower()
    for eid, row in existing.items():
        aliases = row.get("aliases") or _as_list(row.get("aliases_json"))
        names = [row.get("name", ""), *aliases]
        if any(str(n).strip().lower() == key for n in names):
            return eid
    return None


def _as_list(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    import json

    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
