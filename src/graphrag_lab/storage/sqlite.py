from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from graphrag_lab.models import STAGE_ORDER, TextUnit


class IndexStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "exports").mkdir(exist_ok=True)
        (self.index_dir / "logs").mkdir(exist_ok=True)
        self.db_path = self.index_dir / "graph.sqlite"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init()

    def close(self) -> None:
        self.conn.close()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stages (
                name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                stats_json TEXT
            );
            CREATE TABLE IF NOT EXISTS text_units (
                id TEXT PRIMARY KEY,
                chapter TEXT,
                chapter_num INTEGER,
                position INTEGER,
                text TEXT NOT NULL,
                token_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS extract_jobs (
                chunk_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                raw_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS raw_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT,
                name TEXT,
                type TEXT,
                aliases_json TEXT,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS raw_relationships (
                id TEXT PRIMARY KEY,
                chunk_id TEXT,
                source TEXT,
                target TEXT,
                rel_type TEXT,
                description TEXT,
                quote TEXT,
                confidence REAL,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS verify_jobs (
                relationship_id TEXT PRIMARY KEY,
                accepted INTEGER,
                reason TEXT,
                quote TEXT,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                aliases_json TEXT,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,
                chunk_id TEXT,
                surface TEXT
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                rel_type TEXT,
                description TEXT,
                weight REAL,
                confidence REAL,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS relationship_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relationship_id TEXT,
                chunk_id TEXT,
                quote TEXT
            );
            CREATE TABLE IF NOT EXISTS communities (
                id TEXT PRIMARY KEY,
                community INTEGER,
                level INTEGER,
                parent INTEGER,
                title TEXT,
                entity_ids_json TEXT,
                relationship_ids_json TEXT,
                size INTEGER,
                modularity REAL
            );
            CREATE TABLE IF NOT EXISTS community_reports (
                community INTEGER,
                level INTEGER,
                title TEXT,
                summary TEXT,
                findings_json TEXT,
                evidence_json TEXT,
                PRIMARY KEY (community, level)
            );
            CREATE TABLE IF NOT EXISTS merges (
                alias TEXT PRIMARY KEY,
                canonical_id TEXT
            );
            CREATE TABLE IF NOT EXISTS graph_stats (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        for name in STAGE_ORDER:
            self.conn.execute(
                "INSERT OR IGNORE INTO stages(name, status) VALUES (?, 'pending')",
                (name,),
            )
        self.conn.commit()

    def stage_status(self, name: str) -> str:
        row = self.conn.execute("SELECT status FROM stages WHERE name = ?", (name,)).fetchone()
        return row["status"] if row else "pending"

    def all_stages(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT name, status, started_at, finished_at, stats_json FROM stages"
        ).fetchall()
        order = {name: i for i, name in enumerate(STAGE_ORDER)}
        items = [dict(row) for row in rows]
        items.sort(key=lambda item: order.get(item["name"], 99))
        return items

    def set_stage(
        self,
        name: str,
        status: str,
        stats: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        current = self.conn.execute("SELECT * FROM stages WHERE name = ?", (name,)).fetchone()
        stats_json = json.dumps(stats, ensure_ascii=False) if stats is not None else (current["stats_json"] if current else None)
        self.conn.execute(
            """
            UPDATE stages
            SET status = ?, started_at = COALESCE(?, started_at),
                finished_at = COALESCE(?, finished_at),
                stats_json = COALESCE(?, stats_json)
            WHERE name = ?
            """,
            (status, started_at, finished_at, stats_json, name),
        )
        self.conn.commit()

    def mark_stale_after(self, name: str) -> None:
        if name not in STAGE_ORDER:
            return
        idx = STAGE_ORDER.index(name)
        for later in STAGE_ORDER[idx + 1 :]:
            self.conn.execute("UPDATE stages SET status = 'stale' WHERE name = ?", (later,))
        self.conn.commit()

    def replace_text_units(self, units: Iterable[TextUnit]) -> None:
        self.conn.execute("DELETE FROM text_units")
        self.conn.executemany(
            """
            INSERT INTO text_units(id, chapter, chapter_num, position, text, token_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (u.id, u.chapter, u.chapter_num, u.position, u.text, u.token_count)
                for u in units
            ],
        )
        self.conn.commit()

    def text_units(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM text_units ORDER BY position"
        ).fetchall()

    def text_unit(self, chunk_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM text_units WHERE id = ?", (chunk_id,)).fetchone()

    def count(self, table: str, where: str = "1=1", args: tuple[Any, ...] = ()) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", args).fetchone()
        return int(row["n"])

    def pending_extract_chunks(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT t.* FROM text_units t
            LEFT JOIN extract_jobs j ON j.chunk_id = t.id
            WHERE j.chunk_id IS NULL OR j.status = 'pending'
            ORDER BY t.position
            """
        ).fetchall()

    def save_extract_job(self, chunk_id: str, status: str, raw: dict | None, error: str | None) -> None:
        self.conn.execute(
            """
            INSERT INTO extract_jobs(chunk_id, status, raw_json, error)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET status=excluded.status, raw_json=excluded.raw_json, error=excluded.error
            """,
            (chunk_id, status, json.dumps(raw, ensure_ascii=False) if raw else None, error),
        )
        self.conn.commit()

    def replace_raw_for_chunk(self, chunk_id: str, entities: list[dict], rels: list[dict]) -> None:
        self.conn.execute("DELETE FROM raw_entities WHERE chunk_id = ?", (chunk_id,))
        self.conn.execute("DELETE FROM raw_relationships WHERE chunk_id = ?", (chunk_id,))
        self.conn.executemany(
            """
            INSERT INTO raw_entities(chunk_id, name, type, aliases_json, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk_id,
                    item["name"],
                    item.get("type", "concept"),
                    json.dumps(item.get("aliases") or [], ensure_ascii=False),
                    item.get("description", ""),
                )
                for item in entities
            ],
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO raw_relationships(id, chunk_id, source, target, rel_type, description, quote, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'raw')
            """,
            [
                (
                    item["id"],
                    chunk_id,
                    item["source"],
                    item["target"],
                    item.get("type", "related"),
                    item.get("description", ""),
                    item.get("quote", ""),
                    float(item.get("confidence") or 0.5),
                )
                for item in rels
            ],
        )
        self.conn.commit()

    def raw_relationships(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute(
                "SELECT * FROM raw_relationships WHERE status = ?", (status,)
            ).fetchall()
        return self.conn.execute("SELECT * FROM raw_relationships").fetchall()

    def pending_verify(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT r.* FROM raw_relationships r
            LEFT JOIN verify_jobs v ON v.relationship_id = r.id
            WHERE v.relationship_id IS NULL OR v.status = 'pending'
            """
        ).fetchall()

    def save_verify(self, relationship_id: str, accepted: bool, reason: str, quote: str) -> None:
        self.conn.execute(
            """
            INSERT INTO verify_jobs(relationship_id, accepted, reason, quote, status)
            VALUES (?, ?, ?, ?, 'done')
            ON CONFLICT(relationship_id) DO UPDATE SET
                accepted=excluded.accepted, reason=excluded.reason, quote=excluded.quote, status='done'
            """,
            (relationship_id, int(accepted), reason, quote),
        )
        self.conn.execute(
            "UPDATE raw_relationships SET status = ? WHERE id = ?",
            ("accepted" if accepted else "rejected", relationship_id),
        )
        self.conn.commit()

    def accepted_raw(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM raw_relationships WHERE status = 'accepted'"
        ).fetchall()

    def raw_entities(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM raw_entities").fetchall()

    def replace_canonical(
        self,
        entities: list[dict[str, Any]],
        mentions: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        merges: list[tuple[str, str]],
    ) -> None:
        self.conn.execute("DELETE FROM entities")
        self.conn.execute("DELETE FROM entity_mentions")
        self.conn.execute("DELETE FROM relationships")
        self.conn.execute("DELETE FROM relationship_evidence")
        self.conn.execute("DELETE FROM merges")
        self.conn.executemany(
            "INSERT INTO entities(id, name, type, aliases_json, description) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    e["id"],
                    e["name"],
                    e.get("type", "concept"),
                    json.dumps(e.get("aliases") or [], ensure_ascii=False),
                    e.get("description", ""),
                )
                for e in entities
            ],
        )
        self.conn.executemany(
            "INSERT INTO entity_mentions(entity_id, chunk_id, surface) VALUES (?, ?, ?)",
            [(m["entity_id"], m["chunk_id"], m["surface"]) for m in mentions],
        )
        self.conn.executemany(
            """
            INSERT INTO relationships(id, source_id, target_id, rel_type, description, weight, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'canonical')
            """,
            [
                (
                    r["id"],
                    r["source_id"],
                    r["target_id"],
                    r.get("rel_type", "related"),
                    r.get("description", ""),
                    float(r.get("weight") or 1.0),
                    float(r.get("confidence") or 0.5),
                )
                for r in relationships
            ],
        )
        self.conn.executemany(
            "INSERT INTO relationship_evidence(relationship_id, chunk_id, quote) VALUES (?, ?, ?)",
            [(e["relationship_id"], e["chunk_id"], e["quote"]) for e in evidence],
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO merges(alias, canonical_id) VALUES (?, ?)",
            merges,
        )
        self.conn.commit()

    def entities(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM entities ORDER BY name").fetchall()

    def relationships(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM relationships").fetchall()

    def evidence_for(self, relationship_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM relationship_evidence WHERE relationship_id = ?",
            (relationship_id,),
        ).fetchall()

    def mentions(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM entity_mentions").fetchall()

    def set_graph_stat(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO graph_stats(key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def graph_stat(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM graph_stats WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def replace_communities(self, rows: list[dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM communities")
        self.conn.execute("DELETE FROM community_reports")
        self.conn.executemany(
            """
            INSERT INTO communities(id, community, level, parent, title, entity_ids_json, relationship_ids_json, size, modularity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["id"],
                    r["community"],
                    r["level"],
                    r.get("parent"),
                    r.get("title", ""),
                    json.dumps(r.get("entity_ids") or [], ensure_ascii=False),
                    json.dumps(r.get("relationship_ids") or [], ensure_ascii=False),
                    r.get("size", 0),
                    r.get("modularity"),
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def communities(self, level: int | None = None) -> list[sqlite3.Row]:
        if level is None:
            return self.conn.execute("SELECT * FROM communities ORDER BY level, community").fetchall()
        return self.conn.execute(
            "SELECT * FROM communities WHERE level = ? ORDER BY community", (level,)
        ).fetchall()

    def community_levels(self) -> list[int]:
        rows = self.conn.execute("SELECT DISTINCT level FROM communities ORDER BY level").fetchall()
        return [int(r["level"]) for r in rows]

    def upsert_reports(self, reports: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO community_reports(community, level, title, summary, findings_json, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["community"],
                    r["level"],
                    r.get("title", ""),
                    r.get("summary", ""),
                    json.dumps(r.get("findings") or [], ensure_ascii=False),
                    json.dumps(r.get("evidence_chunk_ids") or [], ensure_ascii=False),
                )
                for r in reports
            ],
        )
        self.conn.commit()

    def reports(self, level: int | None = None) -> list[sqlite3.Row]:
        if level is None:
            return self.conn.execute("SELECT * FROM community_reports ORDER BY level, community").fetchall()
        return self.conn.execute(
            "SELECT * FROM community_reports WHERE level = ? ORDER BY community", (level,)
        ).fetchall()

    def rows_as_dicts(self, rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]
