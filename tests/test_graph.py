from graphrag_lab.indexing.communities import run_leiden
from graphrag_lab.indexing.graph import run_graph
from graphrag_lab.indexing.resolve_stage import rebuild_relationships
from graphrag_lab.models import TextUnit
from graphrag_lab.storage.sqlite import IndexStore


def _raw(store: IndexStore, rel_id: str, source: str, target: str, status: str = "verified") -> None:
    store.conn.execute(
        """
        INSERT INTO raw_relationships(id, chunk_id, source, target, rel_type, description, quote, confidence, status)
        VALUES (?, 'c0000', ?, ?, 'knows', '', 'quote', 1, ?)
        """,
        (rel_id, source, target, status),
    )
    store.conn.commit()


def test_graph_and_leiden(store: IndexStore) -> None:
    store.replace_text_units(
        [TextUnit(id="c0000", chapter="Глава 1.", chapter_num=1, position=0, text="x", token_count=1)]
    )
    store.replace_canonical(
        [
            {"id": "a", "name": "A", "type": "person", "aliases": ["A"], "description": ""},
            {"id": "b", "name": "B", "type": "person", "aliases": ["B"], "description": ""},
            {"id": "c", "name": "C", "type": "person", "aliases": ["C"], "description": ""},
            {"id": "d", "name": "D", "type": "person", "aliases": ["D"], "description": ""},
        ],
        [],
        [
            {"id": "e1", "source_id": "a", "target_id": "b", "rel_type": "knows", "description": "", "weight": 2, "confidence": 1},
            {"id": "e2", "source_id": "b", "target_id": "c", "rel_type": "knows", "description": "", "weight": 2, "confidence": 1},
            {"id": "e3", "source_id": "c", "target_id": "a", "rel_type": "knows", "description": "", "weight": 2, "confidence": 1},
            {"id": "e4", "source_id": "c", "target_id": "d", "rel_type": "knows", "description": "", "weight": 1, "confidence": 1},
        ],
        [],
        [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")],
    )
    _raw(store, "r1", "A", "B")
    _raw(store, "r2", "B", "C")
    _raw(store, "r3", "C", "A")
    _raw(store, "r4", "C", "D")
    stats = run_graph(store)
    assert stats["nodes"] == 4
    assert stats["edges"] >= 3
    leiden = run_leiden(store, {"leiden": {"seed": 42, "resolutions": [1.0], "min_community_size": 2}})
    assert leiden["communities"] >= 1


def test_rebuild_verified_keeps_entities(store: IndexStore) -> None:
    store.replace_canonical(
        [
            {"id": "harry", "name": "Гарри", "type": "person", "aliases": ["мальчик"], "description": "герой"},
            {"id": "hermione", "name": "Гермиона", "type": "person", "aliases": [], "description": ""},
        ],
        [],
        [],
        [],
        [("гарри", "harry"), ("гермиона", "hermione")],
    )
    _raw(store, "r-ok", "Гарри", "Гермиона", "verified")
    _raw(store, "r-self", "мальчик", "Гарри", "verified")
    _raw(store, "r-miss", "Гарри", "Неизвестный", "verified")
    stats = rebuild_relationships(store)
    assert stats["relationships_written"] == 1
    assert stats["skipped_self"] == 1
    assert stats["skipped_unmapped"] == 1
    assert store.count("entities") == 2
    assert store.count("merges") == 2
    row = store.conn.execute("SELECT status FROM raw_relationships WHERE id = 'r-ok'").fetchone()
    assert row["status"] == "accepted"
    rels = store.relationships()
    assert len(rels) == 1
    assert {rels[0]["source_id"], rels[0]["target_id"]} == {"harry", "hermione"}
