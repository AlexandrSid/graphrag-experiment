from graphrag_lab.indexing.communities import run_leiden
from graphrag_lab.indexing.graph import run_graph
from graphrag_lab.models import TextUnit
from graphrag_lab.storage.sqlite import IndexStore


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
        [],
    )
    stats = run_graph(store)
    assert stats["nodes"] == 4
    assert stats["edges"] >= 3
    leiden = run_leiden(store, {"leiden": {"seed": 42, "resolutions": [1.0], "min_community_size": 2}})
    assert leiden["communities"] >= 1
