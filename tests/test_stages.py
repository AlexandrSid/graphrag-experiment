from pathlib import Path

import pytest

from graphrag_lab.indexing.runner import run_stage
from graphrag_lab.models import STAGE_ORDER, TextUnit
from graphrag_lab.storage.sqlite import IndexStore


def test_chunk_stage_and_stale(tmp_path: Path) -> None:
    book = tmp_path / "tiny.txt"
    book.write_text("Глава 1. Тест\n\nГерой Кел встречает реку Ирл.\n\n" * 20, encoding="utf-8")
    index = tmp_path / "idx"
    stats = run_stage(index, "chunk", input_path=book)
    assert stats["chunks"] >= 1
    store = IndexStore(index)
    try:
        store.set_stage("extract", "done")
        store.set_stage("verify", "done")
    finally:
        store.close()
    run_stage(index, "chunk", input_path=book, force=True)
    store = IndexStore(index)
    try:
        assert store.stage_status("chunk") == "done"
        assert store.stage_status("extract") == "stale"
        assert store.stage_status("verify") == "stale"
    finally:
        store.close()


def test_cannot_skip_predecessor(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires"):
        run_stage(tmp_path / "idx", "extract")


def test_stage_order_complete() -> None:
    assert STAGE_ORDER[0] == "chunk"
    assert STAGE_ORDER[-1] == "embed"
    assert "leiden" in STAGE_ORDER


def test_text_unit_roundtrip(store: IndexStore) -> None:
    store.replace_text_units(
        [TextUnit(id="c0000", chapter="Глава 1.", chapter_num=1, position=0, text="hi", token_count=1)]
    )
    rows = store.text_units()
    assert rows[0]["id"] == "c0000"
