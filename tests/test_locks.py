from pathlib import Path

from graphrag_lab.indexing.locks import reconcile_running
from graphrag_lab.indexing.runner import abort_stage
from graphrag_lab.storage.sqlite import IndexStore


def test_reconcile_dead_running(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.set_stage("extract", "running")
    (tmp_path / "locks").mkdir(exist_ok=True)
    (tmp_path / "locks" / "extract.pid").write_text("9999999", encoding="utf-8")
    changed = reconcile_running(store, tmp_path)
    assert "extract" in changed
    assert store.stage_status("extract") == "interrupted"
    store.close()


def test_abort_stage(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.set_stage("extract", "running")
    store.close()
    changed = abort_stage(tmp_path, "extract")
    assert changed == ["extract"]
    store = IndexStore(tmp_path)
    assert store.stage_status("extract") == "interrupted"
    store.close()
