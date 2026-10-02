from pathlib import Path

from graphrag_lab.indexing.resolve_stage import ingest_resolve, start_resolve
from graphrag_lab.mailbox import Mailbox
from graphrag_lab.models import TextUnit
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import dump_json


def test_resolve_mailbox_roundtrip(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.replace_text_units(
        [TextUnit(id="c0000", chapter="Глава 1.", chapter_num=1, position=0, text="x", token_count=1)]
    )
    store.replace_raw_for_chunk(
        "c0000",
        [
            {"name": "Кел", "type": "person", "aliases": ["Келли"], "description": "герой"},
            {"name": "Келли", "type": "person", "aliases": [], "description": "тот же"},
        ],
        [
            {
                "id": "r1",
                "source": "Кел",
                "target": "река Ирл",
                "type": "sees",
                "description": "видит реку",
                "quote": "Кел видит реку",
                "confidence": 0.9,
            }
        ],
    )
    store.conn.execute("UPDATE raw_relationships SET status = 'accepted'")
    store.conn.execute(
        "INSERT INTO raw_entities(chunk_id, name, type, aliases_json, description) VALUES ('c0000','река Ирл','place','[]','река')"
    )
    store.conn.commit()
    mailbox = Mailbox(tmp_path)
    start = start_resolve(store, mailbox, {"mailbox": {"max_prompt_tokens": 200000}})
    assert start["waiting"] is True
    assert mailbox.pending() is not None
    prompt = mailbox.prompt_path().read_text(encoding="utf-8")
    assert "Кел" in prompt
    response = {
        "entities": [
            {"canonical_name": "Кел", "type": "person", "aliases": ["Кел", "Келли"], "description": "герой"},
            {"canonical_name": "река Ирл", "type": "place", "aliases": ["река Ирл"], "description": "река"},
        ],
        "rejected_merges": [],
    }
    path = tmp_path / "response.json"
    dump_json(path, response)
    stats = ingest_resolve(store, mailbox, response, {"mailbox": {"max_prompt_tokens": 200000}})
    assert stats["waiting"] is False
    names = {row["name"] for row in store.entities()}
    assert "Кел" in names
    assert mailbox.pending() is None
    store.close()


def test_invalid_ingest_keeps_job(tmp_path: Path) -> None:
    mailbox = Mailbox(tmp_path)
    from graphrag_lab.mailbox import MailboxJob

    mailbox.write_job(
        MailboxJob(job_id="j1", stage="resolve", kind="resolve", schema_name="ResolvePayload"),
        "prompt",
        {"type": "object"},
    )
    store = IndexStore(tmp_path)
    try:
        from graphrag_lab.indexing.resolve_stage import ingest_resolve

        try:
            ingest_resolve(store, mailbox, {"nope": True}, {})
            assert False, "should fail"
        except ValueError:
            assert mailbox.pending() is not None
    finally:
        store.close()
