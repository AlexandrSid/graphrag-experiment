import json
from pathlib import Path

from graphrag_lab.agents.answer import build_answer_messages
from graphrag_lab.query.answer import ask, format_telemetry


class _FakeClient:
    def chat_json_stream(self, model, messages, schema, temperature=0.0):
        assert messages[0]["role"] == "system"
        assert "КОНТЕКСТ:\n\n" in messages[1]["content"]
        return (
            {"answer": "из весов модели", "enough_evidence": True, "used_chunk_ids": []},
            0.2,
            0.5,
        )


def test_prompt_sections_split_context_and_query() -> None:
    _messages, full = build_answer_messages("кто?", "чанк c0001")
    assert "=== SYSTEM ===" in full
    assert "=== RAG CONTEXT ===\nчанк c0001" in full
    assert "=== USER QUERY ===\nкто?" in full


def test_no_rag_dumps_empty_context(tmp_path: Path) -> None:
    index = tmp_path / "idx"
    dump = tmp_path / "queries"
    trace = ask(index, "кто такой Кел?", "global", no_rag=True, dump_dir=dump, client=_FakeClient())
    assert trace.baseline is True
    assert trace.retrieval_time_s == 0.0
    assert trace.retrieved_chunks == []
    assert trace.retrieved_communities == []
    assert "=== RAG CONTEXT ===\n\n=== USER QUERY ===" in trace.full_prompt
    assert trace.llm_ttft_s == 0.2
    assert trace.llm_total_time_s == 0.5
    files = list(dump.glob("*_global_norag.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["query"] == "кто такой Кел?"
    assert payload["llm_response"] == "из весов модели"
    assert payload["baseline"] is True
    assert payload["retrieval_time_s"] == 0.0
    panel = format_telemetry(trace)
    assert "Baseline: no-rag" in panel
    assert "[LLM Pre-fill / TTFT]: 0.20s" in panel
    assert "[Total Generation]: 0.50s" in panel
