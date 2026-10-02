from __future__ import annotations

from pathlib import Path

from graphrag_lab.agents.answer import answer_from_packet
from graphrag_lab.config import load_query_config, ollama_settings
from graphrag_lab.models import AnswerResult
from graphrag_lab.ollama_client import OllamaClient
from graphrag_lab.query.retrieve import retrieve
from graphrag_lab.util import dump_json


def ask(index_dir: Path, question: str, mode: str) -> AnswerResult:
    packet = retrieve(index_dir, question, mode)
    cfg = load_query_config()
    settings = ollama_settings(cfg)
    client = OllamaClient(settings["host"], timeout_s=float(settings.get("timeout_s", 180)))
    answer, _enough, used = answer_from_packet(client, settings["chat_model"], packet)
    if used:
        packet.chunk_ids = [cid for cid in packet.chunk_ids if cid in set(used)] or packet.chunk_ids
        packet.citations = [c for c in packet.citations if c.chunk_id in set(packet.chunk_ids)]
    result = AnswerResult(answer=answer, packet=packet)
    dump_json(index_dir / "logs" / "last_query.json", result.model_dump())
    return result
