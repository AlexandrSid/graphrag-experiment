from __future__ import annotations

import time
from dataclasses import dataclass

from graphrag_lab.agents.schemas import ANSWER_SCHEMA
from graphrag_lab.models import RetrievalPacket
from graphrag_lab.ollama_client import OllamaClient

SYSTEM = (
    "Ответь на вопрос только по переданному контексту. "
    "Если данных недостаточно, так и скажи. Не используй знания вне контекста. "
    "Верни JSON: answer, enough_evidence, used_chunk_ids."
)


@dataclass
class AnswerCall:
    answer: str
    enough: bool
    used: list[str]
    ttft_s: float | None
    total_s: float
    prompt_build_s: float
    full_prompt: str


def build_answer_messages(question: str, context: str) -> tuple[list[dict[str, str]], str]:
    full_prompt = (
        f"=== SYSTEM ===\n{SYSTEM}\n"
        f"=== RAG CONTEXT ===\n{context}\n"
        f"=== USER QUERY ===\n{question}"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}"},
    ]
    return messages, full_prompt


def answer_from_packet(client: OllamaClient, model: str, packet: RetrievalPacket) -> AnswerCall:
    started = time.perf_counter()
    messages, full_prompt = build_answer_messages(packet.question, packet.context)
    prompt_build_s = time.perf_counter() - started
    raw, ttft_s, total_s = client.chat_json_stream(model, messages, ANSWER_SCHEMA, temperature=0.0)
    answer = str(raw.get("answer") or "").strip()
    enough = bool(raw.get("enough_evidence"))
    used = [str(item) for item in raw.get("used_chunk_ids") or []]
    if not enough and "недостаточно" not in answer.lower():
        answer = "Недостаточно данных в извлечённом контексте.\n" + answer
    return AnswerCall(
        answer=answer,
        enough=enough,
        used=used,
        ttft_s=ttft_s,
        total_s=total_s,
        prompt_build_s=prompt_build_s,
        full_prompt=full_prompt,
    )
