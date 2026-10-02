from __future__ import annotations

from graphrag_lab.agents.schemas import ANSWER_SCHEMA
from graphrag_lab.models import RetrievalPacket
from graphrag_lab.ollama_client import OllamaClient


def answer_from_packet(
    client: OllamaClient,
    model: str,
    packet: RetrievalPacket,
) -> tuple[str, bool, list[str]]:
    messages = [
        {
            "role": "user",
            "content": (
                "Ответь на вопрос только по переданному контексту. "
                "Если данных недостаточно, так и скажи. Не используй знания вне контекста. "
                "Верни JSON: answer, enough_evidence, used_chunk_ids.\n\n"
                f"ВОПРОС: {packet.question}\n\nКОНТЕКСТ:\n{packet.context}"
            ),
        }
    ]
    raw = client.chat_json(model, messages, ANSWER_SCHEMA, temperature=0.0)
    answer = str(raw.get("answer") or "").strip()
    enough = bool(raw.get("enough_evidence"))
    used = [str(x) for x in raw.get("used_chunk_ids") or []]
    if not enough and "недостаточно" not in answer.lower():
        answer = "Недостаточно данных в извлечённом контексте.\n" + answer
    return answer, enough, used
