from __future__ import annotations

from graphrag_lab.agents.schemas import VERIFY_SCHEMA
from graphrag_lab.models import VerifyPayload
from graphrag_lab.ollama_client import OllamaClient, OllamaError

SYSTEM = (
    "Ты проверяющий. Смотри только исходный фрагмент. "
    "Для каждой связи скажи, подтверждается ли она текстом. "
    "Не доверяй описанию связи само по себе. Верни JSON."
)


def verify_chunk(
    client: OllamaClient,
    model: str,
    chunk_text: str,
    relationships: list[dict],
    retries: int = 2,
) -> VerifyPayload:
    lines = []
    for item in relationships:
        lines.append(
            f"- key={item['id']}; {item['source']} --{item['rel_type']}--> {item['target']}; "
            f"desc={item['description']}; quote={item['quote']}"
        )
    body = f"{SYSTEM}\n\nФРАГМЕНТ:\n{chunk_text}\n\nСВЯЗИ:\n" + "\n".join(lines)
    messages = [{"role": "user", "content": body}]
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            raw = client.chat_json(model, messages, VERIFY_SCHEMA, temperature=0.0)
            return VerifyPayload.model_validate(raw)
        except (OllamaError, Exception) as exc:  # noqa: BLE001
            last_error = exc
    raise OllamaError(f"verify failed: {last_error}")
