from __future__ import annotations

from graphrag_lab.agents.schemas import EXTRACT_SCHEMA
from graphrag_lab.models import ExtractPayload
from graphrag_lab.ollama_client import OllamaClient, OllamaError

SYSTEM = (
    "Ты экстрактор сущностей. Работай только с переданным фрагментом. "
    "Верни JSON по схеме. Не выдумывай факты вне фрагмента. "
    "Типы сущностей: person, place, event, org, object, concept. "
    "Для каждой связи обязательна короткая цитата из фрагмента."
)


def extract_chunk(
    client: OllamaClient,
    model: str,
    chunk_id: str,
    chapter: str,
    text: str,
    retries: int = 2,
) -> ExtractPayload:
    messages = [
        {"role": "user", "content": f"{SYSTEM}\n\nchunk_id={chunk_id}\nchapter={chapter}\n\n{text}"}
    ]
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            raw = client.chat_json(model, messages, EXTRACT_SCHEMA, temperature=0.0)
            return ExtractPayload.model_validate(raw)
        except (OllamaError, Exception) as exc:  # noqa: BLE001
            last_error = exc
    raise OllamaError(f"extract failed for {chunk_id}: {last_error}")
