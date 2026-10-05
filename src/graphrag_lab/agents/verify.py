# verify.py
from __future__ import annotations

from graphrag_lab.agents.schemas import VERIFY_SCHEMA
from graphrag_lab.agents.cursor_client import CursorCliClient, CursorCliError
from graphrag_lab.models import VerifyPayload

DEFAULT_MODEL = "cursor-grok-4.6-high-fast"

SYSTEM = (
    "Ты проверяющий. Смотри только исходный фрагмент. "
    "Для каждой связи скажи, подтверждается ли она текстом. "
    "Не доверяй описанию связи само по себе. Верни JSON."
)


def verify_chunk(
    client: CursorCliClient,
    model: str = DEFAULT_MODEL,
    chunk_text: str = "",
    relationships: list[dict] | None = None,
    retries: int = 2,
) -> VerifyPayload:
    relationships = relationships or []
    lines = [
        f"- key={item['id']}; {item['source']} --{item['rel_type']}--> {item['target']}; "
        f"desc={item['description']}; quote={item['quote']}"
        for item in relationships
    ]

    body = f"ФРАГМЕНТ:\n{chunk_text}\n\nСВЯЗИ ДЛЯ ПРОВЕРКИ:\n" + "\n".join(lines)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
    ]
    last_error: Exception | None = None

    for _ in range(retries + 1):
        try:
            raw = client.chat_json(model, messages, VERIFY_SCHEMA, temperature=0.0)
            return VerifyPayload.model_validate(raw)
        except (CursorCliError, Exception) as exc:  # noqa: BLE001
            last_error = exc

    raise CursorCliError(f"Verify failed: {last_error}")