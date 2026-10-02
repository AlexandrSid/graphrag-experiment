from __future__ import annotations

import json
from typing import Any

from graphrag_lab.agents.schemas import RESOLVE_SCHEMA


def build_resolve_prompt(entities: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> str:
    payload = {
        "entities": entities,
        "relationships": [
            {
                "source": r["source"],
                "target": r["target"],
                "type": r["rel_type"],
                "description": r["description"],
            }
            for r in relationships
        ],
    }
    return (
        "Ты детерминируешь сущности одной книги. Склей очевидные алиасы одного "
        "персонажа/места/понятия в каноническое имя. Не сливай разных людей с похожими именами.\n"
        "Верни JSON по схеме: entities[{canonical_name, type, aliases, description}], "
        "rejected_merges[{names, reason}].\n"
        "aliases должны включать все встреченные написания, в том числе каноническое имя.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


def resolve_schema() -> dict:
    return RESOLVE_SCHEMA
