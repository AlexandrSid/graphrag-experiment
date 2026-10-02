from __future__ import annotations

import json
from typing import Any

from graphrag_lab.agents.schemas import REPORT_SCHEMA


def build_report_prompt(level: int, communities: list[dict[str, Any]], child_reports: list[dict[str, Any]]) -> str:
    payload = {
        "level": level,
        "communities": communities,
        "child_reports": child_reports,
    }
    extra = ""
    if child_reports:
        extra = " Используй отчёты дочерних сообществ, не выдумывай новых фактов."
    return (
        "Ты пишешь краткие отчёты сообществ графа книги."
        f"{extra} Для каждого сообщества верни title, summary, findings "
        "и evidence_chunk_ids только из переданных данных.\n"
        "JSON: reports[{community, title, summary, findings, evidence_chunk_ids}].\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


def report_schema() -> dict:
    return REPORT_SCHEMA
