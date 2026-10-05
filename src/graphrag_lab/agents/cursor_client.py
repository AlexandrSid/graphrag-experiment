from __future__ import annotations

import json
import re
import subprocess
from typing import Any


class CursorCliError(RuntimeError):
    """Ошибка при вызове CLI Cursor или парсинге ответа."""


class CursorCliClient:
    def __init__(self, executable: str = "cursor") -> None:
        self.executable = executable

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"### {role}:\n{content}")

        schema_hint = ""
        if schema:
            schema_hint = (
                "\n\nОтвет должен строго соответствовать следующей JSON-схеме:\n"
                f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
            )

        full_prompt = (
            "Ты бэкенд-процессор данных. Твой единственный ответ — валидный JSON-объект без пояснений, "
            "префиксов, markdown-разметки или суффиксов.\n\n"
            + "\n\n".join(prompt_parts)
            + schema_hint
        )

        cmd = [
            self.executable,
            "agent",
            "--model",
            model,
            "--prompt",
            full_prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                shell=True,
            )
        except FileNotFoundError as exc:
            raise CursorCliError(
                f"Команда '{self.executable}' не найдена в PATH."
            ) from exc

        if result.returncode != 0:
            raise CursorCliError(
                f"Cursor CLI завершился с кодом {result.returncode}. stderr:\n{result.stderr}"
            )

        return self._extract_json(result.stdout)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if code_block:
            cleaned = code_block.group(1).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise CursorCliError(f"Не удалось распарсить JSON из вывода Cursor CLI:\n{text}") from exc
