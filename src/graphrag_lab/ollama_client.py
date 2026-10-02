from __future__ import annotations

from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, timeout_s: float = 300) -> None:
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.host, timeout=self.timeout_s)

    def health(self) -> dict[str, Any]:
        with self._client() as client:
            try:
                response = client.get("/api/tags")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama is not reachable at {self.host}: {exc}") from exc
            return response.json()

    def require_models(self, *names: str) -> None:
        payload = self.health()
        available = {item.get("name", "") for item in payload.get("models", [])}
        available |= {item.get("model", "") for item in payload.get("models", [])}
        short = {name.split(":")[0] for name in available if name}
        missing = []
        for name in names:
            aliases = {name, name.split(":")[0], name.split(":")[0] + ":latest"}
            if available and aliases.isdisjoint(available | short):
                missing.append(name)
        if missing:
            raise OllamaError(
                f"Ollama at {self.host} is missing models: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(n for n in available if n)) or '(none)'}"
            )

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature},
        }
        with self._client() as client:
            try:
                response = client.post("/api/chat", json=body)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama chat failed: {exc}") from exc
            payload = response.json()
        message = payload.get("message") or {}
        content = message.get("content") or "{}"
        if isinstance(content, dict):
            return content
        import json

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned non-JSON content: {content[:400]}") from exc

    def embed(self, model: str, text: str) -> list[float]:
        with self._client() as client:
            try:
                response = client.post("/api/embed", json={"model": model, "input": text})
                if response.status_code == 404:
                    response = client.post("/api/embeddings", json={"model": model, "prompt": text})
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama embed failed: {exc}") from exc
            payload = response.json()
        if "embeddings" in payload:
            vectors = payload["embeddings"]
            return list(vectors[0] if vectors and isinstance(vectors[0], list) else vectors)
        if "embedding" in payload:
            return list(payload["embedding"])
        raise OllamaError(f"Unexpected embed payload keys: {list(payload)}")
