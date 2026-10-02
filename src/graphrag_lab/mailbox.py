from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from graphrag_lab.util import dump_json, load_json


class MailboxJob(BaseModel):
    job_id: str
    stage: str
    kind: str
    level: int | None = None
    batch: int = 0
    schema_name: str
    input_ids: list[str] = []
    extra: dict[str, Any] = {}


class Mailbox:
    def __init__(self, index_dir: Path) -> None:
        self.root = index_dir / "mailbox"
        self.current = self.root / "current"
        self.inbox = self.root / "inbox"
        self.archive = self.root / "archive"
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(exist_ok=True)
        self.archive.mkdir(exist_ok=True)

    def pending(self) -> MailboxJob | None:
        path = self.current / "job.json"
        if not path.exists():
            return None
        return MailboxJob.model_validate(load_json(path))

    def prompt_path(self) -> Path:
        return self.current / "prompt.md"

    def schema_path(self) -> Path:
        return self.current / "schema.json"

    def write_job(self, job: MailboxJob, prompt: str, schema: dict[str, Any]) -> None:
        if self.current.exists():
            shutil.rmtree(self.current)
        self.current.mkdir(parents=True)
        dump_json(self.current / "job.json", job.model_dump())
        self.prompt_path().write_text(prompt, encoding="utf-8")
        dump_json(self.schema_path(), schema)

    def clear_current(self) -> None:
        if self.current.exists():
            shutil.rmtree(self.current)

    def archive_current(self, response: Any | None = None) -> None:
        job = self.pending()
        if job is None:
            return
        dest = self.archive / job.job_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("job.json", "prompt.md", "schema.json"):
            src = self.current / name
            if src.exists():
                shutil.copy2(src, dest / name)
        if response is not None:
            dump_json(dest / "response.json", response)
        dump_json(
            dest / "meta.json",
            {"archived_at": datetime.now(timezone.utc).isoformat()},
        )

    def load_response(self, path: Path | None = None) -> Any:
        target = path or (self.inbox / "response.json")
        return load_json(target)


def parse_payload(model: type[BaseModel], data: Any) -> BaseModel:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Mailbox response failed schema validation:\n{exc}") from exc
