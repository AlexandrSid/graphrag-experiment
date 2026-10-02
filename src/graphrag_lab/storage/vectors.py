from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

TABLES = ("chunks", "entities", "reports")


class VectorStore:
    def __init__(self, index_dir: Path, dim: int | None = None) -> None:
        self.path = index_dir / "vectors.lancedb"
        self.dim = dim
        self._db = None

    def _db_handle(self):
        if self._db is None:
            import lancedb

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.path))
        return self._db

    def replace_table(self, name: str, rows: list[dict]) -> int:
        if name not in TABLES:
            raise ValueError(name)
        db = self._db_handle()
        if name in db.table_names():
            db.drop_table(name)
        if not rows:
            return 0
        if self.dim is None and rows:
            self.dim = len(rows[0]["vector"])
        db.create_table(name, data=rows)
        return len(rows)

    def search(self, name: str, vector: list[float], k: int) -> list[dict]:
        db = self._db_handle()
        if name not in db.table_names():
            return []
        table = db.open_table(name)
        hits = table.search(np.array(vector, dtype="float32")).limit(k).to_list()
        return hits
