import importlib
import inspect
from pathlib import Path

import pytest

from graphrag_lab.query.retrieve import retrieve
from graphrag_lab.storage.sqlite import IndexStore


def test_query_module_has_no_cloud_imports() -> None:
    module = importlib.import_module("graphrag_lab.query.answer")
    source = inspect.getsource(module)
    assert "google" not in source.lower()
    assert "gemini" not in source.lower()
    retrieve_src = inspect.getsource(importlib.import_module("graphrag_lab.query.retrieve"))
    assert "google" not in retrieve_src.lower()


def test_query_refuses_without_embed(tmp_path: Path) -> None:
    IndexStore(tmp_path)
    with pytest.raises(RuntimeError, match="embed"):
        retrieve(tmp_path, "кто такой кел?", "local")
