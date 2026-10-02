from pathlib import Path

import pytest

from graphrag_lab.storage.sqlite import IndexStore


@pytest.fixture
def tmp_index(tmp_path: Path) -> Path:
    return tmp_path / "index"


@pytest.fixture
def store(tmp_index: Path) -> IndexStore:
    return IndexStore(tmp_index)
