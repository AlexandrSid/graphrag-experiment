from pathlib import Path

from graphrag_lab.indexing.chunking import chunk_book
from graphrag_lab.util import estimate_tokens


def test_chunk_book_uses_chapter_metadata_not_whole_chapter(tmp_path: Path) -> None:
    paragraphs = ["Это предложение номер {} про вымышленный город Зарнвелл.".format(i) for i in range(80)]
    text = "реклама библиотеки\nссылка\n\nГлава 1. Начало\n\n" + "\n\n".join(paragraphs)
    text += "\n\nГлава 2. Продолжение\n\n" + "\n\n".join(paragraphs)
    path = tmp_path / "tiny.txt"
    path.write_text(text, encoding="utf-8")
    units = chunk_book(path, min_tokens=80, max_tokens=160, overlap_tokens=20)
    assert len(units) > 2
    assert all(u.chapter.startswith("Глава") for u in units)
    assert {u.chapter_num for u in units} == {1, 2}
    assert max(u.token_count for u in units) <= 220
    assert not any("реклама библиотеки" in u.text for u in units)
    assert estimate_tokens(units[0].text) > 0
