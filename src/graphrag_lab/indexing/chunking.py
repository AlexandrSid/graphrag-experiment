from __future__ import annotations

import re
from pathlib import Path

from graphrag_lab.models import TextUnit
from graphrag_lab.util import estimate_tokens

CHAPTER_RE = re.compile(r"^Глава\s+(\d+)\.(.*)$")
SCENE_BREAK = re.compile(r"^\s*\*\s*\*\s*\*\s*$")


def _split_paragraphs(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if SCENE_BREAK.match(line) or not line.strip():
            if buf:
                parts.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line.rstrip())
    if buf:
        parts.append("\n".join(buf).strip())
    return [p for p in parts if p]


def _pack(paragraphs: list[str], min_tokens: int, max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current).strip())
            overlap: list[str] = []
            overlap_count = 0
            for item in reversed(current):
                overlap.insert(0, item)
                overlap_count += estimate_tokens(item)
                if overlap_count >= overlap_tokens:
                    break
            current = overlap
            current_tokens = sum(estimate_tokens(item) for item in current)
            if current_tokens + para_tokens > max_tokens:
                current = []
                current_tokens = 0
        current.append(para)
        current_tokens += para_tokens
    if current:
        text = "\n\n".join(current).strip()
        if estimate_tokens(text) >= min(min_tokens, 80) or not chunks:
            chunks.append(text)
        elif chunks:
            chunks[-1] = (chunks[-1] + "\n\n" + text).strip()
    return chunks


def _read_book(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1251", errors="replace")


def chunk_book(
    path: Path,
    target_tokens: int = 650,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[TextUnit]:
    del target_tokens
    raw = _read_book(path)
    lines = raw.splitlines()
    started = False
    chapters: list[tuple[int, str, list[str]]] = []
    current_num = 0
    current_title = ""
    current_lines: list[str] = []
    for line in lines:
        match = CHAPTER_RE.match(line.strip())
        if match:
            if started and current_lines:
                chapters.append((current_num, current_title, current_lines))
            started = True
            current_num = int(match.group(1))
            current_title = line.strip()
            current_lines = []
            continue
        if started:
            current_lines.append(line)
    if started and current_lines:
        chapters.append((current_num, current_title, current_lines))
    if not chapters:
        chapters = [(0, "untitled", lines)]

    units: list[TextUnit] = []
    position = 0
    for num, title, body in chapters:
        paragraphs = _split_paragraphs("\n".join(body))
        packed = _pack(paragraphs, min_tokens, max_tokens, overlap_tokens)
        for piece in packed:
            token_count = estimate_tokens(piece)
            if token_count == 0:
                continue
            units.append(
                TextUnit(
                    id=f"c{position:04d}",
                    chapter=title,
                    chapter_num=num,
                    position=position,
                    text=piece,
                    token_count=token_count,
                )
            )
            position += 1
    return units
