from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from graphrag_lab.agents.schemas import EXTRACT_SCHEMA
from graphrag_lab.models import ExtractPayload
from graphrag_lab.indexing.progress import log
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import append_jsonl, stable_id, write_jsonl

NUM_WORKERS = 3
BATCH_PER_WORKER = 50

SYSTEM = (
    "Ты прецизионный экстрактор графа знаний. Твоя работа подчиняется строгим правилам изоляции:\n"
    "1. ОБРАБОТКА СТРОГО ПО ОДНОМУ ЧАНКУ: обрабатывай каждый чанк как абсолютно изолированный документ.\n"
    "2. ЗАПРЕЩЕНО ОБЪЕДИНЯТЬ ЧАНКИ: запрещено делать общие выводы, сквозные связи между разными chunk_id или объединять сущности на этапе экстракции.\n"
    "3. ЛОКАЛЬНОСТЬ ЦИТАТ: цитата (quote) для каждой связи обязана присутствовать дословно ТОЛЬКО в тексте текущего чанка. Цитаты из соседних чанков недопустимы.\n"
    "4. ПОЛНОТА ВЫХОДА: массив 'results' обязан содержать РОВНО столько же элементов, сколько передано во входном файле. Каждый входной chunk_id обязан иметь собственный отдельный объект в массиве.\n"
    "5. ФОРМАТ: верни валидный JSON без markdown-оберток вида: {\"results\": [ { \"chunk_id\": \"...\", \"payload\": { \"entities\": [...], \"relationships\": [...] } } ]}"
)


def run_extract(store: IndexStore, cfg: dict, log_path) -> dict:
    mailbox_dir = store.index_dir / "mailbox" / "extract"
    mailbox_dir.mkdir(parents=True, exist_ok=True)

    imported_done = 0
    imported_failed = 0
    found_any_output = False

    for w in range(1, NUM_WORKERS + 1):
        worker_dir = mailbox_dir / f"worker_{w}"
        output_file = worker_dir / "output.json"
        if output_file.exists():
            found_any_output = True
            log(f"Найден {output_file}, импортирую данные...")
            stats = _ingest_output(store, output_file, log_path)
            imported_done += stats.get("imported_done", 0)
            imported_failed += stats.get("imported_failed", 0)

    if found_any_output:
        _export(store)
        log(f"Импорт завершен: успешно {imported_done}, с ошибкой {imported_failed}")

    pending = store.pending_extract_chunks()
    if not pending:
        log("Все чанки книги уже успешно извлечены (done)!")
        return {"chunks_total": store.count("text_units"), "status": "completed"}

    total_batch_size = NUM_WORKERS * BATCH_PER_WORKER
    to_process = pending[:total_batch_size]

    log(f"\n{'='*70}")
    log(f"Формирование заданий для {NUM_WORKERS} параллельных воркеров (всего {len(to_process)} чанков):")

    for w in range(NUM_WORKERS):
        w_id = w + 1
        worker_dir = mailbox_dir / f"worker_{w_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        w_chunks = to_process[w * BATCH_PER_WORKER : (w + 1) * BATCH_PER_WORKER]
        if not w_chunks:
            continue

        input_file = worker_dir / "input_chunks.json"
        prompt_file = worker_dir / "prompt.md"
        output_file = worker_dir / "output.json"

        chunks_data = [
            {"chunk_id": u["id"], "chapter": u["chapter"], "text": u["text"]}
            for u in w_chunks
        ]
        input_file.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8")

        prompt_content = (
            f"{SYSTEM}\n\n"
            f"JSON-СХЕМА КАЖДОГО PAYLOAD:\n{json.dumps(EXTRACT_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
            f"ВХОДНЫЕ ДАННЫЕ ВОРКЕРА: `{input_file.as_posix()}`\n\n"
            f"АЛГОРИТМ РАБОТЫ АГЕНТА:\n"
            f"Итерируйся по списку чанков по очереди: `chunk[0] -> chunk[1] -> ...`.\n"
            f"Извлеки сущности и связи для текущего чанка строго по его тексту, зафиксируй результат, и только затем переходи к следующему.\n"
            f"Итоговый файл `{output_file.as_posix()}` должен содержать результаты для всех {len(chunks_data)} чанков без пропусков и объединений."
        )
        prompt_file.write_text(prompt_content, encoding="utf-8")

        log(f"  Воркер {w_id}: {len(chunks_data)} чанков -> {input_file}")

    log(f"{'='*70}\n")
    log("Пайплайн ожидает (waiting_cursor). Запустите обработку в 3 параллельных вкладках Cursor.")

    return {
        "status": "waiting_cursor",
        "pending_total": len(pending),
        "assigned_now": len(to_process),
    }


def _ingest_output(store: IndexStore, output_file: Path, log_path) -> dict:
    try:
        raw_data = json.loads(output_file.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Ошибка чтения {output_file}: {exc}")
        return {"error": str(exc)}

    results = raw_data.get("results", [])
    done = 0
    failed = 0

    for item in results:
        chunk_id = item.get("chunk_id")
        try:
            payload = ExtractPayload.model_validate(item["payload"])
            rels = _unique_rels(chunk_id, payload.relationships)
            store.replace_raw_for_chunk(
                chunk_id,
                [e.model_dump() for e in payload.entities],
                rels,
            )
            store.save_extract_job(chunk_id, "done", payload.model_dump(), None)
            append_jsonl(
                log_path,
                {"ts": datetime.now(timezone.utc).isoformat(), "stage": "extract", "chunk_id": chunk_id, "ok": True},
            )
            done += 1
        except Exception as exc:
            store.save_extract_job(chunk_id, "failed", None, str(exc))
            append_jsonl(
                log_path,
                {"ts": datetime.now(timezone.utc).isoformat(), "stage": "extract", "chunk_id": chunk_id, "ok": False, "error": str(exc)},
            )
            failed += 1

    output_file.unlink(missing_ok=True)
    return {"imported_done": done, "imported_failed": failed}


def _unique_rels(chunk_id: str, relationships) -> list[dict]:
    used: set[str] = set()
    rels: list[dict] = []
    for index, rel in enumerate(relationships):
        rid = stable_id(chunk_id, rel.source, rel.target, rel.type, prefix="r")
        if rid in used:
            rid = stable_id(chunk_id, rel.source, rel.target, rel.type, rel.description, str(index), prefix="r")
        used.add(rid)
        rels.append(
            {
                "id": rid,
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "description": rel.description,
                "quote": rel.quote,
                "confidence": rel.confidence,
            }
        )
    return rels


def _export(store: IndexStore) -> None:
    entities = [dict(r) for r in store.raw_entities()]
    rels = [dict(r) for r in store.raw_relationships()]
    write_jsonl(store.index_dir / "exports" / "extract" / "entities.jsonl", entities)
    write_jsonl(store.index_dir / "exports" / "extract" / "relationships.jsonl", rels)
    failed_rows = store.conn.execute("SELECT * FROM extract_jobs WHERE status = 'failed'").fetchall()
    write_jsonl(store.index_dir / "exports" / "extract" / "failed.jsonl", [dict(r) for r in failed_rows])
