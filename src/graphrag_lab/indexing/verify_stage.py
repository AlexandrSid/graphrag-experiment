from __future__ import annotations

from collections import defaultdict
import json
from datetime import datetime, timezone
from pathlib import Path

from graphrag_lab.agents.schemas import VERIFY_SCHEMA
from graphrag_lab.models import VerifyPayload
from graphrag_lab.indexing.progress import log
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import append_jsonl, write_jsonl

NUM_WORKERS = 6
CHUNKS_PER_WORKER = 20  # 6 воркеров * 20 чанков = 120 чанков со всеми их связями за раз

SYSTEM = (
    "Ты прецизионный верификатор связей графа знаний. Правила валидации:\n"
    "1. СТРОГАЯ ИЗОЛЯЦИЯ: оценивай связи каждого чанка строго по тексту переданного фрагмента.\n"
    "2. БЕЗ ДОМЫСЛОВ: не подтверждай связь (accepted: false), если она не следует прямо из текста фрагмента.\n"
    "3. ЦИТИРОВАНИЕ: если accepted: true, укажи точную короткую цитату из текста чанка, подтверждающую связь.\n"
    "4. ПОЛНОТА: верни вердикты по ВСЕМ связям всех переданных чанков без пропусков.\n"
    "5. ФОРМАТ: валидный JSON вида: {\"results\": [ { \"chunk_id\": \"...\", \"verdicts\": [ { \"relationship_key\": \"...\", \"accepted\": true/false, \"reason\": \"...\", \"quote\": \"...\" } ] } ]}"
)


def run_verify(store: IndexStore, cfg: dict, log_path) -> dict:
    mailbox_dir = store.index_dir / "mailbox" / "verify"
    mailbox_dir.mkdir(parents=True, exist_ok=True)

    # 1. Проверяем готовые ответы воркеров
    imported_ok = 0
    imported_fail = 0
    found_any_output = False

    for w in range(1, NUM_WORKERS + 1):
        worker_dir = mailbox_dir / f"worker_{w}"
        output_file = worker_dir / "output.json"
        if output_file.exists():
            found_any_output = True
            log(f"Найден {output_file}, импортирую верификации...")
            stats = _ingest_output(store, output_file, log_path)
            imported_ok += stats.get("verified_chunks", 0)
            imported_fail += stats.get("failed_chunks", 0)

    if found_any_output:
        write_jsonl(
            store.index_dir / "exports" / "verify" / "relationships.jsonl",
            [dict(r) for r in store.raw_relationships()],
        )
        log(f"Импорт верификации завершен: успешно чанков {imported_ok}, с ошибкой {imported_fail}")

    # 2. Проверяем оставшиеся неподтвержденные связи
    pending = store.pending_verify()
    if not pending:
        log("Все связи графа знаний успешно проверены (done)!")
        return {"status": "completed"}

    # Группируем связи по чанкам
    by_chunk: dict[str, list] = defaultdict(list)
    for row in pending:
        by_chunk[row["chunk_id"]].append(dict(row))

    all_chunk_ids = list(by_chunk.keys())
    total_batch_chunks = NUM_WORKERS * CHUNKS_PER_WORKER
    to_process_ids = all_chunk_ids[:total_batch_chunks]

    log(f"\n{'='*70}")
    log(f"Формирование заданий verify для {NUM_WORKERS} параллельных воркеров (всего {len(to_process_ids)} чанков):")

    for w in range(NUM_WORKERS):
        w_id = w + 1
        worker_dir = mailbox_dir / f"worker_{w_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        w_chunk_ids = to_process_ids[w * CHUNKS_PER_WORKER : (w + 1) * CHUNKS_PER_WORKER]
        if not w_chunk_ids:
            continue

        input_file = worker_dir / "input_rels.json"
        prompt_file = worker_dir / "prompt.md"
        output_file = worker_dir / "output.json"

        batch_data = []
        for cid in w_chunk_ids:
            unit = store.text_unit(cid)
            if unit:
                batch_data.append({
                    "chunk_id": cid,
                    "text": unit["text"],
                    "relationships": by_chunk[cid],
                })

        input_file.write_text(json.dumps(batch_data, ensure_ascii=False, indent=2), encoding="utf-8")

        prompt_content = (
            f"{SYSTEM}\n\n"
            f"JSON-СХЕМА КАЖДОГО ОТВЕТА:\n{json.dumps(VERIFY_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
            f"ВХОДНЫЕ ДАННЫЕ ВОРКЕРА: `{input_file.as_posix()}`\n\n"
            f"ЗАДАЧА: Прочитай `{input_file.as_posix()}`, последовательно проверь связи для каждого чанка "
            f"моделью `cursor-grok-4.6-extra-high-fast` (или `cursor-grok-4.6-high-fast`) и сохрани итоговый JSON строго в `{output_file.as_posix()}`."
        )
        prompt_file.write_text(prompt_content, encoding="utf-8")

        log(f"  Воркер {w_id}: {len(batch_data)} чанков со связями -> {input_file}")

    log(f"{'='*70}\n")
    log("Пайплайн ожидает (waiting_cursor). Запустите верификацию в параллельных вкладках Cursor.")

    return {
        "status": "waiting_cursor",
        "pending_total_rels": len(pending),
        "assigned_chunks": len(to_process_ids),
    }


def _ingest_output(store: IndexStore, output_file: Path, log_path) -> dict:
    try:
        raw_data = json.loads(output_file.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Ошибка чтения {output_file}: {exc}")
        return {"error": str(exc)}

    results = raw_data.get("results", [])
    this_ok = 0
    this_fail = 0

    for item in results:
        chunk_id = item.get("chunk_id")
        raw_verdicts = item.get("verdicts", [])
        try:
            payload = VerifyPayload.model_validate({"verdicts": raw_verdicts})
            verdicts = {v.relationship_key: v for v in payload.verdicts}
            for rel_id, verdict in verdicts.items():
                store.save_verify(rel_id, verdict.accepted, verdict.reason, verdict.quote or "")
            append_jsonl(
                log_path,
                {"ts": datetime.now(timezone.utc).isoformat(), "stage": "verify", "chunk_id": chunk_id, "ok": True},
            )
            this_ok += 1
        except Exception as exc:
            append_jsonl(
                log_path,
                {"ts": datetime.now(timezone.utc).isoformat(), "stage": "verify", "chunk_id": chunk_id, "ok": False, "error": str(exc)},
            )
            this_fail += 1

    output_file.unlink(missing_ok=True)
    return {"verified_chunks": this_ok, "failed_chunks": this_fail}
