# GraphRAG lab

Поэтапный GraphRAG для одной русской книги. Индекс — папка на диске, не Neo4j.
Ответы и массовый extract/verify идут в уже поднятый Ollama. Resolve и отчёты сообществ — через mailbox: пайплайн пишет промпт и ждёт JSON от внешней модели.

Книга: `book.txt` в корне (UTF-8 или Windows-1251). Код её не копирует и query не читает целиком.

## Требования

- Python 3.11+
- Уже запущенный Ollama: `http://localhost:49794/`
- Модели: `gemma3:4b-it-q4_K_M`, `mxbai-embed-large`

Ollama не поднимайте и не останавливайте из этого репозитория. Порт зафиксирован.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Проверка моделей:

```powershell
curl http://localhost:49794/api/tags
```

## Команды

```powershell
graphrag-lab stage list --index indexes/book
graphrag-lab stage status --index indexes/book
graphrag-lab stage show --index indexes/book --stage extract --limit 20
graphrag-lab stage run --index indexes/book --stage chunk --input book.txt
graphrag-lab llm pending --index indexes/book
graphrag-lab llm prompt --index indexes/book
graphrag-lab llm ingest --index indexes/book --file response.json
graphrag-lab ask --index indexes/book --mode local "вопрос"
graphrag-lab chat --index indexes/book
```

По умолчанию одна стадия за запуск. `--from/--to` только для локальных стадий, не для `resolve`/`report`.
Повтор стадии: `--force` (следующие станут `stale`).

## Чеклист оператора

Перед каждой стадией: `stage status`. Дальше только если предыдущая `done`.

1. **chunk** — нарезка, без LLM.
   `graphrag-lab stage run --index indexes/book --stage chunk --input book.txt`
   Проверьте: чанки 500–800 токенов, не «одна глава = один чанк».

2. **extract** — локальная Gemma, можно оставить на ночь. Обрыв продолжается.
   `graphrag-lab stage run --index indexes/book --stage extract`
   Смотрите 20 примеров и долю `failed`. Если failed > ~15% — не гоните verify.

3. **verify** — та же Gemma сверяет связи с чанком.
   `graphrag-lab stage run --index indexes/book --stage verify`
   На следующую стадию идут только accepted.

4. **resolve** — пайплайн пишет промпт и ждёт.
   `graphrag-lab stage run --index indexes/book --stage resolve`
   Затем цикл: `llm prompt` → внешняя модель → JSON по `schema.json` → `llm ingest --file response.json`.
   Пока `llm pending` не пуст, повторяйте. Битый JSON не применяется.

5. **graph** — без LLM.
   `graphrag-lab stage run --index indexes/book --stage graph`
   Смотрите узлы, рёбра, компоненты.

6. **leiden** — авторский `leidenalg`, seed=42.
   `graphrag-lab stage run --index indexes/book --stage leiden`
   Если один кластер на всю книгу или пыль одиночек — меняйте `leiden.resolutions` в `config/index.yaml` и `--force`.

7. **report** — снова mailbox по уровням.
   `graphrag-lab stage run --index indexes/book --stage report`
   Тот же цикл prompt → ingest. После ingest пайплайн сам соберёт следующий уровень.

8. **embed** — локальный `mxbai-embed-large`.
   `graphrag-lab stage run --index indexes/book --stage embed`
   После `done` можно спрашивать.

**Вопросы.** `graphrag-lab chat --index indexes/book` или `ask --mode local|global|vector`.
Если `embed` не `done` — отказ. В ответе должны быть цитаты.

## Артефакт

```
indexes/book/
  manifest.json
  graph.sqlite
  vectors.lancedb/
  logs/index.jsonl
  mailbox/current/prompt.md
  exports/
```

## Тесты

```powershell
pytest
```

Тесты не читают `book.txt` и не требуют модель.
