# GraphRAG lab

Поэтапный GraphRAG для одной русской книги. Индекс — папка на диске, не Neo4j.
Ответы и массовый extract/verify идут в уже поднятый Ollama. Resolve и отчёты сообществ — через mailbox: пайплайн пишет промпт и ждёт JSON от внешней модели.

Книга: `book.txt` в корне (UTF-8 или Windows-1251). Код её не копирует и query не читает целиком.

Голая команда `graphrag-lab` в PATH не обязана быть. Всегда вызывайте exe из `.venv`.

## Требования

- Python 3.11+
- Docker
- Ollama **этого** проекта: контейнер `graphrag-ollama`, порт `11490`
- Модели: `gemma3:4b-it-q4_K_M`, `mxbai-embed-large`

Чужие контейнеры Ollama не трогать. Этот стек живёт в `docker-compose.yml` и не делит том с другими проектами.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
docker compose up -d
.\scripts\pull_models.cmd
Invoke-RestMethod http://localhost:11490/api/tags
```

Стоп только своего контейнера:

```powershell
docker compose stop
docker compose start
docker compose down
```

`down` без `-v` модели в томе `graphrag_ollama` сохраняет. `-v` сотрёт скачанные веса.

## Команды

Из корня репозитория:

```powershell
.\.venv\Scripts\graphrag-lab.exe stage list --index indexes/book
.\.venv\Scripts\graphrag-lab.exe stage status --index indexes/book
.\.venv\Scripts\graphrag-lab.exe stage show --index indexes/book --stage extract --limit 20
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage chunk --input book.txt
.\.venv\Scripts\graphrag-lab.exe llm pending --index indexes/book
.\.venv\Scripts\graphrag-lab.exe llm prompt --index indexes/book
.\.venv\Scripts\graphrag-lab.exe llm ingest --index indexes/book --file response.json
.\.venv\Scripts\graphrag-lab.exe ask --index indexes/book --mode local "вопрос"
.\.venv\Scripts\graphrag-lab.exe chat --index indexes/book
```

По умолчанию одна стадия за запуск. `--from/--to` только для локальных стадий, не для `resolve`/`report`.
Повтор стадии: `--force` (следующие станут `stale`).

## Как остановить

Чужой Ollama не трогать. Свой: `docker compose stop` / `docker compose start`.

| Процесс | Старт | Стоп |
| --- | --- | --- |
| любая `stage run` | команда `stage run` в терминале | в другом окне `stage stop` (Ctrl+C в Cursor часто не доходит) |
| extract / verify после обрыва | снова та же `stage run` | продолжит с незакрытых чанков |
| resolve / report, пока `waiting_llm` | пайплайн сам вышел после записи промпта | процесс уже не бежит; промпт не трогать |
| chat | `chat` | `/quit` или `Ctrl+C` |
| зависший Python, если `Ctrl+C` не взял | — | `Get-Process graphrag-lab, python \| Where-Object { $_.Path -like '*GraphRAG*' } \| Stop-Process` |
| Docker Ollama | уже поднят | не стопать |

После `Ctrl+C` смотрите статус:

```powershell
.\.venv\Scripts\graphrag-lab.exe stage status --index indexes/book
```

Остановка extract/verify из **другого** терминала (надёжнее Ctrl+C):

```powershell
.\.venv\Scripts\graphrag-lab.exe stage stop --index indexes/book --stage extract
.\.venv\Scripts\graphrag-lab.exe stage status --index indexes/book
```

`stop` убивает процесс по pid и ставит `interrupted`. Прогресс чанков не трётся. Потом тот же `stage run` без `--force`.

`status` сам помечает `running` как `interrupted`, если процесса уже нет. Потом снова запустите ту же стадию — она продолжит с незакрытых чанков, без `--force`.

## Чеклист оператора

Перед каждой стадией:

```powershell
.\.venv\Scripts\graphrag-lab.exe stage status --index indexes/book
```

Дальше только если предыдущая `done`.

1. **chunk** — нарезка, без LLM.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage chunk --input book.txt
```

Стоп: `Ctrl+C`. Проверьте: чанки 500–800 токенов, не «одна глава = один чанк».

2. **extract** — локальная Gemma, можно оставить на ночь.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage extract
```

Стоп: `Ctrl+C` в этом окне. Следующий запуск продолжит. Смотрите `failed` через `stage show`. Если failed > ~15% — не гоните verify.

3. **verify** — та же Gemma сверяет связи с чанком.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage verify
```

Стоп: `Ctrl+C`. На следующую стадию идут только accepted.

4. **resolve** — пайплайн пишет промпт и сразу выходит в `waiting_llm`.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage resolve
.\.venv\Scripts\graphrag-lab.exe llm pending --index indexes/book
.\.venv\Scripts\graphrag-lab.exe llm prompt --index indexes/book
.\.venv\Scripts\graphrag-lab.exe llm ingest --index indexes/book --file response.json
```

Стоп: процесс run уже не висит. Не отправляйте ingest, если передумали: статус останется `waiting_llm`. Битый JSON не применяется.

5. **graph** — без LLM.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage graph
```

Стоп: `Ctrl+C`.

6. **leiden** — авторский `leidenalg`, seed=42.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage leiden
```

Стоп: `Ctrl+C`. Если один кластер на всю книгу — меняйте `leiden.resolutions` в `config/index.yaml` и `--force`.

7. **report** — mailbox по уровням.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage report
.\.venv\Scripts\graphrag-lab.exe llm prompt --index indexes/book
.\.venv\Scripts\graphrag-lab.exe llm ingest --index indexes/book --file response.json
```

Стоп: как у resolve. После ingest пайплайн сам соберёт следующий уровень.

8. **embed** — локальный `mxbai-embed-large`.

```powershell
.\.venv\Scripts\graphrag-lab.exe stage run --index indexes/book --stage embed
```

Стоп: `Ctrl+C`. После `done` можно спрашивать.

**Вопросы.**

```powershell
.\.venv\Scripts\graphrag-lab.exe chat --index indexes/book
.\.venv\Scripts\graphrag-lab.exe ask --index indexes/book --mode local "вопрос"
```

Стоп чата: `/quit` или `Ctrl+C`. Если `embed` не `done` — отказ.

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
.\.venv\Scripts\python.exe -m pytest
```
