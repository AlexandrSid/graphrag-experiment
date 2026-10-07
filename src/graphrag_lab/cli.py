from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from graphrag_lab.indexing.locks import reconcile_running
from graphrag_lab.indexing.runner import abort_stage, ingest_mailbox, open_store, run_range, run_stage, stop_stage
from graphrag_lab.mailbox import Mailbox
from graphrag_lab.models import STAGE_ORDER
from graphrag_lab.query.answer import ask as ask_index
from graphrag_lab.query.answer import format_telemetry
from graphrag_lab.storage.manifest import read_manifest

app = typer.Typer(no_args_is_help=True, add_completion=False)
stage_app = typer.Typer(no_args_is_help=True)
llm_app = typer.Typer(no_args_is_help=True)
app.add_typer(stage_app, name="stage")
app.add_typer(llm_app, name="llm")


def _index(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@stage_app.command("list")
def stage_list(index: Path = typer.Option(Path("indexes/book"), "--index")) -> None:
    index = _index(index)
    store = open_store(index)
    try:
        reconcile_running(store, index)
        for row in store.all_stages():
            typer.echo(f"{row['name']:10} {row['status']}")
    finally:
        store.close()


@stage_app.command("status")
def stage_status(index: Path = typer.Option(Path("indexes/book"), "--index")) -> None:
    index = _index(index)
    store = open_store(index)
    try:
        reconcile_running(store, index)
        for row in store.all_stages():
            stats = row.get("stats_json") or ""
            typer.echo(f"{row['name']:10} {row['status']:12} {stats}")
    finally:
        store.close()


@stage_app.command("show")
def stage_show(
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    stage: str = typer.Option(..., "--stage"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    if stage not in STAGE_ORDER:
        raise typer.BadParameter(stage)
    store = open_store(_index(index))
    try:
        typer.echo(f"status={store.stage_status(stage)}")
        mapping = {
            "chunk": store.text_units(),
            "extract": store.raw_entities(),
            "verify": store.raw_relationships(),
            "resolve": store.entities(),
            "graph": store.relationships(),
            "leiden": store.communities(),
            "report": store.reports(),
        }
        rows = mapping.get(stage, [])
        if stage == "embed":
            typer.echo(json.dumps(store.graph_stat("embed", {}), ensure_ascii=False, indent=2))
            return
        typer.echo(f"rows={len(rows)}")
        for row in list(rows)[:limit]:
            item = dict(row)
            item.pop("text", None)
            item.pop("raw_json", None)
            typer.echo(json.dumps(item, ensure_ascii=False, default=str)[:500])
    finally:
        store.close()


@stage_app.command("run")
def stage_run(
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    stage: Optional[str] = typer.Option(None, "--stage"),
    input_path: Path = typer.Option(Path("book.txt"), "--input"),
    force: bool = typer.Option(False, "--force"),
    from_stage: Optional[str] = typer.Option(None, "--from"),
    to_stage: Optional[str] = typer.Option(None, "--to"),
) -> None:
    index = _index(index)
    if from_stage or to_stage:
        start = from_stage or "chunk"
        end = to_stage or from_stage or "chunk"
        results = run_range(index, start, end, input_path, force, None)
        typer.echo(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return
    if not stage:
        raise typer.BadParameter("Provide --stage or --from/--to")
    stats = run_stage(index, stage, input_path=input_path, force=force)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    if stats.get("waiting"):
        mailbox = Mailbox(index)
        typer.echo(f"waiting_llm prompt={mailbox.prompt_path()}")


@stage_app.command("abort")
def stage_abort(
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    stage: Optional[str] = typer.Option(None, "--stage"),
) -> None:
    changed = abort_stage(_index(index), stage)
    if not changed:
        typer.echo("nothing to abort")
        return
    typer.echo("interrupted: " + ", ".join(changed))


@stage_app.command("stop")
def stage_stop(
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    stage: Optional[str] = typer.Option(None, "--stage"),
) -> None:
    result = stop_stage(_index(index), stage)
    typer.echo(json.dumps(result, ensure_ascii=False))


@llm_app.command("pending")
def llm_pending(index: Path = typer.Option(Path("indexes/book"), "--index")) -> None:
    mailbox = Mailbox(_index(index))
    job = mailbox.pending()
    if job is None:
        typer.echo("no pending job")
        raise typer.Exit()
    typer.echo(json.dumps(job.model_dump(), ensure_ascii=False, indent=2))


@llm_app.command("prompt")
def llm_prompt(index: Path = typer.Option(Path("indexes/book"), "--index")) -> None:
    mailbox = Mailbox(_index(index))
    if mailbox.pending() is None:
        typer.echo("no pending job")
        raise typer.Exit(code=1)
    typer.echo(str(mailbox.prompt_path().resolve()))
    typer.echo(str(mailbox.schema_path().resolve()))


@llm_app.command("ingest")
def llm_ingest(
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    file: Path = typer.Option(..., "--file"),
) -> None:
    stats = ingest_mailbox(_index(index), file)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    mailbox = Mailbox(index)
    if mailbox.pending():
        typer.echo(f"next prompt={mailbox.prompt_path()}")


def _print_answer(result) -> None:
    typer.echo(result.answer)
    typer.echo("")
    typer.echo("--- citations ---")
    for cite in result.packet.citations:
        typer.echo(f"[{cite.chunk_id} / {cite.chapter}] {cite.quote}")
    typer.echo(
        f"mode={result.packet.mode} chunks={result.packet.chunk_ids} "
        f"entities={result.packet.entity_ids} communities={result.packet.community_ids}"
    )


def _print_trace(trace, debug_prompt: bool) -> None:
    if debug_prompt:
        typer.echo(trace.full_prompt)
        typer.echo("")
    _print_answer(trace.result)
    typer.echo(format_telemetry(trace))


@app.command()
def ask(
    question: str = typer.Argument(...),
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    mode: str = typer.Option("local", "--mode"),
    debug_prompt: bool = typer.Option(False, "--debug-prompt", "--verbose", help="Print the full prompt sent to the model"),
    dump_dir: Optional[Path] = typer.Option(None, "--dump-dir", help="Directory for per-query JSON traces"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Ask the same model with an empty RAG context"),
) -> None:
    trace = ask_index(index, question, mode, no_rag=no_rag, dump_dir=dump_dir)
    _print_trace(trace, debug_prompt)


@app.command()
def chat(
    index: Path = typer.Option(Path("indexes/book"), "--index"),
    mode: str = typer.Option("local", "--mode"),
    debug_prompt: bool = typer.Option(False, "--debug-prompt", "--verbose", help="Print the full prompt sent to the model"),
    dump_dir: Optional[Path] = typer.Option(None, "--dump-dir", help="Directory for per-query JSON traces"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Ask the same model with an empty RAG context"),
) -> None:
    current = mode
    typer.echo("GraphRAG chat. /mode local|global|vector  /quit")
    while True:
        try:
            line = input(f"[{current}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            break
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in {"local", "global", "vector"}:
                current = parts[1]
                typer.echo(f"mode={current}")
            else:
                typer.echo("use /mode local|global|vector")
            continue
        try:
            trace = ask_index(index, line, current, no_rag=no_rag, dump_dir=dump_dir)
            _print_trace(trace, debug_prompt)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"error: {exc}")


@app.command("check")
def check(index: Path = typer.Option(Path("indexes/book"), "--index")) -> None:
    store = open_store(index)
    try:
        manifest = read_manifest(index)
        typer.echo(json.dumps({"stages": store.all_stages(), "manifest": manifest}, ensure_ascii=False, indent=2, default=str))
    finally:
        store.close()


if __name__ == "__main__":
    app()
