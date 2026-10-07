from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graphrag_lab.agents.answer import answer_from_packet
from graphrag_lab.config import load_query_config, ollama_settings
from graphrag_lab.models import AnswerResult, RetrievalPacket
from graphrag_lab.ollama_client import OllamaClient
from graphrag_lab.query.retrieve import retrieve
from graphrag_lab.util import dump_json, estimate_tokens


@dataclass
class QueryTrace:
    timestamp: str
    mode: str
    query: str
    retrieval_time_s: float
    prompt_build_time_s: float
    llm_ttft_s: float | None
    llm_total_time_s: float
    retrieved_communities: list[str]
    retrieved_chunks: list[str]
    full_prompt: str
    llm_response: str
    baseline: bool
    context_tokens: int
    output_tokens: int
    result: AnswerResult

    def as_dump(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "query": self.query,
            "baseline": self.baseline,
            "retrieval_time_s": _round(self.retrieval_time_s),
            "prompt_build_time_s": _round(self.prompt_build_time_s),
            "llm_ttft_s": None if self.llm_ttft_s is None else _round(self.llm_ttft_s),
            "llm_total_time_s": _round(self.llm_total_time_s),
            "retrieved_communities": self.retrieved_communities,
            "retrieved_chunks": self.retrieved_chunks,
            "full_prompt": self.full_prompt,
            "llm_response": self.llm_response,
        }


def ask(
    index_dir: Path,
    question: str,
    mode: str,
    *,
    no_rag: bool = False,
    dump_dir: Path | None = None,
    client: OllamaClient | None = None,
) -> QueryTrace:
    started = time.perf_counter()
    if no_rag:
        packet = RetrievalPacket(mode=mode, question=question, context="")
        retrieval_time_s = 0.0
    else:
        packet = retrieve(index_dir, question, mode)
        retrieval_time_s = time.perf_counter() - started
    cfg = load_query_config()
    settings = ollama_settings(cfg)
    llm = client or OllamaClient(settings["host"], timeout_s=float(settings.get("timeout_s", 180)))
    call = answer_from_packet(llm, settings["chat_model"], packet)
    if call.used:
        packet.chunk_ids = [cid for cid in packet.chunk_ids if cid in set(call.used)] or packet.chunk_ids
        packet.citations = [cite for cite in packet.citations if cite.chunk_id in set(packet.chunk_ids)]
    result = AnswerResult(answer=call.answer, packet=packet)
    trace = QueryTrace(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        query=question,
        retrieval_time_s=retrieval_time_s,
        prompt_build_time_s=call.prompt_build_s,
        llm_ttft_s=call.ttft_s,
        llm_total_time_s=call.total_s,
        retrieved_communities=list(packet.community_ids),
        retrieved_chunks=list(packet.chunk_ids),
        full_prompt=call.full_prompt,
        llm_response=call.answer,
        baseline=no_rag,
        context_tokens=estimate_tokens(packet.context),
        output_tokens=estimate_tokens(call.answer),
        result=result,
    )
    dump_json(index_dir / "logs" / "last_query.json", result.model_dump())
    _dump_trace(index_dir, trace, dump_dir)
    return trace


def format_telemetry(trace: QueryTrace) -> str:
    communities = ", ".join(trace.retrieved_communities[:12]) or "—"
    if len(trace.retrieved_communities) > 12:
        communities += f" (+{len(trace.retrieved_communities) - 12})"
    ttft = "n/a" if trace.llm_ttft_s is None else f"{trace.llm_ttft_s:.2f}s"
    baseline = " | Baseline: no-rag" if trace.baseline else ""
    return (
        "────────────────── GraphRAG Telemetry ──────────────────\n"
        f"[Retrieval]: {trace.retrieval_time_s:.2f}s | Mode: {trace.mode} | Communities: {communities}{baseline}\n"
        f"[LLM Pre-fill / TTFT]: {ttft} | [Total Generation]: {trace.llm_total_time_s:.2f}s\n"
        f"[Tokens]: Context: ~{trace.context_tokens} tokens | Output: ~{trace.output_tokens} tokens\n"
        "────────────────────────────────────────────────────────"
    )


def _dump_trace(index_dir: Path, trace: QueryTrace, dump_dir: Path | None) -> Path:
    folder = dump_dir or (index_dir / "logs" / "queries")
    stamp = trace.timestamp.replace(":", "-").replace("+", "Z")
    path = folder / f"{stamp}_{trace.mode}{'_norag' if trace.baseline else ''}.json"
    dump_json(path, trace.as_dump())
    return path


def _round(value: float) -> float:
    return round(value, 4)
