from __future__ import annotations

import json

from graphrag_lab.config import ollama_settings
from graphrag_lab.ollama_client import OllamaClient
from graphrag_lab.storage.manifest import read_manifest, write_manifest
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.storage.vectors import VectorStore


def run_embed(store: IndexStore, cfg: dict) -> dict:
    settings = ollama_settings(cfg)
    client = OllamaClient(settings["host"], timeout_s=float(settings.get("timeout_s", 300)))
    client.require_models(settings["embed_model"])
    vectors = VectorStore(store.index_dir)

    chunk_rows = []
    for unit in store.text_units():
        vec = client.embed(settings["embed_model"], unit["text"])
        chunk_rows.append({"id": unit["id"], "text": unit["text"][:500], "vector": vec})
    entity_rows = []
    for ent in store.entities():
        text = f"{ent['name']}. {ent['description']}"
        vec = client.embed(settings["embed_model"], text)
        entity_rows.append({"id": ent["id"], "text": text, "vector": vec})
    report_rows = []
    for report in store.reports():
        text = f"{report['title']}. {report['summary']}"
        vec = client.embed(settings["embed_model"], text)
        report_rows.append(
            {
                "id": f"L{report['level']}-C{report['community']}",
                "community": report["community"],
                "level": report["level"],
                "text": text,
                "vector": vec,
            }
        )
    dim = len(chunk_rows[0]["vector"]) if chunk_rows else None
    vectors.dim = dim
    n_chunks = vectors.replace_table("chunks", chunk_rows)
    n_ents = vectors.replace_table("entities", entity_rows)
    n_reps = vectors.replace_table("reports", report_rows)
    manifest = read_manifest(store.index_dir)
    manifest.update(
        {
            "embed_model": settings["embed_model"],
            "chat_model": settings["chat_model"],
            "vector_dim": dim,
            "ollama_host": settings["host"],
        }
    )
    write_manifest(store.index_dir, manifest)
    store.set_graph_stat("embed", {"chunks": n_chunks, "entities": n_ents, "reports": n_reps, "dim": dim})
    _ = json
    return {"chunks": n_chunks, "entities": n_ents, "reports": n_reps, "dim": dim}
