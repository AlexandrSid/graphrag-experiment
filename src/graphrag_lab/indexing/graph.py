from __future__ import annotations

import igraph as ig

from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import write_jsonl


def build_igraph(store: IndexStore) -> ig.Graph:
    entities = [dict(r) for r in store.entities()]
    rels = [dict(r) for r in store.relationships()]
    graph = ig.Graph()
    graph.add_vertices(len(entities))
    graph.vs["name"] = [e["id"] for e in entities]
    graph.vs["label"] = [e["name"] for e in entities]
    id_to_idx = {e["id"]: i for i, e in enumerate(entities)}
    edges = []
    weights = []
    for rel in rels:
        src = id_to_idx.get(rel["source_id"])
        dst = id_to_idx.get(rel["target_id"])
        if src is None or dst is None or src == dst:
            continue
        edges.append((src, dst))
        weights.append(float(rel["weight"] or 1.0))
    if edges:
        graph.add_edges(edges)
        graph.es["weight"] = weights
    graph.simplify(combine_edges={"weight": "sum"})
    return graph


def run_graph(store: IndexStore) -> dict:
    graph = build_igraph(store)
    components = graph.components().sizes() if graph.vcount() else []
    stats = {
        "nodes": int(graph.vcount()),
        "edges": int(graph.ecount()),
        "components": len(components),
        "largest_component": max(components) if components else 0,
        "isolates": sum(1 for size in components if size == 1),
    }
    store.set_graph_stat("graph", stats)
    write_jsonl(
        store.index_dir / "exports" / "graph" / "stats.jsonl",
        [stats],
    )
    return stats
