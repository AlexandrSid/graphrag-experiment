from __future__ import annotations

from collections import Counter, defaultdict

import leidenalg as la

from graphrag_lab.indexing.graph import build_igraph
from graphrag_lab.storage.sqlite import IndexStore
from graphrag_lab.util import write_jsonl


def run_leiden(store: IndexStore, cfg: dict) -> dict:
    graph = build_igraph(store)
    leiden_cfg = cfg.get("leiden") or {}
    seed = int(leiden_cfg.get("seed", 42))
    n_iterations = int(leiden_cfg.get("n_iterations", 2))
    resolutions = list(leiden_cfg.get("resolutions") or [1.5, 1.0])
    min_size = int(leiden_cfg.get("min_community_size", 2))

    if graph.vcount() == 0:
        store.replace_communities([])
        return {"communities": 0, "levels": 0, "modularity": 0}

    names = list(graph.vs["name"])
    level_maps: list[dict[str, int]] = []
    modularities: list[float] = []
    for resolution in resolutions:
        kwargs = {"weights": "weight"} if graph.ecount() else {}
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            n_iterations=n_iterations,
            seed=seed,
            resolution_parameter=float(resolution),
            **kwargs,
        )
        mapping = {names[i]: int(membership) for i, membership in enumerate(partition.membership)}
        # drop singleton communities by leaving them unassigned as unique negative ids
        counts = Counter(mapping.values())
        next_id = 0
        remapped: dict[str, int] = {}
        remap_old: dict[int, int] = {}
        for entity, comm in mapping.items():
            if counts[comm] < min_size:
                continue
            if comm not in remap_old:
                remap_old[comm] = next_id
                next_id += 1
            remapped[entity] = remap_old[comm]
        level_maps.append(remapped)
        modularities.append(float(partition.modularity))

    # assign parent: finer level 0 -> coarser next levels
    rows = []
    for level, mapping in enumerate(level_maps):
        groups: dict[int, list[str]] = defaultdict(list)
        for entity, comm in mapping.items():
            groups[comm].append(entity)
        parent_map: dict[int, int | None] = {}
        if level + 1 < len(level_maps):
            coarser = level_maps[level + 1]
            for comm, members in groups.items():
                votes = Counter(coarser[m] for m in members if m in coarser)
                parent_map[comm] = votes.most_common(1)[0][0] if votes else None
        else:
            for comm in groups:
                parent_map[comm] = None
        rels = [dict(r) for r in store.relationships()]
        for comm, members in groups.items():
            member_set = set(members)
            rel_ids = [
                r["id"]
                for r in rels
                if r["source_id"] in member_set and r["target_id"] in member_set
            ]
            rows.append(
                {
                    "id": f"L{level}-C{comm}",
                    "community": comm,
                    "level": level,
                    "parent": parent_map.get(comm),
                    "title": f"community {comm} level {level}",
                    "entity_ids": members,
                    "relationship_ids": rel_ids,
                    "size": len(members),
                    "modularity": modularities[level] if level < len(modularities) else None,
                }
            )
    store.replace_communities(rows)
    write_jsonl(store.index_dir / "exports" / "communities.jsonl", rows)
    return {
        "communities": len(rows),
        "levels": len(level_maps),
        "sizes": [r["size"] for r in rows],
        "modularity": modularities,
    }
