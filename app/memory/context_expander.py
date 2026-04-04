"""Graph-first context expansion: seeds + BFS on import edges with caps."""
from __future__ import annotations

from collections import deque

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from app.memory.intel_store import load_import_edges

logger = get_logger(__name__)
settings = get_settings()


def expand_files_on_graph(
    edges: list[tuple[str, str]],
    seeds: set[str],
    max_depth: int,
    max_files: int,
    decay_per_hop: float,
) -> tuple[set[str], list[dict]]:
    if not seeds:
        return set(), []

    adj: dict[str, list[str]] = {}
    for src, tgt in edges:
        adj.setdefault(src, []).append(tgt)

    expanded: set[str] = set(seeds)
    trace: list[dict] = []
    q: deque[tuple[str, int, float]] = deque()
    for s in seeds:
        q.append((s, 0, 1.0))

    while q and len(expanded) < max_files:
        node, depth, decay = q.popleft()
        if depth >= max_depth:
            continue
        for nxt in adj.get(node, ()):
            if nxt in expanded:
                continue
            if len(expanded) >= max_files:
                break
            new_decay = decay * decay_per_hop
            expanded.add(nxt)
            trace.append(
                {
                    "from_file": node,
                    "to_file": nxt,
                    "hop": depth + 1,
                    "decay": round(new_decay, 4),
                    "reason": "graph_import",
                }
            )
            q.append((nxt, depth + 1, new_decay))

    return expanded, trace


async def expand_context_files(
    db: AsyncSession | None,
    repo_name: str | None,
    seed_files: set[str],
) -> tuple[set[str], list[dict], list[str]]:
    fallbacks: list[str] = []
    if not db or not repo_name:
        fallbacks.append("fallback_no_graph")
        return set(seed_files), [], fallbacks

    edges = await load_import_edges(db, repo_name)
    if not edges:
        fallbacks.append("fallback_no_graph")
        return set(seed_files), [], fallbacks

    expanded, gtrace = expand_files_on_graph(
        edges,
        set(seed_files),
        max_depth=settings.GRAPH_MAX_DEPTH,
        max_files=settings.GRAPH_MAX_FILES,
        decay_per_hop=settings.GRAPH_DECAY_PER_HOP,
    )
    return expanded, gtrace, fallbacks
