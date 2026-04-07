"""
Assemble structured edit context after exploration (before EditPlanner / Answer).
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession


def build_edit_context(state: dict[str, Any]) -> dict[str, Any]:
    files = list(state.get("files_read") or [])
    symbols = list(state.get("symbols_found") or [])
    pri = list(state.get("react_priority_files") or [])
    graph_edges: list[dict[str, str]] = []
    for tr in state.get("tool_results") or []:
        if not isinstance(tr, dict):
            continue
        if tr.get("action") not in (
            "get_file_dependencies",
            "get_symbol_dependencies",
        ):
            continue
        data = tr.get("data") or {}
        if not isinstance(data, dict):
            continue
        src = data.get("file") or data.get("primary_file") or ""
        for t in data.get("imports") or []:
            if src and t:
                graph_edges.append({"from": str(src), "to": str(t), "kind": "imports"})
        for im in data.get("imported_by") or []:
            if src and im:
                graph_edges.append({"from": str(im), "to": str(src), "kind": "imports"})

    ctx = {
        "files": files[:24],
        "symbols": symbols[:40],
        "priority_files": pri[:6],
        "dependency_edges": graph_edges[:80],
        "useful_tool_steps": int(state.get("react_useful_steps") or 0),
    }
    state["edit_context"] = ctx
    return ctx


async def enrich_edit_context_dependencies(
    state: dict[str, Any],
    db: AsyncSession | None,
    repo_id: str | None,
) -> None:
    """
    Ensure import-neighborhood edges exist for files_read even when ReAct skipped
    get_file_dependencies (edit mode + indexed repo only).
    """
    build_edit_context(state)
    if not db or not repo_id:
        return
    ctx = state.get("edit_context") or {}
    if not isinstance(ctx, dict):
        return
    covered: set[str] = set()
    for tr in state.get("tool_results") or []:
        if not isinstance(tr, dict):
            continue
        if tr.get("action") != "get_file_dependencies":
            continue
        data = tr.get("data") or {}
        if isinstance(data, dict) and data.get("file"):
            covered.add(str(data["file"]).replace("\\", "/"))
    edges = list(ctx.get("dependency_edges") or [])
    from app.memory.intel_store import get_dependency_bundle_for_file

    for raw in (state.get("files_read") or [])[:6]:
        fn = str(raw).replace("\\", "/").strip()
        if not fn or fn in covered:
            continue
        try:
            bundle = await get_dependency_bundle_for_file(
                db, repo_id, fn, limit_each=14
            )
            for t in bundle.get("imports") or []:
                if t:
                    edges.append(
                        {"from": fn, "to": str(t).replace("\\", "/"), "kind": "imports"}
                    )
            for im in bundle.get("imported_by") or []:
                if im:
                    edges.append(
                        {
                            "from": str(im).replace("\\", "/"),
                            "to": fn,
                            "kind": "imported_by",
                        }
                    )
        except Exception:
            pass
        covered.add(fn)
    ctx["dependency_edges"] = edges[:120]
    state["edit_context"] = ctx
