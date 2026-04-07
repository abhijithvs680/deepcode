"""
High-level agent intent — drives tool binding, ReAct depth, and RAG fallback bias.

Maps existing query_type labels into three buckets:
- structured: narrow targets (paths, symbols, failures, git facts) → allow router tool bind.
- exploratory: open-ended how/what without a sharp file anchor → explore, weak router bind.
- broad: system/architecture questions → favor retrieval (react_force_rag) after a short loop.
"""
from __future__ import annotations

from typing import Any


def derive_agent_intent(classification: dict[str, Any], query: str) -> str:
    qt = (classification.get("query_type") or "factual").lower().strip()
    q = (query or "").lower()
    focus = classification.get("focus_entities") or []

    if qt == "architecture" or any(
        w in q for w in ("how does the system", "overall architecture", "high level")
    ):
        return "broad"

    if qt in ("navigation", "git", "debugging"):
        return "structured"

    if qt == "factual" and focus:
        return "structured"

    if any(
        w in q
        for w in (
            "where is",
            "which file",
            "find the",
            "definition of",
            "symbol",
            "class ",
            "function ",
        )
    ):
        return "structured"

    if any(
        phrase in q
        for phrase in (
            "create a react",
            "create react app",
            "create-react-app",
            "react app",
            "npx create",
            "npm create",
            "yarn create",
            "pnpm create",
            "new react project",
            "initialize a react",
            "vite ",
            "scaffold ",
            "bootstrap ",
        )
    ):
        return "structured"

    return "exploratory"
