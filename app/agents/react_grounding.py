"""
ReAct / answer grounding helpers — strong evidence, goals, relevance, confidence.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"[a-z0-9_]{2,}", re.I)


def token_set(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def extract_goal(query: str) -> str:
    """Compact goal string for relevance checks (no extra LLM)."""
    q = (query or "").strip()
    if len(q) > 240:
        q = q[:240] + "…"
    return q


def keyword_relevance(query: str, *texts: str) -> float:
    """
    Cheap overlap score in [0, 1]: |Q∩T| / max(|Q|, 1), no embeddings.
    """
    g = token_set(query)
    if not g:
        return 0.0
    blob = " ".join(t or "" for t in texts).lower()
    if not blob:
        return 0.0
    hit = sum(1 for t in g if t in blob)
    return min(1.0, hit / max(len(g), 1))


def has_strong_evidence(state: dict[str, Any]) -> bool:
    """File reads or resolved symbols only — primary code anchor."""
    return bool(state.get("files_read")) or bool(state.get("symbols_found"))


def has_strong_merged_context(state: dict[str, Any]) -> bool:
    """Merged exploration is rich enough to skip RAG (2+ files or 2+ symbols)."""
    return has_strong_merged_lists(
        list(state.get("files_read") or []),
        list(state.get("symbols_found") or []),
    )


def has_strong_merged_lists(files: list[str], symbols: list[str]) -> bool:
    return len(files) >= 2 or len(symbols) >= 2


_CODEISH_HINTS = frozenset(
    {
        "file",
        "files",
        "code",
        "error",
        "fix",
        "bug",
        "implement",
        "refactor",
        "patch",
        "edit",
        "repo",
        "git",
        "function",
        "class",
        "import",
        "traceback",
        "exception",
        "test",
        "build",
        "api",
        "stack",
        "trace",
        "login",
        "http",
        "endpoint",
        "deploy",
        "merge",
        "branch",
        "commit",
        "database",
        "sql",
        "json",
        "config",
        "webpack",
        "npm",
        "node",
        "server",
        "client",
        "request",
        "response",
        "state",
        "hook",
        "async",
        "await",
        "promise",
    }
)


def is_trivial_query(query: str) -> bool:
    """
    Very short, non-code queries (e.g. greetings) — skip heavy ReAct / grounding gates.
    Uses word tokens (no substring matches like "git" inside "digital").
    """
    q = (query or "").lower().strip()
    if not q:
        return True
    words = q.split()
    if len(words) > 4:
        return False
    ts = token_set(q)
    if ts & _CODEISH_HINTS:
        return False
    if re.search(r"\b(500|404|403|401|502)\b", q):
        return False
    if "http://" in q or "https://" in q or ".py" in q or "::" in q:
        return False
    if "stack trace" in q or "traceback" in q:
        return False
    return True


def has_strong_rag(ranked_ctx: list[Any] | None) -> bool:
    """Require multiple chunks with non-trivial average score."""
    if not ranked_ctx or len(ranked_ctx) < 2:
        return False
    scores: list[float] = []
    for c in ranked_ctx:
        if not isinstance(c, dict):
            continue
        try:
            scores.append(float(c.get("score", 0)))
        except (TypeError, ValueError):
            continue
    if len(scores) < 2:
        return False
    avg = sum(scores) / len(scores)
    return avg > 0.35


def may_emit_answer(state: dict[str, Any], ranked_ctx: list) -> bool:
    """Write mode, trivial chat, any RAG hits, file/symbol evidence, or strong RAG."""
    q = (state.get("query") or state.get("current_step") or "").strip()
    if is_trivial_query(q):
        return True
    if state.get("write_enabled"):
        return True
    if has_strong_evidence(state):
        return True
    if ranked_ctx:
        return True
    if has_strong_rag(ranked_ctx):
        return True
    return False


def validate_answer(state: dict[str, Any], ranked_ctx: list) -> bool:
    """Same gate as may_emit_answer (post-check / tooling)."""
    return may_emit_answer(state, ranked_ctx)


def goal_progress_delta(
    query: str,
    goal: str,
    res_data: dict[str, Any] | None,
    *,
    action: str | None = None,
) -> float:
    """Set overlap progress, capped; read_file weighted higher."""
    if not res_data or not isinstance(res_data, dict):
        return 0.0
    g = token_set(goal or query)
    q = token_set(query)
    union_q = g | q
    if not union_q:
        return 0.0
    parts: list[str] = []
    if res_data.get("content"):
        parts.append(str(res_data["content"])[:4000])
    for m in res_data.get("matches") or []:
        if isinstance(m, dict):
            parts.append(str(m.get("content", "")))
            parts.append(str(m.get("symbol", "")))
    for e in res_data.get("entries") or []:
        if isinstance(e, dict):
            parts.append(str(e.get("path", "")))
    blob = " ".join(parts).lower()
    cset = token_set(blob)
    overlap = union_q & cset
    base = len(overlap) / max(len(union_q), 1)
    delta = min(base, 0.5)
    if (action or "").strip() == "read_file":
        delta = min(0.5, delta * 1.5)
    if delta >= 0.35:
        return round(min(0.35, delta), 3)
    if delta >= 0.18:
        return round(min(0.28, delta), 3)
    if delta >= 0.08:
        return round(min(0.18, delta), 3)
    return round(delta, 3) if delta > 0 else 0.0


def compute_answer_confidence(state: dict[str, Any]) -> float:
    """Scalar for meta / eval (not model self-reported)."""
    useful = min(1.0, int(state.get("react_useful_steps") or 0) / 3.0)
    files = min(1.0, len(state.get("files_read") or []) / 2.0)
    syms = min(1.0, len(state.get("symbols_found") or []) / 3.0)
    rctx = min(1.0, len(state.get("ranked_ctx") or []) / 8.0)
    raw = 0.35 * useful + 0.3 * files + 0.2 * syms + 0.15 * rctx
    return round(min(1.0, raw), 2)
