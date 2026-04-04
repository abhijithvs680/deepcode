"""Composite retrieval scoring: semantic, keyword, symbol tiers, path, Git recency/churn."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath

from app.memory.symbol_extract import allow_partial_symbol_token

W_SEMANTIC = 0.45
W_KEYWORD = 0.2
W_SYMBOL = 0.2
W_PATH = 0.1
W_GIT = 0.05

TIER_HIGH = 1.0
TIER_PATH = 0.7
TIER_LOW = 0.4


def normalize_semantic(cosine: float) -> float:
    return (cosine + 1.0) / 2.0


def keyword_score(query: str, text: str) -> float:
    words = query.lower().split()
    if not words:
        return 0.0
    score = 0
    text_lower = text.lower()
    for word in words:
        if len(word) > 1 and word in text_lower:
            score += 1
    return score / len(words)


def path_priority(file_path: str, query_lower: str) -> float:
    fp = file_path.replace("\\", "/").lower()
    ql = query_lower
    if "/test" in fp or "/tests/" in fp or fp.startswith("test_") or "conftest" in fp:
        return 0.35
    if "cli/" in fp or fp.endswith("/cli/main.py"):
        return 1.0 if "cli" in ql else 0.55
    if "/core/" in fp or "/app/core/" in fp:
        return 1.0
    return 0.75


def _definition_line_match(symbol: str, line: str) -> bool:
    line = line.strip()
    if not line or line.startswith("#"):
        return False
    esc = re.escape(symbol)
    if re.match(rf"^def {esc}\b", line):
        return True
    if re.match(rf"^class {esc}\b", line):
        return True
    if re.match(rf"^{esc}\s*=", line):
        return True
    if re.match(rf"^async def {esc}\b", line):
        return True
    return False


def symbol_tier_for_chunk(
    identifiers: list[str],
    file_path: str,
    content: str,
    indexed_symbol_hits: frozenset[str] | None = None,
) -> tuple[float, str, str]:
    indexed_symbol_hits = indexed_symbol_hits or frozenset()
    fp_norm = file_path.replace("\\", "/")
    stem = PurePosixPath(fp_norm).stem.lower()

    best = 0.0
    best_tier = "none"
    best_reason = "no_symbol_match"

    for ident in identifiers:
        raw = ident.strip()
        if not raw:
            continue
        low = raw.lower()
        base = low.split(".")[-1]

        if low in indexed_symbol_hits or base in indexed_symbol_hits:
            if TIER_HIGH > best:
                best, best_tier, best_reason = TIER_HIGH, "high", "symbol_index_hit"

        sym_for_def = raw.split(".")[-1]
        for line in content.splitlines():
            if _definition_line_match(sym_for_def, line):
                if TIER_HIGH > best:
                    best, best_tier, best_reason = TIER_HIGH, "high", "definition_in_chunk"
                break

        if base == stem or stem.endswith("_" + base) or base in stem:
            if TIER_PATH > best:
                best, best_tier, best_reason = TIER_PATH, "medium", "file_stem_match"

        if allow_partial_symbol_token(base) and base in content.lower():
            if TIER_LOW > best:
                best, best_tier, best_reason = TIER_LOW, "low", "partial_content_match"

    return best, best_tier, best_reason


def git_channel_score(
    last_commit_time_iso: str | None,
    recent_commits_json: str | None,
    *,
    now: datetime | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Scalar in [0, 1] from recency (dominant) + recent commit count (churn), both bounded.
    """
    churn_n = 0
    if recent_commits_json:
        try:
            churn_n = len(json.loads(recent_commits_json))
        except (json.JSONDecodeError, TypeError):
            pass
    churn_norm = min(float(churn_n), 20.0) / 20.0

    recency = 0.0
    if last_commit_time_iso:
        try:
            s = last_commit_time_iso.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now_d = now or datetime.now(timezone.utc)
            days = max(0.0, (now_d - dt).total_seconds() / 86400.0)
            recency = max(0.0, min(1.0, 1.0 - days / 90.0))
        except (ValueError, TypeError):
            recency = 0.25
    else:
        recency = 0.0

    git_scalar = 0.65 * recency + 0.35 * churn_norm
    subs = {
        "git_recency": round(recency, 4),
        "git_churn": round(churn_norm, 4),
    }
    return round(min(1.0, max(0.0, git_scalar)), 4), subs


def composite_score(
    semantic_norm: float,
    keyword: float,
    symbol_channel: float,
    path_pri: float,
    git_channel: float = 0.0,
) -> float:
    return (
        W_SEMANTIC * semantic_norm
        + W_KEYWORD * keyword
        + W_SYMBOL * symbol_channel
        + W_PATH * path_pri
        + W_GIT * git_channel
    )


def aggregate_file_score(chunk_scores: list[float], symbol_hit_on_file: float) -> float:
    if not chunk_scores:
        return 0.0
    top = sorted(chunk_scores, reverse=True)
    max_c = top[0]
    top3 = top[:3]
    avg3 = sum(top3) / len(top3)
    return 0.6 * max_c + 0.3 * avg3 + 0.1 * symbol_hit_on_file


def context_priority_tuple(chunk: dict) -> tuple[float, float, int]:
    tier_rank = {"high": 3.0, "medium": 2.0, "low": 1.0, "none": 0.0}.get(
        chunk.get("symbol_tier", "none"), 0.0
    )
    if chunk.get("graph_hop") is not None:
        hop = int(chunk["graph_hop"])
    else:
        hop = 99
    score = float(chunk.get("score", 0.0))
    return (tier_rank, score, -hop)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def memory_score(semantic_sim: float, timestamp: datetime, now: datetime | None = None) -> float:
    """
    Score a memory entry based on bounded semantic similarity and recency decay.
    Older memories decay linearly to a floor over 90 days.
    """
    now_d = now or datetime.now(timezone.utc)
    if timestamp.tzinfo and not now_d.tzinfo:
        now_d = now_d.replace(tzinfo=timezone.utc)
    elif not timestamp.tzinfo and now_d.tzinfo:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    days = max(0.0, (now_d - timestamp).total_seconds() / 86400.0)
    # Recency drops from 1.0 to 0.5 over 90 days
    recency_weight = max(0.5, min(1.0, 1.0 - (days / 180.0)))
    
    # Scale raw cosine string matching [0,1]
    return min(1.0, semantic_sim * recency_weight)
