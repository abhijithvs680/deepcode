import time
from typing import Any

from app.tools.base_tool import ToolResult


def max_tool_score_from_dicts(tool_results: list[dict[str, Any]] | None) -> float:
    """Max deterministic tool score across serialized ToolResult dicts."""
    if not tool_results:
        return 0.0
    m = 0.0
    for tr_dict in tool_results:
        try:
            m = max(m, score_tool_result(ToolResult(**tr_dict)))
        except Exception:
            continue
    return m

def score_tool_result(result: ToolResult) -> float:
    """
    Evaluates the precision mapped natively across the deterministic tools.
    Outputs over 0.8 signal the Agent can heavily bypass RAG retrieval.
    Outputs below 0.3 signal the Tool failed or retrieved nothing of value.
    """
    if not result.success or result.error:
        return 0.0
        
    score = 0.5
    action = result.action
    meta = result.metadata
    
    if action == "git_log":
        score += min(0.4, meta.get("count", 0) * 0.05)
    elif action == "git_blame" or action == "git_diff":
        score += 0.45
    elif action == "get_recent_changes":
        score += 0.4
    elif action == "search_logs":
        count = meta.get("found_count", 0)
        if count > 0:
            score += min(0.45, count * 0.05)
        else:
            score = 0.1
    elif action == "get_recent_failures":
        errors = meta.get("total_errors", 0)
        if errors > 0:
            score += 0.45
        else:
            score = 0.1
    elif action == "read_file":
        truncated = meta.get("truncated", False)
        score += 0.2 if truncated else 0.4
    elif action == "search":
        matches = meta.get("total_matches_evaluated", 0)
        if matches > 0:
            score += 0.35
        else:
            score = 0.1
            
    # Apply recency heuristic dynamically.
    age = time.time() - result.timestamp
    if age > 60:
        score -= min(0.2, age / 3600 * 0.05)
        
    return max(0.0, min(1.0, score))
