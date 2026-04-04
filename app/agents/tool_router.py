import json
import logging
from typing import Any, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _heuristic_tool(query: str) -> Optional[str]:
    q = query.lower()
    if "error" in q or "log" in q:
        return "log"
    if "commit" in q or "who changed" in q:
        return "git"
    return None


async def resolve_tool_routing(
    query: str,
    provider: Any,
    *,
    critic_suggested: Optional[str] = None,
    adaptive_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Returns structured routing metadata for observability and downstream gating.

    Keys: selected_tool (str|None), confidence (float), source ("critic"|"llm"|"heuristic"|"none")
    """
    if critic_suggested:
        logger.info(f"[ToolRouter] Critic override tool: '{critic_suggested}'")
        return {
            "selected_tool": critic_suggested,
            "confidence": 1.0,
            "source": "critic",
        }

    settings = get_settings()
    router_model = getattr(settings, "LLM_ROUTER_MODEL", settings.LLM_MODEL)

    prompt = (
        "You are a tool router.\n\n"
        "Select the best tool for the query.\n\n"
        "Available tools:\n"
        "- file: reading code, evaluating file structures, code contents\n"
        "- git: git history, commit blame, authors, timing, commits\n"
        "- log: finding errors, stack traces, exceptions in local error logs\n\n"
        "Return ONLY JSON mapped exactly as follows:\n"
        '{ "tool": "file" | "git" | "log" | null, "confidence": 0.0 }\n\n'
        f"Query: {query}"
    )

    try:
        response = await provider.generate(prompt, model=router_model)

        payload: dict = {}
        if "{" in response and "}" in response:
            try:
                json_str = response[response.find("{") : response.rfind("}") + 1]
                payload = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        if not payload:
            logger.warning("[ToolRouter] Fast model failed proper JSON serialization")
            h = _heuristic_tool(query)
            return {
                "selected_tool": h,
                "confidence": 0.55 if h else 0.0,
                "source": "heuristic",
            }

        tool_val = payload.get("tool")
        confidence = float(payload.get("confidence", 0.0))

        threshold = 0.6
        if adaptive_config:
            threshold = float(adaptive_config.get("router_confidence_threshold", 0.6))
            boost = adaptive_config.get("tool_boost") or {}
            if isinstance(tool_val, str) and tool_val in boost:
                confidence = min(1.0, confidence + float(boost.get(tool_val, 0.0)))

        if confidence < threshold:
            logger.info(
                f"[ToolRouter] Model returned tool '{tool_val}' below confidence ({confidence} < {threshold}). Pure RAG."
            )
            return {"selected_tool": None, "confidence": confidence, "source": "llm"}

        if tool_val in ["file", "git", "log"]:
            logger.info(f"[ToolRouter] Selected '{tool_val}' (confidence: {confidence})")
            return {"selected_tool": tool_val, "confidence": confidence, "source": "llm"}

    except Exception as e:
        logger.error(f"[ToolRouter] Execution failed: {e}")

    h = _heuristic_tool(query)
    return {
        "selected_tool": h,
        "confidence": 0.55 if h else 0.0,
        "source": "heuristic",
    }


async def select_tool_llm(query: str, provider: Any) -> Optional[str]:
    """Backward-compatible: returns only the selected tool name or None."""
    r = await resolve_tool_routing(query, provider)
    return r.get("selected_tool")
