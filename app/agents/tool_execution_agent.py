import logging
import re
from typing import Any
import os

from app.agents.base_agent import BaseAgent
from app.memory.symbol_extract import extract_query_identifiers
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Tokens too vague or unlikely to help a literal repo grep (aligned with symbol_extract noise list).
_GREP_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "whom",
        "this", "that", "these", "those", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "should", "could", "may", "might", "must",
        "can", "not", "and", "but", "or", "nor", "for", "with", "from", "into", "onto",
        "about", "than", "then", "too", "very", "just", "also", "here", "there",
        "when", "where", "why", "how", "all", "each", "every", "some", "any", "few",
        "more", "most", "other", "such", "only", "own", "same", "both", "explain",
        "describe", "tell", "show", "give", "help", "please", "folder", "folders",
        "directory", "directories", "contents", "inside", "overview", "something",
        "anything", "everything", "repo", "repository", "codebase", "project",
        "is", "in", "it", "its", "we", "he", "she", "they", "me", "my", "by",
        "if", "of", "to", "as", "at", "on", "no", "so", "up", "us", "am",
        "work", "works", "working", "does", "did", "doing", "way", "like",
    }
)


def _build_file_search_terms(query: str, focus_entities: list[Any]) -> list[str]:
    """Derive short grep-friendly strings from NL questions + classifier entities."""
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        t = s.strip()
        if len(t) < 2:
            return
        k = t.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(t)

    if isinstance(focus_entities, list):
        for e in focus_entities:
            if isinstance(e, str):
                add(e)

    for ident in extract_query_identifiers(query):
        if ident.lower() in _GREP_STOPWORDS:
            continue
        add(ident)

    for m in re.finditer(
        r"\b[a-z][a-z0-9]{2,}\b|\b[A-Z][a-zA-Z0-9]{2,}\b|\S+\.(?:py|md|ya?ml|json)\b",
        query,
        re.IGNORECASE,
    ):
        token = m.group(0).strip()
        tl = token.lower()
        if tl in _GREP_STOPWORDS:
            continue
        add(token)

    return out[:10]

class ToolExecutionAgent(BaseAgent):
    """
    Safely executes boundaries wrapping explicitly resolved targets via ToolRegistry.
    Handles precise Log/File/Git scrapes bypassing RAG bounds.
    """
    def __init__(self, workspace_root: str = os.getcwd()):
        super().__init__("ToolExecutor")
        self.registry = ToolRegistry(workspace_root)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        target = state.get("selected_tool")
        query = state.get("query", "")
        
        state["tool_results"] = []
        if not target:
            return state

        logger.info(f"[{self.name}] Activating deterministic explicit channel: {target.upper()}")
        
        # Simple heuristic arguments (in production an LLM JSON parsing node extracts these args)
        kwargs = {}
        action = ""
        if target == "git":
            if "blame" in query:
                action = "git_blame"
                # very crude fallback pulling file heuristically
                kwargs["file_path"] = [w for w in query.split() if "." in w][0] if "." in query else "app/main.py" 
            else:
                action = "git_log"
                kwargs["n"] = 10
        elif target == "log":
            if "error" in query or "failure" in query:
                action = "get_recent_failures"
            else:
                action = "search_logs"
                kwargs["query"] = query
        elif target == "file":
            action = "search"
            cls = state.get("classification") or {}
            entities = cls.get("focus_entities") if isinstance(cls, dict) else []
            terms = _build_file_search_terms(query, entities or [])
            if terms:
                kwargs["queries"] = terms
            else:
                kwargs["query"] = query

        if action:
            res = await self.registry.execute_tool(target, action, **kwargs)
            logger.info(f"[{self.name}] Executed mapping '{action}'. Success: {res.success}")
            if res.success:
                state["tool_results"].append(res.model_dump())
                # Assign status identifying explicit success boundaries!
                state["tool_result_status"] = "success" if res.data or res.metadata else "empty"
            else:
                state["tool_result_status"] = "error"
        else:
            state["tool_result_status"] = "error"
        
        return state
