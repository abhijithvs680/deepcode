"""One-shot LLM repair for failed transactional <edit> search/replace."""
from __future__ import annotations

import json
import logging
import os
import re
from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_REPAIR_SYSTEM = """You fix broken search/replace edits. Output ONE JSON object only (no markdown).
Schema:
{"search": "exact substring that must exist in the file once", "replace": "replacement text"}
Rules:
- "search" must match the file literally (whitespace-sensitive unless clearly wrong).
- If the old search was wrong, propose a corrected search that exists exactly once.
- Keep "replace" aligned with the user's intent from the context.
"""


def _parse_repair_json(text: str) -> dict[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    s = obj.get("search")
    r = obj.get("replace")
    if not isinstance(s, str) or not isinstance(r, str):
        return None
    return {"search": s, "replace": r}


async def repair_search_replace(
    provider: LLMProvider,
    *,
    repo_path: str,
    rel_path: str,
    original_search: str,
    original_replace: str,
    error: str,
    user_query: str,
    max_excerpt: int = 14_000,
) -> dict[str, str] | None:
    """
    Ask the model for a corrected search/replace given tool error + file excerpt.
    Returns {"search", "replace"} or None.
    """
    root = os.path.abspath(repo_path)
    rel = rel_path.replace("\\", "/")
    abs_p = os.path.abspath(os.path.join(root, rel))
    if not abs_p.startswith(root) or not os.path.isfile(abs_p):
        return None
    try:
        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None
    if len(content) > max_excerpt:
        content = content[:max_excerpt] + "\n\n… [truncated] …"

    user = (
        f"User task (context): {user_query[:500]}\n"
        f"File: {rel}\n"
        f"Apply tool error: {error[:600]}\n"
        f"Previous search (failed):\n---\n{original_search[:4000]}\n---\n"
        f"Previous replace:\n---\n{original_replace[:4000]}\n---\n"
        f"Current file contents:\n---\n{content}\n---\n"
        "Return JSON with corrected search and replace."
    )
    try:
        raw = await provider.generate(
            user,
            system_prompt=_REPAIR_SYSTEM,
            temperature=0.12,
        )
    except Exception as exc:
        logger.warning("edit_repair LLM failed: %s", exc)
        return None
    out = _parse_repair_json(raw)
    if not out:
        logger.info("edit_repair: could not parse model output")
    return out
