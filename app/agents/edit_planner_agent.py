"""
Structured multi-file edit plan from exploration context (JSON via LLM).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.edit_context import enrich_edit_context_dependencies
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = """You are an edit planner. Output ONE JSON object only (no markdown).
Schema:
{
  "plan": [
    {
      "file": "relative/path.py",
      "change": "one-line summary",
      "target_symbol": "optional",
      "dependencies": ["other/file.py"]
    }
  ],
  "rationale": "short"
}
Rules:
- Every "file" MUST appear in the allowed_files list provided (exact path).
- At most N plan items (N given in user message).
- If allowed_files is empty, return {"plan": [], "rationale": "need file reads first"}.
"""


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return None


class EditPlannerAgent(BaseAgent):
    def __init__(self, provider, db=None):
        super().__init__("EditPlanner")
        self._provider = provider
        self._db = db
        self.settings = get_settings()

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("write_enabled") or state.get("mode") != "edit":
            return state

        await enrich_edit_context_dependencies(
            state, self._db, state.get("repo_id")
        )
        ctx = state.get("edit_context") or {}
        allowed = sorted(
            {f.replace("\\", "/") for f in (ctx.get("files") or []) if f}
        )
        max_items = min(
            5,
            int(getattr(self.settings, "EDIT_MAX_AUTO_FILES", 5) or 5),
        )

        xs = (state.get("cross_subgoal_summary") or "").strip()
        cross_blk = (
            f"Cross-subgoal synthesis (use for consistency):\n{xs[:2400]}\n"
            if xs
            else ""
        )
        user = (
            f"Task: {state.get('query', '')}\n"
            f"{cross_blk}"
            f"allowed_files (you may ONLY reference these): {json.dumps(allowed)}\n"
            f"symbols: {json.dumps(ctx.get('symbols') or [])}\n"
            f"dependency_edges_sample: {json.dumps((ctx.get('dependency_edges') or [])[:24])}\n"
            f"Max plan items: {max_items}\n"
        )
        try:
            raw = await self._provider.generate(
                user,
                system_prompt=_PLAN_SYSTEM,
                temperature=0.1,
            )
            plan_obj = _parse_json_obj(raw) or {}
        except Exception as exc:
            logger.warning("EditPlanner LLM failed: %s", exc)
            plan_obj = {"plan": [], "rationale": str(exc)}

        raw_plan = plan_obj.get("plan") if isinstance(plan_obj, dict) else []
        if not isinstance(raw_plan, list):
            raw_plan = []

        allow_set = set(allowed)
        validated: list[dict[str, Any]] = []
        for item in raw_plan[:max_items]:
            if not isinstance(item, dict):
                continue
            fp = str(item.get("file", "")).replace("\\", "/").strip()
            if fp not in allow_set:
                logger.info("EditPlanner dropped file not in allowed_files: %s", fp)
                continue
            validated.append(
                {
                    "file": fp,
                    "change": str(item.get("change", ""))[:300],
                    "target_symbol": str(item.get("target_symbol", ""))[:120],
                    "dependencies": [
                        z
                        for x in (item.get("dependencies") or [])
                        if (z := str(x).replace("\\", "/")) in allow_set
                    ][:8],
                }
            )

        state["edit_plan"] = {
            "plan": validated,
            "rationale": str(plan_obj.get("rationale", ""))[:500]
            if isinstance(plan_obj, dict)
            else "",
        }
        return state
