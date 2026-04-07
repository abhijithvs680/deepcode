"""Reconcile multiple subgoal traces — structured output for Answer + control hooks."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_SYNTH_SYSTEM = """You reconcile several subgoal exploration traces for one user task.
Output ONE JSON object only (no markdown).
Schema:
{
  "agreements": ["bullet facts multiple subgoals support"],
  "gaps": ["what is still unknown or unverified"],
  "conflicts": ["only if traces disagree; else []"],
  "merged_summary": "3-6 sentences for the final answer agent, neutral tone"
}
Use only information implied by the traces; do not invent paths or behavior."""


def _parse_synth(text: str) -> dict[str, Any] | None:
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


def _format_synthesis(obj: dict[str, Any]) -> str:
    parts = []
    if obj.get("merged_summary"):
        parts.append(str(obj["merged_summary"]).strip())
    ag = obj.get("agreements") or []
    if isinstance(ag, list) and ag:
        parts.append("Agreements: " + "; ".join(str(x) for x in ag[:8]))
    gp = obj.get("gaps") or []
    if isinstance(gp, list) and gp:
        parts.append("Gaps: " + "; ".join(str(x) for x in gp[:6]))
    cf = obj.get("conflicts") or []
    if isinstance(cf, list) and cf:
        parts.append("Conflicts: " + "; ".join(str(x) for x in cf[:5]))
    return "\n".join(parts).strip()


def _norm_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


async def synthesize_cross_subgoals(
    state: dict[str, Any],
    provider: LLMProvider,
) -> dict[str, Any] | None:
    """
    Returns dict with summary_text, gaps, conflicts, agreements; or None if skipped.
    """
    traces = state.get("subgoal_traces") or []
    if len(traces) < 2:
        return None

    payload = {
        "goal": (state.get("query") or "")[:400],
        "subgoals": [
            {
                "id": t.get("id"),
                "confidence": t.get("confidence"),
                "files_implied": [
                    str(x)
                    for x in (t.get("observations") or [])[:3]
                    if isinstance(x, str)
                ],
                "observations_tail": [
                    str(x)[:320] for x in (t.get("observations") or [])[-4:]
                ],
            }
            for t in traces
            if isinstance(t, dict)
        ],
    }
    user = json.dumps(payload, indent=2, default=str)[:6000]
    try:
        raw = await provider.generate(
            user,
            system_prompt=_SYNTH_SYSTEM,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("cross_subgoal synthesis failed: %s", exc)
        return None

    obj = _parse_synth(raw)
    if not isinstance(obj, dict):
        return None
    gaps = _norm_str_list(obj.get("gaps"))
    conflicts = _norm_str_list(obj.get("conflicts"))
    agreements = _norm_str_list(obj.get("agreements"))
    return {
        "summary_text": _format_synthesis(obj),
        "gaps": gaps,
        "conflicts": conflicts,
        "agreements": agreements,
        "merged_summary": str(obj.get("merged_summary") or "").strip(),
    }
