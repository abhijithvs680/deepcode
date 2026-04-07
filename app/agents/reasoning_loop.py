"""
Minimal ReAct-style loop: THINK → ACT (tool) → OBSERVE → repeat → hand off to AnswerAgent.

RAG is not used here; orchestrator runs retrieval only if `enough_context` is False
or `react_force_rag` is set (no-progress escape or broad intent).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from typing import Any

from app.core.config import get_settings
from app.core.event_bus import EventBus
from app.providers.base import LLMProvider
from app.tools.base_tool import ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def valid_observations(obs: list[Any]) -> list[str]:
    """Drop empty / noise strings so prompts are not padded with junk."""
    out: list[str] = []
    for o in obs:
        if isinstance(o, str) and len(o.strip()) > 10:
            out.append(o)
    return out


def enough_context(state: dict[str, Any]) -> bool:
    """
    Grounding only: file reads, resolved symbols, or at least one *useful* tool step.
    Observations alone do not count.
    """
    if state.get("files_read"):
        return True
    if state.get("symbols_found"):
        return True
    if int(state.get("react_useful_steps") or 0) >= 1:
        return True
    return False


def _norm_rel_path(p: str) -> str:
    return str(p).replace("\\", "/")


def _tool_step_useful(res: ToolResult | None) -> bool:
    if not res or not res.success or res.data is None:
        return False
    d = res.data
    if not isinstance(d, dict):
        return bool(d)
    action = (res.action or "").strip()

    if action == "read_file":
        return len((d.get("content") or "")) > 50

    if action == "search":
        m = d.get("matches")
        return isinstance(m, list) and len(m) >= 2

    if action == "find_symbol":
        m = d.get("matches")
        return isinstance(m, list) and len(m) >= 1

    if action == "get_symbols":
        sy = d.get("symbols")
        return isinstance(sy, list) and len(sy) >= 1

    if action == "list_dir":
        ent = d.get("entries")
        return isinstance(ent, list) and len(ent) >= 5

    for key in ("commits", "lines", "entries", "matches", "output", "content", "log_lines"):
        v = d.get(key)
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and len(v.strip()) > 20:
            return True
    return False


def _step_result_summary(res: ToolResult | None) -> str:
    if not res:
        return "no result"
    if not res.success:
        return (res.error or "failed")[:120]
    if not res.data or not isinstance(res.data, dict):
        return "ok"
    d = res.data
    action = res.action or ""
    if action == "read_file":
        return f"read {len(d.get('content') or '')} chars"
    if action == "search":
        return f"{len(d.get('matches') or [])} matches"
    if action == "find_symbol":
        return f"{len(d.get('matches') or [])} symbols"
    if action == "list_dir":
        return f"{len(d.get('entries') or [])} entries"
    return "ok"


def _bump_file_scores(
    scores: dict[str, float],
    last_touch_step: dict[str, int],
    step_i: int,
    tool_name: str,
    action: str,
    args: dict[str, Any],
    data: dict[str, Any] | None,
) -> None:
    if tool_name != "file" or not isinstance(data, dict):
        return

    def add_path(path: str, base_inc: float) -> None:
        p = _norm_rel_path(path)
        prev = last_touch_step.get(p)
        if prev is None:
            w = 1.0
        else:
            w = 1.0 / (1.0 + max(0, step_i - prev))
        scores[p] += base_inc * w
        last_touch_step[p] = step_i

    if action == "read_file":
        p = args.get("path") or args.get("file_path")
        if p:
            add_path(str(p), 3.0)
        return
    if action == "search":
        for m in data.get("matches") or []:
            if not isinstance(m, dict):
                continue
            fp = m.get("file") or m.get("file_path") or m.get("path")
            if isinstance(fp, str) and fp.strip():
                add_path(fp, 1.0)
        return
    if action == "find_symbol":
        for m in data.get("matches") or []:
            if not isinstance(m, dict):
                continue
            fp = m.get("file")
            if isinstance(fp, str) and fp.strip():
                add_path(fp, 2.0)


def _maybe_compress_react_state(state: dict[str, Any]) -> None:
    steps = state.get("react_steps") or []
    if len(steps) <= 3:
        return
    head = steps[:-2]
    bits = [
        f"{s.get('tool', '?')}.{s.get('action', '?')}" for s in head
    ]
    prev = (state.get("react_summary") or "").strip()
    chunk = "; ".join(bits)[:600]
    state["react_summary"] = (prev + "\n" + chunk if prev else chunk)[:1200]
    state["react_steps"] = steps[-2:]
    obs = state.get("react_observations") or []
    if len(obs) > 8:
        state["react_observations"] = obs[-6:]


def _exploration_hint(step_index: int, max_steps: int, intent: str) -> str:
    base = (
        f"Exploration strategy (step {step_index + 1}/{max_steps}): "
        "prefer search or list_dir first to locate paths, "
        "then read_file on the best candidate, "
        "then find_symbol when hunting definitions/APIs."
    )
    if intent == "broad":
        return base + " This task is broad — skim quickly, then rely on retrieval."
    if intent == "exploratory":
        return base + " Explore without assuming a single file."
    return base + " Prefer precise file/symbol tools when possible."


_REACT_SYSTEM = """You are a coding agent step planner. Reply with ONE JSON object only (no markdown).
Schema:
{
  "thought": "short reasoning",
  "tool": "file" | "git" | "log" | null,
  "action": "read_file" | "search" | "list_dir" | "find_symbol" | "get_symbols" | "done" | "git_log" | "git_blame" | "search_logs" | "get_recent_failures",
  "args": { },
  "done": false
}
Rules:
- Use workspace-relative paths in args (e.g. path: "app/main.py").
- For read_file optionally pass start_line, end_line (integers) to limit lines; default read is capped by the tool.
- list_dir: args {"path": ".", "depth": 2}
- find_symbol: args {"name": "SymbolName"} — requires indexed repo
- search: args {"query": "snippet"} or {"queries": ["a","b"]}
- git_log: args {} or {"file_path": "optional", "n": 10}
- Set tool null, action "done", done true when further tools are unnecessary.
"""


async def _emit_react(state: dict[str, Any], event: dict[str, Any]) -> None:
    await EventBus.emit(event)
    cb = state.get("react_event_callback")
    if cb:
        try:
            r = cb(event)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass


async def run_reasoning_loop(
    state: dict[str, Any],
    provider: LLMProvider,
    registry: ToolRegistry,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    max_steps = int(max_steps if max_steps is not None else settings.REACT_MAX_STEPS)
    max_steps = max(1, min(max_steps, 5))

    intent = (state.get("classification") or {}).get("agent_intent", "exploratory")
    if intent == "broad":
        state["react_force_rag"] = True
        max_steps = min(max_steps, 2)

    q = state.get("current_step") or state.get("query") or ""
    router = state.get("selected_tool")
    rconf = float(state.get("router_confidence") or 0.0)

    state["react_steps"] = []
    state["react_observations"] = []
    state["react_summary"] = state.get("react_summary") or ""
    state["files_read"] = []
    state["symbols_found"] = []
    state["react_useful_steps"] = 0
    state["react_failed_steps"] = 0
    state["react_force_rag"] = intent == "broad"
    state["file_scores"] = dict(state.get("file_scores") or {})
    file_scores: dict[str, float] = defaultdict(float)
    last_touch_step: dict[str, int] = {}
    for k, v in (state.get("file_scores") or {}).items():
        try:
            file_scores[_norm_rel_path(str(k))] = float(v)
        except (TypeError, ValueError):
            pass

    prior_tools = list(state.get("tool_results") or [])
    state["tool_results"] = []

    await _emit_react(
        state,
        {"type": "react_session_start", "intent": intent, "max_steps": max_steps},
    )

    last_loop_i = -1
    for i in range(max_steps):
        last_loop_i = i
        vo = valid_observations(state.get("react_observations") or [])
        obs_summary = "\n".join(str(x)[:500] for x in vo[-4:])
        summary_note = ""
        if (state.get("react_summary") or "").strip():
            summary_note = f"Earlier summary: {(state.get('react_summary') or '')[:400]}\n"

        user_block = (
            f"User task: {q}\n"
            f"{_exploration_hint(i, max_steps, intent)}\n"
            f"{summary_note}"
            f"Router hint: tool={router!r} confidence={rconf:.2f}\n"
            f"Workspace root: {registry.workspace_root}\n"
            f"Recent observations (truncated):\n{obs_summary or '(none)'}\n"
            f"Step {i + 1} of {max_steps}."
        )
        prompt = _REACT_SYSTEM + "\n\n" + user_block

        try:
            raw = await provider.generate(
                prompt,
                system_prompt="Output only valid JSON for the next action.",
                temperature=0.15,
            )
        except Exception as exc:
            logger.warning("ReasoningLoop LLM step failed: %s", exc)
            await _emit_react(
                state,
                {"type": "react_llm_error", "step": i, "error": str(exc)[:200]},
            )
            break

        step = _parse_json_object(raw) or {}
        step.setdefault("thought", "")
        step.setdefault("tool", None)
        step.setdefault("action", "done")
        step.setdefault("args", {})
        step.setdefault("done", False)

        state["react_steps"].append(
            {
                "step_index": i,
                "thought": step.get("thought"),
                "tool": step.get("tool"),
                "action": step.get("action"),
                "args": step.get("args"),
            }
        )
        _maybe_compress_react_state(state)

        if step.get("done") or step.get("action") == "done" or not step.get("tool"):
            logger.info("ReasoningLoop: planner signaled done at step %s", i)
            await _emit_react(
                state,
                {"type": "react_done", "step": i, "thought": step.get("thought")},
            )
            break

        tool_name = str(step.get("tool") or "").strip().lower()
        action = str(step.get("action") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}

        await _emit_react(
            state,
            {
                "type": "react_step_start",
                "step": i,
                "tool": tool_name,
                "action": action,
                "args": args,
            },
        )

        if tool_name not in ("file", "git", "log"):
            state["react_observations"].append(
                f"Step {i}: invalid tool {tool_name!r} — stopping."
            )
            state["react_failed_steps"] += 1
            await _emit_react(
                state,
                {
                    "type": "react_step_result",
                    "step": i,
                    "success": False,
                    "summary": "invalid tool",
                },
            )
            if state["react_failed_steps"] >= 2:
                if state["react_useful_steps"] == 0:
                    state["react_force_rag"] = True
                break
            continue

        res: ToolResult | None
        try:
            res = await registry.execute_tool(tool_name, action, **args)
        except Exception as exc:
            res = None
            state["react_observations"].append(f"Step {i}: tool error {exc}")
            state["react_failed_steps"] += 1
            await _emit_react(
                state,
                {
                    "type": "react_step_result",
                    "step": i,
                    "success": False,
                    "summary": str(exc)[:160],
                },
            )
            if state["react_failed_steps"] >= 2:
                if state["react_useful_steps"] == 0:
                    state["react_force_rag"] = True
                logger.info(
                    "ReasoningLoop: stopping after %s failed steps",
                    state["react_failed_steps"],
                )
                break
            continue

        useful = _tool_step_useful(res)
        if useful:
            state["react_useful_steps"] += 1
        else:
            state["react_failed_steps"] += 1

        dumped = res.model_dump() if res is not None else {}
        state["tool_results"].append(dumped)

        ob = f"Step {i}: {tool_name}.{action} success={getattr(res, 'success', False)}"
        if res and res.success and res.data:
            preview = json.dumps(res.data, default=str)[:2000]
            ob += f" data={preview}"
        elif res and res.error:
            ob += f" error={res.error[:400]}"
        state["react_observations"].append(ob)

        await _emit_react(
            state,
            {
                "type": "react_step_result",
                "step": i,
                "success": bool(res and res.success),
                "useful": useful,
                "summary": _step_result_summary(res),
            },
        )

        if res and res.data and isinstance(res.data, dict):
            _bump_file_scores(
                file_scores, last_touch_step, i, tool_name, action, args, res.data
            )

        if tool_name == "file" and action == "read_file":
            p = args.get("path") or args.get("file_path")
            if p and p not in state["files_read"]:
                state["files_read"].append(str(p))
        if tool_name == "file" and action == "find_symbol":
            for m in (res.data or {}).get("matches") or [] if res and res.data else []:
                sym = m.get("symbol")
                if sym and sym not in state["symbols_found"]:
                    state["symbols_found"].append(sym)

        if state["react_failed_steps"] >= 2:
            if state["react_useful_steps"] == 0:
                state["react_force_rag"] = True
            logger.info(
                "ReasoningLoop: stopping after %s low-value/failed steps",
                state["react_failed_steps"],
            )
            break

        if i >= 2 and state["react_useful_steps"] == 0:
            state["react_force_rag"] = True
            logger.info(
                "ReasoningLoop: no useful grounding through step %s → force RAG", i
            )
            break

        if enough_context(state):
            logger.info("ReasoningLoop: enough_context met at step %s", i)
            await _emit_react(
                state,
                {"type": "react_enough_context", "step": i},
            )
            break

    final_i = max(last_loop_i, 0)
    for p in list(file_scores.keys()):
        staleness = final_i - last_touch_step.get(p, final_i)
        if staleness > 0:
            file_scores[p] *= 1.0 / (1.0 + staleness)

    state["file_scores"] = {
        k: round(v, 4) for k, v in sorted(file_scores.items(), key=lambda kv: -kv[1])
    }
    top_files = sorted(file_scores.keys(), key=lambda p: file_scores[p], reverse=True)[:2]
    state["react_priority_files"] = top_files

    state["tool_results"] = prior_tools + state["tool_results"]
    await _emit_react(
        state,
        {
            "type": "react_session_end",
            "useful_steps": state["react_useful_steps"],
            "failed_steps": state["react_failed_steps"],
            "force_rag": bool(state.get("react_force_rag")),
        },
    )
    return state
