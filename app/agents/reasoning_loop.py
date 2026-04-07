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
import time
from collections import defaultdict
from typing import Any

from app.agents.react_grounding import (
    compute_answer_confidence,
    extract_goal,
    goal_progress_delta,
    keyword_relevance,
)
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


def _maybe_force_rag_escape(state: dict[str, Any]) -> None:
    """Escalate to RAG only when exploration is stalled (no useful steps, low progress)."""
    if int(state.get("react_useful_steps") or 0) > 0:
        return
    if float(state.get("goal_progress") or 0) >= 0.3:
        return
    cls_c = float(
        (state.get("classification") or {}).get("confidence", 1.0) or 1.0
    )
    if cls_c >= 0.4:
        return
    state["react_force_rag"] = True


def valid_observations(obs: list[Any]) -> list[str]:
    """Drop empty / noise strings so prompts are not padded with junk."""
    out: list[str] = []
    for o in obs:
        if isinstance(o, str) and len(o.strip()) > 10:
            out.append(o)
    return out


def enough_context(state: dict[str, Any]) -> bool:
    """
    Grounding only: goal progress, file reads, resolved symbols, or one useful tool step.
    """
    if float(state.get("goal_progress") or 0) >= 0.69:
        return True
    if state.get("files_read"):
        return True
    if state.get("symbols_found"):
        return True
    if int(state.get("react_useful_steps") or 0) >= 1:
        return True
    return False


def _norm_rel_path(p: str) -> str:
    return str(p).replace("\\", "/")


_REL_OK = 0.22


def _tool_step_useful(res: ToolResult | None, query: str) -> bool:
    if not res or not res.success or res.data is None:
        return False
    d = res.data
    if not isinstance(d, dict):
        return bool(d)
    action = (res.action or "").strip()
    g = query or ""

    if action == "read_file":
        c = d.get("content") or ""
        if len(c) <= 50:
            return False
        return keyword_relevance(g, c) >= _REL_OK

    if action == "search":
        m = d.get("matches")
        if not isinstance(m, list) or len(m) < 2:
            return False
        blob = " ".join(
            str(x.get("content", "")) for x in m if isinstance(x, dict)
        )
        return keyword_relevance(g, blob, *(str(x.get("file", "")) for x in m if isinstance(x, dict))) >= _REL_OK

    if action == "find_symbol":
        m = d.get("matches")
        if not isinstance(m, list) or len(m) < 1:
            return False
        blob = " ".join(
            str(x.get("symbol", "")) + " " + str(x.get("file", ""))
            for x in m
            if isinstance(x, dict)
        )
        return keyword_relevance(g, blob) >= (_REL_OK * 0.75)

    if action == "get_symbols":
        sy = d.get("symbols")
        if not isinstance(sy, list) or len(sy) < 1:
            return False
        blob = " ".join(str(x.get("name", "")) for x in sy if isinstance(x, dict))
        return keyword_relevance(g, blob) >= (_REL_OK * 0.75)

    if action == "list_dir":
        ent = d.get("entries")
        if not isinstance(ent, list) or len(ent) < 5:
            return False
        blob = " ".join(str(x.get("path", "")) for x in ent if isinstance(x, dict))
        return keyword_relevance(g, blob) >= (_REL_OK * 0.6)

    if action == "get_file_dependencies":
        for k in ("imports", "imported_by"):
            v = d.get(k)
            if isinstance(v, list) and len(v) > 0:
                return True
        return False

    if action == "get_symbol_dependencies":
        if isinstance(d.get("matches"), list) and len(d.get("matches")) > 0:
            return True
        gph = d.get("graph")
        if isinstance(gph, dict) and gph:
            return True
        return False

    if res.tool == "file_edit" and action == "apply_search_replace":
        return bool(res.success)
    if res.tool == "file_edit" and action == "commit_session":
        return bool(res.success)

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
    if action == "get_file_dependencies":
        a = len(d.get("imports") or [])
        b = len(d.get("imported_by") or [])
        return f"deps imports={a} imported_by={b}"
    if action == "get_symbol_dependencies":
        return f"sym matches={len(d.get('matches') or [])}"
    if res.tool == "file_edit" and action == "apply_search_replace":
        return f"edit {d.get('file', '?')}"
    if res.tool == "file_edit" and action == "commit_session":
        return f"commit {len(d.get('files_written') or [])} files"
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
    if not isinstance(data, dict):
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

    if tool_name == "file_edit" and action == "apply_search_replace":
        fp = data.get("file") or args.get("file_path") or args.get("path")
        if fp:
            add_path(str(fp), 2.5)
        return
    if tool_name == "file_edit" and action == "commit_session":
        for fp in data.get("files_written") or []:
            if isinstance(fp, str) and fp.strip():
                add_path(fp, 1.5)
        return

    if tool_name != "file":
        return

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
        return
    if action == "get_file_dependencies":
        for k in ("imports", "imported_by"):
            for fp in data.get(k) or []:
                if isinstance(fp, str) and fp.strip():
                    add_path(fp, 1.2)
        p = args.get("path") or args.get("file_path")
        if p:
            add_path(str(p), 1.5)
        return
    if action == "get_symbol_dependencies":
        for m in data.get("matches") or []:
            if not isinstance(m, dict):
                continue
            fp = m.get("file")
            if isinstance(fp, str) and fp.strip():
                add_path(fp, 1.8)
        gph = data.get("graph")
        if isinstance(gph, dict):
            for _fp, bundle in gph.items():
                if not isinstance(bundle, dict):
                    continue
                for k in ("imports", "imported_by"):
                    for fp in bundle.get(k) or []:
                        if isinstance(fp, str) and fp.strip():
                            add_path(fp, 1.0)


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


def _edit_tool_enabled(state: dict[str, Any]) -> bool:
    if state.get("dry_run"):
        return False
    if not getattr(get_settings(), "REACT_EDIT_TOOL_ENABLED", True):
        return False
    if state.get("write_enabled"):
        return True
    if str(state.get("mode") or "").lower() == "edit":
        return True
    return False


def _edit_allow_paths(state: dict[str, Any]) -> set[str]:
    s: set[str] = set()
    for p in state.get("files_read") or []:
        s.add(str(p).replace("\\", "/"))
    for p in state.get("hint_files") or []:
        s.add(str(p).replace("\\", "/"))
    for p in state.get("react_priority_files") or []:
        s.add(str(p).replace("\\", "/"))
    for p in state.get("active_working_set") or []:
        s.add(str(p).replace("\\", "/"))
    return s


def _path_allowed_for_edit(rel: str, allow: set[str]) -> bool:
    fn = str(rel).replace("\\", "/").strip()
    if not fn:
        return False
    if fn in allow:
        return True
    return any(fn.startswith(a + "/") for a in allow if a)


async def finalize_react_edit_session(
    registry: ToolRegistry,
    state: dict[str, Any],
    *,
    commit: bool,
) -> None:
    """
    Subgoal-level transaction: commit allowed pending edits or rollback working copies.
    Call once after each run_reasoning_loop when file_edit may have been used.
    """
    if not getattr(get_settings(), "REACT_SUBGOAL_EDIT_TRANSACTION", True):
        from app.tools.file_edit_tool import FileEditTool

        ed0 = registry.get_tool("file_edit")
        if isinstance(ed0, FileEditTool) and ed0.has_active_session():
            try:
                await registry.execute_tool("file_edit", "rollback_session")
            except Exception:
                pass
        return
    from app.tools.file_edit_tool import FileEditTool

    ed = registry.get_tool("file_edit")
    if not isinstance(ed, FileEditTool):
        return
    if not ed.has_active_session():
        return
    stg = get_settings()
    if bool(state.get("react_edit_dry_run")) or getattr(
        stg, "REACT_EDIT_DRY_RUN", False
    ):
        commit = False
    allow = _edit_allow_paths(state)
    try:
        if commit and ed.has_pending_changes() and allow:
            await registry.execute_tool(
                "file_edit",
                "commit_session",
                allowed_files=sorted(allow)[:120],
            )
        else:
            await registry.execute_tool("file_edit", "rollback_session")
    except Exception as exc:
        logger.warning("finalize_react_edit_session: %s", exc)


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


_REACT_SYSTEM_NO_EDIT = """You are a coding agent step planner. Reply with ONE JSON object only (no markdown).
Schema:
{
  "thought": "short reasoning",
  "tool": "file" | "git" | "log" | null,
  "action": "read_file" | "search" | "list_dir" | "find_symbol" | "get_symbols" | "get_file_dependencies" | "get_symbol_dependencies" | "done" | "git_log" | "git_blame" | "search_logs" | "get_recent_failures",
  "args": { },
  "done": false
}
Rules:
- Use workspace-relative paths in args (e.g. path: "app/main.py").
- For read_file optionally pass start_line, end_line (integers) to limit lines; default read is capped by the tool.
- list_dir: args {"path": ".", "depth": 2}
- find_symbol: args {"name": "SymbolName"} — requires indexed repo
- get_file_dependencies: args {"path": "app/foo.py", "limit": 30} — import graph (indexed repo + DB)
- get_symbol_dependencies: args {"name": "SymbolName", "limit": 20} — symbol → file + import neighborhood
- search: args {"query": "snippet"} or {"queries": ["a","b"]}
- git_log: args {} or {"file_path": "optional", "n": 10}
- Set tool null, action "done", done true when further tools are unnecessary.
"""

_REACT_SYSTEM_WITH_EDIT = """You are a coding agent step planner. Reply with ONE JSON object only (no markdown).
Schema:
{
  "thought": "short reasoning",
  "tool": "file" | "file_edit" | "git" | "log" | null,
  "action": "read_file" | "search" | "list_dir" | "find_symbol" | "get_symbols" | "get_file_dependencies" | "get_symbol_dependencies" | "apply_search_replace" | "commit_session" | "rollback_session" | "done" | "git_log" | "git_blame" | "search_logs" | "get_recent_failures",
  "args": { },
  "done": false
}
Rules:
- Use workspace-relative paths in args (e.g. path: "app/main.py").
- For read_file optionally pass start_line, end_line (integers) to limit lines; default read is capped by the tool.
- list_dir: args {"path": ".", "depth": 2}
- find_symbol: args {"name": "SymbolName"} — requires indexed repo
- get_file_dependencies / get_symbol_dependencies before editing unknown modules.
- search: args {"query": "snippet"} or {"queries": ["a","b"]}
- file_edit.apply_search_replace: args {"file_path":"app/x.py","search":"exact snippet from file","replace":"new text","expected_occurrences":1}
  Only edit files you already read_file in this subgoal (allowed paths are enforced).
  Edits are batched in memory and committed once at subgoal end if exploration succeeds; mid-subgoal commit_session is optional.
- file_edit.commit_session: args {"allowed_files":["app/x.py",...]} — force early persist (optional).
- file_edit.rollback_session: args {} — discard uncommitted in-session edits.
- git_log: args {} or {"file_path": "optional", "n": 10}
- Set tool null, action "done", done true when further tools are unnecessary.
"""


def _react_system_for(state: dict[str, Any]) -> str:
    if _edit_tool_enabled(state):
        return _REACT_SYSTEM_WITH_EDIT
    return _REACT_SYSTEM_NO_EDIT


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


async def _emit_step_evt(
    state: dict[str, Any],
    *,
    phase: str,
    step: int,
    tool: str | None = None,
    action: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    duration_ms: int | None = None,
    useful: bool | None = None,
    args_preview: str | None = None,
) -> None:
    ev: dict[str, Any] = {
        "type": "step",
        "phase": phase,
        "step": step,
    }
    if tool is not None:
        ev["tool"] = tool
    if action is not None:
        ev["action"] = action
    if status is not None:
        ev["status"] = status
    if summary is not None:
        ev["summary"] = summary
    if duration_ms is not None:
        ev["duration_ms"] = duration_ms
    if useful is not None:
        ev["useful"] = useful
    if args_preview and phase == "start":
        ev["args_preview"] = args_preview[:220]
    await _emit_react(state, ev)


async def run_reasoning_loop(
    state: dict[str, Any],
    provider: LLMProvider,
    registry: ToolRegistry,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    hard_cap = int(getattr(settings, "REACT_MAX_STEPS_CAP", 7) or 7)
    soft = int(max_steps if max_steps is not None else settings.REACT_MAX_STEPS)
    max_steps = max(1, min(soft, hard_cap))
    cls_loop = float(
        (state.get("classification") or {}).get("confidence", 1.0) or 1.0
    )
    if cls_loop < 0.4:
        max_steps = min(max_steps + 2, hard_cap)
    if cls_loop < 0.3:
        max_steps = min(max_steps + 2, hard_cap)

    intent = (state.get("classification") or {}).get("agent_intent", "exploratory")
    if intent == "broad":
        state["react_force_rag"] = True
        max_steps = min(max_steps, 2)

    q = state.get("current_step") or state.get("query") or ""
    router = state.get("selected_tool")
    rconf = float(state.get("router_confidence") or 0.0)
    state["react_goal"] = extract_goal(q)
    state["goal_progress"] = 0.0
    _ACTION_SOFT_LIMITS: dict[str, int] = {
        "search": 3,
        "read_file": 3,
        "find_symbol": 2,
        "list_dir": 2,
        "get_symbols": 2,
        "get_file_dependencies": 2,
        "get_symbol_dependencies": 2,
        "git_log": 3,
        "git_blame": 2,
        "search_logs": 2,
        "get_recent_failures": 2,
    }
    if cls_loop < 0.4:
        _ACTION_SOFT_LIMITS["list_dir"] = max(_ACTION_SOFT_LIMITS["list_dir"], 4)
        _ACTION_SOFT_LIMITS["search"] = max(_ACTION_SOFT_LIMITS["search"], 5)
    action_limits: dict[str, int] = dict(_ACTION_SOFT_LIMITS)
    if _edit_tool_enabled(state):
        action_limits["apply_search_replace"] = 3
        action_limits["commit_session"] = 3
        action_limits["rollback_session"] = 2
        action_limits["begin_session"] = 0
    action_counts: dict[str, int] = defaultdict(int)

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
    carry_fs_raw = state.pop("carryover_file_scores", None)
    carry_fs = dict(carry_fs_raw) if isinstance(carry_fs_raw, dict) else {}
    for k, v in carry_fs.items():
        try:
            file_scores[_norm_rel_path(str(k))] = float(v) * 0.88
        except (TypeError, ValueError):
            pass

    for k, v in (state.get("file_scores") or {}).items():
        try:
            p = _norm_rel_path(str(k))
            file_scores[p] = max(file_scores[p], float(v))
        except (TypeError, ValueError):
            pass

    carry_tr = list(state.get("carryover_tool_results") or [])
    state.pop("carryover_tool_results", None)
    prior_tools = list(state.get("tool_results") or [])
    if carry_tr:
        state["tool_results"] = list(carry_tr)
    else:
        state["tool_results"] = []

    if _edit_tool_enabled(state) and getattr(
        settings, "REACT_SUBGOAL_EDIT_TRANSACTION", True
    ):
        await registry.execute_tool("file_edit", "rollback_session")
        await registry.execute_tool("file_edit", "begin_session")
        fe0 = registry.get_tool("file_edit")
        if fe0 is not None and hasattr(fe0, "set_dry_run"):
            fe0.set_dry_run(
                bool(state.get("react_edit_dry_run"))
                or bool(getattr(settings, "REACT_EDIT_DRY_RUN", False))
            )

    sess_ev: dict[str, Any] = {
        "type": "react_session_start",
        "intent": intent,
        "max_steps": max_steps,
    }
    if state.get("subgoal_id"):
        sess_ev["subgoal_id"] = state["subgoal_id"]
        sess_ev["subgoal_description"] = (state.get("subgoal_description") or "")[:200]
    await _emit_react(state, sess_ev)

    last_loop_i = -1
    for i in range(max_steps):
        last_loop_i = i
        cancel_ev = state.get("react_cancel_event")
        if cancel_ev is not None and getattr(cancel_ev, "is_set", lambda: False)():
            logger.info("ReasoningLoop: cancel requested at step %s", i)
            await _emit_react(state, {"type": "react_cancelled", "step": i})
            break

        vo = valid_observations(state.get("react_observations") or [])
        obs_summary = "\n".join(str(x)[:500] for x in vo[-4:])
        summary_note = ""
        if (state.get("react_summary") or "").strip():
            summary_note = f"Earlier summary: {(state.get('react_summary') or '')[:400]}\n"

        edit_extra = ""
        if (state.get("mode") or "").strip().lower() == "edit":
            edit_extra = (
                "Edit mode: after you have a candidate file, call get_file_dependencies "
                "or get_symbol_dependencies, then read_file on any related paths you need "
                "before signaling done.\n"
            )

        hint_extra = ""
        hf = state.get("hint_files") or []
        hs = state.get("hint_symbols") or []
        if hf:
            hint_extra += f"Shared hint files (prior subgoals): {hf[:6]}\n"
        if hs:
            hint_extra += f"Shared hint symbols: {hs[:8]}\n"
        pi = state.get("prior_insights") or []
        if pi:
            tail = pi[-3:]
            hint_extra += f"Prior subgoal findings (compact): {json.dumps(tail, default=str)[:900]}\n"
        lb = state.get("learned_file_bias") or []
        if lb:
            hint_extra += (
                "Historical edit bias (consider read_file on these): "
                f"{lb[:8]}\n"
            )
        csym = state.get("carryover_symbols") or []
        if csym:
            hint_extra += (
                "Carryover symbols (from parent exploration): "
                f"{json.dumps(csym[:10])}\n"
            )
        cins = state.get("carryover_insights") or []
        if cins:
            hint_extra += (
                "Carryover insights (parent): "
                f"{json.dumps(cins, default=str)[:900]}\n"
            )
        aws = state.get("active_working_set") or []
        if aws:
            hint_extra += (
                "Active working set (recent files + dep neighbors): "
                f"{json.dumps(aws[:12])}\n"
            )
        synh = state.get("synthesis_hints") or []
        if synh:
            hint_extra += (
                "Cross-subgoal gaps to resolve (prioritize): "
                f"{json.dumps([str(x)[:200] for x in synh[:8]])}\n"
            )

        user_block = (
            f"User task: {q}\n"
            f"{edit_extra}"
            f"{hint_extra}"
            f"{_exploration_hint(i, max_steps, intent)}\n"
            f"{summary_note}"
            f"Router hint: tool={router!r} confidence={rconf:.2f}\n"
            f"Workspace root: {registry.workspace_root}\n"
            f"Recent observations (truncated):\n{obs_summary or '(none)'}\n"
            f"Step {i + 1} of {max_steps}."
        )
        prompt = _react_system_for(state) + "\n\n" + user_block

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

        await _emit_step_evt(
            state,
            phase="start",
            step=i,
            tool=tool_name,
            action=action,
            status="pending",
            args_preview=json.dumps(args, default=str) if args else None,
        )

        valid_tools = ("file", "git", "log")
        if _edit_tool_enabled(state):
            valid_tools = ("file", "file_edit", "git", "log")
        if tool_name not in valid_tools:
            state["react_observations"].append(
                f"Step {i}: invalid tool {tool_name!r} — stopping."
            )
            state["react_failed_steps"] += 1
            await _emit_step_evt(
                state,
                phase="result",
                step=i,
                tool=tool_name,
                action=action,
                status="fail",
                summary="invalid tool",
                duration_ms=0,
            )
            if state["react_failed_steps"] >= 2:
                _maybe_force_rag_escape(state)
                break
            continue

        if tool_name == "file_edit" and action == "apply_search_replace":
            fp_chk = args.get("file_path") or args.get("path") or ""
            allow = _edit_allow_paths(state)
            if fp_chk and not _path_allowed_for_edit(str(fp_chk), allow):
                state["react_observations"].append(
                    f"Step {i}: file_edit rejected — read_file {fp_chk!r} first "
                    "(path not in allowed working set)."
                )
                state["react_failed_steps"] += 1
                await _emit_step_evt(
                    state,
                    phase="result",
                    step=i,
                    tool=tool_name,
                    action=action,
                    status="fail",
                    summary="edit path not allowed",
                    duration_ms=0,
                )
                if state["react_failed_steps"] >= 2:
                    _maybe_force_rag_escape(state)
                    break
                continue

        act_limit = action_limits.get(action, 2)
        if action_counts[action] >= act_limit:
            state["react_observations"].append(
                f"Step {i}: repetition guard — {action!r} hit limit ({act_limit}); use another action."
            )
            state["react_failed_steps"] += 1
            await _emit_step_evt(
                state,
                phase="result",
                step=i,
                tool=tool_name,
                action=action,
                status="fail",
                summary="repetition guard",
                duration_ms=0,
            )
            if state["react_failed_steps"] >= 2:
                _maybe_force_rag_escape(state)
                break
            continue
        action_counts[action] += 1

        res: ToolResult | None
        t0 = time.perf_counter()
        try:
            res = await registry.execute_tool(tool_name, action, **args)
        except Exception as exc:
            res = None
            state["react_observations"].append(f"Step {i}: tool error {exc}")
            state["react_failed_steps"] += 1
            await _emit_step_evt(
                state,
                phase="result",
                step=i,
                tool=tool_name,
                action=action,
                status="fail",
                summary=str(exc)[:160],
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            if state["react_failed_steps"] >= 2:
                _maybe_force_rag_escape(state)
                logger.info(
                    "ReasoningLoop: stopping after %s failed steps",
                    state["react_failed_steps"],
                )
                break
            continue

        useful = _tool_step_useful(res, q)
        if res and res.success and tool_name == "file_edit":
            if action in ("apply_search_replace", "commit_session"):
                useful = True
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

        dur_ms = int((time.perf_counter() - t0) * 1000)
        await _emit_step_evt(
            state,
            phase="result",
            step=i,
            tool=tool_name,
            action=action,
            status="success" if (res and res.success) else "fail",
            summary=_step_result_summary(res),
            duration_ms=dur_ms,
            useful=useful,
        )

        if res and res.data and isinstance(res.data, dict):
            _bump_file_scores(
                file_scores, last_touch_step, i, tool_name, action, args, res.data
            )

        if res and res.success and res.data and isinstance(res.data, dict):
            delta = goal_progress_delta(
                q,
                str(state.get("react_goal") or ""),
                res.data,
                action=(res.action or "").strip() or None,
            )
            if delta > 0:
                state["goal_progress"] = min(
                    1.0, float(state.get("goal_progress") or 0) + delta
                )
        if (
            res
            and res.success
            and tool_name == "file_edit"
            and action == "apply_search_replace"
        ):
            state["goal_progress"] = min(
                1.0, float(state.get("goal_progress") or 0) + 0.12
            )

        await _emit_react(
            state,
            {
                "type": "state",
                "subgoal": state.get("subgoal_id"),
                "step": i,
                "goal_progress": float(state.get("goal_progress") or 0),
                "files_read_count": len(state.get("files_read") or []),
                "useful_steps": int(state.get("react_useful_steps") or 0),
            },
        )

        if tool_name == "file" and action == "read_file":
            p = args.get("path") or args.get("file_path")
            if p and p not in state["files_read"]:
                state["files_read"].append(str(p))
        if tool_name == "file_edit" and action == "apply_search_replace":
            p = args.get("file_path") or args.get("path")
            if p and str(p) not in state["files_read"]:
                state["files_read"].append(str(p).replace("\\", "/"))
        if tool_name == "file" and action == "find_symbol":
            for m in (res.data or {}).get("matches") or [] if res and res.data else []:
                sym = m.get("symbol")
                if sym and sym not in state["symbols_found"]:
                    state["symbols_found"].append(sym)

        if (
            compute_answer_confidence(state) >= 0.78
            and int(state.get("react_useful_steps") or 0) >= 1
            and cls_loop >= 0.45
            and len(state.get("files_read") or []) >= 2
        ):
            logger.info(
                "ReasoningLoop: high merged confidence — ending exploration at step %s",
                i,
            )
            await _emit_react(
                state,
                {"type": "react_high_confidence_stop", "step": i},
            )
            break

        if state["react_failed_steps"] >= 2:
            _maybe_force_rag_escape(state)
            logger.info(
                "ReasoningLoop: stopping after %s low-value/failed steps",
                state["react_failed_steps"],
            )
            break

        if i >= 2 and state["react_useful_steps"] == 0:
            cls_esc = float(
                (state.get("classification") or {}).get("confidence", 1.0)
                or 1.0
            )
            if (
                float(state.get("goal_progress") or 0) < 0.3
                and cls_esc < 0.4
            ):
                state["react_force_rag"] = True
                logger.info(
                    "ReasoningLoop: no useful grounding through step %s → force RAG",
                    i,
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
