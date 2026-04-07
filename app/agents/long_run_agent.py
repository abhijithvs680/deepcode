"""
Macro loop: decompose a high-level goal into tasks, run the orchestrator per task (edit mode).

Emits EventBus: cycle_start, task_start, task_complete, goal_complete.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

from app.core.config import get_settings
from app.core.event_bus import EventBus
from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def _persist_long_run_state(
    state: dict[str, Any],
    *,
    persist_dir: str,
    repo_id: str | None,
    log_entry: dict[str, Any],
) -> None:
    base = (persist_dir or ".deepcode/long_run").strip() or ".deepcode/long_run"
    rid = str(state.setdefault("long_run_run_id", str(uuid.uuid4())))
    root = os.path.join(_workspace_root_for_persist(repo_id), base, rid)
    try:
        os.makedirs(root, exist_ok=True)
        tg = os.path.join(root, "task_graph.json")
        art = os.path.join(root, "artifacts.json")
        logf = os.path.join(root, "execution_log.jsonl")
        with open(tg, "w", encoding="utf-8") as f:
            json.dump(
                state.get("task_graph") or state.get("goal_plan") or [],
                f,
                indent=2,
                default=str,
            )
        with open(art, "w", encoding="utf-8") as f:
            json.dump(state.get("artifacts") or {}, f, indent=2, default=str)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
    except OSError as exc:
        logger.debug("long_run persist skipped: %s", exc)


def _workspace_root_for_persist(repo_id: str | None) -> str:
    if repo_id:
        try:
            from app.core.repo_registry import get_repo_entry

            ent = get_repo_entry(repo_id)
            if ent and ent.get("path"):
                return str(ent["path"])
        except Exception:
            pass
    return os.getcwd()


_MACRO_SYSTEM = """You break a software engineering goal into 3–8 ordered tasks for a coding agent.
Output ONE JSON object only (no markdown).
Schema:
{"tasks": [{"id": number, "description": "short imperative, one line", "depends_on": [optional prior task ids]}]}
Rules:
- Tasks must be concrete (files, APIs, UI, wiring).
- Order matters: earlier tasks should not depend on later ones.
- Use the user's wording; do not invent repo-specific paths unless the user gave them."""


def _parse_tasks_json(text: str) -> list[dict[str, Any]] | None:
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
        obj = json.loads(raw[i : j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    tasks = obj.get("tasks")
    if not isinstance(tasks, list):
        return None
    out: list[dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        desc = str(t.get("description", "")).strip()
        if not desc:
            continue
        try:
            nid = int(tid) if tid is not None else len(out) + 1
        except (TypeError, ValueError):
            nid = len(out) + 1
        raw_deps = t.get("depends_on") or t.get("dependencies") or []
        deps: list[int] = []
        if isinstance(raw_deps, list):
            for d in raw_deps:
                try:
                    deps.append(int(d))
                except (TypeError, ValueError):
                    continue
        out.append(
            {
                "id": nid,
                "description": desc,
                "status": "pending",
                "depends_on": deps,
                "outputs": {},
                "retries": 0,
                "attempts": 0,
            }
        )
    return out or None


async def decompose_macro_goal(goal: str, provider: LLMProvider) -> list[dict[str, Any]]:
    g = (goal or "").strip()
    if not g:
        return []
    try:
        raw = await provider.generate(
            g[:2000],
            system_prompt=_MACRO_SYSTEM,
            temperature=0.25,
        )
    except Exception as exc:
        logger.warning("macro decompose LLM failed: %s", exc)
        return [
            {
                "id": 1,
                "description": g,
                "status": "pending",
                "depends_on": [],
                "outputs": {},
                "retries": 0,
                "attempts": 0,
            }
        ]
    parsed = _parse_tasks_json(raw)
    if parsed:
        return parsed
    return [
        {
            "id": 1,
            "description": g,
            "status": "pending",
            "depends_on": [],
            "outputs": {},
            "retries": 0,
            "attempts": 0,
        }
    ]


def _task_dependencies_satisfied(task: dict[str, Any], plan: list[dict[str, Any]]) -> bool:
    deps = task.get("depends_on") or []
    if not deps:
        return True
    for d in deps:
        ok = False
        for o in plan:
            if not isinstance(o, dict):
                continue
            if o.get("id") != d:
                continue
            if o.get("status") == "done":
                ok = True
                break
        if not ok:
            return False
    return True


def pick_next_task(state: dict[str, Any]) -> dict[str, Any] | None:
    plan = [t for t in (state.get("goal_plan") or []) if isinstance(t, dict)]
    for t in plan:
        if t.get("status") not in ("pending", None, ""):
            continue
        if not _task_dependencies_satisfied(t, plan):
            continue
        return t
    return None


def _completed_lines(plan: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for t in plan:
        if not isinstance(t, dict):
            continue
        st = t.get("status")
        if st == "done":
            parts.append(f"- [done] {t.get('description', '')[:200]}")
        elif st == "skipped":
            parts.append(f"- [skipped] {t.get('description', '')[:200]}")
    return "\n".join(parts[-12:])


def build_long_run_context(state: dict[str, Any], task: dict[str, Any]) -> str:
    goal = state.get("long_run_goal") or ""
    prev = _completed_lines(list(state.get("goal_plan") or []))
    art = state.get("artifacts") or {}
    art_line = ""
    if isinstance(art, dict) and art:
        sample = list(art.items())[:10]
        art_line = "Known artifacts (path → source task): " + json.dumps(
            {k: v for k, v in sample}, default=str
        )[:800] + "\n"
    recovery = ""
    if task.get("strategy") == "explore_first":
        recovery = (
            "Recovery mode: last attempt was weak — prioritize search, read_file, "
            "and dependency tools before applying edits.\n"
        )
    return (
        f"Macro goal: {goal}\n"
        f"Cycle: {state.get('cycle', 0)}\n"
        f"{art_line}"
        f"Completed tasks so far:\n{prev or '(none yet)'}\n"
        f"{recovery}"
        f"Current task (focus): {task.get('description', '')}\n"
    )


def is_task_complete(orch_result: dict[str, Any]) -> bool:
    meta = orch_result.get("meta") or {}
    files = meta.get("files_touched") or []
    syms = meta.get("symbols_used") or []
    conf = float(meta.get("answer_confidence") or 0.0)
    return bool(files or syms or conf > 0.7)


def should_replan(state: dict[str, Any]) -> bool:
    return int(state.get("consecutive_task_misses") or 0) >= 2


async def replan_goal(
    state: dict[str, Any],
    provider: LLMProvider,
) -> list[dict[str, Any]]:
    g = state.get("long_run_goal") or ""
    tail = "\n".join(
        str(x.get("summary", ""))[:400]
        for x in (state.get("cycle_summaries") or [])[-4:]
        if isinstance(x, dict)
    )
    prompt = (
        f"Original goal: {g}\n"
        f"Recent outcomes (compact):\n{tail or '(none)'}\n"
        "Produce a revised task list (same JSON schema as before)."
    )
    try:
        raw = await provider.generate(
            prompt[:4000],
            system_prompt=_MACRO_SYSTEM,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("macro replan failed: %s", exc)
        return list(state.get("goal_plan") or [])
    parsed = _parse_tasks_json(raw)
    if parsed:
        return parsed
    return list(state.get("goal_plan") or [])


def update_goal_progress(
    state: dict[str, Any],
    task: dict[str, Any],
    result: dict[str, Any],
) -> None:
    ok = is_task_complete(result)
    meta = result.get("meta") or {}
    conf = float(meta.get("answer_confidence") or 0.0)
    state["last_task_confidence"] = conf
    touched = list(meta.get("files_touched") or [])
    if ok:
        task["status"] = "done"
        state["consecutive_task_misses"] = 0
        outs = dict(task.get("outputs") or {})
        for f in touched:
            if f:
                fp = str(f).replace("\\", "/")
                outs[fp] = "touched"
        task["outputs"] = outs
    else:
        attempts = int(task.get("attempts") or 0) + 1
        task["attempts"] = attempts
        task["retries"] = attempts
        if attempts == 1:
            task["strategy"] = "explore_first"
        if attempts >= 2:
            task["status"] = "skipped"
        state["consecutive_task_misses"] = (
            int(state.get("consecutive_task_misses") or 0) + 1
        )
    art = dict(state.get("artifacts") or {})
    for f in touched:
        if f:
            fn = str(f).replace("\\", "/")
            art[fn] = task.get("id")
    state["artifacts"] = art
    summaries = list(state.get("cycle_summaries") or [])
    summaries.append(
        {
            "task": str(task.get("description", ""))[:300],
            "summary": (result.get("answer") or "")[:1200],
            "confidence": conf,
            "marked_done": ok,
        }
    )
    state["cycle_summaries"] = summaries


def is_goal_complete(state: dict[str, Any]) -> bool:
    plan = state.get("goal_plan") or []
    for t in plan:
        if not isinstance(t, dict):
            continue
        st = t.get("status")
        if st in ("pending", None, ""):
            return False
    return len(plan) > 0


async def run_long_goal(
    goal: str,
    orchestrator: Any,
    *,
    provider: LLMProvider,
    repo_id: str | None = None,
    extra_context: str | None = None,
    **orch_kwargs: Any,
) -> dict[str, Any]:
    settings = get_settings()
    if not getattr(settings, "LONG_RUN_ENABLED", True):
        return await orchestrator.run(
            question=goal,
            repo_id=repo_id,
            mode="edit",
            **orch_kwargs,
        )

    plan = await decompose_macro_goal(goal, provider)
    run_uid = str(uuid.uuid4())
    state: dict[str, Any] = {
        "long_run_run_id": run_uid,
        "long_run_goal": goal,
        "goal_status": "in_progress",
        "cycle": 0,
        "goal_plan": plan,
        "task_graph": plan,
        "consecutive_task_misses": 0,
        "last_task_confidence": 0.0,
        "cycle_summaries": [],
        "artifacts": {},
        "total_tool_steps": 0,
    }
    max_cycles = int(getattr(settings, "LONG_RUN_MAX_CYCLES", 20) or 20)
    max_cycles = max(1, min(max_cycles, 50))
    max_rt = int(getattr(settings, "LONG_RUN_MAX_RUNTIME_SECONDS", 0) or 0)
    max_total_steps = int(getattr(settings, "LONG_RUN_MAX_TOTAL_TOOL_STEPS", 0) or 0)
    t_start = time.monotonic()

    results: list[dict[str, Any]] = []
    combined_answer_parts: list[str] = []

    def _workspace_root() -> str:
        if repo_id:
            try:
                from app.core.repo_registry import get_repo_entry

                ent = get_repo_entry(repo_id)
                if ent and ent.get("path"):
                    return str(ent["path"])
            except Exception:
                pass
        return os.getcwd()

    while state["cycle"] < max_cycles:
        state["cycle"] += 1
        if max_rt > 0 and (time.monotonic() - t_start) > max_rt:
            state["goal_status"] = "stopped_budget_time"
            await EventBus.emit(
                {
                    "type": "goal_complete",
                    "goal": (goal or "")[:200],
                    "reason": "budget_time",
                }
            )
            break
        await EventBus.emit(
            {
                "type": "cycle_start",
                "cycle": state["cycle"],
                "goal": (goal or "")[:200],
            }
        )

        task = pick_next_task(state)
        if not task:
            state["goal_status"] = "completed"
            await EventBus.emit(
                {
                    "type": "goal_complete",
                    "goal": (goal or "")[:200],
                    "reason": "no_remaining_tasks",
                }
            )
            break

        await EventBus.emit(
            {
                "type": "task_start",
                "cycle": state["cycle"],
                "task_id": task.get("id"),
                "description": str(task.get("description", ""))[:240],
            }
        )

        ctx = build_long_run_context(state, task)
        if extra_context and str(extra_context).strip():
            ctx = f"{str(extra_context).strip()}\n\n{ctx}"
        res = await orchestrator.run(
            question=str(task.get("description", "")),
            context=ctx,
            repo_id=repo_id,
            mode="edit",
            **orch_kwargs,
        )
        results.append(res)
        ans = (res.get("answer") or "").strip()
        if ans:
            combined_answer_parts.append(
                f"### Task {task.get('id')}: {task.get('description', '')[:80]}\n{ans}"
            )

        update_goal_progress(state, task, res)
        meta = res.get("meta") or {}
        st_add = int(meta.get("tool_steps_count") or 0)
        state["total_tool_steps"] = int(state.get("total_tool_steps") or 0) + st_add

        await EventBus.emit(
            {
                "type": "task_complete",
                "cycle": state["cycle"],
                "task_id": task.get("id"),
                "done": task.get("status") == "done",
                "confidence": state.get("last_task_confidence"),
            }
        )

        if max_total_steps > 0 and state["total_tool_steps"] > max_total_steps:
            state["goal_status"] = "stopped_budget_steps"
            await EventBus.emit(
                {
                    "type": "goal_complete",
                    "goal": (goal or "")[:200],
                    "reason": "budget_steps",
                }
            )
            break

        if getattr(settings, "LONG_RUN_POST_TASK_VERIFY", False):
            try:
                from app.tools.test_runner_tool import TestRunnerTool

                v = TestRunnerTool(_workspace_root()).execute_tests()
                state["last_verification"] = {
                    "success": bool(v.success),
                    "summary": (v.error or str((v.data or {}).get("message", "")))[
                        :500
                    ],
                }
            except Exception as exc:
                state["last_verification"] = {"success": False, "summary": str(exc)[:300]}

        if getattr(settings, "LONG_RUN_PERSIST_ENABLED", True):
            pd = str(
                getattr(settings, "LONG_RUN_PERSIST_DIR", "") or ".deepcode/runs"
            ).strip()
            if pd:
                _persist_long_run_state(
                    state,
                    persist_dir=pd,
                    repo_id=repo_id,
                    log_entry={
                        "ts": time.time(),
                        "cycle": state.get("cycle"),
                        "task_id": task.get("id"),
                        "task_status": task.get("status"),
                        "confidence": state.get("last_task_confidence"),
                        "goal_status": state.get("goal_status"),
                    },
                )

        if is_goal_complete(state):
            state["goal_status"] = "completed"
            await EventBus.emit(
                {
                    "type": "goal_complete",
                    "goal": (goal or "")[:200],
                    "reason": "all_tasks_done",
                }
            )
            break

        if (
            float(meta.get("answer_confidence") or 0) > 0.85
            and pick_next_task(state) is None
        ):
            state["goal_status"] = "completed"
            await EventBus.emit(
                {
                    "type": "goal_complete",
                    "goal": (goal or "")[:200],
                    "reason": "high_confidence_no_pending",
                }
            )
            break

        if should_replan(state):
            state["goal_plan"] = await replan_goal(state, provider)
            state["task_graph"] = list(state.get("goal_plan") or [])
            state["consecutive_task_misses"] = 0

    if state["goal_status"] not in (
        "completed",
        "stopped_budget_time",
        "stopped_budget_steps",
    ):
        state["goal_status"] = "stopped_max_cycles"

    return {
        "answer": "\n\n".join(combined_answer_parts)
        or (results[-1].get("answer") if results else ""),
        "long_run_state": state,
        "cycle_results": results,
        "meta": {
            "long_run": True,
            "goal_status": state.get("goal_status"),
            "cycles": state.get("cycle"),
            "goal_plan": state.get("goal_plan"),
            "task_graph": state.get("task_graph"),
            "artifacts": state.get("artifacts"),
            "total_tool_steps": state.get("total_tool_steps"),
            "last_task_confidence": state.get("last_task_confidence"),
            "last_verification": state.get("last_verification"),
            "long_run_run_id": state.get("long_run_run_id"),
            "persist_dir": getattr(settings, "LONG_RUN_PERSIST_DIR", ""),
        },
    }
