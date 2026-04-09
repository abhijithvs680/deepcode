"""
Minimal ReAct-style loop: THINK → ACT (tool) → OBSERVE → repeat → hand off to AnswerAgent.

RAG is not used here; orchestrator runs retrieval only if `enough_context` is False
or `react_force_rag` is set (no-progress escape or broad intent).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.agents.hierarchical_planner import is_create_intent
from app.agents.react_grounding import (
    compute_answer_confidence,
    extract_goal,
    goal_progress_delta,
    infer_scaffold_shell_command,
    keyword_relevance,
    looks_like_task_execution_query,
    scaffold_shell_hint,
)
from app.core.config import get_settings
from app.core.event_bus import EventBus
from app.providers.base import LLMProvider
from app.tools.base_tool import ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _minimal_fallback_file_body(goal: str) -> str:
    g = (goal or "").lower()
    if "fastapi" in g:
        return (
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n\n"
            '@app.get("/health")\n'
            "def health():\n"
            '    return {"status": "ok"}\n'
        )
    return (
        "# deepcode fallback — replace with your code\n"
        "def main() -> None:\n"
        '    print("hello")\n\n\n'
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _react_or_node_scaffold_intent(goal: str) -> bool:
    """True when the task is clearly JS/frontend/Node — never use Python main.py fallback."""
    g = (goal or "").lower()
    if any(
        tok in g
        for tok in (
            "react",
            "next.js",
            "nextjs",
            "vite",
            "vue",
            "angular",
            "svelte",
            "frontend",
            "create-react-app",
            "create react app",
            "webpack",
            "tailwind",
            "javascript app",
            "typescript app",
            "spa ",
        )
    ):
        return True
    if re.search(r"\b(npm|npx|yarn|pnpm|node)\b", g):
        return True
    return False


def _goal_triggers_create_fallback(goal: str) -> bool:
    """
    True only for Python/backend create rescue (main.py / stub) — not React/npm/Vite.
    """
    if _react_or_node_scaffold_intent(goal):
        return False
    g = (goal or "").lower()
    if any(
        k in g
        for k in (
            "fastapi",
            "flask",
            "django",
            "starlette",
            "uvicorn",
            "health check",
            "healthcheck",
            "openapi",
        )
    ):
        return True
    if any(
        k in g
        for k in (
            "rest api",
            "restful api",
            "endpoint",
            "api route",
            "graphql",
        )
    ):
        return True
    if ".py" in g or "python file" in g or "python app" in g or "python api" in g:
        return True
    if is_create_intent(goal):
        if any(
            k in g
            for k in (
                "python",
                "pip ",
                "pip install",
                "virtualenv",
                "venv",
                "pytest",
                "backend",
            )
        ):
            return True
        return False
    if "make main" in g or "make the main" in g:
        return True
    if "scaffold" in g and any(
        x in g for x in ("python", "fastapi", "flask", "django", "pip", "uvicorn")
    ):
        return True
    return False


def _scaffold_shell_requires_attempt_before_py_fallback(q: str) -> bool:
    """If set, do not run Python file fallback until shell.run_command appears in react_steps."""
    return bool(scaffold_shell_hint(str(q or "")).strip())


def _react_steps_include_shell_run(state: dict[str, Any]) -> bool:
    for row in state.get("react_steps") or []:
        if not isinstance(row, dict):
            continue
        if (
            str(row.get("tool") or "").lower() == "shell"
            and str(row.get("action") or "").lower() == "run_command"
        ):
            return True
    return False


def _pick_fallback_write_path(goal: str, workspace_root: str) -> tuple[str, str]:
    """Deterministic entry file for create fallback (avoids deepcode_fallback_*.py drift)."""
    _ = workspace_root  # reserved for future repo-type routing
    body = _minimal_fallback_file_body(goal)
    return "main.py", body


def _file_edit_replace_uses_subprocess_pip_install(replace: str) -> bool:
    """Block Python files that shell out to pip instead of using shell.run_command."""
    r = (replace or "").lower()
    return "subprocess" in r and "pip install" in r


def _direct_write_fallback_file(
    workspace_root: str, rel: str, content: str
) -> tuple[bool, str]:
    """
    Write fallback content straight to disk (bypasses file_edit session / commit).
    Returns (ok, error_message).
    """
    if not (workspace_root or "").strip():
        return False, "empty workspace_root"
    rel_n = _norm_rel_path(rel.replace("\\", "/").strip())
    if not rel_n or rel_n.startswith("../"):
        return False, f"invalid rel {rel!r}"
    try:
        root = Path(workspace_root).resolve()
    except OSError as e:
        return False, str(e)
    try:
        dest = (root / rel_n).resolve()
        dest.relative_to(root)
    except (OSError, ValueError) as e:
        return False, str(e)
    logger.info(
        "[WRITE] direct_fallback abs=%s exists_before=%s",
        dest,
        dest.is_file(),
    )
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    except OSError as e:
        logger.warning("[WRITE] direct_fallback failed abs=%s err=%s", dest, e)
        return False, str(e)
    try:
        sz = dest.stat().st_size
    except OSError:
        sz = -1
    logger.info("[WRITE DONE] direct_fallback abs=%s size=%s", dest, sz)
    return True, ""


_EXT = r"(?:py|pyw|js|mjs|cjs|ts|tsx|jsx|java|kt|kts|rs|go|rb|php|cs|fs|cpp|cc|c|h|hpp|vue)"
_ERR_PATH_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r'File "([^"]+)"', re.IGNORECASE),
    re.compile(
        rf"\bat\s+(\S+?\.{_EXT})(?::\d+){{1,3}}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<![\w/])([/.][^\s:|<>\"]+\.{_EXT})(?::\d+)?\b",
        re.IGNORECASE,
    ),
)


def _try_repo_relative(path_str: str, workspace_root: str) -> str | None:
    if not path_str or not workspace_root:
        return None
    try:
        p = Path(str(path_str).strip()).expanduser()
        root = Path(workspace_root).resolve()
        if not p.is_absolute():
            return _norm_rel_path(str(p))
        pr = p.resolve()
        rel = pr.relative_to(root)
        return str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return None


def _scrape_error_paths_from_observation(ob: str) -> set[str]:
    if not ob or len(ob) > 500_000:
        return set()
    ol = (ob or "").lower()
    if not any(
        x in ol
        for x in (
            "traceback",
            "error",
            "syntaxerror",
            "exception",
            "failed",
            "fail",
            "stack",
            "errno",
        )
    ) and 'file "' not in ol:
        return set()
    out: set[str] = set()
    for rx in _ERR_PATH_RES:
        for m in rx.finditer(ob):
            p = m.group(1).replace("\\", "/").strip()
            if p and 2 < len(p) < 400 and not p.startswith("http"):
                out.add(p)
    return out


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


async def _maybe_force_rag_escape_emit(state: dict[str, Any]) -> None:
    before = bool(state.get("react_force_rag"))
    _maybe_force_rag_escape(state)
    if bool(state.get("react_force_rag")) and not before:
        await EventBus.emit(
            {
                "type": "rag_escalation",
                "reason": "no_progress_after_tools",
                "subgoal_id": state.get("subgoal_id"),
            }
        )


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


def _react_evidence_unlocks_execution(state: dict[str, Any], settings: Any) -> bool:
    """
    Hybrid gating: confidence or goal_progress alone cannot unlock — require at least
    one file read with those signals, or strong file/symbol counts.
    Negative gate: very weak grounding cannot unlock until enough distinct reads.
    """
    thr_f = int(getattr(settings, "REACT_EXEC_UNLOCK_MIN_FILES_READ", 2) or 2)
    thr_s = int(getattr(settings, "REACT_EXEC_UNLOCK_MIN_SYMBOLS", 2) or 2)
    conf_thr = float(
        getattr(settings, "REACT_EXEC_UNLOCK_MIN_CONFIDENCE", 0.5) or 0.5
    )
    gp_thr = float(
        getattr(settings, "REACT_EXEC_UNLOCK_MIN_GOAL_PROGRESS", 0.35) or 0.35
    )
    weak_min = int(
        getattr(settings, "REACT_EXEC_WEAK_GROUNDING_MIN_FILES", 3) or 3
    )
    weak_min = max(2, weak_min)
    thr_f = max(1, thr_f)
    thr_s = max(1, thr_s)
    nfr = len(state.get("files_read") or [])
    nsy = len(state.get("symbols_found") or [])
    ac = compute_answer_confidence(state)
    gp = float(state.get("goal_progress") or 0)
    weak_grounding = nfr < 2 and nsy == 0 and ac < 0.5
    if weak_grounding and nfr < weak_min:
        return False
    return (
        nfr >= thr_f
        or nsy >= thr_s
        or (ac >= conf_thr and nfr >= 1)
        or (gp >= gp_thr and nfr >= 1)
    )


_PROBE_ENTRY_BASENAMES: frozenset[str] = frozenset(
    {
        "main.py",
        "app.py",
        "manage.py",
        "index.js",
        "index.ts",
        "index.tsx",
        "index.jsx",
        "server.js",
        "server.ts",
        "app.tsx",
        "app.jsx",
        "main.ts",
        "main.tsx",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "cargo.toml",
        "go.mod",
    }
)


def _probe_path_structure_boost(path: str) -> float:
    """Cheap structural prior on top of lexical relevance."""
    pl = _norm_rel_path(path).lower()
    base = pl.rsplit("/", 1)[-1] if pl else ""
    boost = 0.0
    if base in _PROBE_ENTRY_BASENAMES:
        boost += 3.0
    for frag, w in (
        ("/config", 2.0),
        ("config/", 2.0),
        ("settings", 1.5),
        ("/auth", 2.0),
        ("auth/", 2.0),
        ("middleware", 1.2),
        ("routes/", 1.5),
        ("router", 1.2),
        ("api/", 1.5),
        ("src/", 0.8),
    ):
        if frag in pl:
            boost += w
    return boost


def _rank_probe_search_paths(paths: list[str], query: str) -> list[str]:
    """Order search hits by relevance to the user task (path + filename vs query)."""
    q = (query or "").strip()
    uniq: list[str] = []
    seen: set[str] = set()
    for p in paths:
        n = _norm_rel_path(p).strip()
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    scored: list[tuple[float, int, str]] = []
    for idx, p in enumerate(uniq):
        base = p.split("/")[-1] if p else ""
        rel = keyword_relevance(q, p, base.replace("_", " ")) if q else 0.0
        rel += _probe_path_structure_boost(p)
        scored.append((rel, idx, p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored]


def _workspace_sparse_for_scaffold(workspace_root: str, settings: Any) -> bool:
    """True when the repo root looks empty enough to scaffold without list_dir first."""
    if not workspace_root or not os.path.isdir(workspace_root):
        return False
    cap = int(
        getattr(settings, "REACT_SCAFFOLD_SPARSE_DIR_MAX_ENTRIES", 4) or 4
    )
    cap = max(0, min(cap, 32))
    skip = frozenset(
        {
            ".git",
            ".deepcode",
            ".cursor",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".idea",
            ".vscode",
        }
    )
    try:
        names = os.listdir(workspace_root)
    except OSError:
        return False
    n = 0
    for name in names:
        if name.startswith("."):
            continue
        if name in skip:
            continue
        n += 1
        if n > cap:
            return False
    return n <= cap


def _shell_failure_recovery_hints(entry: dict[str, Any]) -> str:
    """Actionable next steps from the last shell failure (planner + observations)."""
    blob = " ".join(
        [
            str(entry.get("error") or ""),
            str(entry.get("stderr_tail") or ""),
            str(entry.get("stdout_tail") or ""),
            str(entry.get("command") or ""),
        ]
    ).lower()
    hints: list[str] = []
    if any(
        x in blob
        for x in (
            "not recognized",
            "not found",
            "is not recognized",
            "'npx'",
            "`npx`",
            "enoent",
        )
    ):
        if "npx" in blob or "npm" in blob or "node" in blob:
            hints.append(
                "Ensure Node.js is installed and on PATH (try `node -v` / `where npx`)."
            )
    if "python" in blob and (
        "not found" in blob or "not recognized" in blob or "bad interpreter" in blob
    ):
        hints.append(
            "Python not on PATH — activate a venv or use the full path to python."
        )
    if "eacces" in blob or "permission denied" in blob:
        hints.append(
            "Permission error — use a subfolder in the workspace, not system directories."
        )
    if "already exists" in blob or "eexist" in blob or "file exists" in blob:
        hints.append(
            "Target folder exists — pick a new project name or remove/rename the directory."
        )
    if "npm err" in blob or ("enoent" in blob and "package.json" in blob):
        hints.append(
            "npm could not read package.json — check cwd or run from the app root."
        )
    if "create-react-app" in blob and (
        "deprecated" in blob or "unsupported" in blob or "error" in blob
    ):
        hints.append(
            "CRA issues: consider `npm create vite@latest <name> -- --template react`."
        )
    if not hints:
        hints.append("Read stderr; fix the command, cwd, or environment, then retry.")
    return " ".join(hints)


def _shell_recovery_prompt_block(state: dict[str, Any]) -> str:
    lsc = state.get("last_shell_commands") or []
    if not lsc or lsc[-1].get("success"):
        return ""
    hint = _shell_failure_recovery_hints(lsc[-1])
    if not hint:
        return ""
    return f"Last shell failed — recovery guidance:\n{hint}\n"


def _shell_retry_pointless(res: ToolResult) -> bool:
    """True when stderr/stdout indicates a bad command — repeating it will not help."""
    parts: list[str] = [str(res.error or "")]
    d = res.data if isinstance(res.data, dict) else {}
    parts.append(str(d.get("stderr") or ""))
    parts.append(str(d.get("stdout") or ""))
    blob = "\n".join(parts).lower()
    needles = (
        "not found",
        "not recognized",
        "is not recognized",
        "unknown option",
        "unknown command",
        "enoent",
        "no such file",
        "cannot find",
        "is not available",
        "bad interpreter",
        "syntax error",
        "unexpected token",
    )
    return any(n in blob for n in needles)


def _files_read_key_set(state: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for x in state.get("files_read") or []:
        n = _norm_rel_path(str(x)).strip().lstrip("./")
        if n:
            out.add(n)
    return out


_REL_OK = 0.22


def _tool_step_useful(res: ToolResult | None, query: str) -> bool:
    if not res:
        return False
    if res.tool == "shell" and (res.action or "").strip() == "run_command":
        d0 = res.data if isinstance(res.data, dict) else {}
        if d0.get("skipped"):
            return False
        return bool(res.data)
    if not res.success or res.data is None:
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
    if res.tool == "shell" and (res.action or "").strip() == "run_command":
        d = res.data if isinstance(res.data, dict) else {}
        ec = d.get("exit_code")
        lo = len(str(d.get("stdout") or ""))
        le = len(str(d.get("stderr") or ""))
        bits = f"exit={ec} stdout={lo}b stderr={le}b"
        if not res.success:
            err = (res.error or "failed")[:80]
            return f"{bits} ({err})"
        return bits
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
            add_path(str(fp), 4.0)
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
    for p in state.get("_react_fallback_allowed_edit_paths") or []:
        s.add(str(p).replace("\\", "/"))
    return s


def _path_allowed_for_edit(rel: str, allow: set[str]) -> bool:
    fn = str(rel).replace("\\", "/").strip()
    if not fn:
        return False
    if fn in allow:
        return True
    return any(fn.startswith(a + "/") for a in allow if a)


async def _emit_agent_open_after_fallback_commit(
    state: dict[str, Any],
    registry: ToolRegistry,
    commit_result: Any,
) -> None:
    """Open the fallback target file in IDE after commit (stable path, not list order)."""
    if not state.get("_react_forced_file_edit_fallback_done"):
        return
    if commit_result is None or not getattr(commit_result, "success", False):
        return
    data = getattr(commit_result, "data", None) or {}
    if not isinstance(data, dict):
        return
    ws = str(getattr(registry, "workspace_root", "") or "")
    written = [str(x).replace("\\", "/") for x in (data.get("files_written") or [])]
    wset = set(written)
    target = str(state.get("_react_forced_fallback_commit_rel") or "").replace(
        "\\", "/"
    ).strip()
    if target and target in wset:
        path_open = target
    elif written:
        path_open = written[0]
    elif target:
        path_open = target
    else:
        path_open = ""
    if path_open:
        await EventBus.emit(
            {
                "type": "agent_file_open",
                "path": path_open,
                "workspace": ws,
                "subgoal_id": state.get("subgoal_id"),
            }
        )
    state.pop("_react_forced_fallback_commit_rel", None)


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
        logger.debug(
            "finalize_react_edit_session: skip (no active file_edit session) commit=%s",
            commit,
        )
        return
    stg = get_settings()
    if bool(state.get("react_edit_dry_run")) or getattr(
        stg, "REACT_EDIT_DRY_RUN", False
    ):
        commit = False
    allow = _edit_allow_paths(state)
    logger.debug(
        "finalize_react_edit_session: commit=%s pending=%s allow_paths=%s",
        commit,
        ed.has_pending_changes(),
        len(allow) if allow else 0,
    )
    try:
        if commit:
            if ed.has_pending_changes() and allow:
                cr = await registry.execute_tool(
                    "file_edit",
                    "commit_session",
                    allowed_files=sorted(allow)[:120],
                )
                d = getattr(cr, "data", None) or {}
                logger.debug(
                    "finalize_react_edit_session: commit_session success=%s files_written=%s",
                    getattr(cr, "success", None),
                    (d or {}).get("files_written") if isinstance(d, dict) else None,
                )
                await _emit_agent_open_after_fallback_commit(state, registry, cr)
            else:
                logger.debug(
                    "finalize_react_edit_session: rolling back (no pending or empty allow)"
                )
                await registry.execute_tool("file_edit", "rollback_session")
        elif (
            bool(getattr(stg, "REACT_EDIT_ROLLBACK_TO_CHECKPOINT", True))
            and ed.has_checkpoints()
        ):
            try:
                await registry.execute_tool(
                    "file_edit", "rollback_to_last_checkpoint"
                )
            except Exception:
                await registry.execute_tool("file_edit", "rollback_session")
            else:
                if ed.has_pending_changes() and allow:
                    try:
                        cr = await registry.execute_tool(
                            "file_edit",
                            "commit_session",
                            allowed_files=sorted(allow)[:120],
                        )
                        await _emit_agent_open_after_fallback_commit(
                            state, registry, cr
                        )
                    except Exception as exc2:
                        logger.warning(
                            "finalize_react_edit_session partial commit: %s", exc2
                        )
                        await registry.execute_tool(
                            "file_edit", "rollback_session"
                        )
                else:
                    await registry.execute_tool("file_edit", "rollback_session")
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
  "tool": "file" | "git" | "log" | "shell" | null,
  "action": "read_file" | "search" | "list_dir" | "find_symbol" | "get_symbols" | "get_file_dependencies" | "get_symbol_dependencies" | "run_command" | "done" | "git_log" | "git_blame" | "search_logs" | "get_recent_failures",
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
- shell.run_command: only when the task is structured (navigation/git/debugging-style); args {"command": "single-line shell string", "cwd": "optional relative dir", "title": "optional short label"} — user approval in UI; destructive/write commands show extra warnings.
- git_log: args {} or {"file_path": "optional", "n": 10}
- Set tool null, action "done", done true when further tools are unnecessary.
"""

_REACT_SYSTEM_WITH_EDIT = """You are a coding agent step planner. Reply with ONE JSON object only (no markdown).
Schema:
{
  "thought": "short reasoning",
  "tool": "file" | "file_edit" | "git" | "log" | "shell" | null,
  "action": "read_file" | "search" | "list_dir" | "find_symbol" | "get_symbols" | "get_file_dependencies" | "get_symbol_dependencies" | "apply_search_replace" | "checkpoint_session" | "commit_session" | "rollback_session" | "run_command" | "done" | "git_log" | "git_blame" | "search_logs" | "get_recent_failures",
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
- file_edit.apply_search_replace: args {"file_path":"app/x.py","search":"exact snippet from file","replace":"new text","expected_occurrences":1,"dry_run":false}
  Use dry_run true to preview a diff without changing the working copy; only non-dry edits are persisted on commit.
  Only edit files you already read_file in this subgoal (allowed paths are enforced).
  Edits are batched in memory and committed once at subgoal end if exploration succeeds; mid-subgoal commit_session is optional.
- file_edit.checkpoint_session: args {} — save in-session edit state (used on subgoal failure to keep last good checkpoint).
- file_edit.commit_session: args {"allowed_files":["app/x.py",...]} — force early persist (optional).
- file_edit.rollback_session: args {} — discard uncommitted in-session edits.
- shell.run_command: only for structured tasks; args {"command": "...", "cwd": "optional relative", "title": "optional"} — UI approval; writes/destructive commands are flagged.
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
    sg = state.get("subgoal_id")
    if sg:
        ev["subgoal_id"] = sg
    await _emit_react(state, ev)


async def _finalize_create_main_invariant(
    state: dict[str, Any],
    registry: ToolRegistry,
    query_hint: str,
    step_index: int,
) -> None:
    """
    If edit + create-intent finished without main.py on disk, write it once.
    Catches planner/tool paths that never triggered the in-loop fallback.
    """
    goal_final = str(
        state.get("subgoal_description") or state.get("react_goal") or query_hint or ""
    )
    ws_final = str(getattr(registry, "workspace_root", "") or "").strip()
    if not ws_final:
        return
    if str(state.get("mode") or "").lower() != "edit":
        return
    if not bool(state.get("write_enabled")):
        return
    if bool(state.get("dry_run")):
        return
    if not is_create_intent(goal_final):
        return
    if not _goal_triggers_create_fallback(goal_final):
        return
    if _scaffold_shell_requires_attempt_before_py_fallback(
        query_hint
    ) and not _react_steps_include_shell_run(state):
        return
    try:
        root = Path(ws_final).resolve()
        main_abs = (root / "main.py").resolve()
        main_abs.relative_to(root)
    except (OSError, ValueError):
        return
    if main_abs.is_file():
        return
    body = _minimal_fallback_file_body(goal_final)
    ok, err = _direct_write_fallback_file(ws_final, "main.py", body)
    logger.warning(
        "ReasoningLoop: end-of-session create invariant wrote main.py ok=%s err=%r",
        ok,
        err,
    )
    if not ok:
        return
    state["_react_fallback_allowed_edit_paths"] = ["main.py"]
    state["_react_forced_file_edit_fallback_done"] = True
    state["react_useful_steps"] = int(state.get("react_useful_steps") or 0) + 1
    rel_norm = "main.py"
    if rel_norm not in state["files_read"]:
        state["files_read"].append(rel_norm)
    state["react_observations"].append(
        "⚡ End invariant: wrote main.py (create-intent; was missing on disk)."
    )
    tr = ToolResult(
        tool="file_edit",
        action="direct_write_fallback",
        success=True,
        data={
            "file": rel_norm,
            "files_written": [rel_norm],
            "direct": True,
            "invariant": True,
        },
    )
    state["tool_results"].append(tr.model_dump())
    await EventBus.emit(
        {
            "type": "forced_fallback",
            "path": rel_norm,
            "phase": "final_invariant",
            "subgoal_id": state.get("subgoal_id"),
        }
    )
    await EventBus.emit(
        {
            "type": "file_updated",
            "file": rel_norm,
            "workspace": ws_final,
        }
    )
    await EventBus.emit(
        {
            "type": "agent_file_open",
            "path": rel_norm,
            "workspace": ws_final,
            "subgoal_id": state.get("subgoal_id"),
        }
    )
    await _emit_step_evt(
        state,
        phase="result",
        step=step_index,
        tool="file_edit",
        action="direct_write_fallback",
        status="success",
        summary="final invariant main.py",
        duration_ms=0,
        useful=True,
    )


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
    mode_edit_early = str(state.get("mode") or "").lower() == "edit"
    write_on_early = bool(state.get("write_enabled"))
    cls_loop = float(
        (state.get("classification") or {}).get("confidence", 1.0) or 1.0
    )
    if cls_loop < 0.4:
        max_steps = min(max_steps + 2, hard_cap)
    if cls_loop < 0.3:
        max_steps = min(max_steps + 2, hard_cap)
    if cls_loop < 0.5:
        if not (mode_edit_early and write_on_early):
            max_steps = min(max_steps, 2)

    intent = (state.get("classification") or {}).get("agent_intent", "exploratory")
    intent_s = str(intent or "exploratory").lower().strip()
    if intent_s == "broad":
        state["react_force_rag"] = True
        max_steps = min(max_steps, 1)

    amode = str(state.get("agent_mode") or "chat").lower()
    if amode == "execute":
        state.pop("react_low_signal_repo_probe", None)

    if bool(state.get("react_low_signal_repo_probe")):
        pm = int(
            state.get("react_probe_max_steps")
            or getattr(settings, "REACT_LOW_SIGNAL_PROBE_MAX_STEPS", 2)
            or 2
        )
        max_steps = max(1, min(pm, hard_cap))

    if amode == "execute":
        cap_ex = int(getattr(settings, "REACT_MAX_STEPS_EXECUTE_MODE", 3) or 3)
        max_steps = max(1, min(max_steps, cap_ex))
    elif amode == "chat":
        cap_ch = int(getattr(settings, "REACT_MAX_STEPS_CHAT_MODE", 2) or 2)
        max_steps = max(1, min(max_steps, cap_ch))

    q = state.get("current_step") or state.get("query") or ""

    task_like_shell = looks_like_task_execution_query(str(q))

    explore_first = bool(getattr(settings, "REACT_EXPLORATION_FIRST", True))
    mode_ask = str(state.get("mode") or "").lower() == "ask"
    mode_edit = str(state.get("mode") or "").lower() == "edit"
    write_on = bool(state.get("write_enabled"))
    state["react_execution_allowed"] = bool(
        not explore_first
        or mode_edit
        or write_on
        or task_like_shell
        or intent_s == "structured"
        or bool(state.get("long_run_mode"))
        or amode == "execute"
    )

    shell_ok = intent_s == "structured"
    if not shell_ok and mode_ask and intent_s != "broad" and task_like_shell:
        # `looks_like_task_execution_query` is needle-gated; allow shell even when
        # classifier confidence is low so installs/scaffolds are not prose-only.
        shell_ok = True
    if bool(state.get("long_run_mode")):
        shell_ok = shell_ok and bool(
            getattr(settings, "LONG_RUN_SHELL_ENABLED", False)
        )
    router = state.get("selected_tool")
    rconf = float(state.get("router_confidence") or 0.0)
    state["react_goal"] = extract_goal(q)
    scope_goal = str(
        state.get("subgoal_description") or state.get("react_goal") or q or ""
    )
    if _react_or_node_scaffold_intent(scope_goal) and mode_edit:
        shell_ok = True
    if (
        is_create_intent(scope_goal)
        and _goal_triggers_create_fallback(scope_goal)
        and mode_edit
        and write_on
        and not bool(state.get("dry_run"))
    ):
        state["execution_scope"] = "code_only"
    else:
        state.pop("execution_scope", None)
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
        "run_command": 2,
    }
    if cls_loop < 0.4:
        _ACTION_SOFT_LIMITS["list_dir"] = max(_ACTION_SOFT_LIMITS["list_dir"], 4)
        _ACTION_SOFT_LIMITS["search"] = max(_ACTION_SOFT_LIMITS["search"], 5)
    action_limits: dict[str, int] = dict(_ACTION_SOFT_LIMITS)
    if _edit_tool_enabled(state):
        action_limits["apply_search_replace"] = 3
        action_limits["commit_session"] = 3
        action_limits["rollback_session"] = 2
        action_limits["checkpoint_session"] = max(
            1,
            min(
                int(getattr(settings, "REACT_EDIT_CHECKPOINT_MAX", 5) or 5),
                12,
            ),
        )
        action_limits["begin_session"] = 0
    action_counts: dict[str, int] = defaultdict(int)

    state["react_edited_paths"] = []
    state["react_error_paths"] = []

    state["react_steps"] = []
    state["react_observations"] = []
    state["react_summary"] = state.get("react_summary") or ""
    state["files_read"] = []
    state["symbols_found"] = []
    state["react_useful_steps"] = 0
    state["react_failed_steps"] = 0
    state["react_edit_line_estimate"] = 0
    state["react_force_rag"] = intent_s == "broad"
    state["_probe_search_paths"] = []
    state.setdefault("last_shell_commands", [])
    state.pop("_scaffold_opened_shell_early", None)
    state.pop("_react_forced_file_edit_fallback_done", None)
    state.pop("_react_fallback_allowed_edit_paths", None)
    state.pop("_react_forced_fallback_rel_pending", None)
    state.pop("_react_forced_fallback_commit_rel", None)
    state.pop("_react_fallback_injected_at_step", None)
    state.pop("_react_fallback_used", None)
    state.pop("_react_shell_cra_vite_alt_tried", None)
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

    exempt_decay: set[str] = set()
    for p in state.get("prior_react_signal_paths") or []:
        try:
            exempt_decay.add(_norm_rel_path(str(p)))
        except (TypeError, ValueError):
            pass
    decay = float(getattr(settings, "REACT_FILE_SCORE_DECAY", 0.97) or 0.97)
    if 0 < decay < 1 and file_scores:
        for fk in list(file_scores.keys()):
            if fk in exempt_decay:
                continue
            try:
                file_scores[fk] *= decay
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
    if state.get("execution_scope"):
        sess_ev["execution_scope"] = state["execution_scope"]
    await _emit_react(state, sess_ev)

    state["react_early_stop_reason"] = ""
    last_loop_i = -1
    for i in range(max_steps):
        am = str(state.get("agent_mode") or "chat").lower()
        last_loop_i = i
        cancel_ev = state.get("react_cancel_event")
        if cancel_ev is not None and getattr(cancel_ev, "is_set", lambda: False)():
            logger.info("ReasoningLoop: cancel requested at step %s", i)
            state["react_early_stop_reason"] = "user_cancelled"
            await _emit_react(state, {"type": "react_cancelled", "step": i})
            break

        if i >= 1 and int(state.get("react_useful_steps") or 0) == 0:
            mode_ed = str(state.get("mode") or "").lower() == "edit"
            if not (mode_ed and bool(state.get("write_enabled"))):
                logger.info(
                    "ReasoningLoop: early stop no_signal_query (no useful steps after first)"
                )
                state["react_early_stop_reason"] = "no_signal_query"
                await _emit_react(
                    state,
                    {"type": "react_low_signal_stop", "step": i},
                )
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
            _sg = str(
                state.get("subgoal_description") or state.get("react_goal") or ""
            )
            if is_create_intent(_sg):
                edit_extra += (
                    "CREATE/ADD FILE INTENT: use file_edit.apply_search_replace (or "
                    "file_edit.commit_session after edits) to persist changes on disk "
                    "before signaling done; list_dir or read_file alone is not enough.\n"
                )
            if _react_or_node_scaffold_intent(_sg):
                edit_extra += (
                    "FRONTEND / NODE TASK: use shell.run_command for npm/npx/vite/CRA/next "
                    "scaffolds — do not substitute a Python main.py or pip-in-subprocess "
                    "hacks unless the user explicitly asked for Python.\n"
                    "First verify the toolchain: run `node -v` (and `npm -v`). If Node is "
                    "missing, tell the user to install from https://nodejs.org/ — do not "
                    "invent Python fallback files.\n"
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

        shell_mem = ""
        lsc = state.get("last_shell_commands") or []
        if lsc:
            shell_mem = (
                "Recent shell (this subgoal):\n"
                f"{json.dumps(lsc[-4:], default=str)[:1100]}\n"
            )
        shell_rec = _shell_recovery_prompt_block(state)

        mode_ctrl = ""
        if am == "execute":
            mode_ctrl = (
                "Control plane: agent_mode=execute — prioritize shell.run_command for "
                "install/scaffold/test requests; do not signal done before attempting the "
                "command when the task requires a CLI.\n"
            )
        elif am == "explore":
            mode_ctrl = (
                "Control plane: agent_mode=explore — use search/read_file/symbols before "
                "heavy edits or shell.\n"
            )
        elif am == "chat":
            mode_ctrl = (
                "Control plane: agent_mode=chat — answer concisely; use tools only when "
                "the task clearly needs repo evidence.\n"
            )

        goal_text_early = str(
            state.get("subgoal_description") or state.get("react_goal") or q or ""
        )
        skip_llm_for_create = (
            i == 0
            and is_create_intent(goal_text_early)
            and str(state.get("mode") or "").lower() == "edit"
            and bool(state.get("write_enabled"))
            and not bool(state.get("dry_run"))
            and int(state.get("react_useful_steps") or 0) == 0
            and _edit_tool_enabled(state)
            and _goal_triggers_create_fallback(goal_text_early)
            and not bool(state.get("_react_forced_file_edit_fallback_done"))
            and not bool(state.get("react_low_signal_repo_probe"))
        )
        if skip_llm_for_create:
            state["react_observations"].append(
                "Create intent: skipping step-1 planner; deterministic create path runs next."
            )

        user_block = (
            f"User task: {q}\n"
            f"{mode_ctrl}"
            f"{scaffold_shell_hint(q)}"
            f"{edit_extra}"
            f"{hint_extra}"
            f"{shell_mem}"
            f"{shell_rec}"
            f"{_exploration_hint(i, max_steps, intent)}\n"
            f"{summary_note}"
            f"Router hint: tool={router!r} confidence={rconf:.2f}\n"
            f"Workspace root: {registry.workspace_root}\n"
            f"Recent observations (truncated):\n{obs_summary or '(none)'}\n"
            f"Step {i + 1} of {max_steps}."
        )
        prompt = _react_system_for(state) + "\n\n" + user_block

        try:
            if skip_llm_for_create:
                raw = (
                    '{"thought":"create intent (planner skipped)","tool":"file",'
                    '"action":"list_dir","args":{"path":".","depth":1},"done":false}'
                )
            else:
                raw = await provider.generate(
                    prompt,
                    system_prompt="Output only valid JSON for the next action.",
                    temperature=0.15,
                )
        except Exception as exc:
            logger.warning("ReasoningLoop LLM step failed: %s", exc)
            state["react_early_stop_reason"] = "llm_error"
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

        if bool(state.get("react_low_signal_repo_probe")) and not skip_llm_for_create:
            ranked = list(state.get("_probe_search_paths") or [])
            fr = _files_read_key_set(state)
            if i == 0:
                step["tool"] = "file"
                step["action"] = "search"
                step["args"] = {
                    "query": (str(q or "").strip()[:120] or "main entry")
                }
                step["done"] = False
            elif i in (1, 2):
                pick = None
                for p in ranked:
                    pn = _norm_rel_path(p).strip().lstrip("./")
                    if pn and pn not in fr:
                        pick = p
                        break
                if pick:
                    step["tool"] = "file"
                    step["action"] = "read_file"
                    step["args"] = {"path": pick}
                    step["done"] = False
                elif i == 1 and not ranked:
                    step["tool"] = "file"
                    step["action"] = "list_dir"
                    step["args"] = {"path": ".", "depth": 2}
                    step["done"] = False

        _scaf = scaffold_shell_hint(str(q)).strip()
        if (
            _scaf
            and max_steps >= 1
            and not bool(state.get("react_low_signal_repo_probe"))
            and not skip_llm_for_create
            and state.get("execution_scope") != "code_only"
        ):
            premature = (
                not step.get("tool")
                or step.get("done")
                or str(step.get("action") or "").strip().lower() == "done"
            )
            if premature:
                ws_root = getattr(registry, "workspace_root", "") or ""
                cmd0 = infer_scaffold_shell_command(str(q))
                sparse = _workspace_sparse_for_scaffold(ws_root, settings)
                if i == 0:
                    ex_first = am == "execute" and shell_ok and bool(cmd0)
                    if ex_first:
                        state["react_observations"].append(
                            "Execute mode: scaffold — running CLI first (confirm in UI)."
                        )
                        state["_scaffold_opened_shell_early"] = True
                        step = {
                            "thought": "Execute mode — run scaffold CLI immediately.",
                            "tool": "shell",
                            "action": "run_command",
                            "args": {
                                "command": cmd0,
                                "title": "Scaffold project",
                            },
                            "done": False,
                        }
                    elif sparse and shell_ok and cmd0:
                        state["react_observations"].append(
                            "Scaffold request: sparse workspace — running scaffold CLI "
                            "directly (confirm in UI if prompted)."
                        )
                        state["_scaffold_opened_shell_early"] = True
                        step = {
                            "thought": "Sparse workspace — run scaffold CLI without list_dir.",
                            "tool": "shell",
                            "action": "run_command",
                            "args": {
                                "command": cmd0,
                                "title": "Scaffold project",
                            },
                            "done": False,
                        }
                    else:
                        state["react_observations"].append(
                            "Scaffold request: model tried to finish without tools — "
                            "listing workspace first."
                        )
                        step = {
                            "thought": "Scaffold request — inspect workspace before CLI.",
                            "tool": "file",
                            "action": "list_dir",
                            "args": {"path": ".", "depth": 2},
                            "done": False,
                        }
                elif i == 1:
                    if state.get("_scaffold_opened_shell_early"):
                        state["react_observations"].append(
                            "Scaffold: verify generated files (package.json, src, etc.)."
                        )
                        step = {
                            "thought": "Verify scaffold output in the workspace.",
                            "tool": "file",
                            "action": "list_dir",
                            "args": {"path": ".", "depth": 3},
                            "done": False,
                        }
                        state.pop("_scaffold_opened_shell_early", None)
                    else:
                        cmd = infer_scaffold_shell_command(str(q))
                        if shell_ok and cmd:
                            state["react_observations"].append(
                                "Scaffold request: running scaffold CLI "
                                "(confirm in UI if prompted)."
                            )
                            step = {
                                "thought": "Run the scaffold command the user asked for.",
                                "tool": "shell",
                                "action": "run_command",
                                "args": {
                                    "command": cmd,
                                    "title": "Scaffold project",
                                },
                                "done": False,
                            }
                        else:
                            step = {
                                "thought": "Search workspace for relevant files.",
                                "tool": "file",
                                "action": "search",
                                "args": {"query": str(q)[:100]},
                                "done": False,
                            }

        goal_text = str(
            state.get("subgoal_description") or state.get("react_goal") or q or ""
        )
        early_fb = i == 0 and _goal_triggers_create_fallback(goal_text)
        if (
            (i >= 1 or early_fb)
            and str(state.get("mode") or "").lower() == "edit"
            and bool(state.get("write_enabled"))
            and not bool(state.get("dry_run"))
            and int(state.get("react_useful_steps") or 0) == 0
            and _edit_tool_enabled(state)
            and _goal_triggers_create_fallback(goal_text)
            and not bool(state.get("_react_forced_file_edit_fallback_done"))
            and (
                not _scaffold_shell_requires_attempt_before_py_fallback(q)
                or _react_steps_include_shell_run(state)
            )
        ):
            ws = str(getattr(registry, "workspace_root", "") or "").strip()
            if ws:
                fb_blocked = int(state.get("_react_fallback_used", 0) or 0) >= 1
                if fb_blocked:
                    logger.error(
                        "ReasoningLoop: create fallback blocked — already used once this session"
                    )
                    state["react_observations"].append(
                        "⚠ Create fallback skipped (limit 1 per session) — avoid file spam."
                    )
                if not fb_blocked:
                    state["_react_fallback_used"] = 1
                    rel, fbody = _pick_fallback_write_path(goal_text, ws)
                    state["_react_fallback_allowed_edit_paths"] = ["main.py"]

                    direct_ok, direct_err = _direct_write_fallback_file(ws, rel, fbody)
                    logger.info(
                        "ReasoningLoop: create fallback direct write workspace=%r rel=%r ok=%s detail=%r",
                        ws,
                        rel,
                        direct_ok,
                        direct_err,
                    )

                    if direct_ok:
                        rel_norm = _norm_rel_path(rel)
                        state["_react_forced_file_edit_fallback_done"] = True
                        state["react_useful_steps"] += 1
                        state["goal_progress"] = min(
                            1.0, float(state.get("goal_progress") or 0) + 0.12
                        )
                        if rel_norm and rel_norm not in state["files_read"]:
                            state["files_read"].append(rel_norm)
                        state["react_observations"].append(
                            f"⚡ Forced action: wrote {rel_norm} directly "
                            f"(step {i + 1}; bypassed file_edit staging)."
                        )
                        tr = ToolResult(
                            tool="file_edit",
                            action="direct_write_fallback",
                            success=True,
                            data={
                                "file": rel_norm,
                                "files_written": [rel_norm],
                                "direct": True,
                            },
                        )
                        state["tool_results"].append(tr.model_dump())
                        state["react_steps"].append(
                            {
                                "step_index": i,
                                "thought": (
                                    "Deterministic fallback — direct disk write "
                                    "(bypasses tool registry staging)."
                                ),
                                "tool": "file_edit",
                                "action": "direct_write_fallback",
                                "args": {"file_path": rel_norm},
                            }
                        )
                        _maybe_compress_react_state(state)
                        await _emit_step_evt(
                            state,
                            phase="start",
                            step=i,
                            tool="file_edit",
                            action="direct_write_fallback",
                            status="pending",
                            args_preview=json.dumps(
                                {"file_path": rel_norm}, default=str
                            ),
                        )
                        await _emit_step_evt(
                            state,
                            phase="result",
                            step=i,
                            tool="file_edit",
                            action="direct_write_fallback",
                            status="success",
                            summary="direct write",
                            duration_ms=0,
                            useful=True,
                        )
                        preview = json.dumps(tr.data, default=str)[:2000]
                        state["react_observations"].append(
                            f"Step {i}: file_edit.direct_write_fallback success=True data={preview}"
                        )
                        await EventBus.emit(
                            {
                                "type": "forced_fallback",
                                "path": rel_norm,
                                "phase": "direct",
                                "subgoal_id": state.get("subgoal_id"),
                            }
                        )
                        await EventBus.emit(
                            {
                                "type": "file_updated",
                                "file": rel_norm,
                                "workspace": ws,
                            }
                        )
                        await EventBus.emit(
                            {
                                "type": "agent_file_open",
                                "path": rel_norm,
                                "workspace": ws,
                                "subgoal_id": state.get("subgoal_id"),
                            }
                        )
                        await _emit_react(
                            state,
                            {
                                "type": "state",
                                "subgoal": state.get("subgoal_id"),
                                "step": i,
                                "goal_progress": float(state.get("goal_progress") or 0),
                                "files_read_count": len(state.get("files_read") or []),
                                "useful_steps": int(
                                    state.get("react_useful_steps") or 0
                                ),
                            },
                        )
                        _ac = compute_answer_confidence(state)
                        _nfr = len(state.get("files_read") or [])
                        _nsy = len(state.get("symbols_found") or [])
                        await EventBus.emit(
                            {
                                "type": "agent_status",
                                "confidence": _ac,
                                "files_read": _nfr,
                                "symbols": _nsy,
                                "weak_grounding": _nfr < 2
                                and _nsy == 0
                                and _ac < 0.5,
                                "subgoal_id": state.get("subgoal_id"),
                                "agent_execution_phase": (
                                    "execute"
                                    if state.get("react_execution_allowed")
                                    else "explore"
                                ),
                            },
                        )
                        logger.info(
                            "ReasoningLoop: direct fallback complete; stopping "
                            "(forced_write_complete)"
                        )
                        state["react_early_stop_reason"] = "forced_write_complete"
                        await _emit_react(
                            state,
                            {"type": "react_write_complete_stop", "step": i},
                        )
                        break

                    rel_norm = _norm_rel_path(rel)
                    if direct_err:
                        logger.warning(
                            "ReasoningLoop: direct fallback failed; trying file_edit: %s",
                            direct_err,
                        )
                        await EventBus.emit(
                            {
                                "type": "fallback_execution_failed",
                                "path": rel_norm,
                                "step": i,
                                "subgoal_id": state.get("subgoal_id"),
                                "error": f"direct_write:{direct_err}"[:300],
                            }
                        )

                    state["_react_forced_fallback_rel_pending"] = rel
                    state["_react_forced_fallback_commit_rel"] = rel
                    state["_react_fallback_injected_at_step"] = i
                    state["react_observations"].append(
                        f"⚡ Forced action: staging minimal write → {rel} "
                        f"(step {i + 1}, no useful tool steps yet; will commit with subgoal)."
                    )
                    step = {
                        "thought": (
                            "Deterministic fallback — write minimal implementation to disk "
                            "via file_edit."
                        ),
                        "tool": "file_edit",
                        "action": "apply_search_replace",
                        "args": {
                            "file_path": rel,
                            "search": "",
                            "replace": fbody,
                            "expected_occurrences": 1,
                        },
                        "done": False,
                    }
                    state["_react_forced_file_edit_fallback_done"] = True

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

        done_like = (
            step.get("done")
            or str(step.get("action") or "").strip().lower() == "done"
            or not step.get("tool")
        )
        if (
            done_like
            and am == "execute"
            and _scaf
            and shell_ok
            and state.get("execution_scope") != "code_only"
        ):
            cmd_gate = infer_scaffold_shell_command(str(q))
            shell_ran = any(
                isinstance(x, dict) and str(x.get("tool") or "").lower() == "shell"
                for x in (state.get("react_steps") or [])
            )
            if cmd_gate and not shell_ran:
                state["react_observations"].append(
                    "Execute mode: blocked premature done — running scaffold shell."
                )
                step = {
                    "thought": "Execute mode — scaffold requires shell.run_command.",
                    "tool": "shell",
                    "action": "run_command",
                    "args": {
                        "command": cmd_gate,
                        "title": "Scaffold project",
                    },
                    "done": False,
                }
                state["react_steps"][-1] = {
                    "step_index": i,
                    "thought": step.get("thought"),
                    "tool": step.get("tool"),
                    "action": step.get("action"),
                    "args": step.get("args"),
                }
                _maybe_compress_react_state(state)
                done_like = False

        if done_like:
            mode_ed = str(state.get("mode") or "").lower() == "edit"
            writes = bool(state.get("write_enabled"))
            no_useful = int(state.get("react_useful_steps") or 0) == 0
            not_dry = not bool(state.get("dry_run"))
            if (
                mode_ed
                and writes
                and no_useful
                and not_dry
                and i < max_steps - 1
            ):
                logger.info(
                    "ReasoningLoop: ignoring premature planner_done (edit, no useful steps)"
                )
                state["react_observations"].append(
                    "Edit mode: use at least one substantive tool (read_file, search, "
                    "file_edit, …) before signaling done."
                )
                if state.get("react_steps"):
                    state["react_steps"].pop()
                continue
            logger.info("ReasoningLoop: planner signaled done at step %s", i)
            state["react_early_stop_reason"] = "planner_done"
            await _emit_react(
                state,
                {"type": "react_done", "step": i, "thought": step.get("thought")},
            )
            break

        tool_name = str(step.get("tool") or "").strip().lower()
        action = str(step.get("action") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}

        if (
            tool_name == "file_edit"
            and action == "apply_search_replace"
            and not args.get("dry_run")
            and _file_edit_replace_uses_subprocess_pip_install(
                str(args.get("replace") or "")
            )
        ):
            fp_hint = str(args.get("file_path") or args.get("path") or "")[:120]
            logger.warning(
                "ReasoningLoop: blocked file_edit subprocess+pip install workaround path=%r",
                fp_hint,
            )
            state["react_observations"].append(
                "Blocked file_edit: subprocess pip installs are not allowed — "
                "use shell.run_command for installs, or add dependencies to requirements."
            )
            state["react_failed_steps"] += 1
            await EventBus.emit(
                {
                    "type": "tool_blocked",
                    "reason": "subprocess_install_not_allowed",
                    "tool": "file_edit",
                    "path": fp_hint,
                    "subgoal_id": state.get("subgoal_id"),
                }
            )
            await _emit_step_evt(
                state,
                phase="result",
                step=i,
                tool=tool_name,
                action=action,
                status="fail",
                summary="blocked subprocess pip",
                duration_ms=0,
            )
            if state["react_failed_steps"] >= 2:
                await _maybe_force_rag_escape_emit(state)
                state["react_early_stop_reason"] = "failed_steps_limit"
                break
            continue

        await _emit_step_evt(
            state,
            phase="start",
            step=i,
            tool=tool_name,
            action=action,
            status="pending",
            args_preview=json.dumps(args, default=str) if args else None,
        )

        valid_tools: tuple[str, ...] = ("file", "git", "log")
        if _edit_tool_enabled(state):
            valid_tools = ("file", "file_edit", "git", "log")
        if shell_ok:
            valid_tools = valid_tools + ("shell",)
        if state.get("execution_scope") == "code_only":
            valid_tools = ("file_edit",)
        if tool_name not in valid_tools:
            if state.get("execution_scope") == "code_only":
                state["react_observations"].append(
                    f"Step {i}: tool {tool_name!r} blocked — create-intent code_only "
                    "allows file_edit only (no shell / list_dir / pip via other tools)."
                )
            else:
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
                await _maybe_force_rag_escape_emit(state)
                state["react_early_stop_reason"] = "failed_steps_limit"
                break
            continue

        exec_allowed = bool(state.get("react_execution_allowed"))
        if explore_first and not exec_allowed:
            if tool_name == "shell":
                state["react_observations"].append(
                    f"Step {i}: shell deferred — explore the repo (read_file / search / "
                    "dependencies) first, then run commands."
                )
                state["react_failed_steps"] += 1
                await _emit_step_evt(
                    state,
                    phase="result",
                    step=i,
                    tool=tool_name,
                    action=action,
                    status="fail",
                    summary="exploration phase",
                    duration_ms=0,
                )
                if state["react_failed_steps"] >= 2:
                    await _maybe_force_rag_escape_emit(state)
                    state["react_early_stop_reason"] = "failed_steps_limit"
                    break
                continue
            if tool_name == "file_edit" and action in (
                "apply_search_replace",
                "commit_session",
            ):
                if not (action == "apply_search_replace" and args.get("dry_run")):
                    state["react_observations"].append(
                        f"Step {i}: edits deferred — read relevant files first "
                        "(exploration phase)."
                    )
                    state["react_failed_steps"] += 1
                    await _emit_step_evt(
                        state,
                        phase="result",
                        step=i,
                        tool=tool_name,
                        action=action,
                        status="fail",
                        summary="exploration phase",
                        duration_ms=0,
                    )
                    if state["react_failed_steps"] >= 2:
                        await _maybe_force_rag_escape_emit(state)
                        state["react_early_stop_reason"] = "failed_steps_limit"
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
                    await _maybe_force_rag_escape_emit(state)
                    state["react_early_stop_reason"] = "failed_steps_limit"
                    break
                continue
            if not args.get("dry_run"):
                from app.tools.file_edit_tool import FileEditTool

                fe_chk = registry.get_tool("file_edit")
                max_f = int(
                    getattr(settings, "EDIT_MAX_TOTAL_FILES_REACT", 0) or 0
                )
                max_l = int(
                    getattr(settings, "EDIT_MAX_LINES_CHANGED_REACT", 0) or 0
                )
                fp_norm = str(fp_chk).replace("\\", "/").strip()
                if isinstance(fe_chk, FileEditTool) and max_f > 0:
                    touched = set(fe_chk.touched_files())
                    prospective = len(touched) + (
                        1 if fp_norm and fp_norm not in touched else 0
                    )
                    if prospective > max_f:
                        state["react_observations"].append(
                            f"Step {i}: edit budget — max {max_f} files per subgoal "
                            f"(would touch {prospective})."
                        )
                        state["react_failed_steps"] += 1
                        await _emit_step_evt(
                            state,
                            phase="result",
                            step=i,
                            tool=tool_name,
                            action=action,
                            status="fail",
                            summary="edit file budget",
                            duration_ms=0,
                        )
                        if state["react_failed_steps"] >= 2:
                            await _maybe_force_rag_escape_emit(state)
                            state["react_early_stop_reason"] = "edit_budget"
                            break
                        continue
                if max_l > 0:
                    search_s = str(args.get("search") or "")
                    repl_s = str(args.get("replace") or "")
                    add_ln = max(
                        len(search_s.splitlines()),
                        len(repl_s.splitlines()),
                    )
                    cur_ln = int(state.get("react_edit_line_estimate") or 0)
                    if cur_ln + add_ln > max_l:
                        state["react_observations"].append(
                            f"Step {i}: edit budget — max ~{max_l} lines changed per subgoal "
                            f"(estimated {cur_ln + add_ln})."
                        )
                        state["react_failed_steps"] += 1
                        await _emit_step_evt(
                            state,
                            phase="result",
                            step=i,
                            tool=tool_name,
                            action=action,
                            status="fail",
                            summary="edit line budget",
                            duration_ms=0,
                        )
                        if state["react_failed_steps"] >= 2:
                            await _maybe_force_rag_escape_emit(state)
                            state["react_early_stop_reason"] = "edit_budget"
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
                await _maybe_force_rag_escape_emit(state)
                state["react_early_stop_reason"] = "repetition_limit"
                break
            continue
        action_counts[action] += 1

        res: ToolResult | None
        t0 = time.perf_counter()
        exec_args = dict(args)
        if tool_name == "shell":
            exec_args["confirm_shell_callback"] = state.get("confirm_shell_callback")
            exec_args["long_run_mode"] = bool(state.get("long_run_mode"))
        try:
            res = await registry.execute_tool(tool_name, action, **exec_args)
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
                await _maybe_force_rag_escape_emit(state)
                state["react_early_stop_reason"] = "tool_error_limit"
                logger.info(
                    "ReasoningLoop: stopping after %s failed steps",
                    state["react_failed_steps"],
                )
                break
            continue

        if (
            tool_name == "shell"
            and res is not None
            and not res.success
            and bool(getattr(settings, "REACT_SHELL_FAILURE_RETRY_ONCE", True))
        ):
            d0 = res.data if isinstance(res.data, dict) else {}
            err_s = str(res.error or "").lower()
            skip_like = bool(d0.get("skipped")) or "user skipped" in err_s
            no_cb = (
                "interactive session" in err_s
                or "no confirm_shell_callback" in err_s
            )
            timed = "timed out" in err_s or "timeout" in err_s
            futile = _shell_retry_pointless(res)
            if futile:
                alt = ""
                cmd_try = str(args.get("command") or "").lower()
                if "create-react-app" in cmd_try:
                    alt = (
                        " Alternative: `npm create vite@latest my-app -- --template react`."
                    )
                state["react_observations"].append(
                    f"Step {i}: shell failed (no retry — error looks non-transient).{alt}"
                )
            if not skip_like and not no_cb and not timed and not futile:
                state["react_observations"].append(
                    f"Step {i}: shell failed — retrying once ({(res.error or '')[:80]})"
                )
                try:
                    res = await registry.execute_tool(
                        tool_name, action, **exec_args
                    )
                except Exception as exc:
                    state["react_observations"].append(
                        f"Step {i}: shell retry error {exc}"
                    )
            if (
                res is not None
                and not res.success
                and not skip_like
                and not no_cb
                and "create-react-app" in str(args.get("command") or "").lower()
                and not state.get("_react_shell_cra_vite_alt_tried")
            ):
                state["_react_shell_cra_vite_alt_tried"] = True
                state["react_observations"].append(
                    f"Step {i}: CRA failed — trying Vite scaffold once as fallback."
                )
                alt_args: dict[str, Any] = {
                    "command": "npm create vite@latest frontend -- --template react",
                    "title": "Scaffold (Vite fallback after CRA)",
                }
                alt_args["confirm_shell_callback"] = state.get(
                    "confirm_shell_callback"
                )
                alt_args["long_run_mode"] = bool(state.get("long_run_mode"))
                try:
                    res = await registry.execute_tool(
                        "shell", "run_command", **alt_args
                    )
                except Exception as exc:
                    state["react_observations"].append(
                        f"Step {i}: Vite fallback shell error {exc}"
                    )

        if tool_name == "shell" and res is not None:
            dsh = res.data if isinstance(res.data, dict) else {}
            cmd_s = str(args.get("command") or dsh.get("command") or "").strip()
            out_tail = str(dsh.get("stdout") or "")[-600:]
            err_tail = str(dsh.get("stderr") or "")[-600:]
            lst = state.setdefault("last_shell_commands", [])
            lst.append(
                {
                    "command": cmd_s[:500],
                    "success": bool(res.success),
                    "exit_code": dsh.get("exit_code"),
                    "error": (res.error or "")[:300] if not res.success else None,
                    "stdout_tail": out_tail,
                    "stderr_tail": err_tail,
                }
            )
            state["last_shell_commands"] = lst[-12:]
            if not res.success:
                rh = _shell_failure_recovery_hints(lst[-1])
                state["react_observations"].append(
                    f"Step {i}: shell recovery hint — {rh}"
                )
            elif scaffold_shell_hint(str(q)).strip():
                cmd_l = cmd_s.lower()
                if any(
                    k in cmd_l
                    for k in (
                        "create-react-app",
                        "create vite",
                        "create-vite",
                        "create-next-app",
                    )
                ):
                    state["react_observations"].append(
                        "Scaffold shell exited 0 — use list_dir or read_file to confirm "
                        "package.json / src exist before declaring done."
                    )

        useful = _tool_step_useful(res, q)
        if res and res.success and tool_name == "file_edit":
            if action in (
                "apply_search_replace",
                "commit_session",
                "checkpoint_session",
            ):
                useful = True
        if useful:
            state["react_useful_steps"] += 1
        else:
            state["react_failed_steps"] += 1

        fb_at = state.get("_react_fallback_injected_at_step")
        if fb_at is not None and fb_at == i:
            state.pop("_react_fallback_injected_at_step", None)
            if not (res and res.success) or not useful:
                rel_fb = str(state.get("_react_forced_fallback_commit_rel") or "")
                err_s = ""
                if res:
                    err_s = str(res.error or "")[:400]
                logger.warning(
                    "ReasoningLoop: forced file_edit fallback did not succeed at step %s "
                    "(path=%r err=%r)",
                    i,
                    rel_fb,
                    err_s,
                )
                state["react_observations"].append(
                    "⚠ Forced file_edit fallback did not apply — check tool result above."
                )
                await EventBus.emit(
                    {
                        "type": "fallback_execution_failed",
                        "path": rel_fb.replace("\\", "/"),
                        "step": i,
                        "subgoal_id": state.get("subgoal_id"),
                        "error": err_s,
                    }
                )

        if (
            res
            and res.success
            and tool_name == "file"
            and action == "search"
        ):
            data_s = res.data or {}
            probe_paths: list[str] = []
            if isinstance(data_s, dict):
                for m in data_s.get("matches") or []:
                    if not isinstance(m, dict):
                        continue
                    fp = m.get("file") or m.get("path")
                    if isinstance(fp, str) and fp.strip():
                        probe_paths.append(fp.replace("\\", "/"))
            if probe_paths:
                ranked_paths = _rank_probe_search_paths(
                    probe_paths, str(q or "")
                )
                state["_probe_search_paths"] = ranked_paths[:16]

        dumped = res.model_dump() if res is not None else {}
        state["tool_results"].append(dumped)

        ob = f"Step {i}: {tool_name}.{action} success={getattr(res, 'success', False)}"
        if res and res.success and res.data:
            preview = json.dumps(res.data, default=str)[:2000]
            ob += f" data={preview}"
        elif res and res.error:
            ob += f" error={res.error[:400]}"
        state["react_observations"].append(ob)
        ws_root = getattr(registry, "workspace_root", "") or ""
        for raw_path in _scrape_error_paths_from_observation(ob):
            rel = _try_repo_relative(raw_path, ws_root)
            key = rel or _norm_rel_path(raw_path)
            if key:
                err_list = state.setdefault("react_error_paths", [])
                if key not in err_list:
                    err_list.append(key)

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
            pnd = str(state.get("_react_forced_fallback_rel_pending") or "").strip()
            apath = str(p or "")
            if pnd and apath and _norm_rel_path(apath) == _norm_rel_path(pnd):
                if res and res.success:
                    await EventBus.emit(
                        {
                            "type": "forced_fallback",
                            "path": _norm_rel_path(apath),
                            "phase": "staged",
                            "subgoal_id": state.get("subgoal_id"),
                        }
                    )
                state.pop("_react_forced_fallback_rel_pending", None)
        if tool_name == "file" and action == "find_symbol":
            for m in (res.data or {}).get("matches") or [] if res and res.data else []:
                sym = m.get("symbol")
                if sym and sym not in state["symbols_found"]:
                    state["symbols_found"].append(sym)

        if explore_first and not state.get("react_execution_allowed"):
            if _react_evidence_unlocks_execution(state, settings):
                state["react_execution_allowed"] = True

        _ac = compute_answer_confidence(state)
        _nfr = len(state.get("files_read") or [])
        _nsy = len(state.get("symbols_found") or [])
        await EventBus.emit(
            {
                "type": "agent_status",
                "confidence": _ac,
                "files_read": _nfr,
                "symbols": _nsy,
                "weak_grounding": _nfr < 2 and _nsy == 0 and _ac < 0.5,
                "subgoal_id": state.get("subgoal_id"),
                "agent_execution_phase": (
                    "execute" if state.get("react_execution_allowed") else "explore"
                ),
            }
        )

        if (
            state.get("execution_scope") == "code_only"
            and res
            and res.success
            and tool_name == "file_edit"
            and action == "apply_search_replace"
            and not args.get("dry_run")
        ):
            logger.info(
                "ReasoningLoop: code_only — file_edit write succeeded; stopping"
            )
            state["react_early_stop_reason"] = "write_complete"
            await _emit_react(
                state,
                {"type": "react_write_complete_stop", "step": i},
            )
            break

        if (
            res
            and res.success
            and tool_name == "file"
            and action == "read_file"
        ):
            rp = args.get("path") or args.get("file_path")
            if rp:
                await EventBus.emit(
                    {
                        "type": "agent_file_open",
                        "path": str(rp).replace("\\", "/"),
                        "workspace": getattr(registry, "workspace_root", ""),
                        "subgoal_id": state.get("subgoal_id"),
                    }
                )

        if (
            res
            and res.success
            and tool_name == "file_edit"
            and action == "apply_search_replace"
            and not args.get("dry_run")
        ):
            fp_ed = args.get("file_path") or args.get("path")
            if fp_ed:
                ep = _norm_rel_path(str(fp_ed))
                elst = state.setdefault("react_edited_paths", [])
                if ep and ep not in elst:
                    elst.append(ep)
            ss = str(args.get("search") or "")
            rr = str(args.get("replace") or "")
            add_ln = max(len(ss.splitlines()), len(rr.splitlines()))
            state["react_edit_line_estimate"] = int(
                state.get("react_edit_line_estimate") or 0
            ) + add_ln
            conf_thr = float(
                getattr(settings, "REACT_EDIT_CHECKPOINT_MIN_CONFIDENCE", 0.6)
            )
            if compute_answer_confidence(state) >= conf_thr:
                try:
                    await registry.execute_tool(
                        "file_edit", "checkpoint_session"
                    )
                except Exception:
                    pass

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
            state["react_early_stop_reason"] = "high_confidence"
            await _emit_react(
                state,
                {"type": "react_high_confidence_stop", "step": i},
            )
            break

        if state["react_failed_steps"] >= 2:
            await _maybe_force_rag_escape_emit(state)
            state["react_early_stop_reason"] = "failed_steps_limit"
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
                state["react_early_stop_reason"] = "no_progress_rag_escalation"
                await EventBus.emit(
                    {
                        "type": "rag_escalation",
                        "reason": "no_useful_steps_low_classification",
                        "subgoal_id": state.get("subgoal_id"),
                    }
                )
                logger.info(
                    "ReasoningLoop: no useful grounding through step %s → force RAG",
                    i,
                )
            else:
                state["react_early_stop_reason"] = "no_useful_steps_stall"
            break

        if enough_context(state):
            logger.info("ReasoningLoop: enough_context met at step %s", i)
            state["react_early_stop_reason"] = "enough_context"
            await _emit_react(
                state,
                {"type": "react_enough_context", "step": i},
            )
            break

    if not (state.get("react_early_stop_reason") or "").strip():
        state["react_early_stop_reason"] = "max_steps_exhausted"

    await _finalize_create_main_invariant(
        state, registry, str(q or ""), max(last_loop_i, 0)
    )

    final_i = max(last_loop_i, 0)
    ws_merge = getattr(registry, "workspace_root", "") or ""
    for raw_ed in state.get("react_edited_paths") or []:
        pk = _norm_rel_path(str(raw_ed))
        if pk:
            file_scores[pk] = file_scores.get(pk, 0.0) + 3.0
    seen_err_boost: set[str] = set()
    for raw_er in state.get("react_error_paths") or []:
        rel = _try_repo_relative(str(raw_er), ws_merge)
        pk = rel or _norm_rel_path(str(raw_er))
        if not pk or pk in seen_err_boost:
            continue
        seen_err_boost.add(pk)
        file_scores[pk] = file_scores.get(pk, 0.0) + 4.0

    for p in list(file_scores.keys()):
        staleness = final_i - last_touch_step.get(p, final_i)
        if staleness > 0:
            file_scores[p] *= 1.0 / (1.0 + staleness)

    state["file_scores"] = {
        k: round(v, 4) for k, v in sorted(file_scores.items(), key=lambda kv: -kv[1])
    }
    top_files = sorted(file_scores.keys(), key=lambda p: file_scores[p], reverse=True)[:2]
    state["react_priority_files"] = top_files

    new_prior: set[str] = set()
    for raw_ed in state.get("react_edited_paths") or []:
        pk = _norm_rel_path(str(raw_ed))
        if pk:
            new_prior.add(pk)
    for raw_er in state.get("react_error_paths") or []:
        rel = _try_repo_relative(str(raw_er), ws_merge)
        pk = rel or _norm_rel_path(str(raw_er))
        if pk:
            new_prior.add(pk)
    prev_prior = {
        _norm_rel_path(str(x))
        for x in (state.get("prior_react_signal_paths") or [])
        if x
    }
    state["prior_react_signal_paths"] = list(prev_prior | new_prior)[:120]

    state["tool_results"] = prior_tools + state["tool_results"]
    await _emit_react(
        state,
        {
            "type": "react_session_end",
            "useful_steps": state["react_useful_steps"],
            "failed_steps": state["react_failed_steps"],
            "force_rag": bool(state.get("react_force_rag")),
            "early_stop_reason": state.get("react_early_stop_reason") or "",
        },
    )
    return state
