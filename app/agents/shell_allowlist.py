"""Persist shell command allowlist prefixes under the workspace (.ai_assistant)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _allowlist_path(workspace_root: str) -> Path:
    return Path(workspace_root) / ".ai_assistant" / "shell_allowlist.json"


def load_shell_allowlist(workspace_root: str) -> list[str]:
    path = _allowlist_path(workspace_root)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict):
        items = raw.get("prefixes") or raw.get("entries")
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def merge_shell_allowlist(workspace_root: str, new_entries: list[str]) -> list[str]:
    cur = load_shell_allowlist(workspace_root)
    seen: set[str] = set()
    out: list[str] = []
    for x in cur + new_entries:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    path = _allowlist_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"prefixes": out}
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return out


def command_matches_allowlist(command: str, prefixes: list[str]) -> bool:
    cmd = (command or "").strip()
    if not cmd:
        return False
    for p in prefixes:
        if not p:
            continue
        if cmd == p or cmd.startswith(p):
            return True
    return False


def suggest_allowlist_tokens(command: str) -> list[str]:
    s = (command or "").strip()
    if not s:
        return []
    first = s.split(None, 1)[0] if s.split() else s
    ordered: list[str] = []
    for x in (first, s):
        if x and x not in ordered:
            ordered.append(x)
    return ordered
