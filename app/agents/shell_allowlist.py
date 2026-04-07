"""Persist shell allowlist: exact commands vs safe read-only prefixes."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Prefix auto-run is only honored for read-risk commands whose first token is in this set.
# Never include broad interpreters or git here — prefix auto-run is only for
# narrow, read-oriented builtins (see review: `git` prefix → reset --hard risk).
SAFE_PREFIX_TOKENS: frozenset[str] = frozenset(
    {
        "ls",
        "dir",
        "pwd",
        "cd",
        "echo",
        "cat",
        "type",
        "head",
        "tail",
        "wc",
        "grep",
        "rg",
        "which",
        "where",
        "sort",
        "uniq",
        "file",
        "stat",
    }
)


@dataclass
class ShellAllowlist:
    exact: set[str] = field(default_factory=set)
    prefix: list[str] = field(default_factory=list)


def _allowlist_path(workspace_root: str) -> Path:
    return Path(workspace_root) / ".ai_assistant" / "shell_allowlist.json"


def _normalize_exact(s: str) -> str:
    return " ".join((s or "").strip().split())


def _migrate_legacy_prefixes(raw_list: list[str]) -> ShellAllowlist:
    """Old format: flat prefixes list — tighten to exact + safe prefix only."""
    al = ShellAllowlist()
    for x in raw_list:
        e = _normalize_exact(str(x))
        if not e:
            continue
        first = e.split(None, 1)[0].lower()
        if " " not in e and first in SAFE_PREFIX_TOKENS:
            if e not in al.prefix:
                al.prefix.append(e)
        else:
            al.exact.add(e)
    return al


def load_shell_allowlist(workspace_root: str) -> ShellAllowlist:
    path = _allowlist_path(workspace_root)
    if not path.is_file():
        return ShellAllowlist()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ShellAllowlist()

    al = ShellAllowlist()
    if isinstance(raw, dict):
        ex = raw.get("exact")
        pr = raw.get("prefix")
        if isinstance(ex, list):
            al.exact = {_normalize_exact(str(x)) for x in ex if str(x).strip()}
        if isinstance(pr, list):
            al.prefix = []
            for x in pr:
                e = _normalize_exact(str(x))
                if e and e.split(None, 1)[0].lower() in SAFE_PREFIX_TOKENS:
                    if e not in al.prefix:
                        al.prefix.append(e)
        legacy = raw.get("prefixes") or raw.get("entries")
        if isinstance(legacy, list) and not al.exact and not al.prefix:
            return _migrate_legacy_prefixes(
                [str(x) for x in legacy if str(x).strip()]
            )
        return al

    if isinstance(raw, list):
        return _migrate_legacy_prefixes([str(x) for x in raw if str(x).strip()])

    return ShellAllowlist()


def _save_allowlist(workspace_root: str, al: ShellAllowlist) -> None:
    path = _allowlist_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "exact": sorted(al.exact),
        "prefix": list(al.prefix),
    }
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def merge_shell_allowlist(
    workspace_root: str,
    *,
    exact_add: list[str] | None = None,
    prefix_add: list[str] | None = None,
) -> ShellAllowlist:
    al = load_shell_allowlist(workspace_root)
    for x in exact_add or []:
        e = _normalize_exact(str(x))
        if e:
            al.exact.add(e)
    for x in prefix_add or []:
        e = _normalize_exact(str(x))
        if not e:
            continue
        first = e.split(None, 1)[0].lower()
        if first not in SAFE_PREFIX_TOKENS:
            al.exact.add(e)
            continue
        if e not in al.prefix:
            al.prefix.append(e)
    _save_allowlist(workspace_root, al)
    return al


def read_allowlist_allows_command(command: str, al: ShellAllowlist) -> bool:
    """
    Auto-run without prompt only for read-risk paths (caller must verify risk == read).
    Exact match always (for stored full read commands).
    Prefix match only if first token is in SAFE_PREFIX_TOKENS.
    """
    cmd = _normalize_exact(command)
    if not cmd:
        return False
    if cmd in al.exact:
        return True
    first = cmd.split(None, 1)[0].lower()
    if first not in SAFE_PREFIX_TOKENS:
        return False
    for p in al.prefix:
        if not p:
            continue
        if cmd == p or cmd.startswith(p + " ") or cmd.startswith(p + "\t"):
            return True
    return False


def suggest_allowlist_tokens(command: str) -> tuple[list[str], list[str]]:
    """(exact_candidates, prefix_candidates) for UI labels and merge."""
    s = _normalize_exact(command)
    if not s:
        return [], []
    first = s.split(None, 1)[0]
    exact = [s]
    prefixes: list[str] = []
    if first.lower() in SAFE_PREFIX_TOKENS:
        prefixes = [first]
    return exact, prefixes
