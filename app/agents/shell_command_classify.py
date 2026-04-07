"""Classify shell commands for approval policy (read vs write vs destructive)."""
from __future__ import annotations

import re
import shlex
from typing import Literal

ShellRisk = Literal["read", "write", "destructive"]

_REDIRECT_RE = re.compile(
    r"(?:^|[\s;|])(?:[012]?>|>>)\s*\S|"  # > file, 2>>, etc.
    r"\|\s*tee\b",
    re.IGNORECASE,
)

_DEST_SUBSTR = (
    "rm -rf",
    "rm -fr",
    "rm -r -f",
    "rmdir /s",
    "git reset --hard",
    "git clean -fd",
    "git clean -fx",
    "git clean -df",
    ":(){",
    "| bash",
    "mkfs",
    "dd if=",
    "shred ",
    "cipher /w",
    "format ",
)

_WRITE_SUBSTR = (
    "pip install",
    "pip3 install",
    "npm install",
    "yarn add",
    "pnpm add",
    "apt install",
    "apt-get install",
    "dnf install",
    "choco install",
    "scoop install",
)

# First token (lowercase) → treat as filesystem-destructive movers/removers
_DEST_FIRST = frozenset(
    {
        "rm",
        "rmdir",
        "del",
        "erase",
        "deltree",
        "mv",
        "move",
        "ren",
        "rename",
        "remove-item",
        "ri",
        "rd",
    }
)

_WRITE_FIRST = frozenset(
    {
        "cp",
        "copy",
        "xcopy",
        "robocopy",
        "touch",
        "mkdir",
        "md",
        "install",
        "npm",
        "yarn",
        "pnpm",
        "pip",
        "pip3",
        "tee",
        "chmod",
        "chown",
        "attrib",
        "icacls",
    }
)


def _first_token(cmd: str) -> str:
    s = (cmd or "").strip()
    if not s:
        return ""
    try:
        parts = shlex.split(s, posix=True)
    except ValueError:
        parts = s.split()
    if not parts:
        return ""
    return str(parts[0]).lower().strip("\"'")


def classify_shell_risk(command: str) -> ShellRisk:
    """
    Conservative triage: destructive > write > read.
    Heuristic only — user confirmation remains the real gate.
    """
    c = (command or "").strip()
    if not c:
        return "read"
    low = c.lower()

    if any(x in low for x in _DEST_SUBSTR):
        return "destructive"

    if any(x in low for x in _WRITE_SUBSTR):
        return "write"

    if _REDIRECT_RE.search(low):
        return "write"

    if low == "git" or low.startswith("git "):
        if any(
            x in low
            for x in (
                "git reset",
                "git clean",
                "push --force",
                "push -f",
                " branch -d ",
                " branch -D ",
            )
        ):
            return "destructive"

    ft = _first_token(c)
    if ft in _DEST_FIRST:
        return "destructive"
    if ft in _WRITE_FIRST:
        return "write"

    if ft == "git" and any(
        sub in low for sub in (" commit", " merge", " rebase", " cherry-pick", " stash pop")
    ):
        return "write"

    return "read"


def shell_risk_label(risk: ShellRisk) -> str:
    return {"read": "read-only", "write": "writes files/state", "destructive": "destructive"}[
        risk
    ]
