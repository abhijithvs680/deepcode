"""Invoke optional async confirm_shell callbacks with backward-compatible signatures."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

ConfirmShellCallback = Callable[..., Any]

# Returned by callbacks: run / skip / allowlist_run (run and persist suggested tokens).
ShellConfirmChoice = str


async def invoke_confirm_shell_callback(
    confirm_cb: ConfirmShellCallback | None,
    command: str,
    *,
    title: str | None = None,
    cwd: str | None = None,
    tag_labels: list[str] | None = None,
) -> ShellConfirmChoice:
    """
    Ask the UI whether to run a shell command.
    Callback may return:
      - "run" | "skip" | "allowlist_run" (str)
      - legacy True -> "run", False -> "skip"
    """
    if confirm_cb is None:
        return "skip"
    try:
        sig = inspect.signature(confirm_cb)
    except (TypeError, ValueError):
        out = confirm_cb(command)
    else:
        kw: dict[str, Any] = {}
        if "title" in sig.parameters:
            kw["title"] = title
        if "cwd" in sig.parameters:
            kw["cwd"] = cwd
        if "tag_labels" in sig.parameters:
            kw["tag_labels"] = tag_labels
        out = confirm_cb(command, **kw)
    if inspect.isawaitable(out):
        out = await out
    if isinstance(out, str) and out in ("run", "skip", "allowlist_run"):
        return out
    return "run" if bool(out) else "skip"
