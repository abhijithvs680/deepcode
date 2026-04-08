"""Run shell commands in the workspace after optional UI approval and allowlist."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from app.agents.confirm_shell_util import invoke_confirm_shell_callback
from app.agents.shell_allowlist import (
    load_shell_allowlist,
    merge_shell_allowlist,
    read_allowlist_allows_command,
    suggest_allowlist_tokens,
)
from app.agents.shell_command_classify import classify_shell_risk, shell_risk_label
from app.core.config import get_settings
from app.core.event_bus import EventBus
from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 120.0
_MAX_CAPTURE_CHARS = 400_000


def _resolve_cwd(workspace_root: str, cwd: str | None) -> str:
    """Resolve cwd under workspace; symlink targets outside the repo fall back to root."""
    root = Path(workspace_root).resolve()
    if not cwd or not str(cwd).strip():
        return str(root)
    p = Path(str(cwd).strip())
    if not p.is_absolute():
        p = root / p
    try:
        rp = p.resolve()
        try:
            rp.relative_to(root)
        except ValueError:
            return str(root)
        real = Path(os.path.realpath(rp))
        try:
            real.relative_to(os.path.realpath(root))
        except ValueError:
            return str(root)
        return str(real)
    except (ValueError, OSError):
        return str(root)


def _tail_truncate(text: str, max_chars: int, *, label: str) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    chunk = text[-max_chars:]
    nl = chunk.find("\n")
    if 0 < nl < min(160, len(chunk) // 4):
        chunk = chunk[nl + 1 :]
    return (
        f"--- {label} truncated (showing last ~{max_chars} chars) ---\n" + chunk,
        True,
    )


class ShellTool(Tool):
    name = "shell"
    description = (
        "Runs a user-approved shell command in the workspace (stdout/stderr captured)."
    )

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        if action == "run_command":
            return await self.run_command(**kwargs)
        return ToolResult(
            tool=self.name,
            action=action,
            success=False,
            error=f"Unknown shell action: {action!r}",
        )

    async def run_command(
        self,
        command: str | None = None,
        cwd: str | None = None,
        title: str | None = None,
        timeout_s: float | None = None,
        confirm_shell_callback: Any = None,
        long_run_mode: bool = False,
        **_: Any,
    ) -> ToolResult:
        cmd = (command or "").strip()
        if not cmd:
            return ToolResult(
                tool=self.name,
                action="run_command",
                success=False,
                error="Missing non-empty 'command' string.",
            )

        settings = get_settings()
        if long_run_mode and not getattr(settings, "LONG_RUN_SHELL_ENABLED", False):
            return ToolResult(
                tool=self.name,
                action="run_command",
                success=False,
                error=(
                    "Shell is disabled during long-run tasks (set LONG_RUN_SHELL_ENABLED=true "
                    "to allow, with UI confirmation)."
                ),
                retryable=True,
                data={"long_run_blocked": True},
            )

        ws = self.workspace_root
        run_cwd = _resolve_cwd(ws, cwd)
        to = float(timeout_s) if timeout_s is not None else _DEFAULT_TIMEOUT_S
        to = max(1.0, min(to, 600.0))

        risk = classify_shell_risk(cmd)
        al = load_shell_allowlist(ws)
        allowlisted = risk == "read" and read_allowlist_allows_command(cmd, al)

        warning: str | None = None
        if risk == "destructive":
            warning = (
                "[bold red]Destructive command[/bold red] — "
                f"{shell_risk_label(risk)}. Confirm only if you trust this exact line."
            )
        elif risk == "write":
            warning = (
                "[bold yellow]Writes or changes state[/bold yellow] — "
                f"{shell_risk_label(risk)}. Not auto-approved from prefix allowlist."
            )

        need_prompt = True
        if allowlisted and risk == "read":
            need_prompt = False
        elif risk in ("write", "destructive"):
            need_prompt = True

        choice = "run"
        if need_prompt:
            if confirm_shell_callback is None:
                return ToolResult(
                    tool=self.name,
                    action="run_command",
                    success=False,
                    error=(
                        "Terminal commands require an interactive session with UI approval "
                        "(no confirm_shell_callback)."
                    ),
                    retryable=True,
                    data={"risk": risk},
                )
            try:
                await EventBus.emit(
                    {
                        "type": "shell_command_proposed",
                        "command": cmd,
                        "cwd": run_cwd,
                        "title": title or "",
                        "risk": risk,
                    }
                )
            except Exception:
                pass
            exact_cands, prefix_cands = suggest_allowlist_tokens(cmd)
            tag_labels = list(dict.fromkeys(exact_cands + prefix_cands))
            choice = await invoke_confirm_shell_callback(
                confirm_shell_callback,
                cmd,
                title=title,
                cwd=run_cwd,
                tag_labels=tag_labels,
                warning=warning,
                risk=risk,
            )
            if choice == "skip":
                return ToolResult(
                    tool=self.name,
                    action="run_command",
                    success=False,
                    error="User skipped this command.",
                    data={"skipped": True, "command": cmd, "cwd": run_cwd, "risk": risk},
                )
            if choice == "allowlist_run":
                merge_shell_allowlist(
                    ws,
                    exact_add=exact_cands,
                    prefix_add=prefix_cands,
                )

        code, out, err = await self._run_subprocess(cmd, run_cwd, to)
        out, out_cut = _tail_truncate(out, _MAX_CAPTURE_CHARS, label="stdout")
        err, err_cut = _tail_truncate(err, _MAX_CAPTURE_CHARS, label="stderr")
        ok = code == 0
        err_msg = None if ok else (err.strip() or f"exit code {code}")
        return ToolResult(
            tool=self.name,
            action="run_command",
            success=ok,
            data={
                "command": cmd,
                "cwd": run_cwd,
                "exit_code": code,
                "stdout": out,
                "stderr": err,
                "allowlisted": allowlisted,
                "risk": risk,
            },
            error=err_msg,
            metadata={
                "timeout_s": to,
                "exit_code": code,
                "risk": risk,
                "stdout_truncated": out_cut,
                "stderr_truncated": err_cut,
            },
        )

    async def _run_subprocess(
        self, command: str, cwd: str, timeout_s: float
    ) -> tuple[int, str, str]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_CAPTURE_CHARS + 64_000,
            )
        except NotImplementedError:
            logger.error(
                "shell_tool: event loop cannot spawn subprocesses (Windows needs "
                "ProactorEventLoop — use app.cli.main entrypoint)"
            )
            return -1, "", (
                "Subprocesses are not supported by this asyncio event loop on Windows. "
                "Restart via `deepcode` / `python -m app.cli.main` (Proactor policy)."
            )
        try:
            raw_out, raw_err = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            logger.warning("shell_tool: timeout after %.1fs: %s", timeout_s, command[:80])
            return -1, "", f"Command timed out after {timeout_s:.0f}s"

        out = (raw_out or b"").decode(errors="replace")
        err = (raw_err or b"").decode(errors="replace")
        code = int(proc.returncode or 0)
        return code, out, err
