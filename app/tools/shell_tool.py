"""Run shell commands in the workspace after optional UI approval and allowlist."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.agents.confirm_shell_util import invoke_confirm_shell_callback
from app.agents.shell_allowlist import (
    command_matches_allowlist,
    load_shell_allowlist,
    merge_shell_allowlist,
    suggest_allowlist_tokens,
)
from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 120.0
_MAX_CAPTURE_BYTES = 512_000


def _resolve_cwd(workspace_root: str, cwd: str | None) -> str:
    root = Path(workspace_root).resolve()
    if not cwd or not str(cwd).strip():
        return str(root)
    p = Path(str(cwd).strip())
    if not p.is_absolute():
        p = root / p
    try:
        rp = p.resolve()
        rp.relative_to(root)
        return str(rp)
    except (ValueError, OSError):
        return str(root)


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

        ws = self.workspace_root
        run_cwd = _resolve_cwd(ws, cwd)
        to = float(timeout_s) if timeout_s is not None else _DEFAULT_TIMEOUT_S
        to = max(1.0, min(to, 600.0))

        prefixes = load_shell_allowlist(ws)
        allowlisted = command_matches_allowlist(cmd, prefixes)

        choice = "run"
        if not allowlisted:
            if confirm_shell_callback is None:
                return ToolResult(
                    tool=self.name,
                    action="run_command",
                    success=False,
                    error=(
                        "Terminal commands require an interactive session with UI approval "
                        "(no confirm_shell_callback)."
                    ),
                )
            tags = suggest_allowlist_tokens(cmd)
            choice = await invoke_confirm_shell_callback(
                confirm_shell_callback,
                cmd,
                title=title,
                cwd=run_cwd,
                tag_labels=tags,
            )
            if choice == "skip":
                return ToolResult(
                    tool=self.name,
                    action="run_command",
                    success=False,
                    error="User skipped this command.",
                    data={"skipped": True, "command": cmd, "cwd": run_cwd},
                )
            if choice == "allowlist_run":
                merge_shell_allowlist(ws, tags)

        code, out, err = await self._run_subprocess(cmd, run_cwd, to)
        ok = code == 0
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
            },
            error=None if ok else (err.strip() or f"exit code {code}"),
            metadata={"timeout_s": to},
        )

    async def _run_subprocess(
        self, command: str, cwd: str, timeout_s: float
    ) -> tuple[int, str, str]:
        # Windows: shell=True semantics via asyncio.create_subprocess_shell
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_MAX_CAPTURE_BYTES,
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
        if len(out) > _MAX_CAPTURE_BYTES:
            out = out[:_MAX_CAPTURE_BYTES] + "\n…[stdout truncated]"
        if len(err) > _MAX_CAPTURE_BYTES:
            err = err[:_MAX_CAPTURE_BYTES] + "\n…[stderr truncated]"
        code = int(proc.returncode or 0)
        return code, out, err
