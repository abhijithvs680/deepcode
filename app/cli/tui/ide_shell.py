"""
Minimal line-oriented shell for the IDE bottom panel (not a full PTY).

Windows: cmd.exe  ·  Unix: /bin/bash --noprofile

The in-panel ``ShellPromptLine`` types on the same row as the prompt (VS Code–style),
instead of a separate Input below the log.
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static


class IdeSubprocessShell:
    def __init__(
        self,
        *,
        cwd: str | None,
        on_line: Callable[[str], None],
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._on_line = on_line
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            if sys.platform == "win32":
                self._proc = await asyncio.create_subprocess_exec(
                    "cmd.exe",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self._cwd,
                    env=os.environ.copy(),
                )
            else:
                self._proc = await asyncio.create_subprocess_exec(
                    "/bin/bash",
                    "--noprofile",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self._cwd,
                    env=os.environ.copy(),
                )
            self._reader_task = asyncio.create_task(self._read_stdout())

    async def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    text = str(line)
                self._on_line(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._on_line(f"[shell read error] {exc}")

    async def send_line(self, line: str) -> None:
        await self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or not self.running:
            self._on_line("(shell not ready)")
            return
        data = (line.rstrip("\r\n") + os.linesep).encode("utf-8", errors="replace")
        proc.stdin.write(data)
        await proc.stdin.drain()

    async def stop(self) -> None:
        async with self._lock:
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass
            self._reader_task = None
            proc = self._proc
            self._proc = None
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


def format_shell_line(line: str) -> str:
    return escape(line[:4000])


class ShellPromptLine(Static):
    """Focus here to type shell commands; Enter submits (line-oriented shell)."""

    can_focus = True
    DEFAULT_CSS = """
    ShellPromptLine {
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
        border-top: solid $primary 35%;
    }
    ShellPromptLine:focus {
        background: $panel;
        border-top: solid $accent 50%;
    }
    """

    class CommandSubmitted(Message):
        """Bubble to app: user finished a command line."""

        bubble = True

        def __init__(self, line: str) -> None:
            self.line = line
            super().__init__()

    def __init__(
        self,
        tip_fn: Callable[[], str],
        **kwargs: Any,
    ) -> None:
        self._tip_fn = tip_fn
        self._buf = ""
        super().__init__("", shrink=False, **kwargs)
        self._paint()

    def refresh_tip(self) -> None:
        self._paint()

    def _prompt_tip(self) -> str:
        try:
            t = (self._tip_fn() or "").rstrip()
            if not t:
                return "$ "
            if t.endswith("$") or t.endswith(">"):
                return t + " "
            return t + "$ "
        except Exception:
            return "$ "

    def _paint(self) -> None:
        t = Text()
        t.append(self._prompt_tip(), style="bold green")
        t.append(self._buf)
        t.append("▍", style="dim")
        self.update(t)

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "enter":
            line = self._buf
            self._buf = ""
            self._paint()
            self.post_message(self.CommandSubmitted(line))
            event.prevent_default()
            return
        if key in ("backspace", "delete"):
            if key == "backspace" and self._buf:
                self._buf = self._buf[:-1]
            self._paint()
            event.prevent_default()
            return
        if key == "ctrl+u":
            self._buf = ""
            self._paint()
            event.prevent_default()
            return
        if event.is_printable and event.character:
            if len(self._buf) < 16_000:
                self._buf += event.character
                self._paint()
            event.prevent_default()
