"""
Minimal line-oriented shell for the IDE bottom panel (not a full PTY).

Windows: cmd.exe  ·  Unix: /bin/bash --noprofile
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Callable

from rich.markup import escape


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
