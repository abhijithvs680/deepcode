"""Chat formatting helpers (Rich markup + safe display)."""
from __future__ import annotations

import asyncio
import re

from rich.markup import escape


def user_block(text: str) -> str:
    return f"[bold green]You[/bold green]\n{escape(text)}"


def assistant_label() -> str:
    return "[bold cyan]DeepCode[/bold cyan]"


def system_block(text: str) -> str:
    return f"[yellow]{escape(text)}[/yellow]"


def error_block(text: str) -> str:
    return f"[bold red]Error[/bold red]\n{escape(text)}"


def dim_block(text: str) -> str:
    return f"[dim]{escape(text)}[/dim]"


# Chunk by words for a streaming feel without true LLM token streaming yet
_WORD_SPLIT = re.compile(r"(\s+)")


async def stream_plain_text(
    update_cb,
    full_text: str,
    *,
    delay_s: float = 0.012,
) -> None:
    """Call async update_cb(accumulated: str) with growing text (word-bounded)."""
    if not full_text:
        await update_cb("")
        return
    parts = _WORD_SPLIT.split(full_text)
    acc = ""
    for p in parts:
        acc += p
        await update_cb(acc)
        if delay_s > 0:
            await asyncio.sleep(delay_s)
