"""
Lightweight in-process event bus for agent / TUI streaming (ReAct steps, etc.).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

EventListener = Callable[[dict[str, Any]], Any]


class EventBus:
    """Sync or async listeners; safe to emit from async code."""

    _listeners: list[EventListener] = []

    @classmethod
    def subscribe(cls, fn: EventListener) -> None:
        if fn not in cls._listeners:
            cls._listeners.append(fn)

    @classmethod
    def unsubscribe(cls, fn: EventListener) -> None:
        cls._listeners = [f for f in cls._listeners if f is not fn]

    @classmethod
    def clear(cls) -> None:
        cls._listeners.clear()

    @classmethod
    async def emit(cls, event: dict[str, Any]) -> None:
        for fn in list(cls._listeners):
            try:
                out = fn(event)
                if asyncio.iscoroutine(out):
                    await out
            except Exception as exc:
                logger.debug("EventBus listener error: %s", exc)
