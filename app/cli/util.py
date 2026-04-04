"""Shared helpers for CLI commands."""
import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)
