"""Invoke optional async confirm_diff callbacks with backward-compatible signatures."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

ConfirmDiffCallback = Callable[..., Any]


async def invoke_confirm_diff_callback(
    confirm_cb: ConfirmDiffCallback | None,
    combined_diff: str,
    *,
    confirm_reasons: list[str] | None = None,
) -> bool:
    if confirm_cb is None:
        return True
    try:
        sig = inspect.signature(confirm_cb)
    except (TypeError, ValueError):
        out = confirm_cb(combined_diff)
    else:
        if "confirm_reasons" in sig.parameters:
            out = confirm_cb(combined_diff, confirm_reasons=confirm_reasons)
        else:
            out = confirm_cb(combined_diff)
    if inspect.isawaitable(out):
        return bool(await out)
    return bool(out)
