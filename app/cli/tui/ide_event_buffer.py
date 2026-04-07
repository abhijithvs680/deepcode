"""Coalesce high-frequency EventBus events before IDE repaint (~100ms batch)."""
from __future__ import annotations

from typing import Any, Callable


def _select_step_events(
    steps: list[dict[str, Any]], max_n: int
) -> list[dict[str, Any]]:
    """Keep first, last, last failed result (+ its start), then recent middle."""
    if len(steps) <= max_n:
        return steps
    last_fail_i: int | None = None
    for i in range(len(steps) - 1, -1, -1):
        s = steps[i]
        if s.get("phase") != "result":
            continue
        if str(s.get("status") or "").lower() == "fail":
            last_fail_i = i
            break
    keep: set[int] = {0, len(steps) - 1}
    if last_fail_i is not None:
        keep.add(last_fail_i)
        if last_fail_i > 0:
            prev = steps[last_fail_i - 1]
            if (
                prev.get("phase") == "start"
                and prev.get("step") == steps[last_fail_i].get("step")
            ):
                keep.add(last_fail_i - 1)
    rest_i = [i for i in range(len(steps)) if i not in keep]
    need = max_n - len(keep)
    if need > 0:
        keep.update(rest_i[-need:])
    ordered = sorted(keep)
    return [steps[i] for i in ordered]


def normalize_paths(paths: list[str], cap: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        n = str(p or "").replace("\\", "/").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= cap:
            break
    return out


def coalesce_events(
    events: list[dict[str, Any]],
    *,
    max_step_events: int = 48,
) -> list[dict[str, Any]]:
    """Keep last agent_status / state per batch; cap high-frequency step events."""
    out: list[dict[str, Any]] = []
    pending_as: dict[str, Any] | None = None
    pending_state: dict[str, Any] | None = None

    def flush_pending() -> None:
        nonlocal pending_as, pending_state
        if pending_as is not None:
            out.append(pending_as)
            pending_as = None
        if pending_state is not None:
            out.append(pending_state)
            pending_state = None

    for ev in events:
        t = ev.get("type")
        if t == "agent_status":
            pending_as = ev
            continue
        if t == "state":
            pending_state = ev
            continue
        flush_pending()
        out.append(ev)
    flush_pending()
    non_steps = [e for e in out if e.get("type") != "step"]
    steps = [e for e in out if e.get("type") == "step"]
    steps = _select_step_events(steps, max_step_events)
    return non_steps + steps


class IdeEventDebouncer:
    """Single timer; drains queue; reschedules if new events arrived during flush."""

    def __init__(
        self,
        *,
        interval_s: float = 0.1,
        on_flush: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        self.interval_s = interval_s
        self.on_flush = on_flush
        self._queue: list[dict[str, Any]] = []
        self._scheduled = False
        self._app: Any = None

    def bind_app(self, app: Any) -> None:
        self._app = app

    def push(self, event: dict[str, Any]) -> None:
        self._queue.append(event)
        if self._app is None or self._scheduled:
            return
        self._scheduled = True
        self._app.set_timer(self.interval_s, self._flush)

    def _flush(self) -> None:
        self._scheduled = False
        batch = self._queue
        self._queue = []
        if batch:
            self.on_flush(coalesce_events(batch))
        if self._queue and self._app is not None:
            self._scheduled = True
            self._app.set_timer(self.interval_s, self._flush)

    def flush_now(self) -> None:
        """Drain pending events synchronously (e.g. before unsubscribing)."""
        self._scheduled = False
        batch = self._queue
        self._queue = []
        if batch:
            self.on_flush(coalesce_events(batch))
