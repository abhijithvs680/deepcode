"""
Aggregates per-query telemetry into actionable metrics (Week 11).

Persists rolling state to disk so adaptive logic survives process restarts.
Week 11.5: EMA-smoothed signals + atomic persistence.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    xs = sorted(sorted_vals)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return float(xs[f] + (k - f) * (xs[c] - xs[f]))


class MetricsEngine:
    """
    In-memory aggregation + JSON snapshot for dashboards and adaptive control.

    Tracks tool usage/success, router confidence samples, critic retry counts,
    and end-to-end latencies for percentile and rate calculations.
    """

    def __init__(self, state_path: str = ".ai_assistant/logs/metrics_state.json"):
        self.state_path = state_path
        self.tool_usage: dict[str, int] = {}
        self.tool_success: dict[str, int] = {}
        self.router_confidence: list[float] = []
        self.critic_retries: list[int] = []
        self.latencies: list[float] = []
        self.total_queries: int = 0
        self.queries_with_critic_retry: int = 0
        # EMA-smoothed (Week 11.5 — damp oscillation in adaptive rules)
        self.ema_latency_ms: float | None = None
        self.ema_router_confidence: float | None = None
        self.ema_retry_pressure: float | None = None

    def ingest(self, trace: dict[str, Any]) -> None:
        """Consume one telemetry record (same shape as export_metrics payload)."""
        from app.core.config import get_settings

        alpha = float(get_settings().ADAPTIVE_EMA_ALPHA)

        self.total_queries += 1

        tool = trace.get("planner_tool") or trace.get("selected_tool")
        if tool:
            self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1
            status = trace.get("tool_result_status")
            count = int(trace.get("tool_count") or 0)
            ok = status in (None, "success") and count > 0
            if status == "empty":
                ok = False
            if status == "error" or status == "timeout":
                ok = False
            if ok:
                self.tool_success[tool] = self.tool_success.get(tool, 0) + 1

        rc = trace.get("router_confidence")
        if rc is not None:
            try:
                rv = float(rc)
                self.router_confidence.append(rv)
                if self.ema_router_confidence is None:
                    self.ema_router_confidence = rv
                else:
                    self.ema_router_confidence = (
                        alpha * rv + (1.0 - alpha) * self.ema_router_confidence
                    )
            except (TypeError, ValueError):
                pass

        retries = int(trace.get("critic_retries") or 0)
        self.critic_retries.append(retries)
        if retries > 0:
            self.queries_with_critic_retry += 1

        retry_i = 1.0 if retries > 0 else 0.0
        if self.ema_retry_pressure is None:
            self.ema_retry_pressure = retry_i
        else:
            self.ema_retry_pressure = (
                alpha * retry_i + (1.0 - alpha) * self.ema_retry_pressure
            )

        lat = trace.get("latency_ms")
        if lat is not None:
            try:
                lv = float(lat)
                self.latencies.append(lv)
                if self.ema_latency_ms is None:
                    self.ema_latency_ms = lv
                else:
                    self.ema_latency_ms = alpha * lv + (1.0 - alpha) * self.ema_latency_ms
            except (TypeError, ValueError):
                pass

    def tool_success_rate(self, tool: str) -> float | None:
        u = self.tool_usage.get(tool, 0)
        if u == 0:
            return None
        s = self.tool_success.get(tool, 0)
        return s / u

    def avg_router_confidence(self) -> float | None:
        if not self.router_confidence:
            return None
        return sum(self.router_confidence) / len(self.router_confidence)

    def critic_retry_rate(self) -> float:
        if self.total_queries <= 0:
            return 0.0
        return self.queries_with_critic_retry / self.total_queries

    def latency_percentiles(self) -> dict[str, float]:
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0}
        s = sorted(self.latencies)
        return {"p50": _percentile(s, 50), "p95": _percentile(s, 95)}

    def snapshot(self) -> dict[str, Any]:
        rates = {}
        for t, u in self.tool_usage.items():
            if u:
                rates[t] = round(self.tool_success.get(t, 0) / u, 4)
        p = self.latency_percentiles()
        return {
            "total_queries": self.total_queries,
            "tool_usage": dict(self.tool_usage),
            "tool_success": dict(self.tool_success),
            "tool_success_rate": rates,
            "avg_router_confidence": round(self.avg_router_confidence() or 0.0, 4),
            "critic_retry_rate": round(self.critic_retry_rate(), 4),
            "latency_percentiles": p,
            "ema": {
                "latency_ms": round(self.ema_latency_ms, 2) if self.ema_latency_ms is not None else None,
                "router_confidence": round(self.ema_router_confidence, 4)
                if self.ema_router_confidence is not None
                else None,
                "retry_pressure": round(self.ema_retry_pressure, 4)
                if self.ema_retry_pressure is not None
                else None,
            },
        }

    def save(self) -> str | None:
        """Persist aggregate state + snapshot (atomic replace to avoid torn JSON)."""
        abs_path = os.path.abspath(self.state_path)
        dir_name = os.path.dirname(abs_path) or "."
        os.makedirs(dir_name, exist_ok=True)
        payload = {
            "tool_usage": self.tool_usage,
            "tool_success": self.tool_success,
            "router_confidence": self.router_confidence[-5000:],
            "critic_retries": self.critic_retries[-5000:],
            "latencies": self.latencies[-5000:],
            "total_queries": self.total_queries,
            "queries_with_critic_retry": self.queries_with_critic_retry,
            "ema_latency_ms": self.ema_latency_ms,
            "ema_router_confidence": self.ema_router_confidence,
            "ema_retry_pressure": self.ema_retry_pressure,
            "snapshot": self.snapshot(),
        }
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix="metrics_",
                dir=dir_name,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, abs_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.debug("[MetricsEngine] Saved metrics state to %s", abs_path)
        except OSError as e:
            logger.warning("[MetricsEngine] Failed to save: %s", e)
            return None
        return abs_path

    def load(self) -> None:
        """Restore from disk (no-op if missing or corrupt)."""
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[MetricsEngine] Failed to load: %s", e)
            return
        self.tool_usage = dict(data.get("tool_usage") or {})
        self.tool_success = dict(data.get("tool_success") or {})
        self.router_confidence = list(data.get("router_confidence") or [])
        self.critic_retries = list(data.get("critic_retries") or [])
        self.latencies = list(data.get("latencies") or [])
        self.total_queries = int(data.get("total_queries") or 0)
        self.queries_with_critic_retry = int(data.get("queries_with_critic_retry") or 0)
        em = data.get("ema_latency_ms")
        self.ema_latency_ms = float(em) if em is not None else None
        er = data.get("ema_router_confidence")
        self.ema_router_confidence = float(er) if er is not None else None
        ep = data.get("ema_retry_pressure")
        self.ema_retry_pressure = float(ep) if ep is not None else None


metrics_engine = MetricsEngine()
