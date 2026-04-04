"""
Derives runtime knobs from aggregated metrics (Week 11 adaptive loop).

Orchestrator injects ``state["adaptive_config"]``; retriever / planner / router read it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.telemetry.metrics_engine import metrics_engine

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Serializable tuning payload attached to pipeline state."""

    rag_top_k: int = 10
    top_files: int = 12
    top_chunks: int = 15
    max_retries: int = 2
    router_confidence_threshold: float = 0.6
    tool_boost: dict[str, float] = field(default_factory=dict)
    prefer_llm_router: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "rag_top_k": self.rag_top_k,
            "top_files": self.top_files,
            "top_chunks": self.top_chunks,
            "max_retries": self.max_retries,
            "router_confidence_threshold": self.router_confidence_threshold,
            "tool_boost": dict(self.tool_boost),
            "prefer_llm_router": self.prefer_llm_router,
        }


class AdaptiveController:
    """
    Loads metrics history, computes AdaptiveConfig, and exposes ``get_config``.

    Call :meth:`prepare` once per orchestrator run (or at process startup).
    Call :meth:`update_from_metrics` after each completed query to refresh knobs.
    """

    def __init__(self, engine: Any | None = None):
        self._engine = engine or metrics_engine
        self._config = AdaptiveConfig()

    def load_metrics(self) -> None:
        self._engine.load()

    def update_from_metrics(self) -> None:
        self._config = self._compute()

    def prepare(self) -> None:
        """Load persisted aggregates and compute current config."""
        self.load_metrics()
        self.update_from_metrics()

    def get_config(self) -> dict[str, Any]:
        return self._config.as_dict()

    def _compute(self) -> AdaptiveConfig:
        st = get_settings()
        base_k = 10
        base_files = 12
        base_chunks = 15
        max_retries = 2
        threshold = 0.6
        tool_boost: dict[str, float] = {}
        prefer_llm = False

        n = self._engine.total_queries
        if n < st.ADAPTIVE_COLD_START_QUERIES:
            logger.debug(
                "[Adaptive] Cold start (queries=%s < %s) → conservative baseline",
                n,
                st.ADAPTIVE_COLD_START_QUERIES,
            )
            return AdaptiveConfig()

        snap = self._engine.snapshot()
        ema = snap.get("ema") or {}
        p95 = snap.get("latency_percentiles", {}).get("p95") or 0.0
        latency_signal = ema.get("latency_ms")
        if latency_signal is None:
            latency_signal = p95
        else:
            latency_signal = float(latency_signal)

        avg_rc = snap.get("avg_router_confidence") or 0.0
        ema_rc = ema.get("router_confidence")
        router_signal = float(ema_rc) if ema_rc is not None else float(avg_rc)

        retry_rate = snap.get("critic_retry_rate") or 0.0
        ema_retry = ema.get("retry_pressure")
        retry_signal = float(ema_retry) if ema_retry is not None else float(retry_rate)

        # Latency: prefer EMA; high tail → trim RAG breadth
        if latency_signal > 20_000.0:
            base_k = max(3, base_k - st.ADAPTIVE_RAG_TOP_K_TRIM)
            base_files = max(4, base_files - 2)
            base_chunks = max(6, base_chunks - 3)
            logger.info(
                "[Adaptive] High smoothed latency (%.0fms) → rag_top_k=%s",
                latency_signal,
                base_k,
            )

        # Low router confidence (EMA when available)
        if router_signal > 0 and router_signal < 0.6:
            threshold = 0.45
            prefer_llm = True
            logger.info(
                "[Adaptive] Low smoothed router confidence (%.2f) → threshold=%s",
                router_signal,
                threshold,
            )

        # Retry pressure (EMA of per-query retry indicator vs raw rate)
        if retry_signal > 0.4:
            max_retries = 1
            logger.info(
                "[Adaptive] High smoothed retry pressure (%.2f) → max_retries=1",
                retry_signal,
            )

        # Tool reliability → boost routing prior for that tool
        for tool in ("git", "file", "log"):
            sr = self._engine.tool_success_rate(tool)
            if sr is not None and sr > 0.9:
                tool_boost[tool] = round(tool_boost.get(tool, 0.0) + 0.1, 2)

        return AdaptiveConfig(
            rag_top_k=base_k,
            top_files=base_files,
            top_chunks=base_chunks,
            max_retries=max_retries,
            router_confidence_threshold=threshold,
            tool_boost=tool_boost,
            prefer_llm_router=prefer_llm,
        )


adaptive_controller = AdaptiveController()
