"""Tests for Week 11 metrics aggregation."""
import json
import os
import tempfile

import pytest

from app.telemetry.metrics_engine import MetricsEngine


def test_percentile_and_ingest():
    m = MetricsEngine(state_path="__does_not_exist__/noop.json")
    m.latencies = [10.0, 20.0, 30.0, 40.0, 100.0]
    p = m.latency_percentiles()
    assert p["p50"] >= 20.0
    assert p["p95"] >= 40.0

    m2 = MetricsEngine()
    m2.ingest(
        {
            "planner_tool": "git",
            "tool_count": 1,
            "tool_result_status": "success",
            "router_confidence": 0.8,
            "critic_retries": 0,
            "latency_ms": 100.0,
        }
    )
    assert m2.total_queries == 1
    assert m2.tool_usage.get("git") == 1
    assert m2.ema_latency_ms == pytest.approx(100.0)
    assert m2.ema_router_confidence == pytest.approx(0.8)
    snap = m2.snapshot()
    assert "git" in snap["tool_success_rate"]
    assert snap["ema"]["latency_ms"] is not None


def test_atomic_save_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "metrics_state.json")
        m = MetricsEngine(state_path=path)
        m.ingest(
            {
                "tool_count": 0,
                "critic_retries": 0,
                "latency_ms": 50.0,
            }
        )
        m.save()
        m2 = MetricsEngine(state_path=path)
        m2.load()
        assert m2.total_queries == 1
        assert m2.ema_latency_ms == pytest.approx(50.0)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "ema_latency_ms" in data
