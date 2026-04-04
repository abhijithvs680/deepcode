"""
Run benchmark JSON against the live orchestrator (tool routing, latency, retries).

Example:
    deepcode eval evals/benchmark.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.agents.orchestrator import AgentOrchestrator


def _tool_match(expected: str | None, predicted: str | None) -> bool:
    if not expected:
        return True
    return (predicted or "").lower() == expected.lower()


def _answer_heuristic(answer: str, case: dict[str, Any]) -> bool | None:
    """Optional lightweight check: keywords or substring."""
    if "expected_substring" in case:
        return case["expected_substring"].lower() in (answer or "").lower()
    keys = case.get("expected_keywords")
    if not keys:
        return None
    low = (answer or "").lower()
    return all(str(k).lower() in low for k in keys)


async def run_benchmark(
    benchmark_path: str | Path,
    *,
    db=None,
) -> dict[str, Any]:
    path = Path(benchmark_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Benchmark file must be a JSON array of cases")

    orchestrator = AgentOrchestrator(db=db)

    tool_correct = 0
    tool_total = 0
    answer_checked = 0
    answer_ok_n = 0
    latencies: list[float] = []
    retries: list[int] = []

    details: list[dict[str, Any]] = []

    for case in raw:
        if not isinstance(case, dict) or "query" not in case:
            continue
        q = str(case["query"])
        t0 = time.perf_counter()
        result = await orchestrator.run(question=q, source="eval")
        wall_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(wall_ms)

        meta = result.get("meta") or {}
        retries.append(int(meta.get("critic_retry_count") or 0))
        pred = meta.get("selected_tool")

        row: dict[str, Any] = {
            "query": q,
            "type": case.get("type", "unknown"),
            "predicted_tool": pred,
            "latency_ms": round(wall_ms, 2),
            "critic_retries": meta.get("critic_retry_count", 0),
        }

        if case.get("type") == "tool" and case.get("expected_tool"):
            tool_total += 1
            if _tool_match(case.get("expected_tool"), pred):
                tool_correct += 1
                row["tool_match"] = True
            else:
                row["tool_match"] = False

        ah = _answer_heuristic(result.get("answer") or "", case)
        if ah is not None:
            answer_checked += 1
            if ah:
                answer_ok_n += 1
            row["answer_ok"] = ah

        details.append(row)

    n = len(latencies) or 1
    tool_accuracy = (tool_correct / tool_total) if tool_total else None
    answer_accuracy = (answer_ok_n / answer_checked) if answer_checked else None
    avg_latency = sum(latencies) / n
    retry_rate = sum(1 for r in retries if r > 0) / n

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    p95 = sorted_lat[int(max(0, len(sorted_lat) * 0.95 - 1))] if sorted_lat else 0.0

    return {
        "cases": len(details),
        "tool_accuracy": tool_accuracy,
        "answer_accuracy": answer_accuracy,
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "retry_rate": round(retry_rate, 4),
        "details": details,
    }
