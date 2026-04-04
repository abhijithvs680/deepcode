"""Week 11.5 adaptive cold-start and EMA-driven rules."""
from app.agents.adaptive_controller import AdaptiveController, AdaptiveConfig
from app.telemetry.metrics_engine import MetricsEngine


def test_cold_start_returns_baseline():
    eng = MetricsEngine(state_path="__no_save__.json")
    eng.total_queries = 5
    ctrl = AdaptiveController(engine=eng)
    cfg = ctrl._compute()
    assert isinstance(cfg, AdaptiveConfig)
    assert cfg.rag_top_k == 10
    assert cfg.max_retries == 2
    assert cfg.tool_boost == {}


def test_after_cold_start_latency_trims_when_ema_high():
    eng = MetricsEngine(state_path="__no_save__.json")
    eng.total_queries = 25
    eng.ema_latency_ms = 25_000.0
    ctrl = AdaptiveController(engine=eng)
    cfg = ctrl._compute()
    assert cfg.rag_top_k < 10
