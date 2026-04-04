import logging
import json
import os
from datetime import UTC, datetime
from typing import Any, Dict

from app.telemetry.metrics_engine import metrics_engine

logger = logging.getLogger(__name__)

class TelemetryTracker:
    """
    Standard matrix evaluating metrics dynamically across pipeline spans securely 
    persisted across bounded standard logs.
    """
    def __init__(self, log_dir: str = ".ai_assistant/logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def export_metrics(self, session_id: str, state_dict: Dict[str, Any]) -> str:
        """
        Extract latency limits, planner decisions, critic overlaps, and evaluator matrices dynamically
        formatting tracking payloads seamlessly for Dashboarding targets!
        """
        t = datetime.now(UTC)
        stamp = t.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/trace_{session_id}_{stamp}.json"
        
        # Scrape operational nodes natively limiting massive text blocks safely!
        metrics = {
            "session_id": str(session_id),
            "timestamp": t.isoformat(),
            "query": state_dict.get("query"),
            "planner_tool": state_dict.get("selected_tool"),
            "selected_tool": state_dict.get("selected_tool"),
            "tool_routing": state_dict.get("tool_routing"),
            "router_confidence": state_dict.get("router_confidence"),
            "router_source": state_dict.get("router_source"),
            "tool_count": len(state_dict.get("tool_results", [])),
            "tool_result_status": state_dict.get("tool_result_status"),
            "critic_retries": state_dict.get("critic_retry_count", 0),
            "latency_ms": state_dict.get("latency_ms", 0.0),
            "model": state_dict.get("model", "unknown"),
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[Telemetry] Successfully dumped execution metrics into {filename}")
        except Exception as e:
            logger.error(f"[Telemetry] Failed dumping metrics bounds: {e}")

        try:
            metrics_engine.ingest(metrics)
            metrics_engine.save()
            from app.agents.adaptive_controller import adaptive_controller

            adaptive_controller.update_from_metrics()
        except Exception as e:
            logger.warning("[Telemetry] Metrics/adaptive update failed: %s", e)

        return filename

# Global Singleton for quick matrix binding pipelines
telemetry_tracker = TelemetryTracker()
