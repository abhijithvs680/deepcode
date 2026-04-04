import logging
from typing import Any
import os

from app.agents.base_agent import BaseAgent
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

class ToolExecutionAgent(BaseAgent):
    """
    Safely executes boundaries wrapping explicitly resolved targets via ToolRegistry.
    Handles precise Log/File/Git scrapes bypassing RAG bounds.
    """
    def __init__(self, workspace_root: str = os.getcwd()):
        super().__init__("ToolExecutor")
        self.registry = ToolRegistry(workspace_root)

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        target = state.get("selected_tool")
        query = state.get("query", "")
        
        state["tool_results"] = []
        if not target:
            return state

        logger.info(f"[{self.name}] Activating deterministic explicit channel: {target.upper()}")
        
        # Simple heuristic arguments (in production an LLM JSON parsing node extracts these args)
        kwargs = {}
        action = ""
        if target == "git":
            if "blame" in query:
                action = "git_blame"
                # very crude fallback pulling file heuristically
                kwargs["file_path"] = [w for w in query.split() if "." in w][0] if "." in query else "app/main.py" 
            else:
                action = "git_log"
                kwargs["n"] = 10
        elif target == "log":
            if "error" in query or "failure" in query:
                action = "get_recent_failures"
            else:
                action = "search_logs"
                kwargs["query"] = query
        elif target == "file":
             action = "search"
             kwargs["query"] = query

        if action:
            res = await self.registry.execute_tool(target, action, **kwargs)
            logger.info(f"[{self.name}] Executed mapping '{action}'. Success: {res.success}")
            if res.success:
                state["tool_results"].append(res.model_dump())
                # Assign status identifying explicit success boundaries!
                state["tool_result_status"] = "success" if res.data or res.metadata else "empty"
            else:
                state["tool_result_status"] = "error"
        else:
            state["tool_result_status"] = "error"
        
        return state
