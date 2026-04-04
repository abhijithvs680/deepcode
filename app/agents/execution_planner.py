import logging
import json
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class ExecutionPlanner(BaseAgent):
    """
    Phase 2: Multi-Step Execution Engine.
    Breaks complex user queries into deterministic sequential steps reducing context explosion
    and providing strict execution boundaries natively.
    """

    def __init__(self, provider):
        super().__init__("ExecutionPlanner")
        self._provider = provider
        self.settings = get_settings()

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        question = state.get("query", "")
        mode = state.get("mode", "ask")
        
        # Fast path overrides bypassing LLM overhead
        if mode == "ask" or state.get("classification", {}).get("confidence", 10.0) < 0.6:
            logger.info("[ExecutionPlanner] Skipping step generation natively for standard Ask/Low Confidence queries!")
            state["execution_steps"] = [question]
            return state

        sys_prompt = (
            "You are an expert software orchestration architect.\n"
            "Your job is to break the user's intent into a deterministic list of steps.\n"
            "Keep the step count strictly between 1 to 5 max. Output VALID JSON strictly without markdown wrappers.\n"
            "Schema:\n"
            "{\n"
            "  \"steps\": [\"step 1 description\", ...]\n"
            "}\n"
        )
        
        # Inject Gap 6 Memory bounds limits
        memories = state.get("memories", [])
        if memories:
            memo_summary = "\\n".join([f"- {m['summary']}" for m in memories[:5]])
            sys_prompt += f"\n\nContext Memory Summaries:\n{memo_summary}"
            
        from app.agents.query_expander import expand_query
        from app.agents.tool_router import resolve_tool_routing
        queries = expand_query(question)
        tool_name, tool_conf = await resolve_tool_routing(question, queries, self._provider)
        
        if tool_name and tool_conf > 0.6:
            sys_prompt += f"\n\nSystem Tool Target Preference:\nThe orchestrator predicts '{tool_name}' is required to satisfy this request (Confidence: {tool_conf}). Bias step 1 natively utilizing this route gracefully!"
            
        full_prompt = f"{sys_prompt}\n\nUser Query:\n{question}"

        try:
            logger.info("Generating Execution Plans...")
            response = await self._provider.generate(full_prompt)
            # Rip out markdown bounds safely
            clean_res = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_res)
            steps = data.get("steps", [])
            
            if not steps:
                raise ValueError("JSON parsed successfully but 'steps' list was missing natively!")
                
            # Limit strictly avoiding looping explosions naturally!
            state["execution_steps"] = steps[:5]
            logger.info(f"[ExecutionPlanner] Generated {len(state['execution_steps'])} bounded execution steps!")
            
        except Exception as e:
            logger.warning(f"[ExecutionPlanner] Matrix extraction failed ({e}), defaulting to single step!")
            state["execution_steps"] = [question]

        return state
