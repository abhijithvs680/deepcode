import json
import logging
from typing import Any
import asyncio

from app.agents.base_agent import BaseAgent
from app.core.database import async_session_factory
from app.memory.intel_store import store_decision, store_bug

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """
    Evaluates Answer outputs defensively parsing potential logic loops and executing
    background memory updates autonomously out of bound of user interactions.
    """

    def __init__(self, db, provider):
        super().__init__("Critic")
        self._db = db
        self._provider = provider

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Runs validation checks over the LLM output boundaries. 
        Additionally extracts decision/bug schemas autonomously via low-temperature evaluations.
        """
        response = state.get("answer", "")
        repo_id = state.get("repo_id")
        trace = state.get("trace")

        if state.get("skip_critic_llm"):
            logger.info(f"[{self.name}] Skipping critic LLM (high-confidence tool path).")
            state["critic_retry"] = False
            return state

        logger.info(f"[{self.name}] Scrutinizing Answer payload integrities...")

        # For Week 6 MVP, basic formatting checks could run here
        # E.g. state['critic_retry'] = True if output entirely missed constraints.
        state["critic_retry"] = False 

        # Compute maximum deterministic tool confidence directly inside State variables
        from app.tools.tool_scoring import score_tool_result
        tool_results = state.get("tool_results", [])
        max_tool_score = 0.0
        for tr_dict in tool_results:
            try:
                from app.tools.base_tool import ToolResult
                max_tool_score = max(max_tool_score, score_tool_result(ToolResult(**tr_dict)))
            except Exception: pass
            
        state["max_tool_score"] = float(max_tool_score)

        # Fire off autonomous background memory classification
        question = state.get("query", "")
        context = "".join([c.get("content", "") for c in state.get("ranked_ctx", [])])
        
        if self._db and repo_id and response:
            await self._filter_and_log_memory(state, repo_id, response, trace, question, context)

        return state

    async def _filter_and_log_memory(self, state: dict[str, Any], repo_id: str, response: str, trace: Any, question: str, context: str):
        """Asynchronous worker filtering trace strings out from the inference matrix."""
        classify_prompt = (
            "You are a strict Code Critic reviewing an Agent's generated answer.\n"
            "Review the Question, Context and Response. Evaluate if the answer fully resolves the question or if it lacks context/hallucinated.\n"
            "Reply strictly in JSON mapped exactly to this schema:\n"
            "{\n"
            "  \"is_adequate\": true | false,\n"
            "  \"problem\": \"Describe what is missing or wrong (if any)\",\n"
            "  \"action\": \"expand_retrieval\" | \"use_tool\" | \"retry\" | \"accept\",\n"
            "  \"suggested_tool\": \"git\" | \"log\" | \"file\" | null,\n"
            "  \"confidence\": 0.95,\n"
            "  \"is_decision\": true,\n"
            "  \"decision_context\": \"... symbol or context ...\"\n"
            "}\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Response:\n{response[:2000]}"
        )
        try:
            res_eval = await self._provider.generate(classify_prompt, model=self._provider.model)
            payload = {}
            if "{" in res_eval and "}" in res_eval:
                try:
                    json_str = res_eval[res_eval.find("{"):res_eval.rfind("}")+1]
                    payload = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            if not payload:
                payload = {
                    "is_adequate": True,
                    "problem": "LLM failed explicit JSON serialization constraints. Defaulting secure.",
                    "action": "accept",
                    "suggested_tool": None,
                    "confidence": 0.0
                }
            
            # Validation Step
            valid_actions = {"expand_retrieval", "use_tool", "retry", "accept"}
            is_adequate = payload.get("is_adequate", True)
            action = payload.get("action", "accept")
            if action not in valid_actions:
                action = "accept"
            
            c = float(payload.get("confidence", 0.0))
            suggested_tool = payload.get("suggested_tool")
            
            # Explicit tool failure loop guards
            if state.get("tool_result_status") in ("error", "empty"):
                suggested_tool = None 
            
            if not is_adequate and action != "accept" and c > 0.7:
                logger.warning(f"[{self.name}] Critic forcing action '{action}'. Tool: {suggested_tool}")
                state["critic_retry"] = True
                state["critic_feedback"] = payload
                state["critic_feedback"]["suggested_tool"] = suggested_tool
            else:
                state["critic_retry"] = False
                state["critic_feedback"] = payload
                
                t = "decision" if payload.get("is_decision") else "bug_fix"
                sym = payload.get("decision_context")
                if t in ["decision", "bug_fix"]:
                    max_tool_score = state.get("max_tool_score", 0.0)
                    
                    if float(c) >= 0.9 or max_tool_score > 0.85:
                        async with async_session_factory() as local_db:
                            if t == "decision":
                                await store_decision(local_db, repo_id, sym, response[:2000])
                            else:
                                await store_bug(local_db, repo_id, sym, response[:2000])
                        logger.info(f"[{self.name}] Autonomously logged memory type '{t}' with confidence {c} for target '{sym}'")
                    if trace:
                        trace.memory_autolog.append({"type": t, "symbol": sym, "confidence": c})
        except Exception as e:
            logger.warning(f"[{self.name}] Component classification failed: {e}")
