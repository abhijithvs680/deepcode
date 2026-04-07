import logging
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.query_expander import expand_query, merge_queries_with_repo_hints
from app.agents.tool_router import resolve_tool_routing
from app.memory.intel_store import fetch_memory_context, fetch_recent_commit_hints
from app.memory.retriever import get_query_embedding_cached
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Require this to bind a concrete tool hint (below → explore with ReAct / RAG, no narrow router lock-in).
_ROUTER_BIND_CONFIDENCE = 0.7


class PlannerAgent(BaseAgent):
    """
    Orchestration front-runner. 
    Breaks down user requests into actionable subqueries using fast heuristic mapping,
    prioritizes semantic database embeddings, and retrieves long term memory bounds 
    for the Retriever layer.
    """

    def __init__(self, db, provider):
        super().__init__("Planner")
        self._db = db
        self._provider = provider
        self.settings = get_settings()

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Takes raw 'query' and 'current_step' driving 'sub_tasks', 'memories', and 'query_embedding' into state.
        """
        base_query = state.get("query", "")
        question = state.get("current_step") or base_query
        repo_id = state.get("repo_id")
        trace = state.get("trace")
        embed_cache = state.get("embed_cache", {})

        logger.info(f"[{self.name}] Decomposing query: '{question}'")

        # 1. Expand heuristics
        queries = expand_query(question)
        repo_hints: list[str] = []
        if self._db and repo_id:
            repo_hints = await fetch_recent_commit_hints(
                self._db, repo_id, limit=self.settings.GIT_QUERY_HINTS_CAP
            )
            queries = merge_queries_with_repo_hints(queries, repo_hints, max_total=35)
            
        logger.info(f"[{self.name}] Expanded query into {len(queries)} sub-tasks.")

        # 2. Extract base semantic vector
        base_query_embed = []
        if self._provider:
            # We base vectors strictly across the current step context!
            base_query_embed = await get_query_embedding_cached(question, self._provider, embed_cache)
            state["base_query_embed"] = base_query_embed

        # 3. Pull Multi-Session Memory (Bounded if fetched globally by Execution loop already)
        memories = state.get("memories", [])
        if not memories and self._db and repo_id:
            memories = await fetch_memory_context(
                self._db, repo_id, question, base_query_embed, k=3
            )
        
        state["memories"] = memories
        if memories:
            logger.info(f"[{self.name}] Resolved {len(memories)} cognitive memory elements natively.")

        # Trace logs
        if trace:
            trace.expanded_queries = queries
            trace.repo_commit_expansion = repo_hints
            trace.memory_expansion = memories

        # 4. LLM / heuristic tool routing (+ critic retry override)
        critic_feedback = state.get("critic_feedback") or {}
        suggested_tool = critic_feedback.get("suggested_tool") if isinstance(critic_feedback, dict) else None

        routing = await resolve_tool_routing(
            question,
            self._provider,
            critic_suggested=suggested_tool,
            adaptive_config=state.get("adaptive_config"),
        )
        needed_tool = routing.get("selected_tool")
        state["router_confidence"] = float(routing.get("confidence", 0.0))
        state["router_source"] = routing.get("source", "none")

        intent = (state.get("classification") or {}).get("agent_intent", "exploratory")
        bind_tool = bool(
            needed_tool
            and state["router_confidence"] > _ROUTER_BIND_CONFIDENCE
            and intent == "structured"
        )
        state["tool_routing"] = {
            "selected_tool": needed_tool if bind_tool else None,
            "routed_candidate": needed_tool,
            "confidence": state["router_confidence"],
            "source": state["router_source"],
            "bind_threshold": _ROUTER_BIND_CONFIDENCE,
            "bound": bind_tool,
            "agent_intent": intent,
        }
        if trace:
            trace.tool_routing = state["tool_routing"]

        if needed_tool and bind_tool:
            logger.info(
                f"[{self.name}] Tool routing → '{needed_tool}' (conf={state['router_confidence']})"
            )
            state["selected_tool"] = needed_tool
            queries.insert(0, f"TOOL_EXEC:{needed_tool}")
        else:
            if needed_tool and not bind_tool:
                logger.info(
                    "[%s] Router candidate %r not bound (conf=%.2f, intent=%s); explore-first.",
                    self.name,
                    needed_tool,
                    state["router_confidence"],
                    intent,
                )
            state["selected_tool"] = None

        # Commit back to state graph for Retriever
        state["sub_tasks"] = queries
        state["memories"] = memories
        state["embed_cache"] = embed_cache

        return state
