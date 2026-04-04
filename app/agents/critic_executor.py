import logging
from typing import Any

logger = logging.getLogger(__name__)

def apply_critic_action(state: dict[str, Any]) -> bool:
    """
    Consumes validated outputs out of the CriticAgent seamlessly adapting context parameters 
    across Orchestrator boundaries avoiding hallucinated loop crashes natively!
    
    Returns True if the Orchestrator should trigger a sequential execution retry block.
    """
    feedback = state.get("critic_feedback", {})
    action = feedback.get("action", "accept")
    mode = state.get("mode", "ask")
    
    # Gap 4 Constraint limits: Build mode must reject expansion loops dynamically mitigating patches natively!
    if mode == "build" and action == "expand_retrieval":
        logger.warning("CriticExecutor preventing expand_retrieval looping natively during Build mode boundaries!")
        action = "retry"
        
    if action == "expand_retrieval":
        logger.info("[CriticExecutor] Action: Expanding Semantic RAG constraints linearly.")
        # Gap 3 boundaries
        ac = state.get("adaptive_config", {})
        ac["rag_top_k"] = min(ac.get("rag_top_k", 10) + 3, 20)
        return True
        
    elif action == "use_tool":
        # Force a suggested tool route across the planner
        suggested = feedback.get("suggested_tool")
        if suggested:
            logger.info(f"[CriticExecutor] Action: Nudging tool execution fallback -> {suggested}")
            state["force_tool"] = suggested
            state["skip_critic_llm"] = False # Unlock strict traces
        return True
        
    elif action == "retry":
        logger.info("[CriticExecutor] Action: General iteration retry natively bounds.")
        return True
        
    elif action == "accept":
        # Accept clears out retry bounds natively letting Answer agent states gracefully return!
        state["critic_retry"] = False
        return False
        
    # Catch-all
    return False
