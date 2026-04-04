from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
import time

class EvaluationResult(BaseModel):
    """
    Standardizes metric evaluations targeting explicit operational outcomes dynamically identifying
    System Hallucinations and Multi-Agent Orchestrator regressions natively across executions.
    """
    query: str
    success: bool
    hallucination_detected: bool
    tool_usage: List[str]
    score: float
    feedback: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

def score_tool_output(tool_data: dict[str, Any]) -> float:
    """
    Computes heuristic mappings structurally establishing Tool execution confidences natively 
    validating precise RAG extraction overlaps prioritizing actual deterministic bounds.
    """
    if not tool_data.get("success"):
        return 0.0

    score = 0.5 # Baseline for successful execution mappings
    meta = tool_data.get("metadata", {})
    action = tool_data.get("action", "")
    
    # Heuristics tracking tool-specific efficiencies dynamically
    if action == "git_log":
        score += min(0.3, meta.get("count", 0) * 0.05)
    elif action == "search_logs":
        # Heavily prioritize explicit grep exceptions identifying trace boundaries
        score += 0.4 if meta.get("found_count", 0) > 0 else 0.0
    elif action == "read_file":
        if meta.get("truncated", False):
            # Penalize token overflows implicitly signaling improper range targeting
            score -= 0.2
        else:
            score += 0.4
            
    return round(min(1.0, max(0.0, score)), 2)

def score_answer(answer: str, ground_truth: str) -> float:
    """
    Basic string overlap validation limits mapping target evaluations reliably bounding E2E test scripts.
    In real usage, this usually calls a smaller cheap LLM node rating relevance natively.
    """
    import re
    if not answer or not ground_truth:
        return 0.0
        
    truth_tokens = set(re.findall(r'\w+', ground_truth.lower()))
    answer_tokens = set(re.findall(r'\w+', answer.lower()))
    
    if not truth_tokens: return 0.0
    overlap = len(truth_tokens.intersection(answer_tokens))
    
    return round(min(1.0, overlap / len(truth_tokens) * 1.5), 2)
