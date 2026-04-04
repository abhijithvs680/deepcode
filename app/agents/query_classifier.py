"""
Query Classifier Agent (Week 12 Phase 2)

Categorizes user queries into distinct strategic buckets:
- factual: Lookup knowledge, docs, straightforward how-tos
- debugging: Fix errors, trace failures, resolve active bugs
- navigation: Find files, structures, paths, or code references
- architecture: High-level design, component communication, data flows
- git: Commit history, authors, file changes over time

Uses a Hybrid Approach:
1. Fast heuristic matching.
2. LLM fallback if heuristics yield low confidence.
"""
from __future__ import annotations

import json
from enum import Enum

from app.core.config import get_settings
from app.core.logger import get_logger
from app.providers.base import LLMProvider

logger = get_logger(__name__)

class QueryType(str, Enum):
    FACTUAL = "factual"
    DEBUGGING = "debugging"
    NAVIGATION = "navigation"
    ARCHITECTURE = "architecture"
    GIT = "git"


class QueryClassifier:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.settings = get_settings()

    def _heuristic_classify(self, query: str) -> tuple[str | None, float]:
        q = query.lower()
        
        # Git heuristics
        if "who" in q and ("commit" in q or "wrote" in q or "modified" in q or "blame" in q):
            return QueryType.GIT.value, 0.9
        
        # Debugging heuristics
        if ("why" in q or "how" in q) and ("error" in q or "fail" in q or "crash" in q or "bug" in q):
            return QueryType.DEBUGGING.value, 0.85
            
        # Architecture heuristics
        if "architecture" in q or "design" in q or "structure" in q or "system" in q:
            return QueryType.ARCHITECTURE.value, 0.8
            
        # Navigation heuristics
        if "where" in q and ("file" in q or "function" in q or "class" in q):
            return QueryType.NAVIGATION.value, 0.85
            
        return None, 0.0

    async def classify(self, query: str) -> dict:
        """
        Classify the query using heuristics first. 
        If confidence < 0.7, fall back to LLM JSON formulation safely.
        """
        htype, hconf = self._heuristic_classify(query)
        if htype and hconf >= 0.7:
            return {
                "query_type": htype,
                "confidence": hconf,
                "focus_entities": []
            }
            
        prompt = f"""You are a query classifier. Analyze the user query and categorize it into exactly one type: {", ".join([e.value for e in QueryType])}.
Focus entities are important nouns, filenames, or symbols the query revolves around.

Respond strictly with a JSON object:
{{
  "query_type": "...",
  "confidence": <float 0.0-1.0>,
  "focus_entities": ["..."]
}}

Query: {query}
"""
        try:
            res = await self.provider.generate(
                prompt=prompt,
                system_prompt="You are a strict JSON classifier.",
                model=getattr(self.settings, "LLM_ROUTER_MODEL", self.settings.LLM_MODEL),
                temperature=0.0
            )
            val = json.loads(res.strip("` \n").removeprefix("json\n"))
            qtype = val.get("query_type", QueryType.FACTUAL.value)
            conf = float(val.get("confidence", 0.5))
            entities = val.get("focus_entities", [])
            return {
                "query_type": qtype,
                "confidence": conf,
                "focus_entities": entities
            }
        except Exception as e:
            logger.warning(f"LLM Classification failed: {e}. Defaulting to factual.")
            return {
                "query_type": QueryType.FACTUAL.value,
                "confidence": 0.4,
                "focus_entities": []
            }

