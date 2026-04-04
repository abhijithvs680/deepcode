"""
Debug Trace — Observability layer to track the reasoning steps
of the Code Intelligence Engine pipeline.
"""
from typing import Any

from pydantic import BaseModel, Field


class DebugTrace(BaseModel):
    """Tracks the internal state of the orchestrator pipeline."""

    query: str = ""
    expanded_queries: list[str] = Field(default_factory=list)
    retrieval: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    reranking: list[dict[str, Any]] = Field(
        default_factory=list,
        description="LLM rerank trace when enabled; else mirrors composite retrieval.",
    )
    reranking_mode: str = "composite"
    repo_commit_expansion: list[str] = Field(default_factory=list)
    final_context_files: list[str] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)
    graph_expansion: list[dict[str, Any]] = Field(default_factory=list)
    file_ranking: list[dict[str, Any]] = Field(default_factory=list)
    symbol_index_trace: list[dict[str, Any]] = Field(default_factory=list)
    excluded_from_context: list[str] = Field(default_factory=list)
    memory_expansion: list[dict[str, Any]] = Field(default_factory=list)
    memory_autolog: list[dict[str, Any]] = Field(default_factory=list)
    tool_routing: dict[str, Any] = Field(default_factory=dict)
