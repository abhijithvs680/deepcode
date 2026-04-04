import logging
from typing import Any

from app.agents.base_agent import BaseAgent
from app.memory.retriever import merge_chunk_results, rank_files_then_chunks, search as rag_search
from app.memory.scoring import aggregate_file_score
from app.agents.dependency_resolver import resolve_dependencies
from app.core.config import get_settings
from app.tools.tool_scoring import score_tool_result

logger = logging.getLogger(__name__)


def trim_ranked_ctx_hybrid(ranked: list[dict], top_files: int, top_chunks: int) -> list[dict]:
    """Keep top-N files by max chunk score, then cap total chunks (matches low-K hybrid retrieval)."""
    if not ranked:
        return ranked
    by_file: dict[str, list[dict]] = {}
    for c in ranked:
        by_file.setdefault(c["file_path"], []).append(c)
    file_order = sorted(
        by_file.keys(),
        key=lambda fp: max(float(x["score"]) for x in by_file[fp]),
        reverse=True,
    )
    keep_files = set(file_order[:top_files])
    pool = [c for c in ranked if c["file_path"] in keep_files]
    pool.sort(key=lambda x: float(x["score"]), reverse=True)
    return pool[:top_chunks]


class RetrieverAgent(BaseAgent):
    """
    Middle-layer node responsible for mapping Planner sub-tasks onto localized database
    representations utilizing pgvector. Consolidates code paths and dependency networks.
    """

    def __init__(self, db, provider):
        super().__init__("Retriever")
        self._db = db
        self._provider = provider
        self.settings = get_settings()

    def apply_hybrid_postprocess(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        After parallel tool+RAG, shrink RAG when tool score is high (same effect as sequential hybrid).
        """
        tool_results = state.get("tool_results", [])
        max_tool_score = 0.0
        for tr_dict in tool_results:
            try:
                from app.tools.base_tool import ToolResult

                tr_obj = ToolResult(**tr_dict)
                s = score_tool_result(tr_obj)
                max_tool_score = max(max_tool_score, s)
            except Exception:
                pass
        if max_tool_score <= 0.85:
            return state
        ranked_ctx = state.get("ranked_ctx", [])
        for c in ranked_ctx:
            c["score"] = float(c["score"]) * 0.4
        ranked_ctx.sort(key=lambda x: float(x["score"]), reverse=True)
        state["ranked_ctx"] = trim_ranked_ctx_hybrid(ranked_ctx, top_files=2, top_chunks=3)
        logger.info(
            f"[{self.name}] Post-parallel hybrid trim applied (tool_score={max_tool_score:.2f})."
        )
        return state

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Executes parallel searches across planner sub-tasks, ranks vectors, 
        and maps dependencies back onto the state dict under 'ranked_ctx'.
        """
        question = state.get("query", "")
        repo_id = state.get("repo_id")
        trace = state.get("trace")
        sub_tasks = state.get("sub_tasks", [question])
        embed_cache = state.get("embed_cache", {})
        fallbacks_local = state.get("fallbacks_local", [])
        
        # Git dependencies normally populated pre-planner, but fetch missing keys if empty.
        indexed_hits = frozenset()
        git_by_path = state.get("git_by_path", {})

        # Tool+RAG parallel: ignore tool scores for retrieval breadth; post-merge applies hybrid trim.
        parallel_tool_rag = bool(state.get("_parallel_tool_rag"))
        tool_results = [] if parallel_tool_rag else state.get("tool_results", [])
        max_tool_score = 0.0
        for tr_dict in tool_results:
            try:
                from app.tools.base_tool import ToolResult
                tr_obj = ToolResult(**tr_dict)
                s = score_tool_result(tr_obj)
                max_tool_score = max(max_tool_score, s)
            except Exception:
                pass
                
        # Implement Hybrid Logic blending Deterministic Tool + Semantic RAG!
        tool_confidence_override = False
        adaptive = state.get("adaptive_config") or {}
        rag_k_cap = int(adaptive.get("rag_top_k", 10))
        top_files = int(adaptive.get("top_files", 12))
        top_chunks = int(adaptive.get("top_chunks", 15))
        if max_tool_score > 0.85:
            logger.info(f"[{self.name}] Tool Execute confidence extremely high ({max_tool_score}). Employing strict Hybrid Retrieval (low-K).")
            tool_confidence_override = True
            rag_k_cap = 3
            top_files = 2
            top_chunks = 3

        logger.info(f"[{self.name}] Initiating vector retrieval matrix...")

        pool: dict[tuple[str, int], dict] = {}
        
        # Gap 4: Entities Expansion limit natively!
        focus_entities = state.get("classification", {}).get("focus_entities", [])
        if focus_entities:
            # Map robust semantic searches strictly across entities mapping broader domains natively!
            expanded_tasks = [f"{ent} {question}" for ent in focus_entities]
            sub_tasks.extend(expanded_tasks)
        
        # Parallel semantic retrievers
        for q in sub_tasks[:35]:
            part = await rag_search(
                q,
                self._provider,
                repo_name=repo_id,
                top_k=rag_k_cap,
                min_score=0.22,
                full_question=question,
                embed_cache=embed_cache,
                indexed_symbol_hits=indexed_hits,
                git_by_path=git_by_path or None,
                focus_entities=focus_entities,
            )
            merge_chunk_results(pool, part)

        merged = list(pool.values())
        ranked = rank_files_then_chunks(merged, top_files=top_files, top_chunks=top_chunks)
        
        # Expand bounds if chunks are too narrow (unless Hybrid tool override restricted it intentionally)
        if len(ranked) < 5 and not tool_confidence_override:
            ranked = rank_files_then_chunks(merged, top_files=20, top_chunks=20)

        # Execute Deep Dependencies if Fallback triggers
        if self._db and repo_id and "fallback_no_graph" in fallbacks_local and not tool_confidence_override:
            logger.info(f"[{self.name}] Initiating Graph Fallback Dependency Resolution...")
            dep_chunks, dep_traces = await resolve_dependencies(
                ranked[:12], self._db, repo_name=repo_id
            )
            if trace:
                trace.dependencies = dep_traces
            merge_chunk_results(pool, dep_chunks)
            merged = list(pool.values())
            ranked = rank_files_then_chunks(merged, top_files=12, top_chunks=18)
        elif trace:
            trace.dependencies = []

        # Tally file-level trace logging metrics
        by_file: dict[str, list[dict]] = {}
        for c in merged:
            by_file.setdefault(c["file_path"], []).append(c)
            
        file_rows: list[dict] = []
        for fp, clist in by_file.items():
            scores = [float(x["score"]) for x in clist]
            sym_hit = max((float(x["score_breakdown"].get("symbol", 0)) for x in clist), default=0.0)
            fs = aggregate_file_score(scores, sym_hit)
            file_rows.append({"file_path": fp, "file_score": round(fs, 4)})
            
        file_rows.sort(key=lambda x: x["file_score"], reverse=True)
        
        # Weight explicitly avoiding RAG dominating semantic paths over robust Hybrid overrides!
        for c in ranked:
            if tool_confidence_override:
                c["score"] = float(c["score"]) * 0.4  # RAG penalized heavily avoiding tool displacement
            else:
                c["score"] = float(c["score"])

        trace_retrieval = [
            {
                "file_path": r["file_path"],
                "chunk_index": r["chunk_index"],
                "score": r["score"],
                "score_breakdown": r.get("score_breakdown"),
                "symbol_tier": r.get("symbol_tier"),
                "symbol_reason": r.get("symbol_reason"),
                "repo": r.get("repo_name"),
            }
            for r in ranked
        ]

        # Conditionally LLM rerank
        ranked_ctx = ranked
        llm_rerank_trace: list[dict] = []
        if self.settings.RAG_LLM_RERANK and ranked:
            from app.agents.reranker import rerank as llm_rerank
            top_n = min(15, len(ranked))
            logger.info(f"[{self.name}] Executing LLM rerank over Top-{top_n} retrieved documents.")
            reranked, llm_rerank_trace = await llm_rerank(
                question, ranked[:top_n], self._provider, top_k=min(12, top_n)
            )
            seen_k = {(c["file_path"], c["chunk_index"]) for c in reranked}
            tail = [c for c in ranked if (c["file_path"], c["chunk_index"]) not in seen_k]
            ranked_ctx = reranked + tail

        # Populate internal traceability mappings
        if trace:
            trace.retrieval = trace_retrieval
            trace.file_ranking = file_rows[:20]
            if self.settings.RAG_LLM_RERANK and llm_rerank_trace:
                trace.reranking = llm_rerank_trace
                trace.reranking_mode = "llm"
            else:
                trace.reranking = trace_retrieval[: min(12, len(trace_retrieval))]
                trace.reranking_mode = "composite"

        logger.info(f"[{self.name}] Concluded retrieval. Bound {len(ranked_ctx)} active components.")

        # Bind final metrics over into pipeline state
        state["ranked_ctx"] = ranked_ctx
        return state
