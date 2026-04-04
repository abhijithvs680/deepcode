import time
import logging
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.context_builder import (
    ContextBuilder,
    assemble_memory_context,
    assemble_symbol_git_rag_context,
    truncate_to_token_budget,
)
from app.tools.tool_scoring import score_tool_result
from app.tools.base_tool import ToolResult
from app.memory.intel_store import fetch_symbol_anchor_text, fetch_git_context_text
from app.memory.symbol_extract import extract_query_identifiers
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class AnswerAgent(BaseAgent):
    """
    Downstream Node responsible for assembling the prompt context variables
    hierarchically natively restricting token overloads across Code intelligence schemas.
    Executes raw generation across the target LLM provider framework.
    """

    def __init__(self, db, provider, interaction_logger):
        super().__init__("Answer")
        self._db = db
        self._provider = provider
        self._interaction_logger = interaction_logger
        self.settings = get_settings()

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Executes Context Building algorithms dynamically rendering raw context vectors
        prior to final model invocation sequences.
        """
        question = state.get("query", "")
        repo_id = state.get("repo_id")
        trace = state.get("trace")
        ranked_ctx = state.get("ranked_ctx", [])
        memories = state.get("memories", [])
        session_id = state.get("session_id")
        
        identifiers = extract_query_identifiers(question)
        ctx_files = {c["file_path"].replace("\\", "/") for c in ranked_ctx[:18]}
        tool_results = state.get("tool_results", [])
        
        sym_text = ""
        git_slice = ""
        
        if self._db and repo_id and ctx_files:
            sym_text = await fetch_symbol_anchor_text(
                self._db, repo_id, ctx_files, identifiers
            )
            git_slice = await fetch_git_context_text(
                self._db, repo_id, ctx_files, max_files=14
            )

        # Token Constraint Construction Layer
        builder = ContextBuilder(question=question)
        builder.add_context("User Instructions", question, priority=100)
        uc = state.get("user_context")
        if uc and str(uc).strip():
            builder.add_context("User Context", str(uc), priority=95)

        memory_text = assemble_memory_context(
            memories, max_tokens=self.settings.CONTEXT_BUDGET_MEMORY
        )

        rag_text, excluded, included_files = assemble_symbol_git_rag_context(
            sym_text,
            git_slice,
            ranked_ctx,
            question,
            max_tokens=self.settings.MAX_CONTEXT_TOKENS,
            symbol_budget=self.settings.CONTEXT_BUDGET_SYMBOL,
            git_budget=self.settings.CONTEXT_BUDGET_GIT,
            rag_budget=self.settings.CONTEXT_BUDGET_RAG,
        )
        if trace:
            trace.excluded_from_context = excluded
            trace.final_context_files = included_files

        if memory_text.strip():
            builder.add_context("Cognitive Memory", memory_text, priority=10)
            logger.info(f"[{self.name}] Long-term memory context injected (priority=10)")
            
        import json

        if tool_results:
            tool_payload = truncate_to_token_budget(
                json.dumps(tool_results, indent=2),
                self.settings.CONTEXT_BUDGET_TOOL,
            )
            builder.add_context("Tool Execution Results", tool_payload, priority=15)
            logger.info(f"[{self.name}] Tool Execution contexts injected successfully (priority=15). Count: {len(tool_results)}")

        if rag_text.strip():
            builder.add_context("Code intelligence", rag_text, priority=9)
            logger.info(f"[{self.name}] RAG + symbol/Git context injected (token-capped)")

        if state.get("write_enabled"):
            sys_write = (
                "WRITE MODE ENABLED:\n"
                "You are authorized to modify files. If you generate code changes, you MUST output your proposed modifications using exactly the following syntax mapping the entire replacement file:\n\n"
                "<patch file='path/to/file'>\n"
                "new file content entirely replacing original\n"
                "</patch>\n\n"
                "Generate a separate `<patch>` block for each file modified."
            )
            builder.add_context("System Core Directives", sys_write, priority=110)
            
        critic_fallback = state.get("critic_feedback")
        if critic_fallback:
            builder.add_context("CRITICAL: Autonomous Sandbox Testing Failures", critic_fallback, priority=120)
            logger.info("Injected automated Sandbox Error Traces explicitly back into Prompt limits!")
            state.pop("critic_feedback", None)

        built = builder.build()
        state["full_prompt"] = built.full_prompt
        state["context"] = rag_text
        state["pieces_used"] = built.pieces_used

        start_time = time.perf_counter()
        try:
            logger.info(f"[{self.name}] Dispatching payload to Provider engine...")
            
            stream_cb = state.get("stream_callback")
            if stream_cb:
                response_chunks = []
                async for chunk in self._provider.stream_generate(built.full_prompt):
                    response_chunks.append(chunk)
                    await stream_cb(chunk)
                response = "".join(response_chunks)
            else:
                response = await self._provider.generate(built.full_prompt)
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "[%s] LLM responded in %.1fms (model=%s, context_pieces=%s)",
                self.name,
                latency_ms,
                self.settings.LLM_MODEL,
                built.pieces_used,
            )

            if self._interaction_logger and session_id:
                await self._interaction_logger.log_interaction(
                    session_id=session_id,
                    question=question,
                    context=rag_text,
                    full_prompt=built.full_prompt,
                    answer=response,
                    provider=self.settings.LLM_PROVIDER,
                    model=self.settings.LLM_MODEL,
                    latency_ms=latency_ms,
                )

            state["answer"] = response
            state["latency_ms"] = latency_ms

            max_tool_score = 0.0
            for tr_dict in tool_results:
                try:
                    max_tool_score = max(
                        max_tool_score, score_tool_result(ToolResult(**tr_dict))
                    )
                except Exception:
                    pass
            rc = float(state.get("router_confidence") or 0.0)
            if max_tool_score > 0.9 and rc >= 0.9:
                state["skip_critic_llm"] = True
                logger.info(
                    "[%s] High tool score + router confidence → skipping critic LLM",
                    self.name,
                )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            if self._interaction_logger and session_id:
                await self._interaction_logger.log_error(
                    session_id=session_id,
                    question=question,
                    full_prompt=built.full_prompt,
                    provider=self.settings.LLM_PROVIDER,
                    model=self.settings.LLM_MODEL,
                    error_message=str(exc),
                    context=rag_text,
                    latency_ms=latency_ms,
                )
            raise

        return state
