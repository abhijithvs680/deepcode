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
from app.agents.react_grounding import is_trivial_query

logger = logging.getLogger(__name__)


def _filter_tool_results_by_priority(
    tool_results: list[dict[str, Any]],
    priority_files: list[str],
) -> list[dict[str, Any]]:
    """Keep git/log + focused file tools; drop noisy list_dir and off-priority search hits."""
    if not priority_files or not tool_results:
        return tool_results
    pf = {p.replace("\\", "/") for p in priority_files}

    def path_in_priority(path: str) -> bool:
        n = path.replace("\\", "/")
        if n in pf:
            return True
        return any(n.startswith(pref + "/") for pref in pf)

    def touches(tr: dict[str, Any]) -> bool:
        t = tr.get("tool")
        if t in ("git", "log"):
            return True
        if t != "file":
            return True
        data = tr.get("data") or {}
        if not isinstance(data, dict):
            return True
        action = tr.get("action")
        if action in ("read_file", "get_symbols"):
            return True
        if action == "list_dir":
            return False
        if action == "search":
            for m in data.get("matches") or []:
                if not isinstance(m, dict):
                    continue
                fp = m.get("file") or m.get("file_path")
                if isinstance(fp, str) and path_in_priority(fp):
                    return True
            return False
        if action == "find_symbol":
            for m in data.get("matches") or []:
                if not isinstance(m, dict):
                    continue
                fp = m.get("file")
                if isinstance(fp, str) and path_in_priority(fp):
                    return True
            return False
        if action == "get_file_dependencies":
            for k in ("imports", "imported_by"):
                for fp in data.get(k) or []:
                    if isinstance(fp, str) and path_in_priority(fp):
                        return True
            p = data.get("file")
            if isinstance(p, str) and path_in_priority(p):
                return True
            return False
        if action == "get_symbol_dependencies":
            for m in data.get("matches") or []:
                if not isinstance(m, dict):
                    continue
                fp = m.get("file")
                if isinstance(fp, str) and path_in_priority(fp):
                    return True
            gph = data.get("graph")
            if isinstance(gph, dict):
                for fp0 in gph.keys():
                    if isinstance(fp0, str) and path_in_priority(fp0):
                        return True
            return False
        return True

    out = [tr for tr in tool_results if touches(tr)]
    return out if out else tool_results


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
        tool_results_raw = state.get("tool_results", [])
        priority_files = state.get("react_priority_files") or []

        tool_results = _filter_tool_results_by_priority(
            tool_results_raw,
            priority_files,
        )
        
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

        subgoal_traces = state.get("subgoal_traces") or []
        if subgoal_traces:
            sg_text = truncate_to_token_budget(
                json.dumps(
                    {"plan_goal": state.get("query", ""), "subgoals": subgoal_traces},
                    indent=2,
                    default=str,
                ),
                max_tokens=min(1800, self.settings.CONTEXT_BUDGET_TOOL + 400),
            )
            builder.add_context(
                "Hierarchical subgoal traces (merge before answering)",
                sg_text,
                priority=93,
            )
            builder.add_context(
                "Structured synthesis (mandatory when subgoals present)",
                "Your answer MUST: (1) briefly summarize what each subgoal found; "
                "(2) combine those findings into one coherent explanation; "
                "(3) call out any conflicts or gaps between subgoals; "
                "(4) give the final answer to the user's question using only the evidence above.",
                priority=94,
            )
            logger.info(
                "[%s] Injected %s subgoal trace bundle(s)",
                self.name,
                len(subgoal_traces),
            )

        xs = (state.get("cross_subgoal_summary") or "").strip()
        if xs:
            builder.add_context(
                "Cross-subgoal synthesis (reconcile before answering)",
                truncate_to_token_budget(
                    xs,
                    max_tokens=min(900, self.settings.CONTEXT_BUDGET_TOOL // 2),
                ),
                priority=95,
            )
            logger.info("[%s] Cross-subgoal synthesis injected", self.name)

        react_steps = state.get("react_steps") or []
        react_observations = state.get("react_observations") or []
        if react_steps or react_observations or (state.get("react_summary") or "").strip():
            react_payload = {
                "steps": react_steps[-10:],
                "observations": react_observations[-16:],
            }
            rs = (state.get("react_summary") or "").strip()
            if rs:
                react_payload["earlier_summary"] = rs[:800]
            if priority_files:
                react_payload["priority_files"] = priority_files[:2]
            react_text = truncate_to_token_budget(
                json.dumps(react_payload, indent=2, default=str),
                max_tokens=min(2048, self.settings.CONTEXT_BUDGET_TOOL + 512),
            )
            builder.add_context(
                "ReAct exploration (read before answering)",
                react_text,
                priority=92,
            )
            logger.info(
                "[%s] ReAct trace injected (steps=%s, obs=%s)",
                self.name,
                len(react_steps),
                len(react_observations),
            )

        if is_trivial_query(question):
            grounding = (
                "This is a short general or conversational message. Reply briefly and "
                "naturally. You do not need repository evidence for greetings or simple "
                "chit-chat."
            )
        else:
            grounding = (
                "GROUNDING: Prefer ReAct exploration, Tool Execution Results, "
                "Code intelligence / retrieved snippets, and Cognitive Memory when present. "
                "Do not invent repository paths, symbols, or behavior not supported by that "
                "evidence.\n"
                "If context is thin, still give your best-effort answer: state what you "
                "can infer, list reasonable next steps (e.g. which file or search to run), "
                "and clearly label uncertainty. Do not refuse with a stock "
                "\"not enough information\" phrase unless the user explicitly asked only "
                "for facts that are completely absent from all supplied material."
            )
        if state.get("write_enabled"):
            grounding += (
                "\nException for edit mode: you may propose <patch> blocks only for "
                "file paths that appear in the evidence above."
            )
        builder.add_context("Grounding contract", grounding, priority=96)

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
                "EDIT MODE ENABLED:\n"
                "Prefer transactional search/replace (safer) when possible — exact "
                "substring must exist in the file:\n\n"
                "<edit file='relative/path.py'>\n"
                "<search>exact code to find (include enough context to be unique)</search>\n"
                "<replace>replacement code</replace>\n"
                "</edit>\n\n"
                "You may emit multiple <edit> blocks. If a whole-file rewrite is truly "
                "required, you may instead use full-file patches:\n\n"
                "<patch file='path/to/file'>\n"
                "new file content entirely replacing original\n"
                "</patch>\n\n"
                "Only reference paths that appear in Tool Execution Results, ReAct "
                "exploration, Code intelligence, or the structured edit plan below."
            )
            builder.add_context("System Core Directives", sys_write, priority=110)
            ep = state.get("edit_plan") or {}
            if isinstance(ep, dict) and ep.get("plan"):
                plan_blob = truncate_to_token_budget(
                    json.dumps(ep, indent=2, default=str),
                    max_tokens=min(1200, self.settings.CONTEXT_BUDGET_TOOL // 2),
                )
                builder.add_context(
                    "Structured edit plan (follow unless evidence contradicts)",
                    plan_blob,
                    priority=109,
                )
            ec = state.get("edit_context") or {}
            if isinstance(ec, dict) and (
                ec.get("files") or ec.get("symbols") or ec.get("dependency_graph")
            ):
                ctx_blob = truncate_to_token_budget(
                    json.dumps(ec, indent=2, default=str),
                    max_tokens=min(900, self.settings.CONTEXT_BUDGET_TOOL // 3),
                )
                builder.add_context(
                    "Edit context (files read + graph summary)",
                    ctx_blob,
                    priority=108,
                )
            
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
            for tr_dict in tool_results_raw:
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
