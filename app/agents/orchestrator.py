"""
Agent Orchestrator — central coordination layer.

Routes user queries through the appropriate provider and returns the result.
Integrates ContextBuilder, composite RAG + Git scoring, optional LLM rerank, interaction logging.
"""
import asyncio
import os
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.confirm_diff_util import invoke_confirm_diff_callback
from app.agents.context_builder import ContextBuilder
from app.core.config import get_settings
from app.core.interaction_logger import InteractionLogger
from app.core.logger import get_logger
from app.providers.base import LLMProvider
from app.providers.factory import get_provider

logger = get_logger(__name__)
settings = get_settings()


def _workspace_root_for_repo(repo_id: str | None) -> str:
    if repo_id:
        try:
            from app.core.repo_registry import get_repo_entry

            entry = get_repo_entry(repo_id)
            if entry and entry.get("path"):
                return str(entry["path"])
        except Exception:
            pass
    return os.getcwd()


class AgentOrchestrator:
    """Coordinates between the user, context builder, providers, and logger."""

    def __init__(
        self,
        db: AsyncSession | None = None,
        provider: LLMProvider | None = None,
    ):
        self.settings = get_settings()
        self._provider = provider or get_provider()
        self._db = db
        self._interaction_logger = InteractionLogger(db) if db else None

    async def _ask_mode_lightweight_reply(
        self,
        *,
        question: str,
        classification: dict,
        session_id,
        trace,
        kwargs: dict,
        status_cb,
        repo_id: str | None,
        state: dict,
        status_message: str,
        llm_prompt: str,
        event_reason: str,
        router_source: str,
        rag_reason: str,
        agent_intent: str,
    ) -> dict:
        """Direct LLM answer without planner, ReAct, or RAG (trivial or low-signal ask mode)."""
        from app.agents.react_grounding import compute_answer_confidence
        from app.core.event_bus import EventBus

        full_prompt = llm_prompt
        if self._interaction_logger and session_id:
            try:
                hist = await self._interaction_logger.get_session_history(
                    session_id
                )
                tail = hist[-4:]
                lines: list[str] = []
                for h in tail:
                    role = str(h.get("role") or "")
                    c = str(h.get("content") or "").strip()
                    if not c:
                        continue
                    label = "User" if role == "user" else "Assistant"
                    lines.append(f"{label}: {c[:600]}")
                if lines:
                    full_prompt = (
                        "Recent conversation (for continuity):\n"
                        + "\n".join(lines)
                        + "\n\n---\n\n"
                        + llm_prompt
                    )
            except Exception as exc:
                logger.debug("fast_path session history: %s", exc)

        if status_cb:
            await status_cb(status_message)
        t0 = time.perf_counter()
        stream_cb = kwargs.get("stream_callback")
        if stream_cb:
            chunks: list[str] = []
            async for ch in self._provider.stream_generate(full_prompt):
                chunks.append(ch)
                await stream_cb(ch)
            answer = "".join(chunks)
        else:
            answer = await self._provider.generate(full_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        state["answer"] = answer
        state["latency_ms"] = latency_ms
        state["full_prompt"] = full_prompt
        state["ranked_ctx"] = []
        state["tool_results"] = []
        state["context"] = ""
        state["final_confidence"] = min(
            0.95,
            float(classification.get("confidence", 1.0) or 1.0),
        )
        if self._interaction_logger and session_id:
            await self._interaction_logger.log_interaction(
                session_id=session_id,
                question=question,
                context="",
                full_prompt=full_prompt,
                answer=answer,
                provider=self.settings.LLM_PROVIDER,
                model=self.settings.LLM_MODEL,
                latency_ms=latency_ms,
            )
        if self._db and repo_id:
            from app.memory.intel_store import store_query

            await store_query(
                self._db,
                repo_id,
                question,
                answer[:2500],
                state.get("base_query_embed"),
            )
        cls_c = float(classification.get("confidence", 0.85) or 0.85)
        await EventBus.emit(
            {
                "type": "fast_path",
                "reason": event_reason,
                "query": question[:500],
            }
        )
        await EventBus.emit(
            {
                "type": "agent_status",
                "confidence": cls_c,
                "files_read": 0,
                "symbols": 0,
                "weak_grounding": False,
                "subgoal_id": None,
            }
        )
        return_dict = {
            "answer": answer,
            "session_id": session_id,
            "latency_ms": latency_ms,
            "model": self.settings.LLM_MODEL,
            "provider": self.settings.LLM_PROVIDER,
            "meta": {
                "selected_tool": None,
                "router_confidence": state.get("router_confidence"),
                "router_source": router_source,
                "retrieved_files": [],
                "final_confidence": state["final_confidence"],
                "classification": classification,
                "agent_intent": agent_intent,
                "fast_path": True,
                "tool_steps_count": 0,
                "rag_used": False,
                "rag_reason": rag_reason,
                "files_touched": [],
                "symbols_used": [],
                "answer_confidence": compute_answer_confidence(state),
                "require_user_confirm": False,
                "active_working_set": [],
                "file_scores": {},
                "global_insights": [],
                "cross_subgoal_summary": "",
                "early_stop_reason": "",
                "react_early_stop_reason": "",
            },
        }
        if trace:
            return_dict["trace"] = trace.model_dump()
        await EventBus.emit(
            {
                "type": "final_answer",
                "content": answer,
                "session_id": session_id,
            }
        )
        return return_dict

    async def run(
        self,
        question: str,
        context: str | None = None,
        session_id: uuid.UUID | None = None,
        repo_id: str | None = None,
        debug: bool = False,
        source: str = "api",
        mode: str = "ask",
        auto_apply: bool = False,
        dry_run: bool = False,
        **kwargs,
    ) -> dict:
        from app.core.debug_trace import DebugTrace

        trace = DebugTrace(query=question) if debug else None

        if mode == "build":
            logger.warning("Mode 'build' is deprecated; use 'edit'.")
            mode = "edit"

        # Week 12 Phase 2: Start Query Classifier concurrently early.
        from app.agents.query_classifier import QueryClassifier
        classifier = QueryClassifier(self._provider)
        classification_task = asyncio.create_task(classifier.classify(question))

        if self._interaction_logger:
            if session_id:
                exists = await self._interaction_logger.session_exists(session_id)
                if not exists:
                    logger.warning("Session %s not found, creating new session", session_id)
                    session = await self._interaction_logger.create_session(source=source)
                    session_id = session.id
            else:
                session = await self._interaction_logger.create_session(source=source)
                session_id = session.id

        builder = ContextBuilder(question=question)

        if context:
            builder.add_context("User Context", context, priority=10)

        if self._interaction_logger and session_id:
            history = await self._interaction_logger.get_session_history(session_id)
            if history:
                builder.add_conversation_history(history)

        start_time = time.perf_counter()

        if self.settings.ADAPTIVE_ENABLED:
            from app.agents.adaptive_controller import adaptive_controller

            adaptive_controller.prepare()
            ac = adaptive_controller.get_config()
        else:
            from app.agents.adaptive_controller import AdaptiveConfig
            ac = AdaptiveConfig().as_dict()

        from app.core.repo_manager import get_ingest_state
        if repo_id:
            ingest_state = get_ingest_state(repo_id)
            if ingest_state.get("status") == "running":
                ac["rag_top_k"] = max(2, ac.get("rag_top_k", 10) // 2)
                builder.add_context(
                    "System Warning", 
                    "⚡ Repository is still indexing. Results may be incomplete.", 
                    priority=100
                )

        # Resolve classification bounds synchronously
        classification = await classification_task
        from app.agents.agent_intent import derive_agent_intent

        classification["agent_intent"] = derive_agent_intent(classification, question)
        q_conf = classification.get("confidence", 1.0)
        q_type = classification.get("query_type", "factual") if q_conf >= 0.6 else "factual"
        disable_aggressive_overrides = q_conf < 0.6
        
        # Track initial K for Observability
        ac["rag_top_k_before"] = ac.get("rag_top_k", 10)
        
        # Apply Mode Restraints first
        if mode == "plan":
            ac["rag_top_k"] = min(ac.get("rag_top_k", 10), 3)
            
        # Apply Bounded Strategy Overrides (Nudges)
        if not disable_aggressive_overrides:
            if q_type == "debugging":
                ac["rag_top_k"] = min(ac.get("rag_top_k", 10), 5)
                ac["enable_git_bias"] = True
            elif q_type == "architecture":
                ac["top_files"] = min(ac.get("top_files", 3) + 2, 8)
            elif q_type == "git":
                ac["rag_top_k"] = 2
                ac["disable_code_embeddings"] = True

        # Week 6 Matrix Iteration
        from app.agents.agent_registry import AgentRegistry
        from app.agents.planner_agent import PlannerAgent
        from app.agents.tool_execution_agent import ToolExecutionAgent
        from app.agents.retriever_agent import RetrieverAgent
        from app.agents.answer_agent import AnswerAgent
        from app.agents.critic_agent import CriticAgent
        from app.agents.execution_planner import ExecutionPlanner
        from app.agents.edit_planner_agent import EditPlannerAgent

        # Register instances
        AgentRegistry.register(ExecutionPlanner(self._provider))
        AgentRegistry.register(PlannerAgent(self._db, self._provider))
        AgentRegistry.register(ToolExecutionAgent())
        AgentRegistry.register(RetrieverAgent(self._db, self._provider))
        AgentRegistry.register(EditPlannerAgent(self._provider, db=self._db))
        AgentRegistry.register(AnswerAgent(self._db, self._provider, self._interaction_logger))
        AgentRegistry.register(CriticAgent(self._db, self._provider))

        async def run_parallel_tool_rag(s: dict) -> dict:
            """Run tool + RAG concurrently; optionally cancel RAG if tool score is very high (Week 11)."""
            from app.tools.tool_scoring import max_tool_score_from_dicts

            tool_agent = AgentRegistry.get("ToolExecutor")
            retriever_agent = AgentRegistry.get("Retriever")
            st_tool = dict(s)
            st_rag = dict(s)
            st_rag["_parallel_tool_rag"] = True
            st_rag["tool_results"] = []

            async def _tool() -> dict:
                try:
                    return await asyncio.wait_for(
                        tool_agent.run(st_tool),
                        timeout=self.settings.TOOL_EXEC_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning("ToolExecutor exceeded timeout; continuing without tool output.")
                    out = dict(s)
                    out["tool_results"] = []
                    out["tool_result_status"] = "timeout"
                    return out
                except Exception as ex:
                    logger.warning("ToolExecutor failed: %s", ex)
                    out = dict(s)
                    out["tool_results"] = []
                    out["tool_result_status"] = "error"
                    return out

            async def _rag() -> dict:
                try:
                    return await asyncio.wait_for(
                        retriever_agent.run(st_rag),
                        timeout=self.settings.RAG_RETRIEVAL_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Retriever exceeded timeout; returning empty RAG context.")
                    out = dict(s)
                    out["ranked_ctx"] = []
                    fl = list(out.get("fallbacks_local") or [])
                    fl.append("rag_timeout")
                    out["fallbacks_local"] = fl
                    return out
                except Exception as ex:
                    logger.warning("Retriever failed: %s", ex)
                    out = dict(s)
                    out["ranked_ctx"] = []
                    return out

            def _empty_rag(base: dict) -> dict:
                out = dict(base)
                out["ranked_ctx"] = []
                fl = list(out.get("fallbacks_local") or [])
                if "rag_cancelled_high_tool_score" not in fl:
                    fl.append("rag_cancelled_high_tool_score")
                out["fallbacks_local"] = fl
                return out

            def _empty_tool(base: dict) -> dict:
                out = dict(base)
                out["tool_results"] = []
                out["tool_result_status"] = "cancelled"
                fl = list(out.get("fallbacks_local") or [])
                if "tool_cancelled_high_rag_signal" not in fl:
                    fl.append("tool_cancelled_high_rag_signal")
                out["fallbacks_local"] = fl
                return out

            def _rag_retrieval_confidence(ranked_ctx: list) -> float:
                if not ranked_ctx:
                    return 0.0
                scores = [float(c.get("score", 0.0)) for c in ranked_ctx[:8]]
                return max(scores) if scores else 0.0

            tool_task = asyncio.create_task(_tool())
            rag_task = asyncio.create_task(_rag())
            threshold = float(self.settings.EARLY_CANCEL_RAG_TOOL_SCORE)

            done, pending = await asyncio.wait(
                {tool_task, rag_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if tool_task in done and rag_task in done:
                t_out = tool_task.result()
                r_out = rag_task.result()
            elif tool_task in done:
                t_out = tool_task.result()
                score = max_tool_score_from_dicts(t_out.get("tool_results"))
                if score >= threshold and rag_task in pending:
                    rag_task.cancel()
                    try:
                        await rag_task
                    except asyncio.CancelledError:
                        pass
                    r_out = _empty_rag(s)
                    s["early_cancel_rag"] = True
                    logger.info(
                        "Early-cancelled RAG (tool_score=%.2f >= %.2f)",
                        score,
                        threshold,
                    )
                else:
                    r_out = await rag_task
            else:
                r_out = rag_task.result()
                rag_conf = _rag_retrieval_confidence(r_out.get("ranked_ctx") or [])
                rag_cutoff = float(self.settings.EARLY_CANCEL_TOOL_RAG_CONFIDENCE)
                if (
                    not s.get("selected_tool")
                    and rag_conf >= rag_cutoff
                    and tool_task in pending
                ):
                    tool_task.cancel()
                    try:
                        await tool_task
                    except asyncio.CancelledError:
                        pass
                    t_out = _empty_tool(s)
                    s["early_cancel_tool"] = True
                    logger.info(
                        "Early-cancelled tool (no tool route, rag_conf=%.2f >= %.2f)",
                        rag_conf,
                        rag_cutoff,
                    )
                else:
                    t_out = await tool_task

            s["tool_results"] = t_out.get("tool_results", [])
            s["tool_result_status"] = t_out.get("tool_result_status")
            s["ranked_ctx"] = r_out.get("ranked_ctx", [])
            s["embed_cache"] = r_out.get("embed_cache", s.get("embed_cache", {}))
            fl = list(s.get("fallbacks_local") or [])
            fl.extend(x for x in (r_out.get("fallbacks_local") or []) if x not in fl)
            s["fallbacks_local"] = fl
            retriever_agent.apply_hybrid_postprocess(s)
            return s

        # Core DAG: planner → parallel tool+RAG → answer → critic → file_write_test
        workflow = ["Planner", "ParallelRetrieval", "Answer", "Critic", "FileWriteTest"]
        if mode == "edit":
            workflow = [
                "Planner",
                "ParallelRetrieval",
                "EditPlanner",
                "Answer",
                "Critic",
                "FileWriteTest",
            ]
        ws_root = _workspace_root_for_repo(repo_id)
        self._repo_path = ws_root
        state = {
            "query": question,
            "repo_id": repo_id,
            "workspace_root": ws_root,
            "prior_react_signal_paths": [],
            "trace": trace,
            "embed_cache": {},
            "fallbacks_local": [],
            "session_id": session_id,
            "critic_retry_count": 0,
            "max_retries": ac["max_retries"],
            "adaptive_config": ac,
            "user_context": context,
            "classification": classification,
            "disable_aggressive_overrides": disable_aggressive_overrides,
            "mode": mode,
            "write_enabled": mode == "edit",
            "edit_plan": {},
            "edit_context": {},
            "auto_apply": auto_apply,
            "dry_run": dry_run,
            "rag_used": False,
            "rag_reason": "",
            "react_steps": [],
            "react_observations": [],
            "files_read": [],
            "symbols_found": [],
            "react_force_rag": False,
            "react_priority_files": [],
            "react_cancel_event": kwargs.get("react_cancel_event"),
            "global_insights": [],
            "shared_context": {"top_files": [], "top_symbols": []},
            "learned_file_bias": [],
            "cross_subgoal_summary": "",
            "cross_subgoal_meta": {},
            "next_subgoal_bias": [],
            "active_working_set": [],
            "long_run_mode": bool(kwargs.get("long_run_mode")),
        }
        for _cb_key in (
            "stream_callback",
            "status_callback",
            "confirm_diff_callback",
            "confirm_shell_callback",
        ):
            if _cb_key in kwargs:
                state[_cb_key] = kwargs[_cb_key]

        try:
            status_cb = kwargs.get("status_callback") if "kwargs" in locals() else state.get("status_callback")

            from app.agents.react_grounding import (
                compute_answer_confidence,
                is_trivial_query,
                looks_like_repo_exploration_query,
                low_signal_probe_max_steps,
            )
            from app.core.event_bus import EventBus

            if mode == "ask" and is_trivial_query(question, state):
                prompt = (
                    "You are a helpful coding assistant. Reply briefly and naturally "
                    "(one short paragraph or less for greetings and tiny talk).\n\n"
                    f"User: {question}\nAssistant:"
                )
                return await self._ask_mode_lightweight_reply(
                    question=question,
                    classification=classification,
                    session_id=session_id,
                    trace=trace,
                    kwargs=kwargs,
                    status_cb=status_cb,
                    repo_id=repo_id,
                    state=state,
                    status_message="Quick reply (trivial query)…",
                    llm_prompt=prompt,
                    event_reason="trivial_query",
                    router_source="trivial_fast_path",
                    rag_reason="trivial_fast_path",
                    agent_intent="trivial",
                )

            low_sig_thr = float(
                getattr(self.settings, "ORCHESTRATOR_LOW_SIGNAL_CONFIDENCE", 0.35)
                or 0.35
            )
            cls_c0 = float(classification.get("confidence", 1.0) or 1.0)
            if mode == "ask" and cls_c0 < low_sig_thr:
                if looks_like_repo_exploration_query(question):
                    state["react_low_signal_repo_probe"] = True
                    state["react_probe_max_steps"] = low_signal_probe_max_steps(
                        self.settings,
                        state.get("workspace_root"),
                    )
                else:
                    prompt_ls = (
                        "You are a helpful coding assistant. The query had a weak automatic "
                        "match to the indexed codebase, so you are answering without loaded "
                        "repository context. Be concise and helpful using general software "
                        "knowledge; do not invent project-specific file paths, symbols, or "
                        "internals you have not been given.\n\n"
                        f"User: {question}\nAssistant:"
                    )
                    return await self._ask_mode_lightweight_reply(
                        question=question,
                        classification=classification,
                        session_id=session_id,
                        trace=trace,
                        kwargs=kwargs,
                        status_cb=status_cb,
                        repo_id=repo_id,
                        state=state,
                        status_message="Direct reply (weak query signal)…",
                        llm_prompt=prompt_ls,
                        event_reason="low_classification_confidence",
                        router_source="low_signal_fast_path",
                        rag_reason="low_classification_bypass",
                        agent_intent=str(
                            classification.get("agent_intent") or "exploratory"
                        ),
                    )

            # Phase 6: Autonomous Memory mapped natively explicitly into state
            if status_cb: await status_cb("🧠 Analyzing historical vectors natively...")
            from app.memory.intel_store import fetch_memory_context
            if self._db and repo_id:
                mems = await fetch_memory_context(self._db, repo_id, question, state.get("base_query_embed"))
                state["memories"] = [{'summary': f"({m['type'].upper()}) [{m['symbol']}] {m['context']}"} for m in mems[:5]]
                if mode == "edit":
                    try:
                        from app.memory.intel_store import fetch_edit_learning_file_bias

                        lb = await fetch_edit_learning_file_bias(
                            self._db, repo_id, question
                        )
                        if lb:
                            state["learned_file_bias"] = lb
                    except Exception as exc:
                        logger.debug("fetch_edit_learning_file_bias: %s", exc)

            # Phase 2: Execution Planner boundaries
            if status_cb: await status_cb("💬 Structuring execution task sequence natively...")
            ep_agent = AgentRegistry.get("ExecutionPlanner")
            state = await ep_agent.run(state)
            execution_steps = state.get("execution_steps", [question])
            state["step_history"] = []
            
            max_total_iterations = int(
                getattr(self.settings, "ORCHESTRATOR_MAX_ITERATIONS", 16) or 16
            )
            total_iterations = 0
            
            # Persist original stream callback
            original_stream_cb = kwargs.get("stream_callback") if "kwargs" in locals() else state.get("stream_callback")
            
            for step_idx, step in enumerate(execution_steps):
                state["current_step"] = step
                state["current_step_index"] = step_idx
                logger.info(f"--- [Step {step_idx+1}/{len(execution_steps)}] {step} ---")
                
                # Streaming Phase 1: stream only on the final answer step.
                # Edit mode disables streaming so the TUI can show ReAct + structured output first.
                if step_idx == len(execution_steps) - 1 and mode != "edit":
                    if original_stream_cb:
                        state["stream_callback"] = original_stream_cb
                else:
                    state.pop("stream_callback", None)

                state.setdefault("failure_signatures", [])
                    
                ptr = 0
                state["critic_retry_count"] = 0
                
                while ptr < len(workflow):
                    if total_iterations >= max_total_iterations:
                        logger.error(f"Halting orchestrator loop early! Exceeded max total logic steps ({max_total_iterations})!")
                        break
                        
                    node_name = workflow[ptr]
                    total_iterations += 1
                    
                    status_cb = kwargs.get("status_callback") if "kwargs" in locals() else state.get("status_callback")
                    if status_cb:
                        if node_name == "Planner": await status_cb(f"🧠 Planning task sequence [Step {step_idx+1}/{len(execution_steps)}]...")
                        elif node_name == "ParallelRetrieval": await status_cb("⚙️ Retrieving context natively...")
                        elif node_name == "EditPlanner": await status_cb("📝 Planning edits from exploration…")
                        elif node_name == "Answer": await status_cb("🧠 Reasoning and generating...")
                        elif node_name == "FileWriteTest": await status_cb("🛠 Applying patches & evaluating Autonomous Sandboxes...")
                    
                    if node_name == "ParallelRetrieval":
                        retriever_ref = AgentRegistry.get("Retriever")

                        if self.settings.REACT_AGENT_ENABLED and mode != "plan":
                            from app.agents.reasoning_loop import (
                                enough_context,
                                finalize_react_edit_session,
                                run_reasoning_loop,
                            )
                            from app.agents.hierarchical_planner import (
                                Subgoal,
                                build_active_working_set,
                                create_substate,
                                decompose_goal,
                                estimate_dynamic_score,
                                extract_key_findings,
                                merge_react_subgoal_results,
                                merged_files_so_far,
                                merged_symbols_so_far,
                                partial_subgoal_traces_for_synthesis,
                                should_stop_subgoal_loop,
                                subgoal_success,
                                subgoal_synthesis_bias_score,
                            )
                            from app.agents.react_grounding import (
                                compute_answer_confidence,
                                has_strong_merged_context,
                                has_strong_merged_lists,
                            )
                            from app.core.event_bus import EventBus
                            from app.tools.registry import ToolRegistry

                            prior_tools = list(state.get("tool_results") or [])
                            q_for_plan = state.get("current_step") or state.get(
                                "query", ""
                            )
                            plan = decompose_goal(q_for_plan)
                            state["hierarchical_plan"] = {
                                "goal": plan.goal,
                                "subgoals": [
                                    {
                                        "id": sg.id,
                                        "description": sg.description,
                                        "priority": sg.priority,
                                    }
                                    for sg in plan.subgoals
                                ],
                            }

                            if status_cb:
                                nsg = len(plan.subgoals)
                                await status_cb(
                                    f"Exploring codebase ({nsg} subgoal{'s' if nsg != 1 else ''}, tool-first)…"
                                )
                            reg = ToolRegistry(
                                state.get("workspace_root") or os.getcwd(),
                                db=self._db,
                                repo_name=repo_id,
                            )

                            subgoal_runs: list = []
                            hierarchical = bool(
                                getattr(
                                    self.settings,
                                    "HIERARCHICAL_REACT_ENABLED",
                                    True,
                                )
                            )

                            react_ms = int(self.settings.REACT_MAX_STEPS)
                            if state.get("react_low_signal_repo_probe"):
                                react_ms = max(
                                    1,
                                    int(
                                        state.get("react_probe_max_steps")
                                        or low_signal_probe_max_steps(
                                            self.settings,
                                            state.get("workspace_root"),
                                        )
                                    ),
                                )

                            if not hierarchical:
                                try:
                                    state = await run_reasoning_loop(
                                        state,
                                        self._provider,
                                        reg,
                                        max_steps=react_ms,
                                    )
                                finally:
                                    await finalize_react_edit_session(
                                        reg,
                                        state,
                                        commit=subgoal_success(state),
                                    )
                                    state.pop("react_low_signal_repo_probe", None)
                                    state.pop("react_probe_max_steps", None)
                            else:
                                state.setdefault("global_insights", [])
                                state.setdefault(
                                    "shared_context",
                                    {"top_files": [], "top_symbols": []},
                                )
                                remaining = list(plan.subgoals)
                                completed_ids: list[str] = []
                                while remaining:
                                    if state.get("react_cancel_event") is not None:
                                        ev = state["react_cancel_event"]
                                        if getattr(ev, "is_set", lambda: False)():
                                            logger.info(
                                                "Hierarchical ReAct: cancel before next subgoal"
                                            )
                                            break
                                    bias_hints = state.get("next_subgoal_bias") or []
                                    remaining.sort(
                                        key=lambda sg: (
                                            -estimate_dynamic_score(
                                                sg, subgoal_runs
                                            )
                                            - subgoal_synthesis_bias_score(
                                                sg, bias_hints
                                            ),
                                            -sg.priority,
                                        )
                                    )
                                    sg = remaining.pop(0)
                                    await EventBus.emit(
                                        {
                                            "type": "subgoal_start",
                                            "subgoal_id": sg.id,
                                            "description": sg.description[:200],
                                            "priority": sg.priority,
                                        }
                                    )
                                    sub = create_substate(state, sg)
                                    sub_ms = int(self.settings.REACT_MAX_STEPS)
                                    if (
                                        state.get("react_low_signal_repo_probe")
                                        and not subgoal_runs
                                    ):
                                        pms = max(
                                            1,
                                            int(
                                                state.get("react_probe_max_steps")
                                                or low_signal_probe_max_steps(
                                                    self.settings,
                                                    state.get("workspace_root"),
                                                )
                                            ),
                                        )
                                        sub["react_low_signal_repo_probe"] = True
                                        sub["react_probe_max_steps"] = pms
                                        sub_ms = pms
                                    try:
                                        sub = await run_reasoning_loop(
                                            sub,
                                            self._provider,
                                            reg,
                                            max_steps=sub_ms,
                                        )
                                    finally:
                                        await finalize_react_edit_session(
                                            reg,
                                            sub,
                                            commit=subgoal_success(sub),
                                        )
                                    subgoal_runs.append((sg, sub))
                                    state["prior_react_signal_paths"] = list(
                                        sub.get("prior_react_signal_paths") or []
                                    )
                                    state["active_working_set"] = (
                                        build_active_working_set(
                                            subgoal_runs, query=q_for_plan
                                        )
                                    )
                                    state["global_insights"].append(
                                        extract_key_findings(sub)
                                    )
                                    if (
                                        len(subgoal_runs) >= 2
                                        and remaining
                                        and getattr(
                                            self.settings,
                                            "HIERARCHICAL_SYNTH_FEEDBACK",
                                            True,
                                        )
                                    ):
                                        from app.agents.cross_subgoal_synthesis import (
                                            synthesize_cross_subgoals,
                                        )

                                        fb_state = {
                                            **state,
                                            "subgoal_traces": partial_subgoal_traces_for_synthesis(
                                                subgoal_runs
                                            ),
                                            "query": q_for_plan,
                                        }
                                        try:
                                            cs_fb = await synthesize_cross_subgoals(
                                                fb_state, self._provider
                                            )
                                            if cs_fb:
                                                gaps_fb = list(
                                                    cs_fb.get("gaps") or []
                                                )
                                                conf_fb = list(
                                                    cs_fb.get("conflicts") or []
                                                )
                                                if gaps_fb or conf_fb:
                                                    state["next_subgoal_bias"] = (
                                                        gaps_fb + conf_fb
                                                    )[:14]
                                                if (
                                                    conf_fb
                                                    and remaining
                                                    and getattr(
                                                        self.settings,
                                                        "HIERARCHICAL_CONFLICT_RESOLUTION_SUBGOAL",
                                                        True,
                                                    )
                                                    and not state.get(
                                                        "_conflict_resolution_subgoal_added"
                                                    )
                                                ):
                                                    desc = (
                                                        "Reconcile conflicting "
                                                        "traces between subgoals: "
                                                        + "; ".join(
                                                            str(x)[:140]
                                                            for x in conf_fb[:5]
                                                        )
                                                    )[:400]
                                                    _prio = (
                                                        max(
                                                            (
                                                                sg.priority
                                                                for sg in remaining
                                                            ),
                                                            default=0,
                                                        )
                                                        + 1
                                                    )
                                                    remaining.insert(
                                                        0,
                                                        Subgoal(
                                                            "resolve_conflicts",
                                                            desc,
                                                            _prio,
                                                        ),
                                                    )
                                                    state[
                                                        "_conflict_resolution_subgoal_added"
                                                    ] = True
                                        except Exception as exc:
                                            logger.debug(
                                                "mid-hierarchy synthesis: %s",
                                                exc,
                                            )
                                    state["shared_context"] = {
                                        "top_files": merged_files_so_far(
                                            subgoal_runs
                                        )[:3],
                                        "top_symbols": merged_symbols_so_far(
                                            subgoal_runs
                                        )[:5],
                                    }
                                    sconf = compute_answer_confidence(sub)
                                    if subgoal_success(sub):
                                        completed_ids.append(sg.id)
                                    await EventBus.emit(
                                        {
                                            "type": "subgoal_complete",
                                            "subgoal_id": sg.id,
                                            "confidence": sconf,
                                            "files_found": len(
                                                sub.get("files_read") or []
                                            ),
                                            "symbols_found": len(
                                                sub.get("symbols_found") or []
                                            ),
                                            "useful_steps": int(
                                                sub.get("react_useful_steps") or 0
                                            ),
                                        }
                                    )
                                    if should_stop_subgoal_loop(
                                        plan_size=len(plan.subgoals),
                                        completed_ids=completed_ids,
                                        subgoal_runs=subgoal_runs,
                                        strong_merged_predicate=has_strong_merged_lists,
                                        plan_subgoals=plan.subgoals,
                                    ):
                                        logger.info(
                                            "Hierarchical ReAct: early stop after subgoal %s",
                                            sg.id,
                                        )
                                        break

                                state, _traces = merge_react_subgoal_results(
                                    state,
                                    prior_tool_results=prior_tools,
                                    subgoal_runs=subgoal_runs,
                                )
                                state.pop("react_low_signal_repo_probe", None)
                                state.pop("react_probe_max_steps", None)
                                if hierarchical and len(_traces) >= 2:
                                    from app.agents.cross_subgoal_synthesis import (
                                        synthesize_cross_subgoals,
                                    )

                                    cs = await synthesize_cross_subgoals(
                                        state, self._provider
                                    )
                                    if cs:
                                        state["cross_subgoal_summary"] = (
                                            str(cs.get("summary_text") or "").strip()
                                        )
                                        state["cross_subgoal_meta"] = {
                                            "gaps": list(cs.get("gaps") or []),
                                            "conflicts": list(
                                                cs.get("conflicts") or []
                                            ),
                                            "agreements": list(
                                                cs.get("agreements") or []
                                            ),
                                        }
                                        gaps_n = len(state["cross_subgoal_meta"]["gaps"])
                                        conf_n = len(
                                            state["cross_subgoal_meta"]["conflicts"]
                                        )
                                        if (gaps_n or conf_n) and int(
                                            state.get("react_useful_steps") or 0
                                        ) <= 0:
                                            state["react_force_rag"] = True
                                        fac = 1.0
                                        if conf_n:
                                            fac *= max(
                                                0.55,
                                                1.0 - 0.08 * min(conf_n, 4),
                                            )
                                        if gaps_n:
                                            fac *= max(
                                                0.65,
                                                1.0 - 0.04 * min(gaps_n, 6),
                                            )
                                        state["goal_progress"] = float(
                                            state.get("goal_progress") or 0
                                        ) * float(fac)

                            rc = float(state.get("router_confidence") or 0.0)
                            tool_conf_high = rc > 0.90
                            rag_bypass_types = {"git"}
                            force_no_rag = state.get("force_tool") or (
                                tool_conf_high and q_type in rag_bypass_types
                            )
                            from app.agents.hierarchical_planner import (
                                avg_exploration_confidence,
                            )

                            hier_runs = (
                                subgoal_runs
                                if hierarchical and len(subgoal_runs) > 0
                                else None
                            )
                            if hier_runs is not None:
                                merged_react_conf = avg_exploration_confidence(
                                    hier_runs
                                )
                            else:
                                merged_react_conf = compute_answer_confidence(state)
                            cls_conf = float(
                                (state.get("classification") or {}).get(
                                    "confidence", 1.0
                                )
                                or 1.0
                            )
                            low_signal = merged_react_conf < 0.5 or cls_conf < 0.5
                            forced_rag = bool(state.get("react_force_rag"))
                            strong_ctx = has_strong_merged_context(state)
                            enough = enough_context(state)

                            state.setdefault("rag_decision_trace", [])
                            rd = state["rag_decision_trace"]
                            rd.append(
                                {
                                    "phase": "post_react",
                                    "merged_react_conf": merged_react_conf,
                                    "cls_conf": cls_conf,
                                    "forced_rag": forced_rag,
                                    "strong_ctx": strong_ctx,
                                    "enough_context": enough,
                                    "low_signal": low_signal,
                                }
                            )

                            async def _run_rag(reason: str) -> None:
                                nonlocal state
                                state["rag_used"] = True
                                state["rag_reason"] = reason
                                state["rag_focus_files"] = list(
                                    state.get("react_priority_files") or []
                                )
                                try:
                                    state = await asyncio.wait_for(
                                        retriever_ref.run(dict(state)),
                                        timeout=self.settings.RAG_RETRIEVAL_TIMEOUT_S,
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning(
                                        "Retriever exceeded timeout after ReAct (%s).",
                                        reason,
                                    )
                                    state["ranked_ctx"] = []
                                    fl = list(state.get("fallbacks_local") or [])
                                    fl.append("rag_timeout")
                                    state["fallbacks_local"] = fl
                                except Exception as ex:
                                    logger.warning(
                                        "Retriever failed after ReAct (%s): %s",
                                        reason,
                                        ex,
                                    )
                                    state["ranked_ctx"] = []

                            if force_no_rag:
                                state["ranked_ctx"] = []
                                state["rag_used"] = False
                                state["rag_reason"] = (
                                    "git_or_forced_tool_high_confidence_skip_rag"
                                )
                                rd.append(
                                    {
                                        "branch": "skip_rag",
                                        "detail": state["rag_reason"],
                                    }
                                )
                                fl = list(state.get("fallbacks_local") or [])
                                if "rag_skipped_git_bypass" not in fl:
                                    fl.append("rag_skipped_git_bypass")
                                state["fallbacks_local"] = fl
                            elif strong_ctx:
                                state.setdefault("ranked_ctx", [])
                                state["rag_used"] = False
                                state["rag_reason"] = "strong_merged_context_skip_rag"
                                rd.append(
                                    {
                                        "branch": "skip_rag",
                                        "detail": state["rag_reason"],
                                    }
                                )
                            elif enough:
                                state.setdefault("ranked_ctx", [])
                                state["rag_used"] = False
                                state["rag_reason"] = "enough_context_after_react"
                                rd.append(
                                    {
                                        "branch": "skip_rag",
                                        "detail": state["rag_reason"],
                                    }
                                )
                            elif forced_rag and not strong_ctx:
                                rd.append(
                                    {
                                        "branch": "run_rag",
                                        "detail": "no_progress_after_react_tools",
                                    }
                                )
                                await _run_rag("no_progress_after_react_tools")
                            elif low_signal:
                                rd.append(
                                    {
                                        "branch": "run_rag",
                                        "detail": "low_confidence_boost_rag",
                                    }
                                )
                                await _run_rag("low_confidence_boost_rag")
                            else:
                                rd.append(
                                    {
                                        "branch": "run_rag",
                                        "detail": "insufficient_grounding_after_react",
                                    }
                                )
                                await _run_rag("insufficient_grounding_after_react")

                            await EventBus.emit(
                                {
                                    "type": "rag_triggered",
                                    "reason": str(state.get("rag_reason") or ""),
                                    "rag_used": bool(state.get("rag_used")),
                                    "trace": list(rd[-12:]),
                                }
                            )

                            if trace:
                                trace.react_tool_steps_count = len(
                                    state.get("react_steps") or []
                                )
                                trace.rag_used = bool(state.get("rag_used"))
                                trace.rag_reason = str(
                                    state.get("rag_reason") or ""
                                )
                                trace.react_files_touched = list(
                                    state.get("files_read") or []
                                )
                                trace.react_symbols_used = list(
                                    state.get("symbols_found") or []
                                )

                            retriever_ref.apply_hybrid_postprocess(state)
                            ptr += 1
                            continue

                        # Legacy: parallel one-shot tool + RAG (or plan mode)
                        state.pop("react_low_signal_repo_probe", None)
                        state.pop("react_probe_max_steps", None)
                        rc = float(state.get("router_confidence") or 0.0)
                        tool_conf_high = rc > 0.90
                        rag_bypass_types = {"git"}
                        if state.get("force_tool") or (
                            tool_conf_high and q_type in rag_bypass_types
                        ):
                            logger.info(
                                "[Orchestrator] Bypass Phase 5 rules matched (tool_conf=%s, type=%s).",
                                rc,
                                q_type,
                            )
                            try:
                                agent = AgentRegistry.get("ToolExecutor")
                            except KeyError:
                                agent = AgentRegistry.get("ToolExecutionAgent")
                            try:
                                state = await asyncio.wait_for(
                                    agent.run(state),
                                    timeout=self.settings.TOOL_EXEC_TIMEOUT_S,
                                )
                            except asyncio.TimeoutError:
                                state["tool_results"] = []
                            state["ranked_ctx"] = []
                            ptr += 1
                            continue

                        state = await run_parallel_tool_rag(state)
                        ptr += 1
                        continue

                    # Not a registered agent — inline sandbox (must run before AgentRegistry.get)
                    if node_name == "FileWriteTest":
                        if mode == "edit" and state.get("write_enabled"):
                            import re

                            from app.agents.edit_patch_parse import parse_edit_blocks
                            from app.agents.react_grounding import (
                                compute_answer_confidence,
                            )
                            from app.tools.file_edit_tool import FileEditTool
                            from app.tools.file_write_tool import FileWriteTool
                            from app.tools.test_runner_tool import TestRunnerTool

                            rp = self._repo_path or "."
                            fw_tool = FileWriteTool(repo_path=rp)
                            allowed_set: set[str] = set()
                            for c in state.get("ranked_ctx") or []:
                                fp = (c.get("file_path") or "").replace("\\", "/")
                                if fp:
                                    allowed_set.add(fp)
                            for f in state.get("files_read") or []:
                                allowed_set.add(str(f).replace("\\", "/"))
                            for item in (state.get("edit_plan") or {}).get("plan") or []:
                                if isinstance(item, dict) and item.get("file"):
                                    allowed_set.add(
                                        str(item["file"]).replace("\\", "/")
                                    )
                            allowed_files = list(allowed_set)
                            answer = state.get("answer", "")

                            max_edit = int(
                                getattr(
                                    self.settings, "EDIT_MAX_AUTO_FILES", 5
                                )
                                or 5
                            )
                            max_edit = max(1, max_edit)

                            cls_c = float(
                                (state.get("classification") or {}).get(
                                    "confidence", 1.0
                                )
                                or 1.0
                            )
                            post_answer_conf = float(
                                compute_answer_confidence(state)
                            )
                            force_confirm = (
                                post_answer_conf < 0.6
                                or cls_c < 0.6
                                or bool(state.get("require_user_confirm"))
                            )
                            state["require_user_confirm"] = bool(force_confirm)
                            state["require_edit_confirmation"] = force_confirm

                            _confirm_reasons: list[str] = []
                            if post_answer_conf < 0.6:
                                _confirm_reasons.append(
                                    f"Low answer confidence ({post_answer_conf:.2f})"
                                )
                            if cls_c < 0.6:
                                _confirm_reasons.append(
                                    f"Low classification confidence ({cls_c:.2f})"
                                )
                            if bool(state.get("require_user_confirm")) and (
                                post_answer_conf >= 0.6 and cls_c >= 0.6
                            ):
                                _confirm_reasons.append(
                                    "Explicit confirmation requested "
                                    "(require_user_confirm)"
                                )
                            if force_confirm and not _confirm_reasons:
                                _confirm_reasons.append(
                                    "Safety / policy confirmation required"
                                )

                            confirm_cb = (
                                kwargs.get("confirm_diff_callback")
                                if "kwargs" in locals()
                                else state.get("confirm_diff_callback")
                            )

                            def _file_allowed(rel: str) -> bool:
                                fn = rel.replace("\\", "/")
                                if fn in allowed_set:
                                    return True
                                return any(
                                    fn.startswith(a + "/") for a in allowed_set
                                )

                            def _diff_header(paths: list[str]) -> str:
                                if not paths:
                                    return ""
                                lines = "\n".join(
                                    f" - {p}" for p in paths[:max_edit]
                                )
                                return f"Proposed changes:\n{lines}\n\n"

                            edit_ops = parse_edit_blocks(answer)[:max_edit]

                            if edit_ops:
                                fe = FileEditTool(rp)
                                ok_final = False
                                for _attempt in range(2):
                                    fe.begin_session()
                                    branch_ok = True
                                    for ed in edit_ops:
                                        fp = str(ed["file_path"]).replace(
                                            "\\", "/"
                                        )
                                        if not _file_allowed(fp):
                                            logger.warning(
                                                "[FileWriteTest] Skipping edit "
                                                "(path not allowed): %s",
                                                fp,
                                            )
                                            continue
                                        search_t = ed["search"]
                                        replace_t = ed["replace"]
                                        r = fe.apply_search_replace(
                                            fp,
                                            search_t,
                                            replace_t,
                                            expected_occurrences=1,
                                        )
                                        if not r.success:
                                            from app.agents.edit_repair import (
                                                repair_search_replace,
                                            )

                                            max_repairs = int(
                                                getattr(
                                                    self.settings,
                                                    "EDIT_MAX_REPAIR_ATTEMPTS",
                                                    2,
                                                )
                                                or 2
                                            )
                                            max_repairs = max(1, min(max_repairs, 5))
                                            for _repair in range(max_repairs):
                                                fixed = await repair_search_replace(
                                                    self._provider,
                                                    repo_path=rp,
                                                    rel_path=fp,
                                                    original_search=search_t,
                                                    original_replace=replace_t,
                                                    error=str(r.error or ""),
                                                    user_query=state.get(
                                                        "query", question
                                                    ),
                                                )
                                                if not fixed:
                                                    break
                                                r = fe.apply_search_replace(
                                                    fp,
                                                    fixed["search"],
                                                    fixed["replace"],
                                                    expected_occurrences=1,
                                                )
                                                if r.success:
                                                    break
                                        if not r.success:
                                            logger.warning(
                                                "[FileWriteTest] transactional "
                                                "edit failed: %s",
                                                r.error,
                                            )
                                            branch_ok = False
                                            break
                                    if branch_ok and fe.touched_files():
                                        ok_final = True
                                        break
                                    fe.rollback_session()

                                if ok_final and fe.touched_files():
                                    touched = fe.touched_files()
                                    combined_diff = (
                                        _diff_header(touched)
                                        + fe.unified_diff_preview()
                                    )
                                    conf = True
                                    if (
                                        confirm_cb
                                        and (not auto_apply or force_confirm)
                                        and not dry_run
                                    ):
                                        conf = await invoke_confirm_diff_callback(
                                            confirm_cb,
                                            combined_diff,
                                            confirm_reasons=_confirm_reasons
                                            if _confirm_reasons
                                            else None,
                                        )

                                    if dry_run:
                                        fe.rollback_session()
                                        logger.info(
                                            "[FileWriteTest] Dry-run (transactional edits)."
                                        )
                                        ptr += 1
                                        continue

                                    if not state.get("files_read"):
                                        logger.warning(
                                            "[FileWriteTest] Refusing apply: "
                                            "no files_read."
                                        )
                                        fe.rollback_session()
                                        ptr += 1
                                        continue

                                    if conf:
                                        cres = fe.commit_session(allowed_set)
                                        if cres.success:
                                            if self._db and repo_id:
                                                try:
                                                    from app.memory.intel_store import (
                                                        record_edit_learning,
                                                    )

                                                    await record_edit_learning(
                                                        self._db,
                                                        repo_id,
                                                        query=state.get(
                                                            "query", question
                                                        ),
                                                        files=touched,
                                                        success=True,
                                                        detail="transactional_edit",
                                                    )
                                                except Exception as le:
                                                    logger.debug(
                                                        "record_edit_learning: %s",
                                                        le,
                                                    )
                                            tr_res = TestRunnerTool(
                                                rp
                                            ).execute_tests()
                                            if not tr_res.success:
                                                logger.warning(
                                                    "[FileWriteTest] Tests failed: %s",
                                                    tr_res.error[:200],
                                                )
                                                if (
                                                    state["critic_retry_count"]
                                                    < state["max_retries"]
                                                ):
                                                    sig = hash(
                                                        tr_res.error[-500:]
                                                    )
                                                    state[
                                                        "failure_signatures"
                                                    ].append(sig)
                                                    if (
                                                        state[
                                                            "failure_signatures"
                                                        ].count(sig)
                                                        >= 3
                                                    ):
                                                        logger.error(
                                                            "Trajectory Halted: repeated test failure."
                                                        )
                                                        state[
                                                            "abort_reason"
                                                        ] = "repeated_failure"
                                                        state[
                                                            "failure_signature"
                                                        ] = tr_res.error[-500:]
                                                        ptr += 1
                                                        continue
                                                    state["critic_retry"] = True
                                                    state[
                                                        "critic_feedback"
                                                    ] = (
                                                        f"Tests failed after edit:\n{tr_res.error[-1500:]}\nPlease fix."
                                                    )
                                                    state[
                                                        "critic_retry_count"
                                                    ] += 1
                                                    ptr = 0
                                                    continue
                                                else:
                                                    state[
                                                        "test_failure_final"
                                                    ] = tr_res.error
                                                    state[
                                                        "abort_reason"
                                                    ] = "max_retries_exceeded"
                                                    state[
                                                        "failure_signature"
                                                    ] = tr_res.error[-500:]
                                        else:
                                            fe.rollback_session()
                                            if self._db and repo_id:
                                                try:
                                                    from app.memory.intel_store import (
                                                        record_edit_learning,
                                                    )

                                                    await record_edit_learning(
                                                        self._db,
                                                        repo_id,
                                                        query=state.get(
                                                            "query", question
                                                        ),
                                                        files=touched,
                                                        success=False,
                                                        detail=str(
                                                            cres.error
                                                        )[:400],
                                                    )
                                                except Exception:
                                                    pass
                                            logger.error(
                                                "[FileWriteTest] commit failed: %s",
                                                cres.error,
                                            )
                                    else:
                                        fe.rollback_session()
                                        logger.info(
                                            "[FileWriteTest] Transactional edits "
                                            "discarded by user."
                                        )
                                    ptr += 1
                                    continue
                                if edit_ops and self._db and repo_id:
                                    try:
                                        from app.memory.intel_store import (
                                            record_edit_learning,
                                        )

                                        await record_edit_learning(
                                            self._db,
                                            repo_id,
                                            query=state.get("query", question),
                                            files=[
                                                str(
                                                    e.get("file_path", "")
                                                ).replace("\\", "/")
                                                for e in edit_ops
                                            ],
                                            success=False,
                                            detail="transactional_edit_failed",
                                        )
                                    except Exception:
                                        pass

                            patch_matches = re.finditer(
                                r"<patch file=['\"]?(.*?)['\"]?>(.*?)</patch>",
                                answer,
                                re.DOTALL,
                            )
                            patches = []
                            diffs = []
                            for match in patch_matches:
                                fp = match.group(1).strip()
                                nc = match.group(2).strip()
                                df = fw_tool.generate_diff(fp, nc)
                                if df:
                                    patches.append(
                                        {"file_path": fp, "new_content": nc}
                                    )
                                    diffs.append(df)

                            patches = patches[:max_edit]

                            if patches:
                                paths = [p["file_path"] for p in patches]
                                combined_diff = _diff_header(paths) + "\n\n".join(
                                    diffs
                                )
                                conf = True
                                if (
                                    confirm_cb
                                    and (not auto_apply or force_confirm)
                                    and not dry_run
                                ):
                                    conf = await invoke_confirm_diff_callback(
                                        confirm_cb,
                                        combined_diff,
                                        confirm_reasons=_confirm_reasons
                                        if _confirm_reasons
                                        else None,
                                    )

                                if dry_run:
                                    logger.info(
                                        "[FileWriteTest] Dry-run (full-file patches)."
                                    )
                                    ptr += 1
                                    continue

                                if not (state.get("files_read") or dry_run):
                                    logger.warning(
                                        "[FileWriteTest] Refusing apply: no files_read "
                                        "(exploration required before edit)."
                                    )
                                    ptr += 1
                                    continue

                                if conf:
                                    res = fw_tool.apply_patches(
                                        patches, allowed_files
                                    )
                                    if res.success:
                                        if self._db and repo_id:
                                            try:
                                                from app.memory.intel_store import (
                                                    record_edit_learning,
                                                )

                                                await record_edit_learning(
                                                    self._db,
                                                    repo_id,
                                                    query=state.get(
                                                        "query", question
                                                    ),
                                                    files=[
                                                        p["file_path"]
                                                        for p in patches
                                                    ],
                                                    success=True,
                                                    detail="patch_applied",
                                                )
                                            except Exception as le:
                                                logger.debug(
                                                    "record_edit_learning: %s", le
                                                )
                                        state["patch_branch"] = res.data.get(
                                            "branch"
                                        )
                                        tr_res = TestRunnerTool(
                                            rp
                                        ).execute_tests()
                                        if not tr_res.success:
                                            logger.warning(
                                                "[FileWriteTest] Tests failed: %s",
                                                tr_res.error[:200],
                                            )
                                            if (
                                                state["critic_retry_count"]
                                                < state["max_retries"]
                                            ):
                                                sig = hash(tr_res.error[-500:])
                                                state["failure_signatures"].append(
                                                    sig
                                                )
                                                if (
                                                    state["failure_signatures"].count(
                                                        sig
                                                    )
                                                    >= 3
                                                ):
                                                    logger.error(
                                                        "Trajectory Halted: repeated test failure."
                                                    )
                                                    state["abort_reason"] = (
                                                        "repeated_failure"
                                                    )
                                                    state["failure_signature"] = (
                                                        tr_res.error[-500:]
                                                    )
                                                    ptr += 1
                                                    continue
                                                state["critic_retry"] = True
                                                state["critic_feedback"] = (
                                                    f"Tests failed after patch:\n{tr_res.error[-1500:]}\nPlease fix."
                                                )
                                                state["critic_retry_count"] += 1
                                                ptr = 0
                                                continue
                                            else:
                                                state["test_failure_final"] = (
                                                    tr_res.error
                                                )
                                                state["abort_reason"] = (
                                                    "max_retries_exceeded"
                                                )
                                                state["failure_signature"] = (
                                                    tr_res.error[-500:]
                                                )
                                    else:
                                        if self._db and repo_id:
                                            try:
                                                from app.memory.intel_store import (
                                                    record_edit_learning,
                                                )

                                                await record_edit_learning(
                                                    self._db,
                                                    repo_id,
                                                    query=state.get(
                                                        "query", question
                                                    ),
                                                    files=[
                                                        p["file_path"]
                                                        for p in patches
                                                    ],
                                                    success=False,
                                                    detail=str(res.error)[:400],
                                                )
                                            except Exception:
                                                pass
                                        logger.error(
                                            "[FileWriteTest] Patch failed: %s",
                                            res.error,
                                        )
                                else:
                                    logger.info(
                                        "[FileWriteTest] Patch discarded by user."
                                    )
                        ptr += 1
                        continue

                    agent = AgentRegistry.get(node_name)
                    state = await agent.run(state)

                    if node_name == "Critic":
                        from app.agents.critic_executor import apply_critic_action
                        should_retry = apply_critic_action(state)

                        if mode == "edit" and state.get("classification", {}).get("confidence", 1.0) < 0.7:
                            logger.warning(
                                "Aborting edit execution writes due to low confidence."
                            )
                            state["write_enabled"] = False

                        if (
                            state.get("critic_retry") or should_retry
                        ) and state["critic_retry_count"] < state["max_retries"]:
                            logger.warning(
                                "Critic requested retry (attempt %s).",
                                state["critic_retry_count"],
                            )
                            state["critic_retry_count"] += 1
                            ptr = 0
                        else:
                            ptr += 1
                    else:
                        ptr += 1

                # Track steps dynamically natively capturing final context states across jumps!
                state["step_history"].append({
                    "step": step,
                    "answer_snippet": state.get("answer", "")[:100],
                    "status": "failed" if state.get("abort_reason") else "success"
                })

                if state.get("abort_reason") or state.get("test_failure_final"):
                    logger.error(f"Step {step_idx+1} aborted catastrophically natively. Breaking multi-step execution boundaries!")
                    state["step_status"] = "failed"
                    break
                else:
                    state["step_status"] = "success"

            if state.get("critic_retry") and state.get("critic_retry_count", 0) >= state.get("max_retries", 2):
                fb = (
                    "I'm unable to confidently resolve this query after multiple attempts. "
                    "Here is the best available context from the pipeline.\n\n"
                )
                prev = (state.get("answer") or "").strip()
                state["answer"] = fb + (prev[:2000] if prev else "")
                state["critic_retry"] = False

            state["final_confidence"] = min(
                float(state.get("router_confidence", 1.0) or 1.0),
                float(classification.get("confidence", 1.0)),
                0.5 if state.get("abort_reason") else 1.0
            )

            # Log raw foreground memory query
            from app.memory.intel_store import store_query, store_bug, store_decision
            base_query_embed = state.get("base_query_embed")
            if self._db and repo_id:
                await store_query(self._db, repo_id, question, state.get("answer", "")[:2500], base_query_embed)
                
                # Gap 5: Native Memory Loop
                if mode != "plan":
                    fe = classification.get("focus_entities", [])
                    sym = fe[0] if fe else question[:50]
                    ans_slice = state.get("answer", "")[:2000]
                    
                    if q_type == "debugging":
                        await store_bug(self._db, repo_id, sym, ans_slice)
                    elif q_type == "architecture":
                        await store_decision(self._db, repo_id, sym, ans_slice)

            from app.agents.react_grounding import compute_answer_confidence

            return_dict = {
                "answer": state.get("answer", ""),
                "session_id": session_id,
                "latency_ms": state.get("latency_ms", 0.0),
                "model": self.settings.LLM_MODEL,
                "provider": self.settings.LLM_PROVIDER,
                "meta": {
                    "selected_tool": state.get("selected_tool"),
                    "router_confidence": state.get("router_confidence"),
                    "router_source": state.get("router_source"),
                    "retrieved_files": list({c["file_path"] for c in state.get("ranked_ctx", [])}),
                    "patch_branch": state.get("patch_branch"),
                    "final_confidence": state.get("final_confidence"),
                    "abort_reason": state.get("abort_reason"),
                    "failure_signature": state.get("failure_signature"),
                    "retry_count": state.get("critic_retry_count", 0),
                    "adaptive_config": state.get("adaptive_config"),
                    "classification": state.get("classification"),
                    "agent_intent": (state.get("classification") or {}).get(
                        "agent_intent"
                    ),
                    "early_cancel_rag": state.get("early_cancel_rag", False),
                    "early_cancel_tool": state.get("early_cancel_tool", False),
                    "tool_steps_count": len(state.get("react_steps") or []),
                    "rag_used": bool(state.get("rag_used")),
                    "rag_reason": state.get("rag_reason") or "",
                    "files_touched": list(state.get("files_read") or []),
                    "symbols_used": list(state.get("symbols_found") or []),
                    "react_force_rag": bool(state.get("react_force_rag")),
                    "react_useful_steps": int(state.get("react_useful_steps") or 0),
                    "react_failed_steps": int(state.get("react_failed_steps") or 0),
                    "react_priority_files": list(state.get("react_priority_files") or []),
                    "answer_confidence": compute_answer_confidence(state),
                    "require_edit_confirmation": bool(
                        state.get("require_edit_confirmation")
                    ),
                    "learned_file_bias": list(state.get("learned_file_bias") or []),
                    "goal_progress": float(state.get("goal_progress") or 0),
                    "subgoal_confidence": state.get("subgoal_confidence") or {},
                    "hierarchical_react": bool(
                        getattr(
                            self.settings, "HIERARCHICAL_REACT_ENABLED", True
                        )
                    ),
                    "hierarchical_plan": state.get("hierarchical_plan"),
                    "hierarchical_subgoal_count": len(
                        (state.get("hierarchical_plan") or {}).get("subgoals")
                        or []
                    ),
                    "cross_subgoal_meta": state.get("cross_subgoal_meta") or {},
                    "early_stop_reason": state.get("react_early_stop_reason")
                    or "",
                    "react_early_stop_reason": state.get("react_early_stop_reason")
                    or "",
                    "rag_decision_trace": list(state.get("rag_decision_trace") or []),
                    "require_user_confirm": float(
                        compute_answer_confidence(state)
                    )
                    < 0.6
                    or bool(state.get("require_user_confirm")),
                    "active_working_set": list(
                        state.get("active_working_set") or []
                    ),
                    "shared_context": state.get("shared_context") or {},
                    "global_insights": list(
                        (state.get("global_insights") or [])[:24]
                    ),
                    "subgoal_traces": state.get("subgoal_traces") or [],
                    "cross_subgoal_summary": state.get("cross_subgoal_summary")
                    or "",
                    "file_scores": state.get("file_scores") or {},
                },
            }
            if trace:
                return_dict["trace"] = trace.model_dump()
                
            # Log structural metrics directly avoiding LLM payload burdens across raw UI graphs!
            from app.logging.telemetry import telemetry_tracker
            state["model"] = self.settings.LLM_MODEL
            telemetry_tracker.export_metrics(session_id, state)

            from app.core.event_bus import EventBus

            await EventBus.emit(
                {
                    "type": "final_answer",
                    "content": state.get("answer") or "",
                    "session_id": session_id,
                }
            )

            return return_dict

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            if self._interaction_logger and session_id:
                await self._interaction_logger.log_error(
                    session_id=session_id,
                    question=question,
                    full_prompt=state.get("full_prompt", ""),
                    provider=self.settings.LLM_PROVIDER,
                    model=self.settings.LLM_MODEL,
                    error_message=str(exc),
                    context=state.get("context", ""),
                    latency_ms=latency_ms,
                )
            raise
