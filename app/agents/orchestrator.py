"""
Agent Orchestrator — central coordination layer.

Routes user queries through the appropriate provider and returns the result.
Integrates ContextBuilder, composite RAG + Git scoring, optional LLM rerank, interaction logging.
"""
import asyncio
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.context_builder import ContextBuilder
from app.core.config import get_settings
from app.core.interaction_logger import InteractionLogger
from app.core.logger import get_logger
from app.providers.base import LLMProvider
from app.providers.factory import get_provider

logger = get_logger(__name__)
settings = get_settings()


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

        # Register instances
        AgentRegistry.register(ExecutionPlanner(self._provider))
        AgentRegistry.register(PlannerAgent(self._db, self._provider))
        AgentRegistry.register(ToolExecutionAgent())
        AgentRegistry.register(RetrieverAgent(self._db, self._provider))
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
        state = {
            "query": question,
            "repo_id": repo_id,
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
            "write_enabled": mode == "build",
            "auto_apply": auto_apply,
            "dry_run": dry_run,
        }

        try:
            status_cb = kwargs.get("status_callback") if "kwargs" in locals() else state.get("status_callback")
            
            # Phase 6: Autonomous Memory mapped natively explicitly into state
            if status_cb: await status_cb("🧠 Analyzing historical vectors natively...")
            from app.memory.intel_store import fetch_memory_context
            if self._db and repo_id:
                mems = await fetch_memory_context(self._db, repo_id, question, state.get("base_query_embed"))
                state["memories"] = [{'summary': f"({m['type'].upper()}) [{m['symbol']}] {m['context']}"} for m in mems[:5]]

            # Phase 2: Execution Planner boundaries
            if status_cb: await status_cb("💬 Structuring execution task sequence natively...")
            ep_agent = AgentRegistry.get("ExecutionPlanner")
            state = await ep_agent.run(state)
            execution_steps = state.get("execution_steps", [question])
            state["step_history"] = []
            
            max_total_iterations = 6
            total_iterations = 0
            
            # Persist original stream callback
            original_stream_cb = kwargs.get("stream_callback") if "kwargs" in locals() else state.get("stream_callback")
            
            for step_idx, step in enumerate(execution_steps):
                state["current_step"] = step
                state["current_step_index"] = step_idx
                logger.info(f"--- [Step {step_idx+1}/{len(execution_steps)}] {step} ---")
                
                # Streaming Phase 1: Stream ONLY on final answer natively
                # V1 Constraints: Stream explicitly disabled under 'build' mode ensuring Textual UX Stability!
                if step_idx == len(execution_steps) - 1 and mode != "build":
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
                        elif node_name == "Answer": await status_cb("🧠 Reasoning and generating...")
                        elif node_name == "FileWriteTest": await status_cb("🛠 Applying patches & evaluating Autonomous Sandboxes...")
                    
                    if node_name == "ParallelRetrieval":
                        rc = float(state.get("router_confidence") or 0.0)
                        tool_conf_high = rc > 0.90
                        # Phase 5: Bypass RAG only when a deterministic tool is sufficient.
                        # Do not bypass "navigation": the router often picks `file`, and
                        # FileTool.search uses the raw query as a literal grep — natural
                        # language questions then skip RAG and return empty matches.
                        rag_bypass_types = {"git"}
                        if state.get("force_tool") or (tool_conf_high and q_type in rag_bypass_types):
                            logger.info(f"[Orchestrator] Bypass Phase 5 rules matched (tool_conf={rc}, type={q_type}). Tracing exclusively over Tool execution limits!")
                            try:
                                agent = AgentRegistry.get("ToolExecutor")
                            except KeyError:
                                agent = AgentRegistry.get("ToolExecutionAgent")
                            try:
                                state = await asyncio.wait_for(agent.run(state), timeout=self.settings.TOOL_EXEC_TIMEOUT_S)
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
                        if mode == "build" and state.get("write_enabled"):
                            import re
                            from app.tools.file_write_tool import FileWriteTool
                            from app.tools.test_runner_tool import TestRunnerTool

                            fw_tool = FileWriteTool(repo_path=self._repo_path or ".")
                            allowed_files = list({c["file_path"] for c in state.get("ranked_ctx", [])})
                            answer = state.get("answer", "")

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
                                    patches.append({"file_path": fp, "new_content": nc})
                                    diffs.append(df)

                            if patches:
                                combined_diff = "\\n\\n".join(diffs)
                                conf = True
                                confirm_cb = (
                                    kwargs.get("confirm_diff_callback")
                                    if "kwargs" in locals()
                                    else state.get("confirm_diff_callback")
                                )
                                if confirm_cb and not auto_apply and not dry_run:
                                    conf = await confirm_cb(combined_diff)

                                if dry_run:
                                    logger.info(
                                        "[FileWriteTest] Dry-Run mapped flawlessly avoiding executions!"
                                    )
                                    ptr += 1
                                    continue

                                if conf:
                                    res = fw_tool.apply_patches(patches, allowed_files)
                                    if res.success:
                                        state["patch_branch"] = res.data.get("branch")
                                        tr_res = TestRunnerTool(
                                            self._repo_path or "."
                                        ).execute_tests()
                                        if not tr_res.success:
                                            logger.warning(
                                                "[FileWriteTest] Tests Failed! Routing Critic Retries natively: %s",
                                                tr_res.error[:200],
                                            )
                                            if state["critic_retry_count"] < state["max_retries"]:
                                                sig = hash(tr_res.error[-500:])
                                                state["failure_signatures"].append(sig)
                                                if state["failure_signatures"].count(sig) >= 3:
                                                    logger.error(
                                                        "Trajectory Halted: Same autonomous failure boundary hit 3 times dynamically natively. Extracted limit."
                                                    )
                                                    state["abort_reason"] = "repeated_failure"
                                                    state["failure_signature"] = tr_res.error[
                                                        -500:
                                                    ]
                                                    break

                                                state["critic_retry"] = True
                                                state["critic_feedback"] = (
                                                    f"Tests failed mapping patches explicitly: \\n{tr_res.error[-1500:]}\\nPlease fix."
                                                )
                                                state["critic_retry_count"] += 1
                                                ptr = 0
                                                continue
                                            else:
                                                state["test_failure_final"] = tr_res.error
                                                state["abort_reason"] = "max_retries_exceeded"
                                                state["failure_signature"] = tr_res.error[-500:]
                                    else:
                                        logger.error(
                                            "[FileWriteTest] Patch execution boundary hit limits natively: %s",
                                            res.error,
                                        )
                                else:
                                    logger.info(
                                        "[FileWriteTest] Patch Discarded structurally bounds User constraints!"
                                    )
                        ptr += 1
                        continue

                    agent = AgentRegistry.get(node_name)
                    state = await agent.run(state)

                    if node_name == "Critic":
                        from app.agents.critic_executor import apply_critic_action
                        should_retry = apply_critic_action(state)

                        if mode == "build" and state.get("classification", {}).get("confidence", 1.0) < 0.7:
                            logger.warning(
                                "Aborting Build execution writes due to low Confidence evaluations mapped natively!"
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
                    "early_cancel_rag": state.get("early_cancel_rag", False),
                    "early_cancel_tool": state.get("early_cancel_tool", False),
                },
            }
            if trace:
                return_dict["trace"] = trace.model_dump()
                
            # Log structural metrics directly avoiding LLM payload burdens across raw UI graphs!
            from app.logging.telemetry import telemetry_tracker
            state["model"] = self.settings.LLM_MODEL
            telemetry_tracker.export_metrics(session_id, state)
            
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
