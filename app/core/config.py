"""
Central configuration using Pydantic Settings.

Loads values from environment variables → .env file → defaults.
"""
from functools import lru_cache
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — single source of truth."""

    model_config = SettingsConfigDict(
        env_file=os.path.expanduser("~/.deepcode.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "DeepCode Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── Server ────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM Provider ─────────────────────────────────────────────────────
    LLM_PROVIDER: str = "ollama"  # "ollama" | "openai" | "anthropic"
    LLM_MODEL: str = "gemma4:26b"
    LLM_ROUTER_MODEL: str = "gemma4:26b" # Configured for fast low-latency tasks
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # ── Ollama ────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Large local models (e.g. 26B+) often exceed 120s on cold load + first token.
    OLLAMA_HTTP_TIMEOUT_S: float = 300.0
    # Ollama /api/chat "think" flag. Gemma 4 / Qwen-style models need False or `message.content` stays empty.
    OLLAMA_NATIVE_THINKING: bool = False

    # ── Embedding ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "qwen3-embedding:8b"
    # Ollama's qwen3-embedding:8b returns 4096 floats by default (Matryoshka-capable; we use full dim).
    EMBEDDING_DIMENSION: int = 4096

    # ── RAG / code intelligence ───────────────────────────────────────────
    MAX_CONTEXT_TOKENS: int = 6000
    # Per-layer context budgets (approximate tokens, char/4) for answer assembly
    CONTEXT_BUDGET_MEMORY: int = 800
    CONTEXT_BUDGET_TOOL: int = 600
    CONTEXT_BUDGET_RAG: int = 800
    CONTEXT_BUDGET_SYMBOL: int = 400
    CONTEXT_BUDGET_GIT: int = 400
    # Parallel phase timeouts (seconds); tune for your provider/network
    TOOL_EXEC_TIMEOUT_S: float = 45.0
    RAG_RETRIEVAL_TIMEOUT_S: float = 90.0
    RAG_CANDIDATE_CAP: int = 400
    GRAPH_MAX_DEPTH: int = 2
    GRAPH_MAX_FILES: int = 20
    GRAPH_DECAY_PER_HOP: float = 0.85
    CODE_INTEL_PACKAGE_ROOTS: str = "app"
    RAG_LLM_RERANK: bool = False
    GIT_QUERY_HINTS_CAP: int = 12

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/deepcode"

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Adaptive / Week 11 ─────────────────────────────────────────────────
    ADAPTIVE_ENABLED: bool = True
    ADAPTIVE_RAG_TOP_K_TRIM: int = 2
    EARLY_CANCEL_RAG_TOOL_SCORE: float = 0.9
    # Week 11.5: smoothing + cold start + symmetric early cancel
    ADAPTIVE_EMA_ALPHA: float = 0.2
    ADAPTIVE_COLD_START_QUERIES: int = 20
    EARLY_CANCEL_TOOL_RAG_CONFIDENCE: float = 0.9

    # Repo auto-index (repo_manager / TUI)
    REPO_INDEX_MAX_FILES: int = 5000

    # Orchestrator DAG safety cap (critic retries reset the workflow pointer)
    ORCHESTRATOR_MAX_ITERATIONS: int = 16

    # ReAct-style tool-first exploration (RAG only when context is insufficient)
    REACT_AGENT_ENABLED: bool = True
    REACT_MAX_STEPS: int = 5
    # When exploration confidence is low, ReAct may use up to this many steps
    REACT_MAX_STEPS_CAP: int = 7
    # One ReAct loop per heuristic subgoal, then merge context (single subgoal = prior behavior)
    HIERARCHICAL_REACT_ENABLED: bool = True
    # After 2+ subgoals, re-run cross-subgoal synthesis to bias remaining subgoal order
    HIERARCHICAL_SYNTH_FEEDBACK: bool = True
    # Minimum dependency-graph edges seen (imports+imported_by) before early-stop when plan needs deps
    HIERARCHICAL_MIN_DEP_EDGES_FOR_EARLY_STOP: int = 4
    # Expose transactional file_edit in ReAct when write_enabled / edit mode (not dry_run)
    REACT_EDIT_TOOL_ENABLED: bool = True
    # Batch edits per subgoal; commit/rollback once after run_reasoning_loop
    REACT_SUBGOAL_EDIT_TRANSACTION: bool = True
    # If True, ReAct file_edit never persists (working copy only; finalize rolls back)
    REACT_EDIT_DRY_RUN: bool = False
    # Cap active_working_set / carry into subgoals (reduces context drift)
    ACTIVE_WORKING_SET_MAX: int = 8
    # Inject resolve_conflicts subgoal when mid-hierarchy synthesis reports conflicts
    HIERARCHICAL_CONFLICT_RESOLUTION_SUBGOAL: bool = True

    # Multi-file edit mode (CLI/TUI `edit`): max files auto-touched per run
    EDIT_MAX_AUTO_FILES: int = 5
    # LLM repair attempts after a failed transactional <edit> (each hunk)
    EDIT_MAX_REPAIR_ATTEMPTS: int = 2

    # Multi-cycle macro goals (TUI /run, long_run_agent)
    LONG_RUN_ENABLED: bool = True
    LONG_RUN_MAX_CYCLES: int = 20
    # Wall-clock and step budgets for macro runs (0 = no limit)
    LONG_RUN_MAX_RUNTIME_SECONDS: int = 0
    LONG_RUN_MAX_TOTAL_TOOL_STEPS: int = 0
    # After each task, run project tests if True (can be slow)
    LONG_RUN_POST_TASK_VERIFY: bool = False
    # Set non-empty to persist task_graph.json / artifacts.json / execution_log.jsonl per /run
    LONG_RUN_PERSIST_DIR: str = ""


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — avoids re-parsing .env on every call."""
    return Settings()
