"""
Central configuration using Pydantic Settings.

Loads values from environment variables → .env file → defaults.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — single source of truth."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
    LLM_MODEL: str = "llama3.1:latest"
    LLM_ROUTER_MODEL: str = "llama3.1:latest" # Configured for fast low-latency tasks
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # ── Ollama ────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── Embedding ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSION: int = 768

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
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/deepcode"

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


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — avoids re-parsing .env on every call."""
    return Settings()
