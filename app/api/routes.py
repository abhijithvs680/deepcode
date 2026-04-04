"""
API routes for the DeepCode Agent.

Provides REST endpoints for interacting with the LLM.
Now includes:
    - Database-backed sessions
    - Interaction logging
    - Session-aware multi-turn conversations
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import AgentOrchestrator
from app.core.config import get_settings
from app.core.database import get_db

router = APIRouter()
settings = get_settings()


# ──────────────────────────── Request / Response Models ─────────────────────────
class AskRequest(BaseModel):
    """Incoming question payload."""
    question: str = Field(..., min_length=1, max_length=4096, description="The question to ask the LLM")
    context: str | None = Field(None, max_length=8192, description="Optional context to include")
    session_id: str | None = Field(None, description="Optional session ID for multi-turn conversations")
    repo_id: str | None = Field(None, description="Repo name to scope RAG search (from ingest)")
    debug: bool = Field(False, description="Return the internal reasoning trace")


class AskResponse(BaseModel):
    """LLM response payload."""
    answer: str
    model: str
    provider: str
    session_id: str
    latency_hint: str
    trace: dict | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    provider: str
    model: str
    database: str


class SessionResponse(BaseModel):
    """New session creation response."""
    session_id: str
    source: str


# ──────────────────────────── Endpoints ─────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Returns system health and configuration info, including DB connectivity."""
    db_status = "connected"
    try:
        await db.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        database=db_status,
    )


@router.post("/ask", response_model=AskResponse, tags=["llm"])
async def ask_question(
    payload: AskRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a question to the configured LLM and get a response.

    - **question**: The question text (required, 1-4096 chars)
    - **context**: Optional additional context (max 8192 chars)
    - **session_id**: Optional session UUID for multi-turn conversations.
      If omitted, a new session is created automatically.
    """
    # Parse session_id if provided
    session_uuid: uuid.UUID | None = None
    if payload.session_id:
        try:
            session_uuid = uuid.UUID(payload.session_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id format: {payload.session_id}",
            )

    try:
        import time
        start = time.perf_counter()

        orchestrator = AgentOrchestrator(db=db)
        result = await orchestrator.run(
            question=payload.question,
            context=payload.context,
            session_id=session_uuid,
            repo_id=payload.repo_id,
            debug=payload.debug,
            source="api",
        )

        elapsed = (time.perf_counter() - start) * 1000

        return AskResponse(
            answer=result["answer"],
            model=settings.LLM_MODEL,
            provider=settings.LLM_PROVIDER,
            session_id=str(result["session_id"]) if result["session_id"] else "no-db",
            latency_hint=f"{elapsed:.0f}ms",
            trace=result.get("trace"),
        )
    except ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider unavailable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {exc}",
        ) from exc
