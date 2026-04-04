"""
Interaction Logger — persists every LLM call to PostgreSQL.

This is the observability layer. Every question, answer, latency,
and error gets recorded for:
    - Debugging ("why did it say X?")
    - Evaluation ("how accurate are responses?")
    - Analytics ("avg latency per model")
    - Audit trail

Usage:
    logger = InteractionLogger(db_session)
    log = await logger.log_interaction(
        session_id=session_id,
        question="What is Python?",
        context=None,
        full_prompt="...",
        answer="Python is ...",
        provider="ollama",
        model="llama3",
        latency_ms=342.5,
    )
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.core.models import InteractionLog, Session

logger = get_logger(__name__)


class InteractionLogger:
    """Handles persisting interaction data to the database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(self, source: str = "api") -> Session:
        """
        Create a new conversation session.

        Sets UUID manually so it's available immediately without flush.
        The actual INSERT happens when get_db commits at end of request.

        Args:
            source: "api" or "cli"

        Returns:
            The created Session ORM object (with id already set).
        """
        session_id = uuid.uuid4()
        session = Session(id=session_id, source=source)
        self.db.add(session)
        logger.info("Created session %s (source=%s)", session.id, source)
        return session

    async def session_exists(self, session_id: uuid.UUID) -> bool:
        """Check if a session exists in the database."""
        stmt = select(Session).where(Session.id == session_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def log_interaction(
        self,
        session_id: uuid.UUID,
        question: str,
        full_prompt: str,
        answer: str,
        provider: str,
        model: str,
        context: str | None = None,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> InteractionLog:
        """
        Record a single LLM interaction.

        Only calls add() — actual INSERT handled by get_db's commit().
        SQLAlchemy auto-orders parent (sessions) before child (interaction_logs).
        """
        log_entry = InteractionLog(
            id=uuid.uuid4(),
            session_id=session_id,
            question=question,
            context=context,
            full_prompt=full_prompt,
            answer=answer,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status=status,
            error_message=error_message,
        )
        self.db.add(log_entry)

        logger.info(
            "Logged interaction %s → session=%s provider=%s model=%s latency=%.1fms status=%s",
            log_entry.id,
            session_id,
            provider,
            model,
            latency_ms or 0,
            status,
        )
        return log_entry

    async def get_session_history(self, session_id: uuid.UUID) -> list[dict[str, str]]:
        """
        Retrieve conversation history for a session (for multi-turn context).

        Args:
            session_id: The session UUID.

        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts,
            ordered chronologically.
        """
        stmt = (
            select(InteractionLog)
            .where(InteractionLog.session_id == session_id)
            .where(InteractionLog.status == "success")
            .order_by(InteractionLog.created_at.asc())
        )
        result = await self.db.execute(stmt)
        interactions = result.scalars().all()

        history = []
        for interaction in interactions:
            history.append({"role": "user", "content": interaction.question})
            history.append({"role": "assistant", "content": interaction.answer})

        return history

    async def log_error(
        self,
        session_id: uuid.UUID,
        question: str,
        full_prompt: str,
        provider: str,
        model: str,
        error_message: str,
        context: str | None = None,
        latency_ms: float | None = None,
    ) -> InteractionLog:
        """Convenience method to log a failed interaction."""
        return await self.log_interaction(
            session_id=session_id,
            question=question,
            full_prompt=full_prompt,
            answer="",
            provider=provider,
            model=model,
            context=context,
            latency_ms=latency_ms,
            status="error",
            error_message=error_message,
        )
