"""
SQLAlchemy ORM models — the database schema.

Tables:
    - sessions:          Tracks conversation sessions (CLI or API)
    - interaction_logs:  Every LLM call with question, answer, latency, tokens
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Session(Base):
    """
    A conversation session.

    Groups multiple interactions together. Created once per CLI invocation
    or per API session header.
    """

    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(20), nullable=False, default="api")  # "api" | "cli"
    metadata_ = Column("metadata", Text, nullable=True)  # JSON string for flexibility
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    # Relationship
    interactions = relationship(
        "InteractionLog",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Session {self.id} source={self.source}>"


class InteractionLog(Base):
    """
    A single LLM interaction — one question → one answer.

    Captures everything needed for evaluation, debugging, and analytics:
    - What was asked
    - What the LLM returned
    - Which provider/model served it
    - How long it took
    - Token usage (when available)
    """

    __tablename__ = "interaction_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Content
    question = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    full_prompt = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    # Provider info
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)

    # Performance
    latency_ms = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    # Status
    status = Column(String(20), nullable=False, default="success")  # "success" | "error"
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationship
    session = relationship("Session", back_populates="interactions")

    def __repr__(self) -> str:
        return f"<InteractionLog {self.id} status={self.status} latency={self.latency_ms}ms>"
