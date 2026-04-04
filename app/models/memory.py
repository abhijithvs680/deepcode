"""
Memory models — stores long-term cognitive insights across sessions.

Provides schemas for:
- Past architectural decisions (AgentDecision)
- Past bug fixes (AgentBugFix)
- Query history with semantic embeddings (AgentQueryHistory)
"""
import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, String, Text, Float
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.code import _EMBED_DIM, PgVectorOptional


class AgentDecision(Base):
    """Stores past decisions over architecture or codebase constraints."""
    __tablename__ = "agent_decision"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=True, index=True)
    decision_context = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class AgentBugFix(Base):
    """Tracks historical bug remediations for context retention."""
    __tablename__ = "agent_bug_fix"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=True, index=True)
    bug_description = Column(Text, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class AgentQueryHistory(Base):
    """Tracks raw query interactions with their corresponding semantic embeddings."""
    __tablename__ = "agent_query_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    query_embedding = Column(PgVectorOptional(_EMBED_DIM), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
