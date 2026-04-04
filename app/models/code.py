"""
CodeChunk model — stores ingested code with embeddings.

JSON embeddings plus optional pgvector column on PostgreSQL (Text on SQLite).
"""
import uuid

from sqlalchemy import Column, Integer, String, Text, TypeDecorator
from sqlalchemy.dialects.postgresql import UUID

from app.core.config import get_settings
from app.core.database import Base

_EMBED_DIM = get_settings().EMBEDDING_DIMENSION


class PgVectorOptional(TypeDecorator):
    """PostgreSQL pgvector; SQLite/tests use plain Text DDL."""

    impl = Text
    cache_ok = True

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dimension))
        return dialect.type_descriptor(Text())


class CodeChunk(Base):
    """A chunk of code from an ingested repository."""

    __tablename__ = "code_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_name = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String, nullable=True)
    embedding = Column(Text, nullable=True)
    embedding_vec = Column(PgVectorOptional(_EMBED_DIM), nullable=True)

    def __repr__(self) -> str:
        return f"<CodeChunk {self.file_path}:{self.chunk_index}>"
