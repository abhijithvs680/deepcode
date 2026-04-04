"""
Async SQLAlchemy database engine and session management.

Provides:
    - Async engine (PostgreSQL via asyncpg)
    - Session factory
    - Dependency injection helper for FastAPI
    - Table creation utility
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logger import get_logger

_db_logger = get_logger(__name__)

settings = get_settings()

# ── Engine ─────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# ── Session Factory ────────────────────────────────────────────────────────────
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Base Model ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ── FastAPI Dependency ─────────────────────────────────────────────────────────
async def get_db() -> AsyncSession:
    """
    FastAPI dependency — yields a transactional async session.

    Usage in routes:
        @router.post("/ask")
        async def ask(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Table Management ───────────────────────────────────────────────────────────
async def create_tables():
    """Create all tables defined by ORM models. Safe to call repeatedly."""
    async with engine.begin() as conn:
        if conn.engine.dialect.name == "postgresql":
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as exc:  # pragma: no cover
                _db_logger.warning("pgvector extension unavailable: %s", exc)
        await conn.run_sync(Base.metadata.create_all)
        if conn.engine.dialect.name == "postgresql":
            try:
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS code_chunks_embedding_vec_hnsw "
                        "ON code_chunks USING hnsw (embedding_vec vector_cosine_ops)"
                    )
                )
            except Exception as exc:  # pragma: no cover
                _db_logger.debug(
                    "HNSW index not created (expected if no pgvector): %s", exc
                )


async def drop_tables():
    """Drop all tables. USE WITH CAUTION — destroys data."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def dispose_engine():
    """Cleanly shut down the connection pool."""
    await engine.dispose()
