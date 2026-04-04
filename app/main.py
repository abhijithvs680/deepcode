"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Now includes:
    - Database lifecycle (create tables on startup, dispose on shutdown)
    - CORS middleware
    - API router
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown hooks — including DB init."""
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info(
        "%s v%s starting — provider=%s model=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
    )

    # Import models so Base.metadata knows about them, then create tables
    import app.core.models  # noqa: F401
    import app.models.code  # noqa: F401 — register CodeChunk for RAG
    import app.models.repo_intel  # noqa: F401 — symbol index + import graph
    import app.models.memory  # noqa: F401 — long-term memory
    from app.core.database import create_tables, dispose_engine

    try:
        await create_tables()
        logger.info("Database tables ready")
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        logger.warning("Running without database — interactions won't be logged")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    try:
        await dispose_engine()
        logger.info("Database connections closed")
    except Exception:
        pass

    logger.info("%s shutting down", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered code analysis and generation agent with PostgreSQL logging",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint — confirms the server is running."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "database": "postgresql",
    }
