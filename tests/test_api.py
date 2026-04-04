"""
Tests for API routes.

Uses an in-memory SQLite database for testing (via aiosqlite)
so PostgreSQL is NOT required to run tests.

Run with:
    pytest tests/ -v
"""
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.database import Base, get_db
import app.core.models  # noqa: F401  — register models with Base

# ── Create test engine & session factory ─────────────────────────────────────
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
TestSessionFactory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    """Override DB dependency with in-memory SQLite."""
    async with TestSessionFactory() as session:
        yield session
        await session.commit()


# ── Setup: create tables once ────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Create all tables in the in-memory SQLite DB before tests."""
    import asyncio

    async def _create():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())

    # Wire up the override
    from app.main import app
    app.dependency_overrides[get_db] = override_get_db

    yield

    app.dependency_overrides.clear()

    async def _teardown():
        await test_engine.dispose()

    asyncio.run(_teardown())


# ── Test client ──────────────────────────────────────────────────────────────
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


class TestRoot:
    """Tests for the root endpoint."""

    def test_root_returns_app_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["docs"] == "/docs"
        assert data["database"] == "postgresql"


class TestHealth:
    """Tests for the health endpoint."""

    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["provider"] == "ollama"
        assert "database" in data


class TestAsk:
    """Tests for the /ask endpoint."""

    @patch("app.api.routes.AgentOrchestrator")
    def test_ask_returns_answer(self, mock_orchestrator_cls):
        """Test successful ask with mocked provider."""
        mock_instance = mock_orchestrator_cls.return_value
        mock_instance.run = AsyncMock(return_value={
            "answer": "Hello! I'm DeepCode Agent.",
            "session_id": "test-session-id",
            "latency_ms": 100,
            "model": "llama3.1:latest",
            "provider": "ollama"
        })

        response = client.post(
            "/api/v1/ask",
            json={"question": "Hello, who are you?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Hello! I'm DeepCode Agent."
        assert "model" in data
        assert "provider" in data
        assert "session_id" in data
        assert "latency_hint" in data

    def test_ask_rejects_empty_question(self):
        """Validation: empty question must be rejected."""
        response = client.post(
            "/api/v1/ask",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_ask_rejects_missing_question(self):
        """Validation: missing question field must be rejected."""
        response = client.post(
            "/api/v1/ask",
            json={},
        )
        assert response.status_code == 422

    def test_ask_rejects_invalid_session_id(self):
        """Validation: bad session_id format must be rejected."""
        response = client.post(
            "/api/v1/ask",
            json={"question": "Test", "session_id": "not-a-uuid"},
        )
        assert response.status_code == 400

    @patch("app.api.routes.AgentOrchestrator")
    def test_ask_handles_connection_error(self, mock_orchestrator_cls):
        """When provider is down, return 503."""
        mock_instance = mock_orchestrator_cls.return_value
        mock_instance.run = AsyncMock(side_effect=ConnectionError("Ollama down"))

        response = client.post(
            "/api/v1/ask",
            json={"question": "Test question"},
        )
        assert response.status_code == 503

    @patch("app.api.routes.AgentOrchestrator")
    def test_ask_with_context(self, mock_orchestrator_cls):
        """Test ask with optional context field."""
        mock_instance = mock_orchestrator_cls.return_value
        mock_instance.run = AsyncMock(return_value={
            "answer": "Context-aware answer",
            "session_id": "test-session-id",
            "latency_ms": 100,
            "model": "llama3.1:latest",
            "provider": "ollama"
        })

        response = client.post(
            "/api/v1/ask",
            json={
                "question": "Explain this code",
                "context": "def foo(): pass",
            },
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "Context-aware answer"
