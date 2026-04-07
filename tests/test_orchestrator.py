"""
Tests for the Agent Orchestrator.

Updated to work with the new ContextBuilder + InteractionLogger integration.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agents.orchestrator import AgentOrchestrator
from app.core.config import get_settings


class TestOrchestratorWithoutDB:
    """Test orchestrator when no database is provided (fallback mode)."""

    @patch("app.agents.orchestrator.get_provider")
    def test_init_without_db(self, mock_get_provider):
        """Orchestrator should work fine without a DB session."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        orchestrator = AgentOrchestrator()
        assert orchestrator._provider is not None
        assert orchestrator._interaction_logger is None

    @patch("app.agents.orchestrator.get_provider")
    @pytest.mark.asyncio
    async def test_run_without_db(self, mock_get_provider):
        """Run should work without DB — just no logging."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Test answer")
        dim = get_settings().EMBEDDING_DIMENSION
        mock_provider.embed = AsyncMock(return_value=[0.1] * dim)
        mock_get_provider.return_value = mock_provider

        orchestrator = AgentOrchestrator()
        result = await orchestrator.run(question="What is Python?")

        assert result["answer"] == "Test answer"
        assert mock_provider.generate.await_count >= 1


class TestOrchestratorWithDB:
    """Test orchestrator with mocked DB session."""

    @patch("app.agents.orchestrator.get_provider")
    @pytest.mark.asyncio
    async def test_run_with_context(self, mock_get_provider):
        """Run with user context should include it in the prompt."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Answer with context")
        dim = get_settings().EMBEDDING_DIMENSION
        mock_provider.embed = AsyncMock(return_value=[0.1] * dim)
        mock_get_provider.return_value = mock_provider

        orchestrator = AgentOrchestrator()
        result = await orchestrator.run(
            question="Explain this",
            context="def foo(): pass",
        )

        assert result["answer"] == "Answer with context"
        # Verify the prompt sent to the provider included the context
        prompts = [c[0][0] for c in mock_provider.generate.call_args_list if c[0]]
        assert any("def foo(): pass" in p for p in prompts)

    @patch("app.agents.orchestrator.get_provider")
    @pytest.mark.asyncio
    async def test_run_propagates_errors(self, mock_get_provider):
        """Errors from the provider should propagate up."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=ConnectionError("Ollama down"))
        dim = get_settings().EMBEDDING_DIMENSION
        mock_provider.embed = AsyncMock(return_value=[0.1] * dim)
        mock_get_provider.return_value = mock_provider

        orchestrator = AgentOrchestrator()

        with pytest.raises(ConnectionError, match="Ollama down"):
            await orchestrator.run(question="Test")
