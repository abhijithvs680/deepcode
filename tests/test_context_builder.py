"""
Tests for the Context Builder.
"""
import pytest

from app.agents.context_builder import ContextBuilder, SYSTEM_PROMPT


class TestContextBuilder:
    """Test prompt construction via ContextBuilder."""

    def test_basic_question(self):
        """Simplest case — just a question."""
        builder = ContextBuilder(question="What is Python?")
        result = builder.build()

        assert SYSTEM_PROMPT in result.full_prompt
        assert "What is Python?" in result.full_prompt
        assert "system_prompt" in result.pieces_used

    def test_with_user_context(self):
        """Question + user-provided context."""
        builder = ContextBuilder(question="Explain this code")
        builder.add_context("User Context", "def foo(): pass")
        result = builder.build()

        assert "User Context" in result.full_prompt
        assert "def foo(): pass" in result.full_prompt
        assert "User Context" in result.pieces_used

    def test_with_conversation_history(self):
        """Question + conversation history for multi-turn."""
        builder = ContextBuilder(question="Follow up question")
        builder.add_conversation_history([
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ])
        result = builder.build()

        assert "Conversation History" in result.full_prompt
        assert "What is Python?" in result.full_prompt
        assert "Python is a programming language." in result.full_prompt
        assert "conversation_history" in result.pieces_used

    def test_priority_ordering(self):
        """Higher priority context should appear first."""
        builder = ContextBuilder(question="Test")
        builder.add_context("Low Priority", "low stuff", priority=1)
        builder.add_context("High Priority", "high stuff", priority=10)
        result = builder.build()

        high_pos = result.full_prompt.index("High Priority")
        low_pos = result.full_prompt.index("Low Priority")
        assert high_pos < low_pos  # High priority comes first

    def test_empty_context_ignored(self):
        """Empty/whitespace-only context should be silently ignored."""
        builder = ContextBuilder(question="Test")
        builder.add_context("Empty", "")
        builder.add_context("Whitespace", "   ")
        result = builder.build()

        assert "Empty" not in result.full_prompt
        assert "Whitespace" not in result.full_prompt

    def test_method_chaining(self):
        """Builder methods should support chaining."""
        result = (
            ContextBuilder(question="Test")
            .add_context("A", "content a")
            .add_context("B", "content b")
            .build()
        )

        assert "content a" in result.full_prompt
        assert "content b" in result.full_prompt

    def test_built_context_fields(self):
        """BuiltContext should have all expected fields."""
        result = ContextBuilder(question="Test").build()
        assert result.system_prompt == SYSTEM_PROMPT
        assert isinstance(result.full_prompt, str)
        assert isinstance(result.pieces_used, list)
