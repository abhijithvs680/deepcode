"""
Tests for the retriever (cosine similarity).
"""
import pytest

from app.memory.retriever import cosine_similarity


class TestCosineSimilarity:
    """Tests for the cosine similarity function."""

    def test_identical_vectors(self):
        """Same vector → similarity = 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Perpendicular vectors → similarity = 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors → similarity = -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        """Zero vector → similarity = 0.0 (no division error)."""
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_score(self):
        """Slightly different vectors → high similarity."""
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        score = cosine_similarity(a, b)
        assert score > 0.99
