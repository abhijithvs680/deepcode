"""
Tests for configuration system.
"""
from app.core.config import Settings, get_settings


class TestSettings:
    """Verify default configuration values."""

    def test_defaults(self):
        s = Settings()
        assert s.LLM_PROVIDER == "ollama"
        assert s.LLM_MODEL == "llama3.1:latest"
        assert s.OLLAMA_BASE_URL == "http://localhost:11434"
        assert s.LLM_TEMPERATURE == 0.7
        assert s.LLM_MAX_TOKENS == 2048

    def test_cached_settings(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # Same cached instance
