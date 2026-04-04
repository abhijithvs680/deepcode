"""
Provider factory — returns the correct LLMProvider based on configuration.

Keeps instantiation logic in one place. Future providers (OpenAI, Anthropic)
are added here as additional elif branches.
"""
from app.core.config import get_settings
from app.providers.base import LLMProvider


def get_provider() -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    Returns:
        An instance of a concrete LLMProvider subclass.

    Raises:
        ValueError: If the configured provider name is not recognised.
    """
    settings = get_settings()
    provider_name = settings.LLM_PROVIDER.lower().strip()

    if provider_name == "ollama":
        from app.providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    # ── Future providers ──────────────────────────────────────────────
    # elif provider_name == "openai":
    #     from app.providers.openai_provider import OpenAIProvider
    #     return OpenAIProvider()
    #
    # elif provider_name == "anthropic":
    #     from app.providers.anthropic_provider import AnthropicProvider
    #     return AnthropicProvider()

    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. "
        f"Supported: ollama"
    )
