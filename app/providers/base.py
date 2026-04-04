"""
Abstract base class for all LLM providers.

Every provider (Ollama, OpenAI, Anthropic, etc.) must implement this interface.
This ensures the rest of the codebase never couples to a specific vendor.
"""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Contract that every LLM provider must fulfil."""

    @abstractmethod
    async def generate(self, prompt: str, model: str | None = None) -> str:
        """
        Generate a text completion from the given prompt.

        Args:
            prompt: The full prompt string to send to the LLM.
            model: Optional override for the target model (e.g. for fast local routing capabilities).

        Returns:
            The generated text response.

        Raises:
            ConnectionError: If the provider is unreachable.
        """
        ...

    @abstractmethod
    async def stream_generate(self, prompt: str, model: str | None = None): # -> AsyncGenerator[str, None]
        """
        Streams a text completion from the given prompt incrementally.

        Args:
            prompt: The full prompt string to send to the LLM.
            model: Optional override for the target model.

        Yields:
            Chunks of the generated text as they are computed.

        Raises:
            ConnectionError: If the provider is unreachable.
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check whether the provider is reachable and functional.

        Returns:
            True if healthy, False otherwise.
        """
        ...
