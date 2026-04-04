"""
Ollama LLM provider implementation.

Communicates with a locally-running Ollama instance via its REST API.
This is the default (and recommended) provider for local development.
"""
import httpx

from app.core.config import get_settings
from app.core.logger import get_logger
from app.providers.base import LLMProvider

logger = get_logger(__name__)
settings = get_settings()


class OllamaProvider(LLMProvider):
    """Ollama-backed LLM provider using the local REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self._timeout = httpx.Timeout(120.0, connect=10.0)

    # ── LLMProvider interface ─────────────────────────────────────────────
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        """Send a prompt to Ollama and return the generated text."""
        url = f"{self.base_url}/api/generate"
        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.info("Ollama generate → model=%s", target_model)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
        except httpx.ConnectError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}. "
                "Start it with: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ollama returned {exc.response.status_code}: {exc.response.text}") from exc

    async def stream_generate(self, prompt: str, model: str | None = None):
        import json
        url = f"{self.base_url}/api/generate"
        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }

        logger.info("Ollama stream_generate → model=%s", target_model)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk = data.get("response", "")
                                if chunk:
                                    yield chunk
                            except json.JSONDecodeError:
                                pass
        except httpx.ConnectError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}. "
                "Start it with: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Ollama returned {exc.response.status_code}: {exc.response.text}") from exc

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector using Ollama."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text,
        }

        logger.info("Ollama embed → model=%s", self.embedding_model)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("embedding", [])
        except httpx.ConnectError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}."
            ) from exc

    async def health_check(self) -> bool:
        """Ping Ollama to verify it's alive."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
