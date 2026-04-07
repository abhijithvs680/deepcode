"""
Ollama LLM provider implementation.

Uses /api/chat (not /api/generate) so reasoning models like Gemma 4 return text in
`message.content` when native thinking is off. See ollama/ollama#15288.
"""
import json

import httpx

from app.core.config import get_settings
from app.core.logger import get_logger
from app.providers.base import LLMProvider

logger = get_logger(__name__)
settings = get_settings()


def _chat_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": prompt})
    return messages


def _text_from_chat_response(data: dict) -> str:
    msg = data.get("message")
    if isinstance(msg, dict):
        content = (msg.get("content") or "").strip()
        if content:
            return content
        thinking = msg.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            return thinking.strip()
    # Rare: top-level (older shapes)
    r = (data.get("response") or "").strip()
    if r:
        return r
    t = data.get("thinking")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""


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
        to = float(getattr(settings, "OLLAMA_HTTP_TIMEOUT_S", 300.0))
        self._timeout = httpx.Timeout(to, connect=10.0)

    def _gen_options(
        self, temperature: float | None = None
    ) -> dict[str, float | int]:
        return {
            "temperature": temperature
            if temperature is not None
            else settings.LLM_TEMPERATURE,
            "num_predict": settings.LLM_MAX_TOKENS,
        }

    def _think_flag(self) -> bool:
        return bool(getattr(settings, "OLLAMA_NATIVE_THINKING", False))

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        """Chat completion via /api/chat (compatible with Gemma 4 + thinking models)."""
        url = f"{self.base_url}/api/chat"
        target_model = model or self.model
        payload: dict = {
            "model": target_model,
            "messages": _chat_messages(prompt, system_prompt),
            "stream": False,
            "options": self._gen_options(temperature),
            "think": self._think_flag(),
        }

        logger.info("Ollama chat → model=%s think=%s", target_model, payload["think"])

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                text = _text_from_chat_response(data)
                if text:
                    return text
                # Fallback for models that only implement /api/generate well
                return await self._legacy_generate(
                    client, prompt, target_model, system_prompt, temperature
                )
        except httpx.ConnectError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}. "
                "Start it with: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text}"
            ) from exc

    async def _legacy_generate(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        target_model: str,
        system_prompt: str | None,
        temperature: float | None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        body: dict = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": self._gen_options(temperature),
        }
        if system_prompt:
            body["system"] = system_prompt
        logger.info("Ollama generate (fallback) → model=%s", target_model)
        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return (data.get("response") or "").strip()

    async def stream_generate(self, prompt: str, model: str | None = None):
        url = f"{self.base_url}/api/chat"
        target_model = model or self.model
        payload = {
            "model": target_model,
            "messages": _chat_messages(prompt, None),
            "stream": True,
            "options": self._gen_options(),
            "think": self._think_flag(),
        }

        logger.info("Ollama chat stream → model=%s", target_model)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    prev_content = ""
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = data.get("message")
                        if not isinstance(msg, dict):
                            continue
                        cur = msg.get("content")
                        if not isinstance(cur, str) or not cur:
                            continue
                        if cur == prev_content:
                            continue
                        if cur.startswith(prev_content):
                            piece = cur[len(prev_content) :]
                            prev_content = cur
                            if piece:
                                yield piece
                        else:
                            prev_content = prev_content + cur
                            yield cur
        except httpx.ConnectError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}. "
                "Start it with: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text}"
            ) from exc

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
                if "embedding" in data:
                    return data["embedding"] or []
                embs = data.get("embeddings")
                if isinstance(embs, list) and embs:
                    first = embs[0]
                    if isinstance(first, list):
                        return first
                return []
        except httpx.ConnectError as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            raise ConnectionError(
                f"Ollama is not running at {self.base_url}."
            ) from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
