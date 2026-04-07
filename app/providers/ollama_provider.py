"""
Ollama LLM provider implementation.

Uses /api/chat (not /api/generate) so `message.content` is populated for most instruct models
when `think` is false. Some reasoning-only stacks put text in `thinking` — see _text_from_chat_response.
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
        """Chat completion via /api/chat (works with DeepSeek, Llama, Qwen, etc.)."""
        url = f"{self.base_url}/api/chat"
        target_model = model or self.model
        payload: dict = {
            "model": target_model,
            "messages": _chat_messages(prompt, system_prompt),
            "stream": False,
            "options": self._gen_options(temperature),
            "think": self._think_flag(),
        }

        logger.info("Ollama chat: model=%s think=%s", target_model, payload["think"])

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
        logger.info("Ollama generate (fallback): model=%s", target_model)
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

        logger.info("Ollama chat stream: model=%s", target_model)

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

    @staticmethod
    def _parse_embedding_body(data: dict) -> list[float]:
        """Normalize Ollama JSON (legacy + /api/embed) to a single vector."""
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs:
            first = embs[0]
            if isinstance(first, list) and first and isinstance(first[0], (int, float)):
                return [float(x) for x in first]
        single = data.get("embedding")
        if isinstance(single, list) and single and isinstance(single[0], (int, float)):
            return [float(x) for x in single]
        return []

    async def embed(self, text: str) -> list[float]:
        """
        Embedding via Ollama. Prefer /api/embed; fall back to /api/embeddings.
        Matryoshka models can use EMBEDDING_OLLAMA_PASS_DIMENSIONS + EMBEDDING_DIMENSION;
        fixed-size models (e.g. nomic-embed-text) should leave that flag False.
        """
        raw = (text or "").replace("\x00", "")
        if not raw.strip():
            return []

        # Avoid pathological payloads (embedding models still tokenize this)
        max_chars = 120_000
        if len(raw) > max_chars:
            raw = raw[:max_chars]
            logger.warning("Embedding input truncated to %d chars", max_chars)

        model = self.embedding_model
        dim = getattr(settings, "EMBEDDING_DIMENSION", None)
        logger.info("Ollama embed: model=%s", model)

        async def _post(client: httpx.AsyncClient, path: str, body: dict) -> httpx.Response:
            return await client.post(f"{self.base_url}{path}", json=body)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                attempts: list[tuple[str, dict]] = []

                base_embed: dict = {
                    "model": model,
                    "input": raw,
                    "truncate": True,
                }
                use_dim_param = bool(
                    getattr(settings, "EMBEDDING_OLLAMA_PASS_DIMENSIONS", False)
                )
                if (
                    use_dim_param
                    and isinstance(dim, int)
                    and dim > 0
                ):
                    attempts.append(
                        ("/api/embed", {**base_embed, "dimensions": dim})
                    )
                attempts.append(("/api/embed", dict(base_embed)))

                # Legacy
                attempts.append(
                    (
                        "/api/embeddings",
                        {"model": model, "prompt": raw},
                    )
                )

                last_err = ""
                for path, body in attempts:
                    try:
                        r = await _post(client, path, body)
                        if r.status_code != 200:
                            last_err = f"{path} {r.status_code}: {r.text[:400]}"
                            logger.warning("Ollama embed %s", last_err)
                            continue
                        vec = self._parse_embedding_body(r.json())
                        if vec:
                            if (
                                isinstance(dim, int)
                                and dim > 0
                                and len(vec) != dim
                            ):
                                logger.warning(
                                    "Ollama embed vector len=%d != EMBEDDING_DIMENSION=%d "
                                    "(model=%s); fix model or dimension setting.",
                                    len(vec),
                                    dim,
                                    model,
                                )
                            return vec
                        last_err = f"{path} returned empty embedding"
                        logger.warning("Ollama %s", last_err)
                    except httpx.HTTPError as exc:
                        last_err = str(exc)
                        logger.warning("Ollama embed %s failed: %s", path, exc)

                logger.error(
                    "All Ollama embedding attempts failed for model=%s. Last: %s",
                    model,
                    last_err,
                )
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
