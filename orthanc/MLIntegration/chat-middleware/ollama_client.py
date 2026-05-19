"""
Async streaming client for OpenAI-compatible /v1/chat/completions endpoint.
Supports both Ollama and llama.cpp backends.
"""
import json
import asyncio
import logging
from typing import List, Dict, AsyncGenerator, Optional

import aiohttp

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Async streaming client for /v1/chat/completions endpoint.
    Supports both Ollama and llama.cpp backends via backend_type.
    """

    def __init__(self, base_url: str, model: str, backend_type: str = "ollama"):
        """
        Args:
            base_url: Base URL for the LLM server (e.g., "http://localhost:11434")
            model: Model name to use (e.g., "medgemma-128k")
            backend_type: "ollama" or "llamacpp"
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.backend_type = backend_type

    async def chat_stream(
        self,
        messages: List[dict],
        cancel_event: Optional[asyncio.Event] = None,
        runtime_options: Optional[dict] = None
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Stream chat completion tokens from Ollama's OpenAI-compatible endpoint.

        Args:
            messages: List of message dicts with role and content (string or content array)
            cancel_event: Event to signal cancellation (optional)
            runtime_options: Optional dict with supported OpenAI params (max_tokens, temperature, etc.)

        Yields:
            Dicts with "type" ("content" | "thinking") and "text" keys
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        # Only add supported OpenAI-compatible parameters
        if runtime_options:
            for key in ("max_tokens", "temperature", "top_p", "stop", "seed",
                        "presence_penalty", "frequency_penalty", "think"):
                if runtime_options.get(key) is not None:
                    payload[key] = runtime_options[key]

        logger.info(f"Starting chat stream to {url} with model {self.model}")
        logger.debug(f"Messages count: {len(messages)}")

        timeout = aiohttp.ClientTimeout(total=300)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")

                    logger.debug("SSE stream connected, receiving tokens...")
                    cancelled = False

                    try:
                        async for raw_line in response.content:
                            if cancel_event and cancel_event.is_set():
                                logger.info("Chat stream cancelled by user")
                                cancelled = True
                                break

                            if not raw_line:
                                continue

                            line = raw_line.decode('utf-8').strip()

                            if not line:
                                continue

                            # SSE format: "data: {...}" or "data: [DONE]"
                            if not line.startswith("data: "):
                                continue

                            data = line[len("data: "):]

                            if data == "[DONE]":
                                logger.debug("SSE stream completed ([DONE])")
                                break

                            try:
                                chunk = json.loads(data)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    reasoning = delta.get("reasoning_content")
                                    if reasoning:
                                        yield {"type": "thinking", "text": reasoning}
                                    content = delta.get("content")
                                    if content:
                                        yield {"type": "content", "text": content}
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse SSE chunk: {e}")
                                continue
                    finally:
                        if cancelled:
                            logger.debug("Releasing HTTP response due to cancellation")
                            response.close()

                    logger.info(f"Chat stream finished (cancelled={cancelled})")

        except asyncio.CancelledError:
            logger.info("Chat stream cancelled")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if the LLM backend is reachable.
        Ollama: GET /api/tags  |  llama.cpp: GET /health
        """
        endpoint = "/health" if self.backend_type == "llamacpp" else "/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed ({self.backend_type}): {e}")
            return False

    async def list_models(self) -> List[str]:
        """
        List available models.
        Ollama: GET /api/tags  |  llama.cpp: GET /v1/models
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.backend_type == "llamacpp":
                    async with session.get(f"{self.base_url}/v1/models") as response:
                        if response.status != 200:
                            return []
                        data = await response.json()
                        return [m["id"] for m in data.get("data", [])]
                else:
                    async with session.get(f"{self.base_url}/api/tags") as response:
                        if response.status != 200:
                            return []
                        data = await response.json()
                        return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list models ({self.backend_type}): {e}")
            return []


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client(base_url: str = None, model: str = None) -> OllamaClient:
    """
    Get the global Ollama client singleton.

    Args:
        base_url: Override base URL (used on first call to initialize)
        model: Override model name (used on first call to initialize)

    Returns:
        OllamaClient instance
    """
    global _ollama_client
    if _ollama_client is None:
        from config import get_config
        config = get_config()
        _ollama_client = OllamaClient(
            base_url=base_url or config.ollama_url,
            model=model or config.ollama_model,
            backend_type=config.backend_type,
        )
    else:
        effective_url = base_url or ""
        effective_model = model or ""
        if effective_url and effective_url != _ollama_client.base_url:
            logger.warning(
                f"OllamaClient already initialized with base_url={_ollama_client.base_url}, "
                f"ignoring requested base_url={effective_url}"
            )
        if effective_model and effective_model != _ollama_client.model:
            logger.warning(
                f"OllamaClient already initialized with model={_ollama_client.model}, "
                f"ignoring requested model={effective_model}"
            )
    return _ollama_client


def reset_ollama_client() -> None:
    """Reset the Ollama client singleton (useful for testing)"""
    global _ollama_client
    _ollama_client = None
