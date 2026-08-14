"""
Async streaming client for OpenAI-compatible /v1/chat/completions endpoint.
Supports both Ollama and llama.cpp backends.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)

# Model listing budget. The requests themselves are fast (~0.2s each against
# ollama.com), so this bounds a stalled connection rather than slow work.
MODEL_LIST_TIMEOUT_SECONDS = 30
# Attempts for the /api/tags catalogue request; see the retry comment in
# list_models_detailed for why a remote host needs more than one.
MODEL_LIST_ATTEMPTS = 2


class ModelListError(Exception):
    """Raised when a model listing fails.

    Distinct from returning an empty list: the chat panel has to tell "this key
    is rejected / the host is unreachable" apart from "this account genuinely has
    no models", and an empty list conflates the two.
    """


class CloudBackendUnavailableError(Exception):
    """Raised when the cloud provider is requested but cannot be used.

    Either the operator has not set ALLOW_CLOUD_BACKEND, or no API key is
    configured. The message is safe to surface to a client — it never contains
    the key.
    """


class UpstreamChatError(Exception):
    """A non-200 from the LLM backend's chat endpoint.

    Carries a reader-facing message rather than the raw response body: the chat
    panel shows this text directly, and hosted backends answer with a JSON
    envelope that is unreadable as-is.
    """


def _upstream_message(body: str) -> str:
    """Pull the human-readable reason out of an LLM backend's error response.

    Ollama Cloud answers with an OpenAI-shaped envelope,

        {"error": {"message": "this model requires a subscription, upgrade for
                   access: https://ollama.com/upgrade (ref: ...)", ...}}

    and a local Ollama with a bare {"error": "..."}. Surfacing the whole blob put
    JSON punctuation in front of the one sentence that tells the user what to do
    — here, that the chosen model needs a paid plan. Falls back to the trimmed
    body when it is not JSON (llama.cpp returns plain text for some failures).
    """
    text = (body or "").strip()
    if not text:
        return "the backend returned an error with no detail"
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text[:400]

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:400]
    if isinstance(error, str) and error.strip():
        return error.strip()[:400]
    return text[:400]


def _describe(exc: Exception) -> str:
    """Human-readable one-liner for an exception, safe to show a client.

    Several failure modes here stringify to nothing — `str(TimeoutError())` is
    "" — which produced a bare "Model listing failed: " in the chat panel with no
    indication of what went wrong. Fall back to the class name so the reason is
    never blank. Only exception type/message is used, so a URL-embedded
    credential cannot leak in via a repr.
    """
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return type(exc).__name__


class OllamaClient:
    """
    Async streaming client for /v1/chat/completions endpoint.
    Supports both Ollama and llama.cpp backends via backend_type.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        backend_type: str = "ollama",
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            base_url: Base URL for the LLM server (e.g., "http://localhost:11434")
            model: Model name to use (e.g., "medgemma-128k")
            backend_type: "ollama" or "llamacpp"
            api_key: Bearer token, required by Ollama Cloud and unused locally
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.backend_type = backend_type
        self.api_key = api_key or None

    def _auth_headers(self) -> dict[str, str]:
        """Authorization header when an API key is configured, else nothing.

        Returned rather than stored so the key is never part of this object's
        repr and never lands in a logged payload.
        """
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def chat_stream(
        self,
        messages: list[dict],
        cancel_event: asyncio.Event | None = None,
        runtime_options: dict | None = None,
    ) -> AsyncGenerator[dict[str, str], None]:
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
            for key in (
                "max_tokens",
                "temperature",
                "top_p",
                "stop",
                "seed",
                "presence_penalty",
                "frequency_penalty",
                "think",
            ):
                if runtime_options.get(key) is not None:
                    payload[key] = runtime_options[key]

        logger.info(f"Starting chat stream to {url} with model {self.model}")
        logger.debug(f"Messages count: {len(messages)}")

        timeout = aiohttp.ClientTimeout(total=300)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
                async with session.post(
                    url, json=payload, headers=self._auth_headers()
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise UpstreamChatError(
                            f"{_upstream_message(error_text)} (HTTP {response.status})"
                        )

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

                            line = raw_line.decode("utf-8").strip()

                            if not line:
                                continue

                            # SSE format: "data: {...}" or "data: [DONE]"
                            if not line.startswith("data: "):
                                continue

                            data = line[len("data: ") :]

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
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(f"{self.base_url}{endpoint}", headers=self._auth_headers()) as response,
            ):
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed ({self.backend_type}): {e}")
            return False

    async def list_models(self) -> list[str]:
        """
        List available models.
        Ollama: GET /api/tags  |  llama.cpp: GET /v1/models
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = self._auth_headers()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.backend_type == "llamacpp":
                    async with session.get(
                        f"{self.base_url}/v1/models", headers=headers
                    ) as response:
                        if response.status != 200:
                            return []
                        data = await response.json()
                        return [m["id"] for m in data.get("data", [])]
                else:
                    async with session.get(
                        f"{self.base_url}/api/tags", headers=headers
                    ) as response:
                        if response.status != 200:
                            return []
                        data = await response.json()
                        return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list models ({self.backend_type}): {e}")
            return []

    async def list_models_detailed(self) -> list[dict]:
        """
        List available models with their capabilities.

        Capabilities come from /api/show, one request per model, not from the
        `capabilities` array that /api/tags also returns. The two disagree:
        verified against Ollama 0.32.11, /api/tags reported ["completion"] for
        thiagomoraes/medgemma-1.5-4b-it:Q4_K_M while /api/show reported
        ["completion", "vision"] — and that model demonstrably reads images. Trusting
        /api/tags would mislabel vision models as text-only, which is precisely the
        judgement the chat panel needs to get right. /v1/models carries no
        capability data at all.

        A per-model /api/show failure yields an empty capability list for that
        model rather than dropping it or failing the whole listing.

        Returns:
            List of {"name": str, "capabilities": list[str], "supports_vision": bool},
            sorted by name. Empty `capabilities` means "unknown", not "text-only".
        """
        if self.backend_type == "llamacpp":
            # llama.cpp serves a single preloaded model and exposes no capability data.
            return [
                {"name": m, "capabilities": [], "supports_vision": False}
                for m in await self.list_models()
            ]

        headers = self._auth_headers()
        timeout = aiohttp.ClientTimeout(total=MODEL_LIST_TIMEOUT_SECONDS)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Retry the catalogue request: connecting to a remote cloud host
                # intermittently stalls in the TLS handshake (observed against
                # ollama.com from inside Docker — a request that timed out
                # succeeded in 0.2s immediately afterwards). One retry on a fresh
                # connection turns that blip into a slow success rather than an
                # error the user has to notice and manually refresh past.
                # Only the catalogue is retried; a per-model /api/show failure
                # already degrades to "capabilities unknown" on its own.
                data = None
                last_exc: Exception | None = None
                for attempt in range(1, MODEL_LIST_ATTEMPTS + 1):
                    try:
                        async with session.get(
                            f"{self.base_url}/api/tags", headers=headers
                        ) as response:
                            if response.status != 200:
                                body = await response.text()
                                logger.warning(
                                    f"Failed to list models: HTTP {response.status} - {body[:200]}"
                                )
                                # An HTTP error is a definitive answer (bad key,
                                # wrong host); retrying it would just stall the UI.
                                raise ModelListError(
                                    f"Model listing failed: HTTP {response.status}"
                                )
                            data = await response.json()
                            break
                    except ModelListError:
                        raise
                    except Exception as e:
                        last_exc = e
                        if attempt < MODEL_LIST_ATTEMPTS:
                            logger.info(
                                f"Model catalogue attempt {attempt} failed "
                                f"({_describe(e)}); retrying"
                            )
                        else:
                            raise

                if data is None:  # pragma: no cover - defensive
                    raise ModelListError(
                        f"Model listing failed: {_describe(last_exc) if last_exc else 'no response'}"
                    )

                names = []
                for m in data.get("models", []):
                    name = m.get("name") or m.get("model")
                    if name:
                        names.append(name)

                # Bounded concurrency: a cloud account can list dozens of models and
                # each needs its own /api/show.
                semaphore = asyncio.Semaphore(8)

                async def capabilities_for(name: str) -> list[str]:
                    async with semaphore:
                        try:
                            async with session.post(
                                f"{self.base_url}/api/show",
                                json={"model": name},
                                headers=headers,
                            ) as show_response:
                                if show_response.status != 200:
                                    logger.debug(
                                        f"/api/show for {name}: HTTP {show_response.status}"
                                    )
                                    return []
                                show_data = await show_response.json()
                                return show_data.get("capabilities") or []
                        except Exception as e:
                            logger.debug(f"/api/show for {name} failed: {e}")
                            return []

                all_caps = await asyncio.gather(*(capabilities_for(n) for n in names))

            models = [
                {
                    "name": name,
                    "capabilities": caps,
                    "supports_vision": "vision" in caps,
                }
                for name, caps in zip(names, all_caps, strict=True)
            ]
            models.sort(key=lambda m: m["name"])
            return models
        except ModelListError:
            raise
        except Exception as e:
            logger.warning(f"Failed to list models with capabilities: {_describe(e)}")
            raise ModelListError(f"Model listing failed: {_describe(e)}") from e


# Global client instance
_ollama_client: OllamaClient | None = None


def get_ollama_client(base_url: str | None = None, model: str | None = None) -> OllamaClient:
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


def build_cloud_client(model: str | None = None) -> OllamaClient:
    """
    Build a client pointed at Ollama Cloud.

    Deliberately not a singleton: the cloud client carries an API key and a
    user-selected model, and the local singleton above is shared process-wide and
    mutated per request. Constructing a fresh instance keeps the key off shared
    state and keeps the local client's configuration untouched.

    Args:
        model: Cloud model tag. Falls back to OLLAMA_CLOUD_MODEL.

    Raises:
        CloudBackendUnavailableError: cloud disabled by the operator, no API key set,
            or no model resolved.
    """
    from config import get_config

    config = get_config()

    if not config.allow_cloud_backend:
        raise CloudBackendUnavailableError(
            "The Ollama Cloud backend is disabled. An operator must set "
            "ALLOW_CLOUD_BACKEND=1 on the chat-middleware service to enable it."
        )
    if not config.ollama_cloud_api_key:
        raise CloudBackendUnavailableError(
            "No Ollama Cloud API key is configured. Set OLLAMA_API_KEY on the "
            "chat-middleware service."
        )

    effective_model = model or config.ollama_cloud_model
    if not effective_model:
        raise CloudBackendUnavailableError(
            "No cloud model selected. Pick one in the chat panel settings or set "
            "OLLAMA_CLOUD_MODEL."
        )

    return OllamaClient(
        base_url=config.ollama_cloud_url,
        model=effective_model,
        backend_type="ollama",
        api_key=config.ollama_cloud_api_key,
    )


def get_client_for_provider(provider: str, model: str | None = None) -> OllamaClient:
    """
    Resolve the client for a provider name.

    Args:
        provider: "local" or "cloud"
        model: Model tag override

    Raises:
        CloudBackendUnavailableError: provider is "cloud" and it is not usable.
    """
    if provider == "cloud":
        return build_cloud_client(model)

    client = get_ollama_client()
    if model:
        client.model = model
    return client
