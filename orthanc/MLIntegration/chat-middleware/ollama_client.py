"""
Async streaming client for OpenAI-compatible /v1/chat/completions endpoint.
Supports Ollama, llama.cpp and OpenRouter backends.
"""

import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)

# Model listing budget. The requests themselves are fast (~0.2s each against
# ollama.com), so this bounds a stalled connection rather than slow work.
MODEL_LIST_TIMEOUT_SECONDS = 30
# Attempts for the catalogue request; see the retry comment in _fetch_catalogue
# for why a remote host needs more than one.
MODEL_LIST_ATTEMPTS = 2

# Whole-turn budget for a chat. Generous: a large model on CPU can take minutes.
CHAT_TIMEOUT_SECONDS = 300
# Connection-establishment budget, TLS handshake included. Well under the 300s
# total so a stalled handshake surfaces quickly instead of consuming the turn.
CONNECT_TIMEOUT_SECONDS = 15
# Attempts to *establish* the connection. Generation itself is never repeated.
CHAT_CONNECT_ATTEMPTS = 2
CHAT_RETRY_BASE_DELAY_SECONDS = 0.5
CHAT_RETRY_JITTER_SECONDS = 0.5

# Backends whose catalogue is the OpenAI-shaped GET /v1/models: llama.cpp serves
# its one preloaded model there, OpenRouter its whole brokered catalogue.
_OPENAI_CATALOGUE_BACKENDS = frozenset({"llamacpp", "openrouter"})

# Cheapest endpoint that proves each backend is reachable. OpenRouter has no
# /api/tags and no dedicated health route, so its catalogue doubles as one.
_HEALTH_ENDPOINTS = {
    "llamacpp": "/health",
    "openrouter": "/v1/models",
    "ollama": "/api/tags",
}

# Failures that mean "no connection was established", so the request provably
# never reached the model and can safely be sent again. ClientConnectorError
# covers DNS and TCP/TLS connect failures (ClientConnectorSSLError included);
# ConnectionTimeoutError is sock_connect expiring.
RETRYABLE_CONNECT_ERRORS = (
    aiohttp.ClientConnectorError,
    aiohttp.ConnectionTimeoutError,
)


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


def _reasoning_text(delta: dict) -> str:
    """Pull streamed reasoning text out of a delta, whichever dialect it speaks.

    Ollama — local and cloud — sends `reasoning_content`. OpenRouter sends
    `reasoning` as a plain string, plus `reasoning_details` (an array of typed
    parts) for models whose reasoning is structured. Reading only
    `reasoning_content` dropped OpenRouter's thinking tokens silently: the answer
    streamed normally and the thinking pane simply stayed empty.

    Returns "" when the delta carries no reasoning, which the caller treats the
    same as an absent field.
    """
    for key in ("reasoning_content", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value

    details = delta.get("reasoning_details")
    if isinstance(details, list):
        return "".join(
            part["text"]
            for part in details
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


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
    Supports Ollama, llama.cpp and OpenRouter backends via backend_type.

    Chat is identical across all three — one OpenAI-compatible POST. The backends
    differ only in how they answer "which models do you have", and OpenRouter
    additionally accepts routing options the others do not (see `extra_payload`).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        backend_type: str = "ollama",
        api_key: str | None = None,
        extra_payload: dict | None = None,
        model_filter: tuple[str, ...] = (),
    ) -> None:
        """
        Args:
            base_url: Base URL for the LLM server (e.g., "http://localhost:11434")
            model: Model name to use (e.g., "medgemma-128k")
            backend_type: "ollama", "llamacpp" or "openrouter"
            api_key: Bearer token, required by the hosted backends and unused locally
            extra_payload: Backend-specific keys merged into every chat request.
                Kept out of this class's own logic so provider policy — OpenRouter's
                data-collection setting, say — is decided by whoever builds the
                client, not buried in the transport.
            model_filter: Case-insensitive substrings; when non-empty, only models
                whose id contains one of them are listed.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.backend_type = backend_type
        self.api_key = api_key or None
        self.extra_payload = dict(extra_payload or {})
        self.model_filter = tuple(model_filter)

    def _keep_model(self, name: str) -> bool:
        """Whether a model id survives `model_filter`."""
        if not self.model_filter:
            return True
        lowered = name.lower()
        return any(needle.lower() in lowered for needle in self.model_filter)

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
            **self.extra_payload,
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

        # sock_connect is set explicitly. Supplying a custom ClientTimeout replaces
        # aiohttp's defaults wholesale, which leaves sock_connect=None -- so a
        # connection that stalls in the TLS handshake fell through to asyncio's
        # 60s handshake watchdog instead of failing promptly. Observed as a 61.3s
        # "Cannot connect to host ollama.com:443" against a path whose MTU cannot
        # carry OpenSSL 3.5's ~1.5 KB ClientHello.
        timeout = aiohttp.ClientTimeout(
            total=CHAT_TIMEOUT_SECONDS, sock_connect=CONNECT_TIMEOUT_SECONDS
        )

        # Whether anything has been handed to the caller yet. Once a token is
        # yielded the request is not repeatable, so retries stop.
        yielded_any = False

        for attempt in range(1, CHAT_CONNECT_ATTEMPTS + 1):
            try:
                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.post(url, json=payload, headers=self._auth_headers()) as response,
                ):
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        # Deliberately not retried: an HTTP status is a real answer
                        # from the model host (entitlement, bad key, bad model), and
                        # repeating it just delays the message.
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
                                    reasoning = _reasoning_text(delta)
                                    if reasoning:
                                        yielded_any = True
                                        yield {"type": "thinking", "text": reasoning}
                                    content = delta.get("content")
                                    if content:
                                        yielded_any = True
                                        yield {"type": "content", "text": content}
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse SSE chunk: {e}")
                                continue
                    finally:
                        if cancelled:
                            logger.debug("Releasing HTTP response due to cancellation")
                            response.close()

                    logger.info(f"Chat stream finished (cancelled={cancelled})")
                    return

            except asyncio.CancelledError:
                logger.info("Chat stream cancelled")
                raise
            except RETRYABLE_CONNECT_ERRORS as e:
                # Only failures to establish the connection are repeated, and only
                # while nothing has been streamed. Deliberately excluded:
                # ServerDisconnectedError and a bare TimeoutError, either of which
                # can mean the request did reach the model and generation is already
                # under way -- resending would bill and run it twice.
                if attempt < CHAT_CONNECT_ATTEMPTS and not yielded_any:
                    delay = CHAT_RETRY_BASE_DELAY_SECONDS + random.uniform(
                        0, CHAT_RETRY_JITTER_SECONDS
                    )
                    logger.info(
                        f"Chat connect attempt {attempt} failed ({_describe(e)}); "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Ollama connection error: {_describe(e)}")
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
        Ollama: GET /api/tags | llama.cpp: GET /health | OpenRouter: GET /v1/models
        """
        endpoint = _HEALTH_ENDPOINTS.get(self.backend_type, "/api/tags")
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
        Ollama: GET /api/tags | llama.cpp and OpenRouter: GET /v1/models
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = self._auth_headers()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.backend_type in _OPENAI_CATALOGUE_BACKENDS:
                    async with session.get(
                        f"{self.base_url}/v1/models", headers=headers
                    ) as response:
                        if response.status != 200:
                            return []
                        data = await response.json()
                        return [m["id"] for m in data.get("data", []) if self._keep_model(m["id"])]
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

        Ollama takes two round trips per model. Capabilities come from /api/show,
        one request per model, not from the `capabilities` array that /api/tags
        also returns. The two disagree: verified against Ollama 0.32.11, /api/tags
        reported ["completion"] for thiagomoraes/medgemma-1.5-4b-it:Q4_K_M while
        /api/show reported ["completion", "vision"] — and that model demonstrably
        reads images. Trusting /api/tags would mislabel vision models as text-only,
        which is precisely the judgement the chat panel needs to get right.
        /v1/models carries no capability data at all.

        OpenRouter takes one: GET /v1/models returns the whole catalogue with
        `architecture.input_modalities` per entry, so there is nothing to fan out
        to and capabilities are always reported.

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
                if self.backend_type == "openrouter":
                    catalogue = await self._fetch_catalogue(session, "/v1/models", headers)
                    models = self._openrouter_models(catalogue)
                else:
                    models = await self._ollama_models(session, headers)

            models.sort(key=lambda m: m["name"])
            return models
        except ModelListError:
            raise
        except Exception as e:
            logger.warning(f"Failed to list models with capabilities: {_describe(e)}")
            raise ModelListError(f"Model listing failed: {_describe(e)}") from e

    async def _fetch_catalogue(
        self, session: aiohttp.ClientSession, path: str, headers: dict[str, str]
    ) -> dict:
        """GET a catalogue endpoint, retrying a connection that never opened.

        Connecting to a remote cloud host intermittently stalls in the TLS
        handshake (observed against ollama.com from inside Docker — a request that
        timed out succeeded in 0.2s immediately afterwards). One retry on a fresh
        connection turns that blip into a slow success rather than an error the
        user has to notice and manually refresh past.

        Only the catalogue is retried; a per-model /api/show failure already
        degrades to "capabilities unknown" on its own.
        """
        for attempt in range(1, MODEL_LIST_ATTEMPTS + 1):
            try:
                async with session.get(f"{self.base_url}{path}", headers=headers) as response:
                    if response.status != 200:
                        body = await response.text()
                        logger.warning(
                            f"Failed to list models: HTTP {response.status} - {body[:200]}"
                        )
                        # An HTTP error is a definitive answer (bad key, wrong
                        # host); retrying it would just stall the UI.
                        raise ModelListError(f"Model listing failed: HTTP {response.status}")
                    return await response.json()
            except ModelListError:
                raise
            except Exception as e:
                if attempt >= MODEL_LIST_ATTEMPTS:
                    raise
                logger.info(f"Model catalogue attempt {attempt} failed ({_describe(e)}); retrying")

        raise ModelListError("Model listing failed: no response")  # pragma: no cover - defensive

    def _openrouter_models(self, catalogue: dict) -> list[dict]:
        """Shape OpenRouter's /v1/models payload into the common listing form.

        Vision comes from `architecture.input_modalities` — what the model
        accepts, e.g. ["text", "image", "file"]. Every entry carries it, so unlike
        Ollama there is no "capabilities unknown" case. The modalities are passed
        through as the capability list because they are what this backend
        genuinely knows about a model.
        """
        models = []
        for entry in catalogue.get("data", []):
            name = entry.get("id")
            if not name or not self._keep_model(name):
                continue
            architecture = entry.get("architecture") or {}
            modalities = architecture.get("input_modalities") or []
            models.append(
                {
                    "name": name,
                    "capabilities": list(modalities),
                    "supports_vision": "image" in modalities,
                }
            )
        return models

    async def _ollama_models(
        self, session: aiohttp.ClientSession, headers: dict[str, str]
    ) -> list[dict]:
        """Catalogue plus a per-model /api/show fan-out, the Ollama way."""
        catalogue = await self._fetch_catalogue(session, "/api/tags", headers)

        names = []
        for m in catalogue.get("models", []):
            name = m.get("name") or m.get("model")
            if name and self._keep_model(name):
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
                            logger.debug(f"/api/show for {name}: HTTP {show_response.status}")
                            return []
                        show_data = await show_response.json()
                        return show_data.get("capabilities") or []
                except Exception as e:
                    logger.debug(f"/api/show for {name} failed: {e}")
                    return []

        all_caps = await asyncio.gather(*(capabilities_for(n) for n in names))

        return [
            {
                "name": name,
                "capabilities": caps,
                "supports_vision": "vision" in caps,
            }
            for name, caps in zip(names, all_caps, strict=True)
        ]


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


def cloud_disabled_message() -> str:
    """Why the cloud slot is closed, naming the service the operator configured."""
    from config import get_config

    return (
        f"The {get_config().cloud_label} backend is disabled. An operator must set "
        "ALLOW_CLOUD_BACKEND=1 on the chat-middleware service."
    )


def cloud_key_missing_message() -> str:
    """Which env var to set, named for the service the key comes from."""
    from config import get_config

    config = get_config()
    return (
        f"No {config.cloud_label} API key is configured. Set {config.cloud_key_env} "
        "on the chat-middleware service."
    )


def build_cloud_client(model: str | None = None) -> OllamaClient:
    """
    Build a client pointed at the configured cloud provider.

    Which provider that is — Ollama Cloud or OpenRouter — is an operator choice
    (CLOUD_PROVIDER); both are OpenAI-compatible, so only the base URL, the
    catalogue dialect and OpenRouter's routing options differ.

    Deliberately not a singleton: the cloud client carries an API key and a
    user-selected model, and the local singleton above is shared process-wide and
    mutated per request. Constructing a fresh instance keeps the key off shared
    state and keeps the local client's configuration untouched.

    Args:
        model: Cloud model tag. Falls back to the configured default.

    Raises:
        CloudBackendUnavailableError: cloud disabled by the operator, no API key set,
            or no model resolved.
    """
    from config import CLOUD_PROVIDER_OPENROUTER, get_config

    config = get_config()

    if not config.allow_cloud_backend:
        raise CloudBackendUnavailableError(cloud_disabled_message())
    if not config.cloud_api_key:
        raise CloudBackendUnavailableError(cloud_key_missing_message())

    effective_model = model or config.cloud_model
    if not effective_model:
        raise CloudBackendUnavailableError(
            f"No cloud model selected. Pick one in the chat panel settings or set "
            f"{config.cloud_provider.upper()}_MODEL."
        )

    extra_payload = {}
    if config.cloud_provider == CLOUD_PROVIDER_OPENROUTER:
        # OpenRouter brokers the request to one of several inference providers, so
        # this is where the deployment's data policy has to be stated: "deny"
        # restricts routing to providers that do not retain or train on prompts.
        # Sent on every chat rather than relying on the account's dashboard
        # setting, so the guarantee travels with the request.
        extra_payload["provider"] = {"data_collection": config.openrouter_data_collection}

    return OllamaClient(
        base_url=config.cloud_url,
        model=effective_model,
        backend_type=config.cloud_provider,
        api_key=config.cloud_api_key,
        extra_payload=extra_payload,
        model_filter=config.cloud_model_filter,
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
