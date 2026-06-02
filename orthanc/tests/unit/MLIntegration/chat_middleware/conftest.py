"""Chat-middleware test fixtures.

Prepends the chat-middleware service directory to sys.path so that
ollama_client, config, etc. are importable by tests in this package.

Provides ollama_fake — a FakeOllamaClient injected via monkeypatch.
"""
import os
import sys
from typing import Iterator

import pytest

from _colliders import ML_SERVICE_COLLIDERS

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHAT_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "..", "MLIntegration", "chat-middleware")
)
if _CHAT_DIR not in sys.path:
    sys.path.insert(0, _CHAT_DIR)


@pytest.fixture(autouse=True)
def _force_chat_path() -> Iterator[None]:
    """Ensure chat-middleware dir is at sys.path[0] and evict colliding sibling names."""
    saved = list(sys.path)
    if _CHAT_DIR in sys.path:
        sys.path.remove(_CHAT_DIR)
    sys.path.insert(0, _CHAT_DIR)
    for k in list(sys.modules):
        top = k.split(".", 1)[0]
        if top in ML_SERVICE_COLLIDERS:
            del sys.modules[k]
    try:
        yield
    finally:
        sys.path[:] = saved


class FakeOllamaClient:
    """In-memory fake matching OllamaClient's async interface."""

    def __init__(self):
        self.chunks = [{"type": "content", "text": "test response"}]
        self.healthy = True
        self.models = ["medgemma-128k"]
        self.chat_calls = []

    async def chat_stream(self, messages, cancel_event=None, runtime_options=None):
        self.chat_calls.append(messages)
        for chunk in self.chunks:
            yield chunk

    async def health_check(self):
        return self.healthy

    async def list_models(self):
        return list(self.models)


@pytest.fixture
def ollama_fake(monkeypatch) -> Iterator[FakeOllamaClient]:
    """Provide a FakeOllamaClient and wire it into the ollama_client singleton.

    Usage:
        async def test_x(ollama_fake):
            ollama_fake.chunks = [{"type": "content", "text": "hi"}]
            client = get_ollama_client()
            tokens = [t async for t in client.chat_stream([...])]
    """
    fake = FakeOllamaClient()
    monkeypatch.setattr("ollama_client.get_ollama_client", lambda *a, **kw: fake)
    yield fake
