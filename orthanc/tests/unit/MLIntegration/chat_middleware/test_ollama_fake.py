"""Self-tests for the ollama_fake fixture."""
import pytest


async def test_ollama_fake_chat_stream_yields_configured_chunks(ollama_fake):
    ollama_fake.chunks = [
        {"type": "thinking", "text": "analyzing..."},
        {"type": "content", "text": "The scan shows normal findings."},
    ]
    tokens = [t async for t in ollama_fake.chat_stream([{"role": "user", "content": "analyze"}])]
    assert len(tokens) == 2
    assert tokens[0] == {"type": "thinking", "text": "analyzing..."}
    assert tokens[1] == {"type": "content", "text": "The scan shows normal findings."}


async def test_ollama_fake_tracks_chat_calls(ollama_fake):
    messages = [{"role": "user", "content": "hello"}]
    _ = [t async for t in ollama_fake.chat_stream(messages)]
    assert ollama_fake.chat_calls == [messages]


async def test_ollama_fake_health_check_returns_configured_value(ollama_fake):
    assert await ollama_fake.health_check() is True
    ollama_fake.healthy = False
    assert await ollama_fake.health_check() is False


async def test_ollama_fake_list_models_returns_configured_list(ollama_fake):
    assert await ollama_fake.list_models() == ["medgemma-128k"]
    ollama_fake.models = ["model-a", "model-b"]
    assert await ollama_fake.list_models() == ["model-a", "model-b"]


async def test_ollama_fake_get_ollama_client_returns_fake(ollama_fake):
    from ollama_client import get_ollama_client

    client = get_ollama_client()
    assert client is ollama_fake
