import pytest
from starlette.websockets import WebSocket, WebSocketDisconnect
from .test_app_and_debug import _fresh_app


@pytest.mark.parametrize("origin,allowed", [
    ("http://testserver", True), ("https://testserver", True),
    ("https://viewer.example", True), ("http://192.168.1.50:8081", True),
    ("https://evil.example", False), ("null", False), (None, False),
    ("https://viewer.example.evil.test", False), ("https://viewer.example/path", False),
    ("https://user@viewer.example", False),
])
def test_origin_policy_before_session_allocation(tmp_path, monkeypatch, origin, allowed):
    monkeypatch.setenv("CHAT_WEBSOCKET_ORIGIN_CHECK", "1")
    monkeypatch.setenv("CHAT_ALLOWED_ORIGINS", "https://viewer.example,http://192.168.1.50:8081")
    app, client = _fresh_app(tmp_path, monkeypatch)
    calls = []
    async def handler(socket, session_id):
        calls.append(session_id)
        await socket.accept()
        await socket.send_json({"ok": True})
        await socket.close()
    monkeypatch.setattr(app, "handle_websocket", handler)
    headers = {} if origin is None else {"origin": origin}
    if allowed:
        with client.websocket_connect("/ws/chat/new", headers=headers) as ws:
            assert ws.receive_json() == {"ok": True}
        assert calls == ["new"]
    else:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/chat/new", headers=headers): pass
        assert calls == []
    assert client.get("/debug/config").status_code == 200


def test_default_preserves_existing_clients(monkeypatch):
    from origin_policy import websocket_origin_allowed
    monkeypatch.delenv("CHAT_WEBSOCKET_ORIGIN_CHECK", raising=False)
    socket = WebSocket({"type": "websocket", "headers": [(b"origin", b"https://custom-lan.test")]}, receive=None, send=None)
    assert websocket_origin_allowed(socket)


def test_duplicate_origin_and_forged_forwarding_rejected(monkeypatch):
    from origin_policy import websocket_origin_allowed
    monkeypatch.setenv("CHAT_WEBSOCKET_ORIGIN_CHECK", "1")
    monkeypatch.setenv("CHAT_ALLOWED_ORIGINS", "")
    for headers in [
        [(b"origin", b"https://testserver"), (b"origin", b"https://evil.test")],
        [(b"origin", b"https://evil.test"), (b"x-forwarded-host", b"evil.test")],
    ]:
        socket = WebSocket({"type": "websocket", "headers": [(b"host", b"testserver"), *headers]}, receive=None, send=None)
        assert not websocket_origin_allowed(socket)
