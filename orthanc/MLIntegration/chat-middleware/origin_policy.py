"""Optional browser WebSocket origin policy; does not authenticate clients."""

import os
from urllib.parse import urlsplit

from starlette.websockets import WebSocket


def _origin(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if (
            any(c.isspace() for c in value)
            or "\\" in value
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            return None
        return (
            parsed.scheme,
            parsed.hostname.lower(),
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
    except ValueError:
        return None


def websocket_origin_allowed(websocket: WebSocket) -> bool:
    if os.getenv("CHAT_WEBSOCKET_ORIGIN_CHECK", "0") != "1":
        return True
    origins = websocket.headers.getlist("origin")
    if len(origins) != 1:
        return False
    candidate = _origin(origins[0])
    if candidate is None:
        return False
    # Same-host HTTP/HTTPS origins also cover a same-origin TLS-terminating proxy.
    # Ignore forwarded headers: only an explicitly configured origin or Host is trusted here.
    host = websocket.headers.get("host", "")
    allowed = {_origin(f"http://{host}"), _origin(f"https://{host}")}
    allowed.update(
        _origin(value.strip()) for value in os.getenv("CHAT_ALLOWED_ORIGINS", "").split(",")
    )
    allowed.discard(None)
    return candidate in allowed
