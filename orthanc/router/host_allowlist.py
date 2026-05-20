"""
Optional hostname allowlist for outbound URLs in UPS routing endpoints.

Reads `ROUTER_HOST_ALLOWLIST` (comma-separated) from the environment.
Empty / unset => no enforcement (current research behaviour).
Non-empty   => URLs whose hostname is not in the set are rejected.

See docs/production-hardening.md.
"""
import os
from urllib.parse import urlparse


def _load_allowlist():
    raw = os.environ.get("ROUTER_HOST_ALLOWLIST", "").strip()
    if not raw:
        return None
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _extract_host(url: str) -> str:
    """
    Extract a hostname from either:
      * a proper URL ("http://orthanc-router-mst:8042/dicom-web") — via urlparse
      * a DICOM modality target ("orthanc-router-mst:4242/AET") — by splitting
        on ':' / '/' (urlparse treats the first token as the scheme here)
    Returns "" if no host can be identified.
    """
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if host:
        return host.lower()
    token = url.split("://", 1)[-1]
    token = token.split("/", 1)[0]
    token = token.split(":", 1)[0]
    return token.lower()


def host_is_allowed(url: str) -> bool:
    """
    Return True if `url`'s hostname is allowed (or the allowlist is empty /
    unset). Returns False for unparseable URLs or hosts not in the allowlist.
    """
    allow = _load_allowlist()
    if allow is None:
        return True
    host = _extract_host(url)
    return bool(host) and host in allow
