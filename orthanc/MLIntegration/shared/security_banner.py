"""
Startup banner that warns operators when an AI service is running with
research / demo defaults. Printed to stderr so it survives most logging
configurations and is visible in `docker compose logs`.

Suppress with `ODELIA_SUPPRESS_SECURITY_BANNER=1` once you have read
docs/security/production-hardening.md and intentionally accept the posture.
"""

import os
import sys

_BANNER_LINES = [
    "",
    "=" * 72,
    "WARNING — Odelia research / demo defaults are in effect.",
    "",
    "This service ships with insecure-by-design defaults intended for local",
    "research and demos. Do NOT expose this stack to an untrusted network",
    "without working through docs/security/production-hardening.md first.",
    "",
    "Known defaults that need changing for non-local deployments:",
    "  * Published ports bind to 0.0.0.0 (LAN-shared)",
    "  * Orthanc auth permissive; Keycloak in start-dev mode",
    "  * Hardcoded admin passwords for Keycloak / Grafana / Postgres",
    "  * CORS = '*' on chat-middleware; debug API on by default",
    "  * Routing endpoints accept arbitrary hostnames unless",
    "    ROUTER_HOST_ALLOWLIST is set",
    "",
    "See docs/security/production-hardening.md.",
    "Suppress this banner with ODELIA_SUPPRESS_SECURITY_BANNER=1.",
    "=" * 72,
    "",
]


def print_security_banner(service_name: str = "") -> None:
    """Print the research-defaults warning banner to stderr, once."""
    if os.environ.get("ODELIA_SUPPRESS_SECURITY_BANNER") == "1":
        return
    if service_name:
        sys.stderr.write(f"[{service_name}] ")
    for line in _BANNER_LINES:
        sys.stderr.write(line + "\n")
    sys.stderr.flush()
