# Security & hardening

The stack ships **insecure-by-design** defaults for frictionless local research. Work through
these before exposing any part of it beyond a trusted machine.

- [production-hardening.md](production-hardening.md) — full checklist: credentials, auth, HTTPS, CORS, SSRF, debug API
- [restrict-to-localhost.md](restrict-to-localhost.md) — bind every published port to loopback with `BIND_HOST`
