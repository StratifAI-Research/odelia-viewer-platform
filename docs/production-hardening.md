# Production hardening checklist

> **The default configuration of this repository is intended for local
> research and demos.** It is optimized for `docker compose up` ease of
> setup, not for production. If you intend to expose any part of this
> stack beyond a trusted local network, work through this checklist
> first — every default below is *insecure-by-design* for the sake of
> the research UX.

The threat model the defaults assume is:

> "Keep easy local setup; prevent surprise production exposure. LAN
> sharing with colleagues is a feature, not a bug."

If your deployment violates that assumption (multi-tenant network, VPS,
cloud VM, anything reachable from the public internet), do everything
in the **Must-do** section. The **Should-do** section is recommended
for any long-lived hosted deployment.

---

## Must-do before any non-local exposure

### 1. Restrict published ports

Default: every published port binds to `0.0.0.0`. Set `BIND_HOST=127.0.0.1:`
(trailing colon required) in `.env` to flip the whole stack to loopback
binding — see [`restrict-to-localhost.md`](restrict-to-localhost.md) for
details. A managed firewall / security group is fine in addition, not as
a replacement.

### 2. Enable Orthanc authentication

The Orthanc instances run with permissive auth defaults to make the
demo painless. Before production:

* Set `RegisteredUsers` in `orthanc/viewer/orthanc.json` (and the
  router instances) with strong credentials.
* Set `AuthenticationEnabled` to `true`.
* Configure HTTPS in front of Orthanc (nginx, Traefik, Caddy).

### 3. Rotate every default credential

The repo ships with these *known-public* credentials so the stack
starts out-of-the-box. **All of them must be changed.**

| Service   | Default user | Default password | Source                                        |
| --------- | ------------ | ---------------- | --------------------------------------------- |
| Keycloak admin | `admin` | `admin`     | [`docker-compose.yml`](../docker-compose.yml) `KEYCLOAK_ADMIN_PASSWORD` |
| Keycloak DB    | `keycloak` | `password` | [`docker-compose.yml`](../docker-compose.yml) `KC_DB_PASSWORD` |
| Postgres       | `keycloak` | `password` | [`docker-compose.yml`](../docker-compose.yml) `POSTGRES_PASSWORD`      |
| Grafana admin  | `admin` | `odelia`    | [`docker-compose.yml`](../docker-compose.yml) `GF_SECURITY_ADMIN_PASSWORD` |
| Viewer login   | `viewer` | `viewer`   | Keycloak realm import [`config/ohif-keycloak-realm.json`](../config/ohif-keycloak-realm.json) |

The preferred approach is to switch the compose file from hardcoded
values to `${VAR:-default}` form and supply real secrets via a `.env`
file that is **not** committed:

```yaml
# docker-compose.yml
environment:
  KEYCLOAK_ADMIN_PASSWORD: ${KEYCLOAK_ADMIN_PASSWORD:-admin}
  KC_DB_PASSWORD:          ${KC_DB_PASSWORD:-password}
  POSTGRES_PASSWORD:       ${POSTGRES_PASSWORD:-password}
  GF_SECURITY_ADMIN_PASSWORD: ${GF_SECURITY_ADMIN_PASSWORD:-odelia}
```

```bash
# .env (gitignored)
KEYCLOAK_ADMIN_PASSWORD=<long random>
KC_DB_PASSWORD=<long random>
POSTGRES_PASSWORD=<long random>
GF_SECURITY_ADMIN_PASSWORD=<long random>
```

Also create new realm users in Keycloak and remove the default
`viewer/viewer` test user.

### 4. Switch Keycloak out of dev mode

`docker-compose.yml` runs Keycloak with `command: 'start-dev --import-realm'`,
which disables HTTPS enforcement and brute-force protection. For
production, switch to `start` and configure:

* HTTPS termination (either Keycloak-native via `KC_HTTPS_*` env vars,
  or a TLS-terminating reverse proxy in front).
* `KC_HOSTNAME_STRICT_HTTPS=true`.
* Brute-force detection in the realm settings.
* Remove `KC_HOSTNAME_DEBUG=true` from environment.
* Re-issue realm signing keys (the import file ships a key the world
  can read).

### 5. Validate routing target URLs

Set `ROUTER_HOST_ALLOWLIST` (comma-separated hostnames) in the
environment of `orthanc-viewer` and the `orthanc-router-*` services.
When non-empty, the router REST handlers will reject any `target_url`,
`wado_rs_base`, or `subscriber_url` whose hostname is not in the
allowlist — preventing SSRF against `keycloak`, `grafana`, or cloud
metadata endpoints. Empty / unset preserves the current research
behaviour.

Example:

```yaml
environment:
  ROUTER_HOST_ALLOWLIST: "orthanc-viewer,orthanc-router-mst,orthanc-router-medgemma"
```

### 6. Disable debug API on chat-middleware

The chat-middleware ships a debug router at `/debug` that exposes
runtime configuration and cache inspection. Set
`DEBUG_API_ENABLED=false` (or simply leave it unset and gate the
blueprint registration on the env var) in any non-local profile.

```yaml
chat-middleware:
  environment:
    DEBUG_API_ENABLED: "false"
```

### 7. Tighten CORS

`orthanc/MLIntegration/chat-middleware/app.py` allows `*` for
`allow_origins`. Replace with an explicit list of the OHIF viewer
origin(s) the chat-middleware should serve:

```python
allow_origins=[os.environ.get("CHAT_CORS_ORIGIN", "https://viewer.example.org")],
```

The Flask AI services (breast-cancer-classification, MST-classification,
medgemma-mri) use `flask_cors.CORS(app)` with no whitelist; lock these
down the same way if they will be reachable from a browser.

---

## Should-do for any hosted deployment

### 8. Replace verbose error messages with correlation IDs

Several Flask handlers return `str(e)` in JSON error responses. The
exposure is low (exception messages, not stack traces) but a hosted
deployment should swap to a generic error message and log the real
exception against a correlation ID returned to the client.

### 9. Bump container images on a schedule

Some images are pinned to older versions for reproducible demos
(notably `grafana/grafana:11.1.0` and
`quay.io/keycloak/keycloak:24.0.5`). Long-lived hosted deployments
should track upstream releases and rebuild periodically to pick up
security fixes. Keep the pinned tag in the repo, but document the
expected refresh cadence for your deployment.

### 10. Remove TLS-verify bypass when wiring HTTPS

`orthanc/MLIntegration/breast-cancer-classification/app.py` (legacy)
calls Orthanc with `verify=False`. This is a no-op today because
`ORTHANC_URL` is always plain HTTP inside the Docker network, but the
flag is a latent footgun the moment anyone points the service at an
HTTPS endpoint. Remove `verify=False` as part of the production pass.

---

## Where to look in the code

* Compose file: [`docker-compose.yml`](../docker-compose.yml)
* Orthanc viewer config: [`orthanc/viewer/orthanc.json`](../orthanc/viewer/orthanc.json)
* Keycloak realm: [`config/ohif-keycloak-realm.json`](../config/ohif-keycloak-realm.json)
* Router REST handlers: [`orthanc/viewer/router.py`](../orthanc/viewer/router.py), [`orthanc/router/ups/routes.py`](../orthanc/router/ups/routes.py), [`orthanc/router/ups/processor.py`](../orthanc/router/ups/processor.py)
* Chat-middleware CORS / debug: [`orthanc/MLIntegration/chat-middleware/app.py`](../orthanc/MLIntegration/chat-middleware/app.py), [`orthanc/MLIntegration/chat-middleware/debug_routes.py`](../orthanc/MLIntegration/chat-middleware/debug_routes.py)
