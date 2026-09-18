# Updating an existing deployment

Update a selected release while preserving the services, profiles, image/model versions
and local overrides used by your deployment. A repository update is not an instruction
to rotate credentials, enable a security policy, or upgrade your language model.

## Before updating

Record the current Git revision, container image digests, selected Compose services/profiles,
and any shell-only overrides such as `VIEWER_TAG`. Keep secret values out of logs and
support tickets. Record the current `ohif_viewer` client's PKCE setting and your rollback
release. Compare the release's Compose/config changes with your own configuration.

Prepare a recoverable backup of PostgreSQL, Orthanc data/index and feedback databases.
Use database-native dumps or a coordinated consistent snapshot; a live recursive copy
of changing database files is not a verified backup. Verify the restore procedure in an
isolated environment before relying on it. Never remove persistent volumes to update.

## Build and replace only the selected services

Fetch the intended release and review its changes before switching your checkout. Resolve
local modifications explicitly. Do not overwrite your `.env` with an example.

Build/pull the affected images before interruption, using your existing project name,
profiles, release pins and overrides. For example, for a viewer-only update:

```bash
# Set VIEWER_TAG to the reviewed release; retain other deployment overrides.
docker compose pull viewer
docker compose up -d --no-deps viewer
docker compose ps viewer
```

For locally built platform changes, build the selected services first, then recreate those
services using the same project/options. This security release changes viewer configuration,
MST loading, chat origin checks, and routing/WADO helpers: rebuild the relevant platform
images if you use source builds. Router and ML shared-code changes are baked into images.
The viewer's `host_allowlist.py` and Python plugin are bind-mounted in the default stack,
so restart that service after updating those files.

Do not use an unqualified `docker compose up` that enables services you previously omitted.
Do not change the Ollama/OpenRouter model as part of this update. Avoid taking down
Keycloak/PostgreSQL or the whole stack for a viewer/configuration-only change.

## Security changes in this release

### Existing Keycloak realm: require S256 explicitly

The committed realm now requires PKCE S256 for the public `ohif_viewer` client.
New installations import it automatically. **Startup `--import-realm` skips an existing
realm**; restarting Keycloak does not apply this field.

After verifying your deployed viewer sends `code_challenge_method=S256`, sign in to the
Keycloak admin console, select realm **ohif**, client **ohif_viewer**, and its advanced
**Proof Key for Code Exchange Code Challenge Method** setting. Record its previous value,
set it to **S256**, save, and test a fresh login and logout in Chrome and Firefox.
If a customized client fails, restore only the recorded setting while correcting that client.

Do not recreate/import over the realm, remove the PostgreSQL volume, or replace users,
keys, issuer, redirect URIs, or authentication flows. Passwords, optional OTP enrollment,
session durations, silent renewal, and logout cleanup are unchanged.

### MST model integrity

MST code, configuration and weights are pinned to revision
`fb680403cf5e15958ddf6182f0b7bfff8b71a0f4` of `ODELIA-AI/MST`.
The integrity metadata `orthanc/MLIntegration/MST-classification/model-integrity.json`
contains expected SHA-256 values for all four files. These bytes match the previously
inspected running MST bundle and its secondary config/weight cache. Loading checks
the complete bundle before importing Python; configuration and weights are loaded directly
from verified local files.

Check your own deployment's model baseline before adopting this release. If you intentionally
use different bytes, review and validate the updated integrity metadata and full commit together.
Do not disable verification to accept unexplained mismatches. This is integrity against a
reviewed checksums, not a signature from the model publisher. Keep `transformers==5.3.0`;
preprocessing and model architecture are unchanged.

### Optional controls and credentials

All existing default credentials and published port bindings remain the same until an
operator supplies overrides. Environment variables do not constitute a secret vault.

| Input | Effect |
| --- | --- |
| `KEYCLOAK_DB_PASSWORD` | Shared Keycloak/PostgreSQL password input; default `password`. |
| `KEYCLOAK_ADMIN_PASSWORD` | Keycloak bootstrap admin password; default `admin`. |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin initialization password; default `odelia`. |
| `BIND_HOST=127.0.0.1` | Restrict published ports to loopback; breaks direct LAN access unless another ingress is provided. |
| `ROUTER_HOST_ALLOWLIST` | Comma-separated allowed destination hostnames for routing/WADO retrieval. Empty preserves existing routing. When set, HTTP redirects are refused on protected paths; include every intended PACS/router/subscriber host. |
| `CHAT_WEBSOCKET_ORIGIN_CHECK=1` | Check browser origins before chat session allocation. Default `0` preserves existing clients. |
| `CHAT_ALLOWED_ORIGINS` | Comma-separated viewer origins for HTTP CORS and the optional WebSocket check; use exact scheme/host/port. |

When upgrading from the old `BIND_HOST` prefix syntax, remove the trailing colon
from existing `.env` and shell values (for example, `127.0.0.1:` becomes `127.0.0.1`).

Changing database password environment variables does **not** change the password of an
existing PostgreSQL role. Coordinate actual rotation first or Keycloak loses database access.
Likewise bootstrap admin variables do not rotate existing Keycloak/Grafana users, and these
variables do not modify realm users such as `viewer` and `pacsadmin`.

When enabling WebSocket checks, same-host HTTP/HTTPS origins are accepted for same-origin
proxying, plus explicitly listed origins. Other origins, missing/duplicate Origin headers
and `null` origins are rejected. Non-browser clients need an accepted Origin header.
The check does not trust forwarded-host headers and does not authenticate a client:
a non-browser caller can forge Origin. Keep the viewer's public Host header on the proxy
hop; configure `CHAT_ALLOWED_ORIGINS` for other legitimate origins. Debug remains available.

Routing allowlists do not provide complete network isolation or DNS-rebinding protection.
Use network egress controls where this threat matters. No new authentication/role boundary,
TLS deployment, MFA requirement, account lockout or five-minute application revocation
guarantee is introduced by these changes.

## Verify and roll back

Check fresh login/logout, study listing/upload/viewing, send-to-AI and results, feedback,
chat model selection/history/deletion, and all configured local/cloud backends. Verify
the new headers and optional policies on real deployment paths, including direct ports.

For an application-only rollback, restore the recorded image/configuration versions and
recreate just affected services. Restore the recorded client PKCE field only if necessary.
Keep data volumes intact. A version upgrade that changes a database schema needs its own
tested restore/migration procedure; an image downgrade alone is not a safe rollback.

See [production hardening](../security/production-hardening.md) for remaining controls.
