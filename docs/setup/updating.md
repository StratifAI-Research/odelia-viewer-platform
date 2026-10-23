# Updating an existing deployment

**1. Back up persistent data.** `volumes/` holds DICOM images and the feedback database; the
`postgres_data` named volume holds the Keycloak database. Dump Postgres while the containers are
still up (a `pg_dumpall` is robust against the project-prefixed volume name):

```bash
cp -r volumes/ volumes-backup-$(date +%F)/
docker compose exec -T postgres pg_dumpall -U keycloak > postgres-backup-$(date +%F).sql
```

**2. Stop containers** (this does not delete volumes):

```bash
docker compose down
```

**3. Pull the latest repository** and review config changes before restarting:

```bash
git pull origin main
git diff HEAD@{1} -- docker-compose.yml config/   # check for new vars, ports, routes
```

> Re-apply any local customizations (e.g. image overrides, production hostnames) — a pull may surface
> upstream changes to `docker-compose.yml` or `config/`.

**4. Pull and restart.** The default services use published images; model weights
are embedded and no HF token is needed. Image digests are updated manually in
`docker-compose.yml`. Existing model `*_IMAGE`, `VIEWER_IMAGE`, `VIEWER_TAG`,
`ORTHANC_VIEWER_IMAGE`, `ORTHANC_ROUTER_IMAGE`, `CHAT_IMAGE`, and `TAG` overrides
take precedence: remove or update them to adopt the new pins.

Published defaults always use `stratifai`; `DOCKER_HUB_USERNAME` now affects only
locally built model previews. For a custom registry or namespace, set the full
`*_IMAGE` reference for each service you replace. `TAG` selects a shared tag for
Orthanc/chat under `stratifai`; a full image override takes precedence over it.

```bash
docker compose pull
docker compose up -d --no-build --remove-orphans
docker compose ps          # confirm everything is healthy
```

`--remove-orphans` removes retired services, including the MedGemma MRI service
and router. It preserves their data volumes. The optional `llamacpp-server`
is pulled with its profile: `docker compose --profile llamacpp pull llamacpp-server`.

For development builds of Orthanc or chat, set a writable tag, for example
`TAG=dev docker compose build orthanc-viewer chat-middleware`, then start those
services with the same `TAG=dev` override. Digest references cannot be build tags.

**5. Update the Ollama model** if you use Chat AI: `ollama pull thiagomoraes/medgemma-1.5-4b-it:F16`.

**Condensed sequence for experienced operators:**

```bash
docker compose down
git pull origin main
docker compose pull
docker compose up -d --no-build --remove-orphans
```
