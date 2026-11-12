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

`main` always equals the newest release; [`CHANGELOG.md`](../../CHANGELOG.md) lists what
changed. To go back to an older release, see
[Go back to an older release](../releases.md#go-back-to-an-older-release).

**4. Pull and restart.** The stack runs the published images listed under `x-images`
in `docker-compose.yml`; model weights are embedded and no HF token is needed. A
`docker-compose.override.yml` takes precedence over that list: remove it, or update
it, to adopt the new images. Image variables in `.env` (`TAG`, `VIEWER_TAG`,
`*_IMAGE` and similar) are no longer read; move any you still need into
`docker-compose.override.yml` (see `docker-compose.override.example.yml`).

```bash
docker compose pull
docker compose up -d --remove-orphans
docker compose ps          # confirm everything is healthy
```

`--remove-orphans` removes retired services, including the MedGemma MRI service
and router. It preserves their data volumes. The optional `llamacpp-server`
is pulled with its profile: `docker compose --profile llamacpp pull llamacpp-server`.

To build the Orthanc, router and chat images from your checkout instead, add
`docker-compose.dev.yml`:
`docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build`.

**5. Update the Ollama model** if you use Chat AI: `ollama pull thiagomoraes/medgemma-1.5-4b-it:F16`.

**Condensed sequence for experienced operators:**

```bash
docker compose down
git pull origin main
docker compose pull
docker compose up -d --remove-orphans
```
