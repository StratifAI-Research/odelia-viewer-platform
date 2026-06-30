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

> Re-apply any local customizations (e.g. `HF_TOKEN`, production hostnames) — a pull may surface
> upstream changes to `docker-compose.yml` or `config/`.

**4. Rebuild and restart.** Services built from source (`orthanc-viewer`, `orthanc-router-*`,
`mst-classifier`, `medgemma-mri`, `chat-middleware`) need a rebuild; pre-built images
(`viewer`, `grafana`, `keycloak`, `postgres`) just need a pull. The optional `llamacpp-server` is
pulled with its profile: `docker compose --profile llamacpp pull llamacpp-server`.

```bash
docker compose build --no-cache
docker compose pull viewer grafana keycloak postgres
docker compose up -d
docker compose ps          # confirm everything is healthy
```

**5. Update the Ollama model** if you use Chat AI: `ollama pull thiagomoraes/medgemma-1.5-4b-it:F16`.

**Condensed sequence for experienced operators:**

```bash
docker compose down && git pull origin main && docker compose build --no-cache && docker compose up -d
```
