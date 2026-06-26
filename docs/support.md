# Getting support

Report bugs and request features by [opening an issue](https://github.com/StratifAI-Research/odelia-viewer-platform/issues)
in this repository.

## Collecting logs for a support request

When you open an issue, please attach the following so the problem can be reproduced:

```bash
mkdir -p logs
docker compose logs --no-color > logs/compose.log
docker compose ps -a          > logs/containers.log
docker system info            > logs/docker-info.log
cp config/*                     logs/
```

Compress the `logs/` directory and attach it to the issue, together with a description of
what you did, what you expected, and what happened instead.

> [!CAUTION]
> GitHub issues are **public**. Logs and config files can contain credentials, Hugging Face
> tokens, hostnames, or the Keycloak realm signing key (`config/ohif-keycloak-realm.json`).
> Redact anything sensitive before attaching, or share it privately if your deployment uses
> real secrets.
