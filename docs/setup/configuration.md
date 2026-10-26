# Configuration

The stack runs as-is for local use. This page covers the common adjustments. For hosting beyond your
own machine, start with [`production-hardening.md`](../security/production-hardening.md).

## Key files

| File | Controls |
| --- | --- |
| [`docker-compose.yml`](../../docker-compose.yml) | Services, ports, images, and environment variables (image overrides, `BIND_HOST`, chat backend, Keycloak hostnames) |
| [`config/app-config.js`](../../config/app-config.js) | OHIF viewer settings — DICOMweb data source, AI endpoints, OIDC |
| [`config/nginx.conf`](../../config/nginx.conf) | Reverse-proxy routes (`/`, `/pacs/`, `/keycloak/`, chat, …); set `server_name` for a real domain |
| [`config/ohif-keycloak-realm.json`](../../config/ohif-keycloak-realm.json) | Keycloak `ohif` realm, imported on first start |
| [`config/orthanc-router.json`](../../config/orthanc-router.json) | Orthanc router behaviour |

When changing the domain or going to production, update the `ohif_viewer` client (redirect URIs, web
origins) in Keycloak and the `KC_HOSTNAME_URL` / `KC_HOSTNAME_ADMIN_URL` variables in
`docker-compose.yml`. Details in [`production-hardening.md`](../security/production-hardening.md).

## Login theme

Keycloak uses the [ODELIA login theme](../../themes/odelia/README.md), matching the viewer's
dark surfaces, blue buttons, cyan focus indicators, and Inter font. Compose mounts the theme
read-only, and the imported realm selects it for new installations.

For an existing installation, run the updated Compose configuration to add the theme mount:

    docker compose up -d --no-deps keycloak

Then select **odelia** under **Realm settings → Themes → Login theme** in the **ohif** realm.
Startup import skips an existing realm, so changing the realm JSON alone does not switch its theme.
Existing users, authentication flows, and client settings are retained. To revert the appearance,
select the previous login theme in the same settings.

## Hugging Face access token (`HF_TOKEN`)

The five default trained models include weights in their Docker images and need
no runtime HF token. Building a new image from a private HF repository requires
a token with read access, passed as a BuildKit secret. For local builds and GitHub
Actions configuration, see [loading trained model packages](../usage/loading_model_weights.md).
The chat backend uses Ollama or local GGUF files; see [`setup_chat.md`](setup_chat.md).

## Chat AI backend

The Chat AI panel needs a local LLM backend (Ollama by default, or llama.cpp on a GPU).
Step-by-step setup — in order — is in [`setup_chat.md`](setup_chat.md).
