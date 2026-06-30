# Configuration

The stack runs as-is for local use. This page covers the common adjustments. For hosting beyond your
own machine, start with [`production-hardening.md`](../security/production-hardening.md).

## Key files

| File | Controls |
| --- | --- |
| [`docker-compose.yml`](../../docker-compose.yml) | Services, ports, images, and environment variables (`HF_TOKEN`, `BIND_HOST`, chat backend, Keycloak hostnames) |
| [`config/app-config.js`](../../config/app-config.js) | OHIF viewer settings — DICOMweb data source, AI endpoints, OIDC |
| [`config/nginx.conf`](../../config/nginx.conf) | Reverse-proxy routes (`/`, `/pacs/`, `/keycloak/`, chat, …); set `server_name` for a real domain |
| [`config/ohif-keycloak-realm.json`](../../config/ohif-keycloak-realm.json) | Keycloak `ohif` realm, imported on first start |
| [`config/orthanc-router.json`](../../config/orthanc-router.json) | Orthanc router behaviour |

When changing the domain or going to production, update the `ohif_viewer` client (redirect URIs, web
origins) in Keycloak and the `KC_HOSTNAME_URL` / `KC_HOSTNAME_ADMIN_URL` variables in
`docker-compose.yml`. Details in [`production-hardening.md`](../security/production-hardening.md).

## Hugging Face access token (`HF_TOKEN`)

Any model whose Hugging Face repo is **gated** — i.e. requires you to request access or accept a
license before downloading — needs a Hugging Face access token, supplied as `HF_TOKEN`. The
services pass it to the Hugging Face download. Ungated models need no token, and neither does the
chat backend (it loads models from Ollama / local GGUF — see [`setup_chat.md`](setup_chat.md)).

Setup — example: the gated MedGemma model used by the breast-MRI classification service
(`google/medgemma-1.5-4b-it`):

1. Create a **Read** access token in your Hugging Face account settings. *(A fine-grained token must
   also enable "Read access to contents of all public gated repos you can access", or the download
   fails with a 403 that `transformers` misreports as "We couldn't connect to https://huggingface.co".)*
2. Request access / accept the license for the model you intend to use, while logged in to Hugging
   Face — for MedGemma, [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it).
3. Provide the token via a gitignored `.env` next to `docker-compose.yml`, then restart:

   ```bash
   echo 'HF_TOKEN=hf_your_token_here' >> .env
   docker compose up -d
   ```


## Chat AI backend

The Chat AI panel needs a local LLM backend (Ollama by default, or llama.cpp on a GPU).
Step-by-step setup — in order — is in [`setup_chat.md`](setup_chat.md).
