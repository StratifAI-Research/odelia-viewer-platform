# Changelog

What changed in each release of the ODELIA Viewer Platform, newest first. Every version
bump adds a section here, and that section becomes the release's notes on GitHub. See
`.github/RELEASING.md`.

## 2.3.0

First versioned release of the ODELIA Viewer Platform: the self-hostable stack for
ingesting DICOM studies, viewing them in the browser, running them through AI models,
and reviewing the results.

Earlier images were published only as `latest` and `0.1.0`, with nothing identifying
the commit they were built from. From this release on, a version corresponds to an
exact commit, recorded both here and in the images CI built from it. This release
ships ODELIA Viewer 2.3.0.

### What's in it

- **Viewer + Orthanc PACS** — browser-based DICOM viewing backed by a local Orthanc
  store, with feedback capture and Grafana dashboards for reader studies.
- **AI analysis** — send a breast MRI study to a model and get a three-class result
  (no lesion / benign / malignant), with annotations and a report written back as
  DICOM alongside the original images.
- **Chat AI** — ask free-form questions about the open study. Slice selection follows
  the reader's window and the viewport, a region of interest can be drawn, and the
  context sent with each message is visible. An optional cloud backend (Ollama Cloud
  or OpenRouter) is available and off by default; enabling it sends images to a third
  party.
- **Keycloak** — authentication and user separation.

### Fixes

- **Multiphase series no longer stay black while loading.** Study prefetching is now
  off by default (`studyPrefetcher.enabled` in `config/app-config.js`). Its restarts
  could drop pending volume requests from the shared Cornerstone queue, leaving series
  such as BRAID Water black and stuck loading. Unopened series now load when selected
  rather than in the background; re-enabling prefetching brings the stall back.

### Installing or updating

Back up first, as `docs/setup/updating.md` describes, then:

```bash
docker compose down
git pull origin main
docker compose pull
docker compose up -d --remove-orphans
```

`main` always matches the newest release. Its `docker-compose.yml` names every image it
runs, under `x-images` at the top: this release's three platform images
(`odelia-orthanc-viewer`, `odelia-orthanc-router` and `odelia-chat`, all `2.3.0`),
ODELIA Viewer 2.3.0 and the five trained swarm models pinned by digest. The trained
weights are baked into their images, so no `.env`, Hugging Face token or model download
is needed. Images are built for **linux/amd64** only.

Changes for existing deployments:

- **The MedGemma MRI service and its router are retired.** `--remove-orphans` removes
  their containers; their data volumes are kept.
- **Image settings in `.env` are no longer read.** Earlier setups selected images with
  `TAG`, `VIEWER_TAG`, `VIEWER_IMAGE` and the `*_IMAGE` variables; they are now
  ignored. To run other images on purpose, use `docker-compose.override.yml` (see
  `docker-compose.override.example.yml`).
- **Building from source uses `docker-compose.dev.yml`.** The default compose file only
  pulls published images.

### Setup you have to do yourself

- **Chat AI needs a model backend.** It is not usable from Compose alone: install
  Ollama on the host and pull the default model (several GB), enable the GPU-only
  llama.cpp profile, or configure a cloud backend. See `docs/setup/setup_chat.md`.

### Known limitations

- **Five trained models are selectable by default:** MST, agaldran, SwinUNETR and
  ABMIL swarm May 2026, plus MeisenMeister swarm August 2026. Pimed remains an
  untrained opt-in preview behind `--profile model-previews`.
- **MST no longer generates heatmaps.** The generalized service returns classification
  SRs without attention-map Secondary Captures. Existing stored results are retained.
- **Trained images are deliberate model releases.** The workflow runs manually;
  publishing requires selecting `publish`. Public images expose their embedded weights.
- **Breast MRI only.** The classification models assume breast MRI input; other
  anatomies will produce unreliable results. No performance metrics or bias evaluation
  are published with this release.
- **The defaults are insecure by design.** Every published port binds to all
  interfaces, Orthanc authentication is disabled, and the default credentials are
  public. Read `docs/security/production-hardening.md` and
  `docs/security/restrict-to-localhost.md` before running this anywhere but a trusted
  machine.
- **Model licences differ from this repository's.** Consult the selected swarm
  package's notices; MedGemma, the default Chat AI model, is under gated Health AI
  Developer Foundation terms. See `docs/licensing.md`.

### Intended use

**Research use only.** This platform and its AI workflows are research software. They
are **not a medical device**, are **not CE-marked or FDA-cleared**, and must **not be
used for clinical diagnosis or patient care**. Model outputs may be inaccurate.
