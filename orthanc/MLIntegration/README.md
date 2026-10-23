# MLIntegration

ML services that the ODELIA Viewer Platform routes DICOM studies to, plus the shared
library they build on. Each service is a Flask (or async) microservice that retrieves a
series via WADO-RS, runs inference, and returns JSON results the Orthanc router converts
into DICOM SR / Secondary Capture.

## Layout

| Path | What it is |
| --- | --- |
| `shared/` | Reusable library (WADO-RS retrieval, DICOM storage, config, exceptions, timing & security helpers) — installed into every service image via `pip install -e .` |
| `odelia-classification/` | Trained swarm classification service — **deployed** as `odelia-classification-mst` and the four other trained model services; classification results and DICOM SRs, without heatmaps |
| `MST-classification/` | Legacy MST classifier with attention maps — retained for reference, **not deployed or built/published by CI** |
| `medgemma-mri/` | MedGemma vision-language model, applied to breast-MRI classification — **deployed** as `medgemma-mri` |
| `chat-middleware/` | WebSocket chat backend (Ollama / llama.cpp) — **deployed** as `chat-middleware` |
| `breast-cancer-classification/` | Bilateral breast-cancer classifier (port 5555) — present in the tree but **not deployed** by the current `docker-compose.yml`; remains the default `MODEL_BACKEND_URL` target in the router |
| `pyproject.toml` | Packaging for the `shared` module (`mlintegration` wrapper) |

The services are built and run by the root [`docker-compose.yml`](../../docker-compose.yml);
you do not start them from this directory directly.

The default MST endpoint now uses the generalized service with the May 2026
swarm weights. It keeps the existing endpoint address but no longer generates
attention-map overlays or heatmap Secondary Captures. Existing stored results
are retained. See the [MST model card](../../docs/models/mst-classification.md).

## Documentation

- **What each model does, its inputs and limits** — [`docs/models/`](../../docs/models/)
- **Integrate your own model** — [`docs/usage/adding_custom_models.md`](../../docs/usage/adding_custom_models.md)
- **Working on the Python services** (linting, tests, dependency pinning) — [`orthanc/CONTRIBUTING.md`](../CONTRIBUTING.md)
