# MLIntegration

ML services that the ODELIA Viewer Platform routes DICOM studies to, plus the shared
library they build on. Each service is a Flask (or async) microservice that retrieves a
series via WADO-RS, runs inference, and returns JSON results the Orthanc router converts
into DICOM SR / Secondary Capture.

## Layout

| Path | What it is |
| --- | --- |
| `shared/` | Reusable library (WADO-RS retrieval, DICOM storage, config, exceptions, timing & security helpers) — installed into every service image via `pip install -e .` |
| `MST-classification/` | MST (DINOv2) bilateral classifier with attention maps — **deployed** as `mst-classifier` |
| `medgemma-mri/` | MedGemma vision-language model, applied to breast-MRI classification — **deployed** as `medgemma-mri` |
| `chat-middleware/` | WebSocket chat backend (Ollama / llama.cpp) — **deployed** as `chat-middleware` |
| `breast-cancer-classification/` | Bilateral breast-cancer classifier (port 5555) — present in the tree but **not deployed** by the current `docker-compose.yml`; remains the default `MODEL_BACKEND_URL` target in the router |
| `pyproject.toml` | Packaging for the `shared` module (`mlintegration` wrapper) |

The services are built and run by the root [`docker-compose.yml`](../../docker-compose.yml);
you do not start them from this directory directly.

## Documentation

- **What each model does, its inputs and limits** — [`docs/models/`](../../docs/models/)
- **Integrate your own model** — [`docs/usage/adding_custom_models.md`](../../docs/usage/adding_custom_models.md)
- **Working on the Python services** (linting, tests, dependency pinning) — [`orthanc/CONTRIBUTING.md`](../CONTRIBUTING.md)
