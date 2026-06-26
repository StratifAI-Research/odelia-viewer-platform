# Licensing

This repository's own source — the Docker Compose stack, the configuration files, and the Python
services under [`orthanc/`](../orthanc/) — is released under the [MIT License](../LICENSE)
(© 2023–2026 StratifAI and the ODELIA project contributors).

The platform runs and builds on a number of independent components, each under **its own license**.
Running or distributing the full stack means complying with all of them — and because several are
copyleft or non-commercial, the assembled platform is **research use only** (see
[License & intended use](../README.md#license--intended-use)).

| Component | Role | License |
| --- | --- | --- |
| [OHIF Viewer](https://github.com/OHIF/Viewers) | Viewer base (via the [ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer) image) | MIT |
| [Orthanc](https://www.orthanc-server.com/) | DICOM server / PACS | GPLv3+ |
| [Keycloak](https://www.keycloak.org/) | Authentication | Apache-2.0 |
| [Grafana](https://grafana.com/) | Dashboards | AGPLv3 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | Chat LLM backend | MIT |
| [MST weights](https://huggingface.co/ODELIA-AI/MST) | MST classification model | CC-BY-NC-4.0 (non-commercial) |
| [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) | MedGemma model | Health AI Developer Foundation terms (gated) |

Orthanc, Keycloak, and Grafana run as **separate Docker containers** that the stack merely orchestrates
(their source is not bundled or modified here), so their copyleft terms do not extend to this
repository's MIT-licensed code. The viewer itself is a fork of OHIF and preserves OHIF's MIT copyright
notice (© [OHIF](https://github.com/OHIF)) in the [ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer) repository.
