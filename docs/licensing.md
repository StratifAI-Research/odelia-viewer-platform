# Licensing

This repository's own source — the Docker Compose stack, the configuration files, the documentation,
and the Python services under [`orthanc/`](../orthanc/) — is released under the
[GNU General Public License v3.0 or later](../LICENSE) (`GPL-3.0-or-later`,
© 2023–2026 StratifAI and the ODELIA project contributors).

GPLv3 is required because the stack does not merely *run* Orthanc — it builds **custom Orthanc images**
([`orthanc/viewer/Dockerfile`](../orthanc/viewer/Dockerfile),
[`orthanc/router/Dockerfile`](../orthanc/router/Dockerfile)) that load our own Python **plugins** into
the Orthanc process (`router.py` / `server.py`, registered via `ORTHANC__PYTHON_SCRIPT`). That plugin
code links against the `orthanc` API, so the images this repository distributes are a combined work
with GPLv3 Orthanc. To keep the whole platform unambiguous and compliant, all of this repository's
source is GPLv3.

The platform also runs and builds on a number of independent components, each under **its own
license**. Running or distributing the full stack means complying with all of them — and because
several are copyleft or non-commercial, the assembled platform is **research use only** (see
[License & intended use](../README.md#license--intended-use)).

| Component | Role | License |
| --- | --- | --- |
| [OHIF Viewer](https://github.com/OHIF/Viewers) | Viewer base (via the [ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer) image) | MIT |
| [Orthanc](https://www.orthanc-server.com/) | DICOM server / PACS | GPLv3+ |
| [Keycloak](https://www.keycloak.org/) | Authentication | Apache-2.0 |
| [Inter](https://rsms.me/inter/) | Bundled login-theme font | [SIL Open Font License 1.1](../themes/odelia/login/resources/fonts/OFL.txt) |
| [Grafana](https://grafana.com/) | Dashboards | AGPLv3 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | Chat LLM backend | MIT |
| [Original MST weights](https://huggingface.co/ODELIA-AI/MST) | Legacy MST model, no longer the default deployment | CC-BY-NC-4.0 (non-commercial) |
| Trained swarm weights | Five default classification models, baked into selected images | Release-specific model cards and notices; separate from the original challenge weights |
| [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) | MedGemma model | Health AI Developer Foundation terms (gated) |

Keycloak and Grafana run as **separate, unmodified Docker containers** that the stack only
orchestrates (their source is neither bundled nor modified here), so their terms apply to those
containers rather than reaching this repository's code. Orthanc is different: because we ship plugin
code that runs inside it, the Orthanc images we build are a derivative work — which is why this
repository as a whole is GPLv3 (see above). Model weights are not included in this source repository.
The trained swarm images include their weights; publishing one publicly distributes those weights.
Consult the selected release's notices before publishing. MedGemma still downloads its gated weights
at runtime. The viewer is a
fork of OHIF and preserves OHIF's MIT copyright notice (© [OHIF](https://github.com/OHIF)) in the
[ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer) repository.
