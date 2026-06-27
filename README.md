<div align="center">
  <h1>ODELIA Viewer Platform</h1>
  <p>
    <strong>The full, runnable deployment stack for the <a href="https://odelia.ai/">ODELIA</a> medical-imaging platform.</strong><br/>
    Viewer + Orthanc PACS + AI services + authentication + monitoring, wired together with Docker Compose. Maintained by <a href="https://www.stratifai.com/">StratifAI</a>.
  </p>

  <p>
    <a href="https://odelia.ai/">ODELIA Project</a> ·
    <a href="https://www.stratifai.com/">StratifAI</a> ·
    <a href="https://github.com/StratifAI-Research/odelia-viewer">ODELIA Viewer</a> ·
    <a href="#documentation">Documentation</a>
  </p>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"/></a>
    <a href="https://github.com/StratifAI-Research/odelia-viewer-platform/actions/workflows/docker-build-push.yml"><img src="https://github.com/StratifAI-Research/odelia-viewer-platform/actions/workflows/docker-build-push.yml/badge.svg" alt="Docker Build & Push"/></a>
    <a href="https://github.com/StratifAI-Research/odelia-viewer-platform/actions/workflows/python-tests.yml"><img src="https://github.com/StratifAI-Research/odelia-viewer-platform/actions/workflows/python-tests.yml/badge.svg" alt="Python tests"/></a>
    <a href="https://github.com/StratifAI-Research/odelia-viewer-platform/actions/workflows/python-lint.yml"><img src="https://github.com/StratifAI-Research/odelia-viewer-platform/actions/workflows/python-lint.yml/badge.svg" alt="Python lint"/></a>
    <img src="https://img.shields.io/badge/use-research%20only-orange" alt="Research use only"/>
  </p>
</div>

---

The **ODELIA Viewer Platform** is the complete, self-hostable stack behind the ODELIA medical-imaging
workflow. With one `docker compose up` it brings up everything you need to ingest DICOM studies, view
them in the browser, run them through AI models, and review the results — all on your own machine.

It bundles:

- **[ODELIA Viewer](https://github.com/StratifAI-Research/odelia-viewer)** — a web DICOM viewer (an [OHIF](https://ohif.org/) fork).
- **[Orthanc](https://www.orthanc-server.com/)** — a local PACS that stores DICOMs.
- **AI services** — send DICOM studies to [ODELIA models](https://huggingface.co/ODELIA-AI/models) and wrap the results back into DICOM.
- **[Keycloak](https://www.keycloak.org/)** — user authentication and separation.
- **[Grafana](https://grafana.com/)** — dashboards to collect and evaluate reader studies.



## Quick start

**Prerequisites:** [Git](https://git-scm.com/) and [Docker](https://docs.docker.com/get-docker/)

```bash
git clone https://github.com/StratifAI-Research/odelia-viewer-platform.git
cd odelia-viewer-platform
docker compose up -d
```

**The first start builds several images and may take a while.** \
Check progress with `docker compose ps` and `docker compose logs -f`.

Then open **<http://localhost:8081>** and sign in with the default credentials **`viewer` / `viewer`**.

To use the **chat** feature:
[`docs/setup/setup_chat.md`](docs/setup/setup_chat.md).
To update an existing deployment: [`docs/setup/updating.md`](docs/setup/updating.md).




## Using the platform

Once the stack is up, the typical workflow is:

1. **Upload** a study — via the viewer's upload button, the Orthanc web UI, or DICOM C-STORE.
2. **Send to AI** — open a study, pick a series and model in the **AI Analysis** panel, and run it.
3. **Review** — annotations, classification, and report appear next to the original images.
4. **Chat** — ask free-form questions about the open study in the **Chat AI** panel.

Full step-by-step instructions, screenshots, and user management are in
[`docs/usage/`](docs/usage/).

## Configuration
To customize the setup take a look at [`docs/setup/configuration.md`](docs/setup/configuration.md).

The defaults are **insecure-by-design** for frictionless local research.\
Before exposing any part of this stack beyond a trusted machine, read
[`docs/security/production-hardening.md`](docs/security/production-hardening.md).



## Documentation

| Guide | What it covers |
| --- | --- |
| [`docs/`](docs/) | Documentation index |
| [`docs/architecture.md`](docs/architecture.md) | Services, repo layout, and how a study flows through the stack |
| [`docs/setup/`](docs/setup/) | Configuration & `HF_TOKEN`, the Chat AI backend, and updating an existing deployment |
| [`docs/usage/`](docs/usage/) | Upload, run AI, chat, manage users, add custom models — the how-to guides |
| [`docs/models/`](docs/models/) | Bundled-model cards (MST, MedGemma, chat) — inputs, classes, and limits |
| [`docs/security/`](docs/security/) | Production hardening (credentials, auth, HTTPS, CORS, SSRF) and binding ports to localhost |
| [`docs/licensing.md`](docs/licensing.md) | Per-component licenses and why the assembled stack is research-only |
| [`docs/support.md`](docs/support.md) | Collecting logs for a support request |





## About ODELIA

[ODELIA](https://odelia.ai/) (Open Consortium for Decentralized Medical Artificial Intelligence) unites
partners across Europe to build the first open-source **swarm learning** framework — training medical
AI across institutions without sharing patient data — and to develop and validate AI for breast-cancer
detection in MRI.

- [ODELIA website](https://odelia.ai/)
- [Hugging Face](https://huggingface.co/ODELIA-AI)
- [Zenodo](https://zenodo.org/communities/odelia/)

This project has received funding from the European Union's Horizon Europe research and innovation
programme under grant agreement [No 101057091](https://cordis.europa.eu/project/id/101057091).



## Support

- **Bugs & feature requests:** [open an issue](https://github.com/StratifAI-Research/odelia-viewer-platform/issues) in this repository.
- **Further support information** in [`docs/support.md`](docs/support.md)



## License & intended use

**Research use only.** Model outputs may be inaccurate.

This repository's own source is released under the [GNU General Public License v3.0 or later](LICENSE)
(© 2023–2026 StratifAI and the ODELIA project contributors). The stack builds custom Orthanc images
that load our own Orthanc plugins, so what it distributes is a combined work with GPLv3 Orthanc —
hence the whole repository is GPLv3. It also bundles further independent components under their own
licenses — several copyleft or non-commercial. See [`docs/licensing.md`](docs/licensing.md) for the
per-component breakdown.