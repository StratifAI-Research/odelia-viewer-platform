# Using the platform

A walkthrough of the day-to-day workflow once the stack is running (see the
[README quick start](../../README.md#quick-start) to bring it up). The viewer is at
<http://localhost:8081> — sign in with the default credentials `viewer` / `viewer`.

## 1. Upload DICOM studies

Get `.dcm` files into Orthanc by any of:

- **Viewer (recommended)** — use the upload button at <http://localhost:8081>.
- **Orthanc web UI** — drag-and-drop at <http://localhost:8000>.
- **DICOM C-STORE** — send from an external PACS/modality to host `localhost`, port `2000`
  (AE Title `ORTHANC`; the called AE title is not checked).

<p align="center"><img src="../images/orthanc-upload.jpg" alt="Uploading a study in the viewer" width="720"/></p>

## 2. Send a study to AI

1. Open a study. The **AI Analysis** panel (right sidebar) detects the active study.
2. Choose the series to analyze and click **Next**.
3. Pick a model and click **Send to AI**.
4. The processed study — annotations, classification, and report — appears in your study list
   when inference finishes.

The five trained swarm endpoints are registered in
[`config/app-config.js`](../../config/app-config.js) (`aiEndpoints`): **MST,
agaldran, SwinUNETR and ABMIL (May 2026)**, and **MeisenMeister (August 2026)**.
See [the deployment roster](loading_model_weights.md#default-viewer-deployment)
for their model/router pairs and optional image overrides.

<p align="center">
  <img src="../images/ai-analysis-panel.jpg" alt="AI analysis panel" width="360"/>
  <img src="../images/select-ai-model.jpg" alt="Selecting an AI model" width="360"/>
</p>

See [`models/`](../models/) for what each model does, its inputs, and its limits.

## 3. Chat about a study

The **Chat AI** panel runs a vision-language model (MedGemma by default) against the open study. It requires a local
LLM backend (Ollama or llama.cpp) — see [Setting up the Chat AI](../setup/setup_chat.md).

## 4. Manage users

Add users in Keycloak (default admin `admin` / `admin`). Step-by-step:
[`adding_new_users.md`](adding_new_users.md).

## Where this fits in the stack

The services behind this workflow — the viewer, Orthanc PACS, the AI routers, the model
services, and Grafana — and a diagram of how a study flows through them are described in
[`architecture.md`](../architecture.md).