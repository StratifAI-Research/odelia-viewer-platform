# Bundled AI models

> **All AI models in the ODELIA Viewer Platform are for research use only.**
> Outputs may be inaccurate and must not be used for clinical decision-making.

The default viewer exposes five trained swarm classifiers: MST, agaldran,
SwinUNETR and ABMIL from May 2026, plus MeisenMeister from August 2026.
See [trained model deployment](../usage/loading_model_weights.md) for the release
configuration. The model cards below describe MST and the chat service:

| Model | Architecture | Task | Classes | Input | Card |
|-------|--------------|------|---------|-------|------|
| MST Classification | DINOv2 Vision Transformer | 3-class classification | No lesion / Benign / Malignant | Breast DCE-MRI (pre+post, subtraction, or multi-phase) | [mst-classification.md](mst-classification.md) |
| Chat Middleware | Vision-Language Model (Ollama / llama.cpp) | Free-form chat | N/A | Any DICOM series | [chat-middleware.md](chat-middleware.md) |

The five swarm endpoints are registered in the viewer out of the box; see
[`usage/`](../usage/README.md#2-send-a-study-to-ai) for how to run them.

## Common input requirements

All models receive DICOM data via WADO-RS retrieval. These tags are universally required:

| Tag | Keyword | Purpose |
|-----|---------|---------|
| `(0020,000D)` | StudyInstanceUID | Study-level retrieval |
| `(0020,000E)` | SeriesInstanceUID | Series-level retrieval |
| `(7FE0,0010)` | PixelData | Image data |

## Critical DICOM tags by model

| Tag | Keyword | MST | Chat |
|-----|---------|-----|------|
| `(0020,0100)` | TemporalPositionIdentifier | Primary | — |
| `(0018,1060)` | TriggerTime | Fallback | — |
| `(0020,1041)` | SliceLocation | Primary | Via GDCM |
| `(0020,0032)` | ImagePositionPatient | Fallback | Via GDCM |
| `(0020,0013)` | InstanceNumber | Fallback | Via GDCM |

## Key limitations

### All models

- **Research use only** — not validated for clinical use.
- **Breast MRI only** — the classification models assume breast MRI input; other anatomies will produce unreliable results.
- **No demographic bias testing** — performance across patient demographics has not been systematically evaluated.
- **No performance metrics published** — evaluation results are not tracked or displayed within the viewer.

### MST Classification

- Loads baked May 2026 swarm weights without runtime HF credentials or downloads.
- Returns classification SRs without the legacy service's attention-map overlays or heatmap Secondary Captures. Existing stored results are retained.
- Classifies a contrast-subtraction volume; the subtraction is either supplied directly or computed by the service depending on the selected input configuration (pre+post, subtraction, or multi-phase) — see the [card](mst-classification.md#input-configurations).
- With both `SliceLocation` and `ImagePositionPatient` missing, `InstanceNumber` is the only remaining ordering key; if that is also unreliable the volume can be **silently mis-ordered**.

### Chat Middleware

- Free-form text responses — no structured classification guarantee.
- No explicit 4D temporal series handling; multi-phase series may be merged incorrectly.
- Depends on an external Ollama (or llama.cpp) backend being available with the model loaded.
- Any confidence or certainty expressed in chat responses is not statistically calibrated.

## Legacy reference

The [MedGemma MRI model card](medgemma-mri.md) documents retained source only.
Its classification service and router are no longer deployed or published by CI.
MedGemma remains an option for the separate [Chat AI backend](../setup/setup_chat.md).
