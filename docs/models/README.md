# Bundled AI models

> **All AI models in the ODELIA Viewer Platform are for research use only.**
> Outputs may be inaccurate and must not be used for clinical decision-making.

This stack ships three AI services. Each has a model card describing its inputs,
behaviour, and failure modes:

| Model | Architecture | Task | Classes | Input | Card |
|-------|--------------|------|---------|-------|------|
| MST Classification | DINOv2 Vision Transformer | 3-class classification | No lesion / Benign / Malignant | Breast DCE-MRI (pre+post, subtraction, or multi-phase) | [mst-classification.md](mst-classification.md) |
| MedGemma | Vision-Language Model | 3-class classification | No lesion / Benign / Malignant | Breast MRI (single 3D series) | [medgemma-mri.md](medgemma-mri.md) |
| Chat Middleware | Vision-Language Model (Ollama / llama.cpp) | Free-form chat | N/A | Any DICOM series | [chat-middleware.md](chat-middleware.md) |

Only the **MST** endpoint is registered in the viewer out of the box; see
[`usage/`](../usage/README.md#2-send-a-study-to-ai) for how to expose MedGemma too.

## Common input requirements

All models receive DICOM data via WADO-RS retrieval. These tags are universally required:

| Tag | Keyword | Purpose |
|-----|---------|---------|
| `(0020,000D)` | StudyInstanceUID | Study-level retrieval |
| `(0020,000E)` | SeriesInstanceUID | Series-level retrieval |
| `(7FE0,0010)` | PixelData | Image data |

## Critical DICOM tags by model

| Tag | Keyword | MST | MedGemma | Chat |
|-----|---------|-----|----------|------|
| `(0020,0100)` | TemporalPositionIdentifier | Primary | Primary | — |
| `(0018,1060)` | TriggerTime | Fallback | Fallback | — |
| `(0020,1041)` | SliceLocation | Primary | Primary | Via GDCM |
| `(0020,0032)` | ImagePositionPatient | Fallback | Fallback | Via GDCM |
| `(0020,0013)` | InstanceNumber | Fallback | Fallback | Via GDCM |

## Key limitations

### All models

- **Research use only** — not validated for clinical use.
- **Breast MRI only** — the classification models assume breast MRI input; other anatomies will produce unreliable results.
- **No demographic bias testing** — performance across patient demographics has not been systematically evaluated.
- **No performance metrics published** — evaluation results are not tracked or displayed within the viewer.

### MST Classification

- Downloads model weights from Hugging Face on first run (requires network access; `HF_TOKEN` is **optional** — [ODELIA-AI/MST](https://huggingface.co/ODELIA-AI/MST) is a public repo and downloads unauthenticated).
- Classifies a contrast-subtraction volume; the subtraction is either supplied directly or computed by the service depending on the selected input configuration (pre+post, subtraction, or multi-phase) — see the [card](mst-classification.md#input-configurations).
- With both `SliceLocation` and `ImagePositionPatient` missing, `InstanceNumber` is the only remaining ordering key; if that is also unreliable the volume can be **silently mis-ordered**.

### MedGemma

- The bundled stack sets `NUM_SLICES=100` ([`docker-compose.yml`](../../docker-compose.yml)); the model samples the central 60% of the volume, but falls back to the whole volume when that window is narrower than `NUM_SLICES` (volumes under ~167 slices). The code default if unset is 5. See the [card](medgemma-mri.md#slice-extraction) for details.
- Confidence scores are **self-reported by the language model**, not calibrated probability distributions.
- Output depends on JSON parsing of free-text generation — malformed model responses cause errors.
- Requires a Hugging Face token and license acceptance for the gated model.

### Chat Middleware

- Free-form text responses — no structured classification guarantee.
- No explicit 4D temporal series handling; multi-phase series may be merged incorrectly.
- Depends on an external Ollama (or llama.cpp) backend being available with the model loaded.
- Any confidence or certainty expressed in chat responses is not statistically calibrated.
