# Model Card: MST Classification (DINOv2 Vision Transformer)

## Model Details

| Field | Value |
|-------|-------|
| **Name** | ODELIA MST (Multi-Scale Transformer) |
| **Architecture** | DINOv2-based Vision Transformer |
| **Task** | 3-class classification per breast: No lesion / Benign / Malignant |
| **Output** | Bilateral classification with confidence scores + attention map overlays |
| **Source** | [ODELIA-AI/MST on HuggingFace](https://huggingface.co/ODELIA-AI/MST) (public model) |
| **License** | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial, attribution required |
| **Service port** | 5556 |

## Intended Use

Research-only analysis of breast MRI with interpretability via attention maps. Not approved for clinical decision-making.

## Input Requirements

### Modality & Anatomy

- **Modality:** breast MRI (dynamic contrast-enhanced)
- **Anatomy:** Breast

The model classifies a **contrast-subtraction** volume. Depending on the input
configuration selected in the viewer, the subtraction is either supplied directly or
computed by the service (see *Input configurations* below).

### Required DICOM Tags

| Tag | Keyword | Required | Purpose |
|-----|---------|----------|---------|
| `(0020,000D)` | StudyInstanceUID | Yes | Study-level retrieval |
| `(0020,000E)` | SeriesInstanceUID | Yes | Series-level retrieval |
| `(0020,0100)` | TemporalPositionIdentifier | Recommended | Primary tag for temporal phase detection |
| `(0018,1060)` | TriggerTime | Fallback | Used if TemporalPositionIdentifier is absent |
| `(0020,1041)` | SliceLocation | Recommended | Primary tag for spatial slice ordering |
| `(0020,0032)` | ImagePositionPatient | Fallback | Z-coordinate used if SliceLocation is absent |
| `(0020,0013)` | InstanceNumber | Fallback | Used if both spatial tags are absent |
| `(7FE0,0010)` | PixelData | Yes | Image data |

### Input configurations

The service dispatches on the `input_configuration_id` / `input_mapping` sent by the
viewer (defined in the router's [`manifest.json`](../../orthanc/MLIntegration/MST-classification/manifest.json)):

| Configuration | Inputs | What the service does |
|---------------|--------|-----------------------|
| **Pre + Post Contrast** (`pre_post`) | Two series: pre-contrast and first post-contrast | Converts each to NIfTI and computes the subtraction (post − pre) |
| **Subtraction** (`subtraction`) | One pre-computed subtraction series | Converts it to NIfTI directly |
| **Multi-phase** (`multiphase`) | One multi-phase (dynamic) series | Extracts temporal groups and computes the subtraction from the 1st and 2nd phases |

If no configuration is supplied (legacy flat fallback), the first series in
`wado_rs_retrieval` is retrieved and converted to NIfTI as-is, with no subtraction.

### Conversion Pipeline

1. The series for the selected configuration are retrieved via WADO-RS
2. Each is converted to a NIfTI volume via SimpleITK; the subtraction is computed where the configuration requires it
3. The resulting NIfTI is loaded as a TorchIO `ScalarImage`
4. Further preprocessing and inference is handled by the HuggingFace model code (`predict_attention.py`)

## Input Limitations & Failure Modes

| Condition | Behavior |
|-----------|----------|
| No `.dcm` files in folder | `ValueError` raised |
| GDCM cannot recognize series | `ValueError` raised |
| Missing `SliceLocation` and `ImagePositionPatient` | Slice position defaults to 0.0, leaving `InstanceNumber` as the only ordering key; if that is also unreliable, the 3D volume can be **silently mis-ordered** |
| Missing `TemporalPositionIdentifier` and `TriggerTime` | All files grouped as single phase (temporal position 0) — safe for single-phase input, but multi-phase data will be merged incorrectly |
| Non-breast anatomy | Model will still run but predictions are meaningless |
| Very small volumes | Depends on model's internal handling (from HuggingFace repo) |

## Authentication

None required. [ODELIA-AI/MST](https://huggingface.co/ODELIA-AI/MST) is a public (non-gated) repository,
so the weights download automatically on first start without a Hugging Face token. The model is licensed
CC-BY-NC-4.0 — non-commercial use only, with attribution.
