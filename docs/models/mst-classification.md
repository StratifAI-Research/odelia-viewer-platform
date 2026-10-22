# Model Card: MST Swarm Classification (May 2026)

The default `mst-classifier` service runs the generalized
`odelia-classification` implementation with the May 2026 swarm weights.
It returns classification results and DICOM SRs **without attention maps or
heatmap overlays**. The endpoint address is unchanged, but new MST analyses no
longer create heatmap Secondary Captures. Previously stored results remain in
Orthanc.

The older `MST-classification/` implementation is retained in the source tree
for reference. It is no longer deployed by Compose or built and promoted by CI.
Heatmap support has not been ported to the generalized service.

## Model Details

| Field | Value |
|-------|-------|
| **Name** | ODELIA MST (Multi-Scale Transformer) |
| **Architecture** | DINOv2-based Vision Transformer |
| **Task** | 3-class classification per breast: No lesion / Benign / Malignant |
| **Output** | Per-breast class probabilities and classification SR; no heatmap overlays |
| **Weights** | May 2026 swarm release, baked into the selected `MST_IMAGE` |
| **Original model** | [ODELIA-AI/MST on Hugging Face](https://huggingface.co/ODELIA-AI/MST); the swarm checkpoint is a separate release |
| **Weight terms** | Consult the selected swarm package's model card and notices; do not infer its terms from the original challenge model |
| **Service port** | 5556 |

## Intended Use

Research-only analysis of breast MRI. Not approved for clinical decision-making.
Loading and pipeline checks establish technical compatibility; training
preprocessing and class ordering still require confirmation for this release
before interpreting predictions. This deployment provides no heatmap-based
interpretability output.

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
viewer (defined in the router's [`manifest.json`](../../orthanc/MLIntegration/odelia-classification/manifest.json)):

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
4. The current default MediSwarm preprocessing canonicalizes and resamples the volume, splits it into left/right views, crops/pads and normalizes each view to a `(1, 1, 32, 224, 224)` tensor
5. The baked MST model produces three logits per view; softmax probabilities and bilateral classifications are returned for the router to write a DICOM SR

The active preprocessing implementation is in
[`preprocessing/pipeline.py`](../../orthanc/MLIntegration/odelia-classification/preprocessing/pipeline.py).
The deployed service does not invoke the legacy `predict_attention.py` workflow.

## Input Limitations & Failure Modes

| Condition | Behavior |
|-----------|----------|
| No `.dcm` files in folder | `ValueError` raised |
| GDCM cannot recognize series | `ValueError` raised |
| Missing `SliceLocation` and `ImagePositionPatient` | Slice position defaults to 0.0, leaving `InstanceNumber` as the only ordering key; if that is also unreliable, the 3D volume can be **silently mis-ordered** |
| Missing `TemporalPositionIdentifier` and `TriggerTime` | All files grouped as single phase (temporal position 0) — safe for single-phase input, but multi-phase data will be merged incorrectly |
| Non-breast anatomy | Model will still run but predictions are meaningless |
| Very small volumes | Cropping/padding can produce model-shaped input, but does not establish that sufficient anatomy is present for meaningful predictions |

## Authentication

Runtime needs no HF token: weights and pinned DINOv2 source are baked into the
image, and startup verifies and loads the local artifact. Building from a private
HF package requires a token supplied through a BuildKit secret and an immutable
model revision. See [Loading trained model packages](../usage/loading_model_weights.md).
