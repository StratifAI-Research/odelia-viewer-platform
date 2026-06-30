# Model Card: MedGemma (Vision-Language Model)

> Deployed as the `medgemma-mri` service, which applies the general-purpose MedGemma
> vision-language model to breast-MRI classification.

## Model Details

| Field | Value |
|-------|-------|
| **Name** | MedGemma |
| **Architecture** | Vision-Language Model (Google MedGemma 1.5-4B-IT) |
| **Model ID** | `google/medgemma-1.5-4b-it` |
| **Task** | 3-class classification per breast: No lesion / Benign / Malignant |
| **Output** | Bilateral classification with confidence scores (no attention maps) |
| **Inference mode** | Deterministic (`do_sample=False`), max 500 new tokens |
| **Source** | [google/medgemma-1.5-4b-it on HuggingFace](https://huggingface.co/google/medgemma-1.5-4b-it) (gated model) |
| **Service port** | 5557 |

## Intended Use

Research-only analysis of breast MRI using a general-purpose medical vision-language model. The model is prompted to behave as a radiologist. Not approved for clinical decision-making.

## Input Requirements

### Modality & Anatomy

- **Modality:** MRI (breast MRI, single 3D series)
- **Anatomy:** Breast (the prompt assumes breast MRI — other anatomies will be misclassified)

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

### Temporal Handling (4D Series)

- If multiple temporal phases are detected, **only the first temporal position** is used
- Single-phase series are used as-is

### Slice Extraction

| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of slices | `100` (bundled default) | Set via `NUM_SLICES` in [`docker-compose.yml`](../../docker-compose.yml); the code default if unset is `5` |
| Region | Central 60%, with whole-volume fallback | Indices 20%–80% of the depth; if `NUM_SLICES` does not fit inside that window, sampling expands to the whole volume |
| Spacing | Even | Evenly spaced within the selected region |

The whole-volume fallback triggers when the central-60% window is narrower than
`NUM_SLICES` — i.e. when the volume has fewer than ~`NUM_SLICES / 0.6` slices. With the
bundled `NUM_SLICES=100`, volumes under ~167 slices are sampled across their full depth,
while larger volumes still sample 100 evenly-spaced slices from the central 60%. If the
volume has fewer than `NUM_SLICES` slices in total, every slice is used.

### Image Preprocessing

1. DICOM volume is read via SimpleITK
2. `NUM_SLICES` evenly-spaced axial slices are extracted (central 60% if the count fits, otherwise the whole volume)
3. Each slice is normalized using percentile-based windowing (1st–99th percentile) to 0–255
4. Grayscale is converted to RGB by tripling the channel
5. Delivered as PIL `Image` objects to the HuggingFace processor

### Prompt Structure

The model receives an interleaved sequence of every extracted slice:
```
[instruction text] [image1] "SLICE 1" [image2] "SLICE 2" ... [imageN] "SLICE N" [query text]
```

The instruction prompt sets the role as a radiologist analyzing breast MRI. The query requests a JSON response with bilateral classification (left/right), each with classification, confidence, and reasoning.

## Input Limitations & Failure Modes

| Condition | Behavior |
|-----------|----------|
| No `.dcm` files in folder | `ValueError` raised |
| Volume with fewer slices than `num_slices` | `num_slices` is silently reduced to available count |
| Missing `SliceLocation` and `ImagePositionPatient` | Slice position defaults to 0.0, leaving `InstanceNumber` as the only ordering key; if that is also unreliable, the volume can be **silently mis-ordered** |
| Missing temporal tags on multi-phase data | All phases merged into one volume — may produce incorrect slice ordering |
| Non-breast anatomy | Model is prompted for breast MRI — will attempt breast classification on any anatomy, producing **unreliable results** |
| Model returns malformed JSON | `ResponseParsingError` raised |
| Model returns invalid classification label | `ResponseParsingError` raised (only "No lesion", "Benign", "Malignant" accepted) |

### Confidence Score Caveat

Confidence scores are **self-reported by the language model**, not derived from calibrated probability distributions. They should not be interpreted as true statistical confidence.

## Authentication

Requires a valid HuggingFace token (`HF_TOKEN` environment variable). The MedGemma license must be accepted at https://huggingface.co/google/medgemma-1.5-4b-it before first use.
