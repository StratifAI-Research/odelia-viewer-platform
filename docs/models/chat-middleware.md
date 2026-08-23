# Model Card: Chat Middleware (vision-language model via Ollama / llama.cpp)

## Model Details

| Field | Value |
|-------|-------|
| **Name** | Chat Middleware |
| **Architecture** | Any vision-language model served via Ollama or llama.cpp (default: MedGemma 1.5 4B) |
| **Default model** | `thiagomoraes/medgemma-1.5-4b-it:F16` (Ollama registry; the `OLLAMA_MODEL` value set in `docker-compose.yml`) — swap in any other VLM your backend can serve |
| **Task** | Free-form medical image chat / interactive analysis |
| **Interface** | WebSocket (streaming tokens) |
| **Output** | Streaming text responses |
| **Service port** | 5560 |

## Intended Use

Research-only interactive chat about medical images (primarily breast MRI). Users can ask free-form questions about loaded DICOM series. Not approved for clinical decision-making.

## Input Requirements

### Modality & Anatomy

- **Modality:** Any DICOM series (general-purpose chat)
- **Anatomy:** No restriction, but the underlying model is optimized for medical images

### Required DICOM Tags

| Tag | Keyword | Required | Purpose |
|-----|---------|----------|---------|
| `(0020,000D)` | StudyInstanceUID | Yes | Study-level retrieval |
| `(0020,000E)` | SeriesInstanceUID | Yes | Series-level retrieval |
| `(7FE0,0010)` | PixelData | Yes | Image data |

Spatial ordering relies on GDCM's built-in series reader, which uses standard DICOM tags internally (ImagePositionPatient, SliceLocation, InstanceNumber).

### Slice Extraction Strategies

Configurable via the debug API:

| Strategy | Description |
|----------|-------------|
| `CENTRAL` (default) | Extract from central N% of volume (default: central 60%) |
| `UNIFORM` | Evenly spaced across entire volume |
| `FIRST_N` | First N slices |
| `LAST_N` | Last N slices |

Default: 5 slices from central 60% of the volume.

### Image Preprocessing

1. DICOM series retrieved via WADO-RS
2. Read as 3D volume via SimpleITK GDCM series reader (no explicit 4D temporal handling)
3. Slices extracted per configured strategy
4. Percentile-based normalization (1st–99th percentile) to 0–255
5. Grayscale to RGB conversion
6. Encoded as base64 PNG for delivery to Ollama API

### Prompt Structure

Uses OpenAI-compatible `/v1/chat/completions` format with interleaved content:
```
[system prompt] [conversation history] [image1] "SLICE 1" [image2] "SLICE 2" ... [user message]
```

System prompt and generation options (temperature, max_tokens, top_p, stop sequences, seed) are adjustable at runtime via the debug API.

## Input Limitations & Failure Modes

| Condition | Behavior |
|-----------|----------|
| No `.dcm` files in folder | `ValueError` raised |
| GDCM cannot recognize series | `ValueError` raised |
| No explicit 4D temporal handling | Multi-phase series may be read as a single merged volume |
| Ollama server not reachable | Connection error |
| Model not loaded in Ollama | Generation error |
| Non-medical images | Model may still respond but quality is unverified |

### Confidence Caveat

This is a free-form chat model. Any confidence or certainty expressed in text responses is self-reported by the language model and is not statistically calibrated.

## Infrastructure Dependency

Requires a running LLM backend with the model loaded. The default is **Ollama**
(`http://host.docker.internal:11434`, model `thiagomoraes/medgemma-1.5-4b-it:F16`); the
optional **llama.cpp** backend (`BACKEND_TYPE=llamacpp`) serves the same model from a
GGUF file inside Docker. Override via the `OLLAMA_URL` / `OLLAMA_MODEL` / `BACKEND_TYPE`
environment variables — see [`setup_chat.md`](../setup/setup_chat.md).

### Data egress

Both backends above run **on your own hardware**: no image data leaves the deployment.

An optional **Ollama Cloud** backend (`ALLOW_CLOUD_BACKEND=1`) instead routes chat to Ollama's
hosted models, which means **the preprocessed DICOM slices are uploaded to a third party**. It is
disabled by default, the service always starts on the local backend, and the panel warns whenever
it is active. Note also that many cloud models are text-only and cannot accept the slices at all —
see [`setup_chat.md`](../setup/setup_chat.md#option-c--ollama-cloud-optional-sends-images-off-site).
