# Setting up the Chat AI

The **Chat AI** panel runs a vision-language model (VLM) against the open study. Any VLM your
backend can serve works — the bundled default is **MedGemma** (`thiagomoraes/medgemma-1.5-4b-it:F16`),
used as the example throughout this guide. The model needs a local LLM backend — pick **one**:

- **Ollama** (default) — runs on the host. Simplest; works without a GPU.
- **llama.cpp** (optional) — runs in Docker on an NVIDIA GPU. Typically faster.

> [!NOTE]
> **No Hugging Face token is needed for the chat.** Ollama and llama.cpp load models from the
> Ollama registry or local GGUF files, not via `HF_TOKEN`. (A token is only needed for *gated*
> Hugging Face weights elsewhere in the stack — see
> [Hugging Face access token](configuration.md#hugging-face-access-token-hf_token).)

---

## Option A — Ollama (default)

Do these **in order**:

### 1. Install Ollama

```bash
# Linux (macOS / Windows: https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a vision-language model (default: MedGemma, ~8 GB)

```bash
ollama pull thiagomoraes/medgemma-1.5-4b-it:F16
```

To use a different VLM, pull its tag instead and set `OLLAMA_MODEL` to match (step 4).

### 3. Start the Ollama server

```bash
ollama serve            # serves http://localhost:11434
```

Keep it running before you bring the stack up.

### 4. (Re)start the stack

```bash
docker compose up -d
```

The `chat-middleware` service is preconfigured to reach Ollama on the host via
`host.docker.internal`. Relevant variables (override in `.env` or the shell):

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API URL (change if Ollama is remote) |
| `OLLAMA_MODEL` | `thiagomoraes/medgemma-1.5-4b-it:F16` | Model tag (use a smaller quantization to save VRAM) |
| `NUM_SLICES` | `5` | DICOM slices sent to the model per study |

> On Linux, the `extra_hosts: host.docker.internal:host-gateway` mapping (already in the compose
> file) is what lets the container reach the host — don't remove it.

---

## Option B — llama.cpp (optional, GPU)

llama.cpp runs the same model fully inside Docker (GGUF format), typically with faster inference.
It needs an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Do these **in order**:

### 1. Download the model + vision projector

Any GGUF vision-language model with its projector works; MedGemma is the example here. Place the
files in the path the compose file mounts (`./volumes/models`):

```bash
mkdir -p volumes/models
wget -P volumes/models/ \
  https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-BF16.gguf
wget -P volumes/models/ \
  https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/mmproj-BF16.gguf
```

### 2. Point the chat middleware at llama.cpp

In `.env` (next to `docker-compose.yml`):

```bash
BACKEND_TYPE=llamacpp
OLLAMA_URL=http://llamacpp-server:8090
OLLAMA_MODEL=medgemma-1.5-4b-it-BF16
```

### 3. Start both, enabling the `llamacpp` profile

```bash
docker compose --profile llamacpp up -d chat-middleware llamacpp-server
```

Configuration variables for the `llamacpp-server`:

| Variable | Default | Description |
| --- | --- | --- |
| `GGUF_MODEL_FILE` | `medgemma-1.5-4b-it-BF16.gguf` | Model filename in `volumes/models/` |
| `MMPROJ_FILE` | `mmproj-BF16.gguf` | Vision projector filename (same across quantizations) |
| `LLAMA_CTX_SIZE` | `131072` | Context window (tokens) |
| `LLAMA_N_GPU_LAYERS` | `99` | Layers offloaded to GPU (`99` = all) |

To use a smaller quantization (e.g. `Q8_0`, `Q4_K_M`) download it from
[unsloth/medgemma-1.5-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF) and set
`GGUF_MODEL_FILE` / `OLLAMA_MODEL` accordingly. A benchmark comparing the two backends lives at
[`orthanc/MLIntegration/chat-middleware/benchmark.py`](../../orthanc/MLIntegration/chat-middleware/benchmark.py).

---

## Verify

Open a study in the viewer, open the **Chat AI** panel, and ask a question. See
[the usage guide](../usage/README.md#3-chat-about-a-study) for the workflow and
[`models/chat-middleware.md`](../models/chat-middleware.md) for what the model does and its limits.
