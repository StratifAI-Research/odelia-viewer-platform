# Setting up the Chat AI

The **Chat AI** panel runs a vision-language model (VLM) against the open study. Any VLM your
backend can serve works — the bundled default is **MedGemma** (`thiagomoraes/medgemma-1.5-4b-it:F16`),
used as the example throughout this guide. The model needs a local LLM backend — pick **one**:

- **Ollama** (default) — runs on the host. Simplest; works without a GPU.
- **llama.cpp** (optional) — runs in Docker on an NVIDIA GPU. Typically faster.

There is also an optional [**Ollama Cloud** backend](#option-c--ollama-cloud-optional-sends-images-off-site),
disabled by default. It runs no model locally, but **sends the study's images to a third party** —
see the warning in that section before enabling it.

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

## Option C — Ollama Cloud (optional, sends images off-site)

> [!WARNING]
> **This sends patient imaging outside your network.** The chat uploads the preprocessed DICOM
> slices of the open study to Ollama's hosted service for analysis. Everything else in this stack
> runs locally by design. Do not enable this on a deployment holding patient data unless your
> institution explicitly permits it, and prefer it only for non-patient or already-public data.

The cloud backend runs no model on your hardware, so it needs no GPU and no multi-GB download. It
is **disabled by default** and an operator has to opt in.

### 1. Create an API key

Create one at <https://ollama.com/settings/keys>.

The key is held only by the `chat-middleware` service. It is never sent to the browser, never
returned by any endpoint, and never written to the logs — so chat users select a model but never
see or enter the key.

### 2. Enable it in `.env`

```bash
ALLOW_CLOUD_BACKEND=1
OLLAMA_API_KEY=<your key>
# Optional: preselect a model. Otherwise users pick one in the chat panel.
OLLAMA_CLOUD_MODEL=qwen3.5
```

```bash
docker compose up -d chat-middleware
```

### 3. Select it in the viewer

Open the **Chat AI** panel → settings (gear) → **Backend** → *Provider* → **Ollama Cloud**, then
pick a **Cloud Model**. The list is fetched live from your account.

> [!IMPORTANT]
> **Pick a model marked “vision”.** The chat sends slices as images, and many Ollama Cloud models
> are text-only — at the time of writing roughly half. A text-only model cannot see the study at
> all. The panel marks vision-capable models and warns if you select one that is not.

Capabilities are read from Ollama's `/api/show`, not the `capabilities` array in `/api/tags`; the
two disagree, and `/api/tags` under-reports vision.

| Variable | Default | Description |
| --- | --- | --- |
| `ALLOW_CLOUD_BACKEND` | `0` | Operator gate. While `0`, the UI hides the option and the middleware refuses cloud requests. |
| `OLLAMA_API_KEY` | *(empty)* | Ollama Cloud API key. Stays server-side. |
| `OLLAMA_CLOUD_URL` | `https://ollama.com` | Cloud host. Ollama Cloud behaves as a remote Ollama host. |
| `OLLAMA_CLOUD_MODEL` | *(empty)* | Optional preselected cloud model. |

### Scope and caveats

- **The provider is a deployment-wide setting**, like the existing model and system-prompt
  settings: switching to cloud affects every chat user of this deployment, not just you. The
  panel shows which backend is active.
- `/chat-api/` and `/ws/chat/` are proxied **without authentication** (see
  [production hardening](../security/production-hardening.md)). Anyone who can reach the viewer
  host can therefore flip the provider whenever the gate is on. Leaving `ALLOW_CLOUD_BACKEND=0`
  is what prevents that.
- The service always starts on the **local** backend, even with cloud enabled, so a restart never
  silently resumes sending data off-site.
- Billing and rate limits are attached to the single operator key.

---

## Verify

Open a study in the viewer, open the **Chat AI** panel, and ask a question. See
[the usage guide](../usage/README.md#3-chat-about-a-study) for the workflow and
[`models/chat-middleware.md`](../models/chat-middleware.md) for what the model does and its limits.
