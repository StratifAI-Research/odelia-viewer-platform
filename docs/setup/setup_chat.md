# Setting up the Chat AI

The **Chat AI** panel runs a vision-language model (VLM) against the open study. Any VLM your
backend can serve works — the bundled default is **MedGemma** (`thiagomoraes/medgemma-1.5-4b-it:F16`),
used as the example throughout this guide. The model needs a local LLM backend — pick **one**:

- **Ollama** (default) — runs on the host. Simplest; works without a GPU.
- **llama.cpp** (optional) — runs in Docker on an NVIDIA GPU. Typically faster.

There is also an optional [**hosted cloud backend**](#option-c--hosted-cloud-model-optional-sends-images-off-site)
— Ollama Cloud or OpenRouter — disabled by default. It runs no model locally, but **sends the
study's images to a third party** — see the warning in that section before enabling it.

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

## Option C — Hosted cloud model (optional, sends images off-site)

> [!WARNING]
> **This sends patient imaging outside your network.** The chat uploads the preprocessed DICOM
> slices of the open study to a hosted service for analysis. Everything else in this stack
> runs locally by design. Do not enable this on a deployment holding patient data unless your
> institution explicitly permits it, and prefer it only for non-patient or already-public data.

The cloud backend runs no model on your hardware, so it needs no GPU and no multi-GB download. It
is **disabled by default** and an operator has to opt in.

Pick **one** hosted provider per deployment with `CLOUD_PROVIDER`:

| | `CLOUD_PROVIDER=ollama` (default) | `CLOUD_PROVIDER=openrouter` |
| --- | --- | --- |
| Service | [Ollama Cloud](https://ollama.com) | [OpenRouter](https://openrouter.ai) |
| Catalogue | A couple of dozen models, roughly half vision-capable | ~430 models from many vendors (Gemini, GPT, Claude, Qwen-VL, …), ~260 of them vision-capable |
| Who runs the model | Ollama | One of several third-party inference providers OpenRouter routes to |
| Billing | Ollama plan/subscription | Per-token, prepaid credits |

Both are OpenAI-compatible, so the chat itself is identical; they differ in the catalogue and in
where the slices ultimately land. The panel offers a single **Cloud** section either way — the
counterpart of **Local** — with the configured host shown under it.

### 1. Create an API key

- **Ollama Cloud** — <https://ollama.com/settings/keys>
- **OpenRouter** — <https://openrouter.ai/settings/keys>

The key is held only by the `chat-middleware` service. It is never sent to the browser, never
returned by any endpoint, and never written to the logs — so chat users select a model but never
see or enter the key.

### 2. Enable it in `.env`

Ollama Cloud:

```bash
ALLOW_CLOUD_BACKEND=1
CLOUD_PROVIDER=ollama
OLLAMA_API_KEY=<your key>
# Optional: preselect a model. Otherwise users pick one in the chat panel.
OLLAMA_CLOUD_MODEL=qwen3.5
```

OpenRouter:

```bash
ALLOW_CLOUD_BACKEND=1
CLOUD_PROVIDER=openrouter
OPENROUTER_API_KEY=<your key>
# Optional: preselect a model. Otherwise users pick one in the chat panel.
OPENROUTER_MODEL=google/gemini-2.5-pro
# Strongly recommended with OpenRouter: its catalogue is several hundred models.
CLOUD_MODEL_FILTER=gemini,qwen,medgemma
```

Only the selected provider's variables are read, so both sets can sit in `.env` side by side.

```bash
docker compose up -d chat-middleware
```

### 3. Select it in the viewer

Open the **Chat AI** panel and pick a model from the **Cloud** section of the model menu in the
header. The list is fetched live from the provider; settings (gear) → **Models available in chat**
prunes which of them the menu offers.

> [!IMPORTANT]
> **Pick a model marked “vision”.** The chat sends slices as images, and a large part of either
> catalogue is text-only — roughly half of Ollama Cloud's, and about four in ten of OpenRouter's. A
> text-only model cannot see the study at all. The panel marks vision-capable models and warns if
> you select one that is not.

Where the vision flag comes from differs by provider. Ollama's is read from `/api/show`, not the
`capabilities` array in `/api/tags`; the two disagree, and `/api/tags` under-reports vision.
OpenRouter reports `architecture.input_modalities` for every model in one catalogue request, so its
flags are always known.

| Variable | Default | Description |
| --- | --- | --- |
| `ALLOW_CLOUD_BACKEND` | `0` | Operator gate. While `0`, the UI hides the option and the middleware refuses cloud requests. |
| `CLOUD_PROVIDER` | `ollama` | Which hosted service the cloud slot points at: `ollama` or `openrouter`. An unrecognized value falls back to `ollama`. |
| `CLOUD_MODEL_FILTER` | *(empty)* | Comma-separated substrings; only models whose id contains one of them are offered. Empty offers the whole catalogue. |
| `OLLAMA_API_KEY` | *(empty)* | Ollama Cloud API key. Stays server-side. |
| `OLLAMA_CLOUD_URL` | `https://ollama.com` | Cloud host. Ollama Cloud behaves as a remote Ollama host. |
| `OLLAMA_CLOUD_MODEL` | *(empty)* | Optional preselected cloud model. |
| `OPENROUTER_API_KEY` | *(empty)* | OpenRouter API key. Stays server-side. |
| `OPENROUTER_URL` | `https://openrouter.ai/api` | OpenRouter host. |
| `OPENROUTER_MODEL` | *(empty)* | Optional preselected cloud model, e.g. `google/gemini-2.5-pro`. |
| `OPENROUTER_ALLOW_DATA_COLLECTION` | `0` | While `0`, every request carries `provider.data_collection=deny`, restricting routing to providers that do not retain or train on prompts. Setting it to `1` lifts that and widens model availability. |

> [!NOTE]
> **OpenRouter is a broker, not the model host.** It forwards each request to one of several
> third-party inference providers, so "off-site" is a wider set of companies than with a
> single-hop service. The deny-by-default above is sent with every request rather than relying on
> the account dashboard, but it is a contractual guarantee from those providers, not a technical
> one. Review OpenRouter's privacy settings before enabling this on anything sensitive.
>
> A model with no compliant provider answers `No endpoints found matching your data policy`, which
> the chat panel shows verbatim. Pick another model, or set
> `OPENROUTER_ALLOW_DATA_COLLECTION=1` if the deployment's data permits it.

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
- Switching `CLOUD_PROVIDER` invalidates any preselected cloud model: the tag formats differ
  (`qwen3.5` vs `google/gemini-2.5-pro`), so pick a model again after the switch.

---

## Verify

Open a study in the viewer, open the **Chat AI** panel, and ask a question. See
[the usage guide](../usage/README.md#3-chat-about-a-study) for the workflow and
[`models/chat-middleware.md`](../models/chat-middleware.md) for what the model does and its limits.
