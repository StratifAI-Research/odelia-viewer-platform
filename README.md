# Odelia Viewer

## Overview 🎯
The Odelia Viewer is a cutting-edge medical imaging platform developed under the **EU ODELIA Project**. This platform facilitates the integration and analysis of medical images using advanced AI models, specifically designed for Odelia AI components.

> 🔗 For more information about the project's vision and goals, visit the official **[ODELIA Project Website](https://odelia.ai/)**.

---

## Installation and Setup 🚀
This section guides you through the process of getting the Odelia Viewer up and running on your local machine using Docker.

### Prerequisites
1.  **Git:** Installed to clone the repository.
2.  **Docker:** Installed on your computer (includes Docker Compose).

### Quick Start (Local Deployment)
Follow these steps for a complete local deployment:

<details>
<summary>View Quick Start Steps</summary>

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/StratifAI-Research/odelia-deployment
    cd odelia-deployment
    ```

2.  **Run the Deployment:**
    Open a terminal in the project directory and start all services in detached mode:
    ```bash
    docker compose up -d
    ```
    > **Note:** Initial run warnings about not being able to pull images for `odelia-orthanc-router`, `odelia-orthanc-viewer`, and the AI models (`odelia-breast-cancer-classification`, `odelia-mst-classifier`) are **normal**. These images will be built automatically from source.

3.  **Access the Viewer:**
    Open your web browser and navigate to (use viewer/viewer as default credentials):
    ```
    http://localhost:8081
    ```
    > **Configuration Note:** All settings are pre-set for `http://localhost:8081` for local deployment. No changes are required.
</details>

### Directory Structure
<details>
<summary>View Deployment Directory Structure</summary>

```
├── docker-compose.yml
├── config/                     # Deployment configuration files
│   ├── app-config.js           # OHIF viewer settings (OIDC, DICOMWeb)
│   ├── nginx.conf              # Nginx reverse proxy settings
│   ├── ohif-keycloak-realm.json# Keycloak realm import (ohif)
│   └── orthanc-router.json     # Orthanc router configuration
├── grafana/                    # Grafana provisioning for monitoring
│   └── provisioning/...
└── volumes/                    # Persistent data volumes
    ├── orthanc-viewer-db/      # Local Orthanc DICOM data
    └── feedback-db/            # AI feedback data
```


</details>

---


## Getting Started 💡
Once the viewer is running, here are the main use cases for interacting with the platform.

### Uploading Images to Orthanc
<details>
<summary>View Image Upload Methods</summary>

You can upload DICOM images (`.dcm` files) using three methods:

* **Method A: OHIF Viewer (Recommended)**
    * Navigate to the viewer at `http://localhost:8081`.
    * Use the **upload button** to drag and drop your DICOM files.
* **Method B: Orthanc Web Interface**
    * Access the Orthanc dashboard at `http://localhost:8000`.
    * Click **"Upload"** and select your DICOM files.
* **Method C: DICOM Send (C-STORE)**
    * Configure your external DICOM source/PACS with:
        * **AE Title:** `ORTHANC-VIEWER`
        * **Host:** `localhost`
        * **Port:** `2000`
</details>

### AI Routing and Analysis
<details>
<summary>View AI Analysis Workflow</summary>

Send studies to the integrated AI models for analysis:

1.  **Open a Study** in the Odelia Viewer. The **AI Analysis** panel (right sidebar) will automatically detect the active study.
2.  **Select Series** - Choose the specific series you want to analyze and click **"Next"**.
3.  **Select AI Model** and click **"Send to AI"**:
    * Classification model (breast cancer)
    * MST AI model (requires `HF_TOKEN` configuration - see Configuration section).
4.  **View Results** - The AI-processed studies, including annotations and results, will appear in your study list.
</details>

### User Management
<details>
<summary>View User Management Details</summary>

Please see the documentation: [Adding New Users](docs/adding_new_users.md)
</details>

---

## Updating 🔄

This section covers how to update an existing Odelia Viewer deployment to the latest version.

<details>
<summary>View Full Update Procedure</summary>

#### 1. Back Up Persistent Data

Before updating, back up your persistent data. The `volumes/` directory contains DICOM images and the feedback database. The `postgres_data` named Docker volume holds the Keycloak database.

```bash
# Back up the volumes directory
cp -r volumes/ volumes-backup-$(date +%F)/

# Back up the Keycloak Postgres database
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

#### 2. Stop Running Containers

```bash
docker compose down
```

> **Note:** `docker compose down` does **not** delete volumes. Your data remains intact.

#### 3. Pull Latest Deployment Repository

```bash
git pull origin main
```

Replace `main` with the branch you are tracking if different.

#### 4. Review Configuration Changes

If you have customized any configuration files, check for upstream changes before restarting:

```bash
git diff HEAD@{1} -- docker-compose.yml config/
```

Files to watch for changes:
- `docker-compose.yml` — new environment variables, ports, or services
- `config/nginx.conf` — reverse proxy routes
- `config/app-config.js` — OHIF viewer settings
- `config/orthanc-router.json` — Orthanc routing configuration

> **Important:** Re-apply any custom settings (e.g., `HF_TOKEN`, production domain URLs) after pulling, as they may be overwritten by the update.

#### 5. Rebuild Docker Images

Rebuild all images that are built from source:

```bash
docker compose build --no-cache
```

Or rebuild only specific services:

```bash
docker compose build <service-name>
```

**Services built from source** (require rebuild): `orthanc-viewer`, `orthanc-router`, `orthanc-router-mst`, `orthanc-router-medgemma`, `breast-cancer-classification`, `mst-classifier`, `medgemma-mri`, `chat-middleware`

**Pre-built images** (update via pull): `viewer`, `grafana`, `keycloak`, `postgres`

```bash
docker compose pull viewer grafana keycloak postgres
```

#### 6. Update Ollama Model (if applicable)

If you use the Chat AI feature, update the Ollama model on the host machine:

```bash
ollama pull thiagomoraes/medgemma-1.5-4b-it:F16
```

#### 7. Start Services

```bash
docker compose up -d
```

#### 8. Verify Deployment

```bash
# Check all containers are running
docker compose ps

# Check logs for errors
docker compose logs --tail=50
```

Then open `http://localhost:8081` in your browser to confirm the viewer loads correctly.

</details>

<details>
<summary>Quick-Reference Commands</summary>

For experienced users, here is the condensed update sequence:

```bash
docker compose down
git pull origin main
docker compose build --no-cache
docker compose up -d
docker compose ps
```

</details>

---

## Architecture Components 🏗️
The Odelia Viewer deployment consists of several interconnected components, each serving a specific purpose:

| Component | Purpose |
| :--- | :--- |
| **Odelia Viewer** | The frontend application (OHIF-based) for viewing medical images, customized with features for AI routing and feedback. |
| **Keycloak** | Handles **User Authentication** and **Authorization** (OIDC). Admin Console is available at `http://localhost:8081/keycloak`. |
| **Local Orthanc Instance** | The core **DICOM Server**. It acts as a local PACS, receiving, storing, and managing all DICOM medical images. |
| **Orthanc Router** | The **Traffic Controller** for the AI pipeline. It receives studies, routes them to the appropriate AI model, wraps the inference results in DICOM, and sends everything back to the viewer. |
| **AI Models** | Services that receive studies from the Router and return inference results (e.g., `odelia-breast-cancer-classification`, `odelia-mst-classifier`). |
| **Nginx** | Acts as a **Reverse Proxy**, routing traffic from port `8081` to the appropriate backend services (Viewer, Keycloak, etc.). |
| **Grafana** | An optional component for **Monitoring** and visualization of system metrics. |

---


## Configuration Details ⚙️
The viewer is pre-configured, but you may need to adjust settings for production or advanced use.

### Main Viewer and Server Configuration
<details>
<summary>View Configuration Files and Settings</summary>

| File | Purpose | Keycloak/OIDC Settings | Other Settings |
| :--- | :--- | :--- | :--- |
| `config/app-config.js` | **OHIF Viewer Settings** | `oidc[0].authority`, `oidc[0].redirect_uri`, `oidc[0].post_logout_redirect_uri` | DICOMWeb endpoints, OHIF features. |
| `config/nginx.conf` | **Web Server Reverse Proxy** | Routes `/keycloak/` to the Keycloak service. | Maps `/` to the viewer and static content. Set `server_name` for production. |
| `docker-compose.yml` | **Service Environment Variables** | `KC_HOSTNAME_URL`, `KC_HOSTNAME_ADMIN_URL` (for production domain changes). | `HF_TOKEN` (for MST classifier), volumes, ports, etc. |
</details>

### Keycloak Configuration 🔐
<details>
<summary>View Keycloak Setup Details</summary>

The Odelia Viewer uses OHIF's internal OIDC module for authentication.

* **Admin Console:** Access at `http://localhost:8081/keycloak`
* **Default Credentials:** `admin` / `admin`
* **Realm:** The `ohif` realm is auto-imported from `config/ohif-keycloak-realm.json`.
* **Production Deployment:** When deploying to a server or changing the domain, you **must** update the `ohif_viewer` client settings in the Keycloak UI (or `ohif-keycloak-realm.json`):
    * `Root URL`, `Home URL`
    * `Valid Redirect URIs` (e.g., `https://YOUR_DOMAIN/viewer/callback`)
    * `Web Origins`
    * Also update `KC_HOSTNAME_URL` and `KC_HOSTNAME_ADMIN_URL` in `docker-compose.yml`.
</details>

### AI Model Configuration (Hugging Face) 🤖
<details>
<summary>View MST Classifier Setup (HF_TOKEN)</summary>

The **MST Classification model** requires a Hugging Face token for access. The default **breast-cancer-classification** model does not.

1.  **Obtain Token:** Get a "Read" access token from your Hugging Face account settings.
2.  **Accept Licenses:** **Crucially**, you must log in to Hugging Face and accept the licenses for:
    * [ODELIA-AI/MST](https://huggingface.co/ODELIA-AI/MST) (Model Usage Agreement)
    * [DINOv3](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) (DINOv3 License terms)
3.  **Configure Token:** Set the `HF_TOKEN` environment variable for the `mst-classifier` service:
    * **Option 1 (Recommended):** Edit `docker-compose.yml` and replace the placeholder:
        ```yaml
        environment:
          HF_TOKEN: "your_actual_token_here" # REPLACE with your token
        ```
    * **Option 2 (Environment Variable):**
        ```bash
        export HF_TOKEN=your_actual_token_here
        docker compose up -d
        ```
    > ⚠️ **Security Warning:** Never commit your actual token to a version control system like Git. Use environment variables or a `.env` file that is in your `.gitignore`.
</details>

### Ollama + MedGemma Setup (Chat AI) 💬
<details>
<summary>View Ollama Installation and MedGemma Configuration</summary>

The **Chat AI** panel in the viewer uses a local [Ollama](https://ollama.com/) instance running the **MedGemma** vision-language model. Ollama runs on the **host machine** (not inside Docker) and is accessed by the `chat-middleware` container via `host.docker.internal`.

#### 1. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS / Windows:**
Download the installer from [https://ollama.com/download](https://ollama.com/download).

Verify the installation:
```bash
ollama --version
```

#### 2. Pull the MedGemma Model

Pull the MedGemma model that the chat middleware expects by default:
```bash
ollama pull thiagomoraes/medgemma-1.5-4b-it:F16
```

> **Note:** This model is ~8 GB. Ensure you have sufficient disk space and a GPU with enough VRAM for acceptable performance. Smaller quantizations (e.g., `Q8_0`, `Q4_K_M`) are available if resources are limited — adjust the `OLLAMA_MODEL` variable accordingly.

Verify the model is available:
```bash
ollama list
```

#### 3. Start Ollama

Make sure the Ollama server is running before starting the Docker stack:
```bash
ollama serve
```

By default, Ollama listens on `http://localhost:11434`. You can confirm it is running:
```bash
curl http://localhost:11434/api/tags
```

#### 4. Docker Compose Configuration

The `chat-middleware` service in `docker-compose.yml` is pre-configured to connect to Ollama on the host:

```yaml
chat-middleware:
  environment:
    OLLAMA_URL: "http://host.docker.internal:11434"
    OLLAMA_MODEL: "thiagomoraes/medgemma-1.5-4b-it:F16"
    OLLAMA_NUM_CTX: "128000"  # 128k context window
    NUM_SLICES: "5"
  extra_hosts:
    - "host.docker.internal:host-gateway"
```

**Key variables you may want to adjust:**

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | URL of the Ollama API. Change if Ollama runs on a remote machine. |
| `OLLAMA_MODEL` | `thiagomoraes/medgemma-1.5-4b-it:F16` | Model tag in Ollama. Change if you pulled a different quantization or model. |
| `OLLAMA_NUM_CTX` | `128000` | Context window size in tokens. Reduce if you run into memory issues. |
| `NUM_SLICES` | `5` | Number of DICOM slices sent to the model per study for analysis. |

> **Linux note:** The `extra_hosts` mapping (`host.docker.internal:host-gateway`) is required on Linux to allow containers to reach the host network. This is included in the default `docker-compose.yml` and should not be removed.

#### 5. Using a Different Model

To use a different Ollama-compatible model:

1. Pull the desired model:
    ```bash
    ollama pull <model_name>:<tag>
    ```
2. Update the `OLLAMA_MODEL` variable in `docker-compose.yml`:
    ```yaml
    OLLAMA_MODEL: "<model_name>:<tag>"
    ```
3. Restart the chat middleware:
    ```bash
    docker compose up -d chat-middleware
    ```

</details>

### llama.cpp Alternative Backend (Chat AI) 🦙
<details>
<summary>View llama.cpp Setup and Configuration</summary>

[llama.cpp](https://github.com/ggerganov/llama.cpp) is an alternative backend for the Chat AI feature. It runs the same MedGemma model but uses the GGUF format and runs entirely inside Docker — no host-level installation required.

**Why choose llama.cpp over Ollama?**
* Typically **faster inference** (lower time-to-first-token and higher tokens/sec) on the same hardware.
* Runs in a Docker container — nothing to install on the host besides the NVIDIA driver.
* Supports quantized models (Q4, Q8, etc.) for reduced VRAM usage.

**Trade-offs:**
* You need to download the GGUF model file manually (Ollama handles downloads automatically).
* Configuration is done via environment variables rather than a CLI tool.

#### 1. Prerequisites

* **NVIDIA GPU** with compatible drivers installed on the host.
* **NVIDIA Container Toolkit** — required so Docker containers can access the GPU. Follow the official installation guide: [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

  Verify it works:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```

#### 2. Download the Model Files

The Chat AI uses MedGemma 1.5 4B. You need two files — the main model and the vision (multimodal) projector:

```bash
# Create the models directory
sudo mkdir -p custom/deploy/volumes/models

# Download the main model (~7.8 GB)
sudo wget -P custom/deploy/volumes/models/ \
  https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-BF16.gguf

# Download the multimodal projector (~812 MB)
sudo wget -P custom/deploy/volumes/models/ \
  https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/mmproj-BF16.gguf
```

> **Disk space:** You need approximately **9 GB** of free space for both files. Smaller quantized models (e.g., `Q8_0` at 4.1 GB, `Q4_K_M` at 2.5 GB) are available from the same repository if space or VRAM is limited.

Verify the files are in place:
```bash
ls -lh custom/deploy/volumes/models/
# Expected output:
#   medgemma-1.5-4b-it-BF16.gguf   (~7.8G)
#   mmproj-BF16.gguf                (~812M)
```

#### 3. Switch from Ollama to llama.cpp

Edit the `chat-middleware` section in `docker-compose.yml`. Change the three highlighted variables:

```yaml
chat-middleware:
    environment:
      OLLAMA_URL: "http://llamacpp-server:8090"          # was: http://host.docker.internal:11434
      OLLAMA_MODEL: "medgemma-1.5-4b-it-BF16"            # was: thiagomoraes/medgemma-1.5-4b-it:F16
      BACKEND_TYPE: "llamacpp"                            # was: ollama
      # ... leave the rest unchanged
```

Then start both the chat middleware and the llama.cpp server:
```bash
docker compose --profile llamacpp up -d chat-middleware llamacpp-server
```

> **Important:** The `--profile llamacpp` flag is required to start the `llamacpp-server` container. Without it, only the `chat-middleware` will start and it won't be able to reach the llama.cpp server.

Verify both containers are running:
```bash
docker compose --profile llamacpp ps
```

You should see both `odelia-chat-middleware` and `odelia-llamacpp-server` with status `Up`.

#### 4. Switch Back to Ollama

Revert the three variables in `docker-compose.yml` to their original values:

```yaml
chat-middleware:
    environment:
      OLLAMA_URL: "http://host.docker.internal:11434"
      OLLAMA_MODEL: "thiagomoraes/medgemma-1.5-4b-it:F16"
      BACKEND_TYPE: "ollama"
      # ... leave the rest unchanged
```

Then restart without the llama.cpp profile:
```bash
docker compose down chat-middleware llamacpp-server
docker compose up -d chat-middleware
```

#### 5. Configuration Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BACKEND_TYPE` | `ollama` | Backend to use: `ollama` or `llamacpp`. |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | URL of the LLM server. For llama.cpp: `http://llamacpp-server:8090`. |
| `OLLAMA_MODEL` | `thiagomoraes/medgemma-1.5-4b-it:F16` | Model identifier. For llama.cpp: `medgemma-1.5-4b-it-BF16`. |
| `GGUF_MODEL_FILE` | `medgemma-1.5-4b-it-BF16.gguf` | GGUF model filename inside `volumes/models/`. Only used by `llamacpp-server`. |
| `MMPROJ_FILE` | `mmproj-BF16.gguf` | Vision projector filename. Only used by `llamacpp-server`. |
| `LLAMA_CTX_SIZE` | `131072` | Context window size in tokens. Reduce if you run into VRAM issues. |
| `LLAMA_N_GPU_LAYERS` | `99` | Number of model layers to offload to GPU. `99` means all layers. |

#### 6. Using a Remote llama.cpp Server

The llama.cpp server does not have to run on the same machine. If you have a remote GPU server running `llama-server`, simply point `OLLAMA_URL` to it:

```bash
# In .env:
BACKEND_TYPE=llamacpp
OLLAMA_URL=http://192.168.1.50:8090
OLLAMA_MODEL=medgemma-1.5-4b-it-BF16
```

Then start the chat middleware **without** the `llamacpp` profile (since the server is remote):
```bash
docker compose up -d chat-middleware
```

#### 7. Using a Different Quantization

To use a smaller (faster, less VRAM) or larger model variant:

1. Download the desired GGUF from [unsloth/medgemma-1.5-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF). For example, `Q8_0` (~4.1 GB):
    ```bash
    sudo wget -P custom/deploy/volumes/models/ \
      https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/medgemma-1.5-4b-it-Q8_0.gguf
    ```

2. Set the filename via environment variable:
    ```bash
    # In .env:
    GGUF_MODEL_FILE=medgemma-1.5-4b-it-Q8_0.gguf
    OLLAMA_MODEL=medgemma-1.5-4b-it-Q8_0
    ```

3. Restart the llama.cpp server:
    ```bash
    docker compose --profile llamacpp up -d llamacpp-server
    ```

> **Note:** The `mmproj-BF16.gguf` vision projector is the same regardless of which quantization you choose for the main model.

#### 8. Benchmarking Ollama vs llama.cpp

A benchmark script is included to compare performance between backends:

```bash
# Install the only dependency
pip install requests

# Run against both backends (requires both to be running)
python orthanc/MLIntegration/chat-middleware/benchmark.py \
  --ollama http://localhost:11434 \
  --llamacpp http://localhost:8090 \
  --show-responses

# With real DICOM images from Orthanc
python orthanc/MLIntegration/chat-middleware/benchmark.py \
  --ollama http://localhost:11434 \
  --llamacpp http://localhost:8090 \
  --series-uid <SeriesInstanceUID> \
  --show-responses
```

The script measures time-to-first-token (TTFT), tokens per second, and total generation time across text and multimodal prompts.

</details>

---

## Support 🤝
If you encounter any issues, please follow the steps below to gather diagnostic information before reaching out.

### Collecting Logs for Support
<details>
<summary>View Log Collection Commands</summary>

Create a `logs` directory and run the following commands to collect logs from all components:

1.  **Viewer Logs**
    ```bash
    docker logs odelia-viewer > logs/viewer.log
    ```
2.  **Orthanc Logs**
    ```bash
    docker logs odelia-orthanc-viewer > logs/orthanc.log
    ```
3.  **Router Logs**
    ```bash
    docker logs odelia-orthanc-router > logs/router.log
    ```
4.  **System & Container Info**
    ```bash
    docker system info > logs/docker-info.log
    docker ps -a > logs/containers.log
    ```
5.  **Configuration Files**
    ```bash
    cp config/* logs/
    ```
</details>

### Contact
<details>
<summary>View Contact Information</summary>

Once all files are collected:
1.  Compress the `logs` directory.
2.  Provide a clear description of the issue.
3.  Attach the compressed logs and send them to: **b.maksudov@lua.tatar**
</details>


## Acknowledgement 🎗️
* This platform, the Odelia Viewer, is built upon the foundational work of the [Open Health Imaging Foundation (OHIF) Viewer](https://ohif.org/). We gratefully acknowledge the OHIF community for developing this platform, which serves as the base for our specialized AI integration features.

* This work was funded by the European Union's Horizon Europe research and innovation programme under Grant Agreement No. [101057091](https://doi.org/10.3030/101057091).
