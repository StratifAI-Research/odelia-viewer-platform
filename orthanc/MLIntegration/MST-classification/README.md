# MST Classification Service

Flask microservice for breast MRI malignancy classification using the ODELIA MST model (DINOv2-based Vision Transformer).

## Architecture

- `app_refactored.py` - Flask entry point: a thin HTTP layer exposing `/health` and `/analyze/mri`, delegating to the model service
- `model_service.py` - Orchestrates retrieval, preprocessing, inference, and response building
- `model_loader.py` - HuggingFace model download and loading logic
- `config.py` - Environment-driven configuration
- `retrieval_strategy.py` / `wado_helper.py` - WADO-RS series retrieval
- `dicom_converter.py` / `dicom_utils.py` - DICOM to NIfTI conversion utilities
- `preprocessing.py` - Volume preprocessing for the model
- `response_builder.py` - Formats the bilateral classification response

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_FOLDER` | `./images` | Temporary storage for DICOM files |
| `MODEL_PATH` | `./mst_model` | Directory for model files |
| `HF_TOKEN` | None | HuggingFace API token (optional; not needed for the public `ODELIA-AI/MST` repo) |
| `HTTP_PROXY` | None | HTTP proxy server (e.g., `http://user:pass@proxy.example.com:8080`) |
| `HTTPS_PROXY` | None | HTTPS proxy server (falls back to HTTP_PROXY if not set) |

## Proxy Support

To use HTTP proxy for HuggingFace downloads:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
python app_refactored.py
```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### MRI Analysis

```bash
POST /analyze/mri
Content-Type: application/json
```

Request body. Each `wado_rs_retrieval` entry carries `retrieval_url`, `study_uid`, and `series_uid`. `input_configuration_id` and `input_mapping` are optional — supported modes are `pre_post`, `subtraction`, and `multiphase`; when omitted, the service falls back to flat single-series retrieval.

```json
{
  "wado_rs_retrieval": [
    {
      "retrieval_url": "http://orthanc-viewer:8042/dicom-web/studies/{study}/series/{series}",
      "study_uid": "1.2.3...",
      "series_uid": "1.2.3..."
    }
  ],
  "study_uid": "1.2.3...",
  "input_configuration_id": "pre_post",
  "input_mapping": {
    "pre":  {"series_uid": "...", "wado_rs_url": "..."},
    "post": {"series_uid": "...", "wado_rs_url": "..."}
  }
}
```

Response. A bilateral, three-class classification (`No lesion` / `Benign` / `Malignant`) with `confidence` as a percentage (0–100), plus model metadata and attention maps:

```json
{
  "left":  {"prediction": "Malignant", "confidence": 87.0},
  "right": {"prediction": "No lesion", "confidence": 98.2},
  "model_metadata": {
    "model_name": "ODELIA-AI",
    "architecture": "Vision Transformer",
    "version": "1.0"
  },
  "attention_maps": {"data": "...", "shape": [...], "dtype": "..."}
}
```

## Model Details

- **Model**: ODELIA MST (Multi-Scale Transformer)
- **Architecture**: Vision Transformer based on DINOv2
- **Task**: 3-class classification (No lesion / Benign / Malignant), reported per breast (bilateral)
- **Input**: 3D breast MRI (NIfTI format)
- **Repository**: https://huggingface.co/ODELIA-AI/MST

## Usage

1. Start the service:
```bash
python app_refactored.py
```

2. The service will automatically:
   - Download model files from HuggingFace (on first run)
   - Load the model into memory
   - Start Flask server on port 5556

3. Send MRI series for analysis via POST request

## Requirements

See `requirements.txt` for dependencies.

Key dependencies:
- PyTorch
- Flask
- HuggingFace Hub
- TorchIO
- PyDICOM
- NiBabel

## License

Research-only use. See model card on HuggingFace for full license terms.
