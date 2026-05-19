# MST Classification Service

Flask microservice for breast MRI malignancy classification using the ODELIA MST model (DINOv2-based Vision Transformer).

## Architecture

- `app.py` - Main Flask application with REST API endpoints
- `model_loader.py` - HuggingFace model download and loading logic
- `dicom_utils.py` - DICOM to NIfTI conversion utilities

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORTHANC_URL` | `http://orthanc:8042` | Orthanc PACS server URL |
| `IMAGE_FOLDER` | `./images` | Temporary storage for DICOM files |
| `MODEL_PATH` | `./mst_model` | Directory for model files |
| `HF_TOKEN` | None | HuggingFace API token (required for gated models) |
| `HTTP_PROXY` | None | HTTP proxy server (e.g., `http://user:pass@proxy.example.com:8080`) |
| `HTTPS_PROXY` | None | HTTPS proxy server (falls back to HTTP_PROXY if not set) |

## Proxy Support

To use HTTP proxy for HuggingFace downloads:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
python app.py
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

{
  "seriesInstanceUID": "1.2.840.113619.2...."
}
```

Response:
```json
{
  "classification": {
    "prediction": "Malignant",
    "probability": 0.87,
    "confidence": 0.74,
    "model_name": "MST (DINOv2-based)",
    "version": "1.0",
    "architecture": "Vision Transformer"
  },
  "attention_maps": [...]
}
```

## Model Details

- **Model**: ODELIA MST (Multi-Scale Transformer)
- **Architecture**: Vision Transformer based on DINOv2
- **Task**: Binary classification (Benign vs Malignant)
- **Input**: 3D breast MRI (NIfTI format)
- **Repository**: https://huggingface.co/ODELIA-AI/MST

## Usage

1. Start the service:
```bash
python app.py
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
