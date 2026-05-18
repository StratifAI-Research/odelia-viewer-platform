# Orthanc AI Router

This component is a Python plugin for Orthanc that routes medical images to AI models for analysis and creates DICOM outputs based on the model results.

## Features

- Automatically detects when a study becomes stable in Orthanc
- Sends the study to a configured AI model backend for analysis
- Creates DICOM Secondary Capture (SC) images with annotations
- Creates DICOM Structured Reports (SR) with the model's findings
- Handles both left and right side analysis results
- Configurable through environment variables

## Configuration

The following environment variables can be used to configure the plugin:

- `MODEL_BACKEND_URL`: URL of the AI model backend (default: "http://breast-cancer-classification:5555")
- `AI_TEXT`: Text to overlay on the SC images (default: "PROCESSED BY AI")
- `AI_COLOR`: Color for the text overlay (default: "red")
- `AI_NAME`: Name of the AI model to include in the SR report (default: "Breast Cancer Classification Model")

## DICOM Output

The plugin creates two types of DICOM outputs:

1. **Secondary Capture (SC)**: An annotated version of the original image with text overlay
2. **Structured Report (SR)**: A report containing the model's findings for both left and right sides, including:
   - Classification (Benign/Malignant)
   - Confidence scores
   - Model metadata

## Integration with AI Model Backend

The plugin expects the AI model backend to provide a REST API endpoint at `/analyze/mri` that accepts a POST request with a JSON body containing a `seriesInstanceUID`. The response should be a JSON object with results for both left and right sides, including:

```json
{
  "left": {
    "prediction": "Cancerous" or "Not Cancerous",
    "confidence": 95.7
  },
  "right": {
    "prediction": "Cancerous" or "Not Cancerous",
    "confidence": 98.2
  }
}
```

## Usage

This component is designed to be used as part of a Docker Compose setup with Orthanc and an AI model backend. See the main project's docker-compose.yml for the complete setup. 