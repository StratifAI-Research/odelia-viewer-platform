# 🧠 Integration of ML Model into Medical Imaging Viewer

This project integrates a machine learning model with an ORTHANC DICOM server to classify MRI images for breast cancer detection. The backend is built with Flask and designed to run inside a Docker container.

## 🚀 Getting Started

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/mkhoobi/MLIntegration.git
cd breast_cancer_classification
```

### 2. Prepare Directory Structure

Create the following folders inside the main project directory:

- `images/` – for temporary storage of DICOM images
- `models/` – to store your trained ML model

Example:

```
breast_cancer_classification/
├── app.py
├── images/
└── models/
    └── your_model.pth
```

### 3. Configure Environment Variables

Edit `app.py` to set your environment or override them with your custom values:

```python
ORTHANC_URL = os.getenv("ORTHANC_URL", "http://[ip_address_of_orthanc]")
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "./images")
MRI_MODEL_PATH = os.getenv("MODEL_PATH", "./models/your_model.pth")
```

Alternatively, you can create a `.env` file and pass it to Docker if preferred.

### 4. Build and Run the Docker Container

```bash
docker build -t analyze-cancer-api .
docker run -p 5555:5555 analyze-cancer-api
```

### 5. Test the API

Use `curl` to send a test request (replace with a valid SeriesInstanceUID):

```bash
curl -X POST http://localhost:5555/analyze/mri \
     -H "Content-Type: application/json" \
     -d '{"seriesInstanceUID": "1.3.36.670*******"}'
```
