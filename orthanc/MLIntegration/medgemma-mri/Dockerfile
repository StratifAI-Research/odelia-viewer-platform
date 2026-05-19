# Dockerfile for MedGemma MRI Classification Service
# Build from MLIntegration directory:
#   cd MLIntegration
#   docker build -f medgemma-mri/Dockerfile -t medgemma-mri .

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install shared package
COPY shared/ ./shared/
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy MedGemma service code
COPY medgemma-mri/ ./medgemma-service/

# Install service dependencies
WORKDIR /app/medgemma-service
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p images

# Environment variables (override at runtime)
ENV HF_TOKEN=""
ENV IMAGE_FOLDER="./images"
ENV NUM_SLICES="5"
ENV MAX_NEW_TOKENS="500"
ENV TORCH_DTYPE="bfloat16"

# Expose port
EXPOSE 5557

# Run service
CMD ["python", "app_refactored.py"]
