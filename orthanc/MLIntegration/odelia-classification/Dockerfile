# Dockerfile for the generalized ODELIA model service (ODV-214)
# Build from MLIntegration directory:
#   cd MLIntegration
#   docker build -f odelia-classification/Dockerfile -t odelia-classification .

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install shared package
COPY shared/ ./shared/
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy service code (includes the vendored mediswarm/ tree)
COPY odelia-classification/ ./odelia-service/

# Install service dependencies
WORKDIR /app/odelia-service
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p model images

# MODEL_NAME selects which model this container serves (e.g. ResNet50, MST).
ENV MODEL_NAME=MST
# MODEL_DEVICE (cpu|cuda) is REQUIRED and intentionally not defaulted here — it
# must be supplied at runtime (deploy/compose, ODV-218); the service fails
# loudly at startup if it is unset.

# Expose port (ODV-218 overrides via PORT for per-model allocation)
EXPOSE 5556

# Run service
CMD ["python", "app_refactored.py"]
