# Dockerfile for MST Classification Service
# Build from MLIntegration directory:
#   cd MLIntegration
#   docker build -f MST-classification/Dockerfile -t mst-classification .

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install shared package
COPY shared/ ./shared/
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy MST service code
COPY MST-classification/ ./mst-service/

# Install service dependencies
WORKDIR /app/mst-service
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p mst_model images

# Expose port
EXPOSE 5556

# Run service
CMD ["python", "app_refactored.py"]
