# Dockerfile for Breast Cancer Classification Service
# Build from MLIntegration directory:
#   cd MLIntegration
#   docker build -f breast-cancer-classification/Dockerfile -t breast-cancer-classification .

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

# Copy breast cancer service code
COPY breast-cancer-classification/ ./bc-service/

# Install service dependencies
WORKDIR /app/bc-service
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p models images

# Expose port
EXPOSE 5555

# Run service
CMD ["python", "app_refactored.py"]
