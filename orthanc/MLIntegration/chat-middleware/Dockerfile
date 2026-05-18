# Dockerfile for Chat Middleware Service
# Build from MLIntegration directory:
#   cd MLIntegration
#   docker build -f chat-middleware/Dockerfile -t chat-middleware .

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for SimpleITK and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglx-mesa0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install shared package first (for caching)
COPY shared/ ./shared/
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy chat-middleware requirements and install
COPY chat-middleware/requirements.txt ./chat-middleware/
RUN pip install --no-cache-dir -r ./chat-middleware/requirements.txt

# Copy chat-middleware source code
COPY chat-middleware/ ./chat-middleware/

WORKDIR /app/chat-middleware

# Create image folder
RUN mkdir -p /tmp/chat-middleware-images

# Environment variables (override at runtime)
ENV OLLAMA_URL="http://host.docker.internal:11434"
ENV OLLAMA_MODEL="MedAIBase/MedGemma1.5:4b"
ENV WADO_BASE_URL="http://orthanc-viewer:8042/dicom-web"
ENV NUM_SLICES="5"
ENV MAX_CACHE_ENTRIES="100"
ENV IMAGE_FOLDER="/tmp/chat-middleware-images"

# Expose port
EXPOSE 5560

# Run the application
CMD ["python", "app.py"]
