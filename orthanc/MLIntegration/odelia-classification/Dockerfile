# Dockerfile for the generalized ODELIA model service (ODV-214).
#
# One image = one model. MODEL selects the vendored subunit; the others are
# pruned so the runtime serves exactly one, with no runtime model selection.
# WEIGHT_PATH (optional) bakes + checks a trained weights file; unset => an
# init-only image. Strict-loading baked weights into the model is a later ticket.
#
# Build from the MLIntegration/ directory:
#   docker build -f odelia-classification/Dockerfile \
#     --build-arg MODEL=BCN_AIM \
#     [--build-arg WEIGHT_PATH=weights/bcn_aim.safetensors] \
#     -t odelia-model-bcnaim .
FROM python:3.10-slim

# MODEL: canonical model name (e.g. MST, BCN_AIM). Required.
ARG MODEL
# WEIGHT_PATH: path (relative to the build context) to a .safetensors weights
# file to bake into the subunit. Unset => init-only image.
ARG WEIGHT_PATH=

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

# Copy service code (includes the owned models/ package + build tools)
COPY odelia-classification/ ./odelia-service/

# Install service dependencies
WORKDIR /app/odelia-service
RUN pip install --no-cache-dir -r requirements.txt

# Select the model: prune every subunit except the one MODEL names.
RUN test -n "$MODEL" || (echo "ERROR: MODEL build-arg is required" >&2 && exit 1)
RUN python -m tools.select_subunit "$MODEL"

# MST builds a DINOv2 backbone at construction; pre-cache it so runtime is
# network-free (TORCH_HOME is baked into the image).
ENV TORCH_HOME=/app/odelia-service/.torch
RUN if [ "$MODEL" = "MST" ]; then \
      python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"; \
    fi

# Bake + check weights when provided (the staged file is ignored when WEIGHT_PATH
# is empty). Default source keeps the COPY valid for init-only builds.
COPY ${WEIGHT_PATH:-pyproject.toml} /tmp/weight.in
RUN python -m tools.bake_weights "$MODEL" /tmp/weight.in "$WEIGHT_PATH"

# Create runtime dirs
RUN mkdir -p images

# Expose port (ODV-218 overrides via PORT for per-model allocation)
EXPOSE 5556

# Run service
CMD ["python", "app_refactored.py"]
