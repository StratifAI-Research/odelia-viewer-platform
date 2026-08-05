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

# Dependencies first: this layer keys only on requirements.txt, so ordinary
# service-code edits reuse it, and it is identical across every MODEL --
# neither build-arg is in scope yet at this point in the file.
COPY odelia-classification/requirements.txt ./odelia-service/requirements.txt
WORKDIR /app/odelia-service
RUN pip install --no-cache-dir -r requirements.txt

# Service code (includes the owned models/ package + build tools).
# WORKDIR is already /app/odelia-service, so the destination is "." --
# using "./odelia-service/" here would nest it under itself.
COPY odelia-classification/ .

# MODEL: canonical model name (e.g. MST, BCN_AIM). Required. Declared here,
# immediately before first use, so it does not key any layer above.
ARG MODEL
# Select the model: prune every subunit except the one MODEL names.
RUN test -n "$MODEL" || (echo "ERROR: MODEL build-arg is required" >&2 && exit 1)
RUN python -m tools.select_subunit "$MODEL"

# MST builds a DINOv2 backbone at construction; pre-cache it so runtime is
# network-free (TORCH_HOME is baked into the image).
ENV TORCH_HOME=/app/odelia-service/.torch
RUN if [ "$MODEL" = "MST" ]; then \
      python -c "import torch; torch.hub._validate_not_a_forked_repo = lambda a, b, c: True; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"; \
    fi

# WEIGHT_PATH: path (relative to the build context) to a .safetensors weights
# file to bake into the subunit. Unset => init-only image. Declared here,
# immediately before use, so it does not key any layer above.
ARG WEIGHT_PATH=
# Bake + check weights when provided (the staged file is ignored when WEIGHT_PATH
# is empty). Default source keeps the COPY valid for init-only builds.
COPY ${WEIGHT_PATH:-pyproject.toml} /tmp/weight.in
RUN python -m tools.bake_weights "$MODEL" /tmp/weight.in "$WEIGHT_PATH"

# Create runtime dirs
RUN mkdir -p images

# Expose port: every container listens on 5556 regardless of model; ODV-218
# maps each to a distinct host port in the compose files, not via this Dockerfile.
EXPOSE 5556

# Run service
CMD ["python", "app_refactored.py"]
