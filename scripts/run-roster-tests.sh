#!/usr/bin/env bash
# Bring up the five trained model pairs and run the per-model roster suites.
set -euo pipefail

cd "$(dirname "$0")/.."

export DIVIDE_AND_CONQUER_PORT="${DIVIDE_AND_CONQUER_PORT:-5561}"
ROSTER_PORTS=(5556 5558 5559 "$DIVIDE_AND_CONQUER_PORT" 5562)
declare -A ROSTER_SLUGS=(
  [5558]=agaldran
  [5559]=bcn-aim
  [$DIVIDE_AND_CONQUER_PORT]=divide-and-conquer
  [5562]=lme-abmil
  [5556]=mst
)

# Scope up to the roster pairs: unscoped `up` also starts every profile-less
# service, whose fixed container names collide with a viewer stack running
# under a different compose project.
roster_services=()
for port in "${ROSTER_PORTS[@]}"; do
  slug="${ROSTER_SLUGS[$port]}"
  if [[ "$slug" == mst ]]; then
    roster_services+=(orthanc-router-mst mst-classifier)
  else
    roster_services+=("orthanc-router-odelia-${slug}" "odelia-classification-${slug}")
  fi
done
docker compose up -d "${roster_services[@]}"

if [[ -n "${VIEWER_NETWORK:-}" ]]; then
  echo "Re-attaching roster containers to viewer network '${VIEWER_NETWORK}'..."
  for port in "${ROSTER_PORTS[@]}"; do
    slug="${ROSTER_SLUGS[$port]}"
    containers=("odelia-orthanc-router-odelia-${slug}" "odelia-classification-${slug}")
    if [[ "$slug" == mst ]]; then
      containers=(odelia-orthanc-router-mst odelia-model-mst)
    fi
    for container in "${containers[@]}"; do
      if ! err=$(docker network connect "${VIEWER_NETWORK}" "${container}" 2>&1); then
        if [[ "${err}" == *"already exists in network"* ]]; then
          echo "  ${container}: already attached"
        else
          echo "ERROR: failed to attach ${container} to ${VIEWER_NETWORK}: ${err}" >&2
          exit 1
        fi
      else
        echo "  ${container}: attached"
      fi
    done
  done
else
  echo "VIEWER_NETWORK not set -- skipping cross-project network attachment."
  echo "If the viewer stack runs under a different compose project, attach both"
  echo "containers per model pair manually (see docs/usage/adding_custom_models.md)."
fi

HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-180}"
HEALTH_INTERVAL_S=5

echo "Waiting for roster services to report healthy (up to ${HEALTH_TIMEOUT_S}s each)..."
for port in "${ROSTER_PORTS[@]}"; do
  slug="${ROSTER_SLUGS[$port]}"
  elapsed=0
  until curl -sf "http://localhost:${port}/health" >/dev/null; do
    if (( elapsed >= HEALTH_TIMEOUT_S )); then
      echo "ERROR: ${slug} (port ${port}) did not become healthy within ${HEALTH_TIMEOUT_S}s" >&2
      exit 1
    fi
    echo "  waiting on ${slug} (port ${port})... ${elapsed}s"
    sleep "${HEALTH_INTERVAL_S}"
    elapsed=$(( elapsed + HEALTH_INTERVAL_S ))
  done
  echo "  backend ${port} (${slug}) ready"
done

VIEWER_URL="${ORTHANC_VIEWER_BASE_URL:-http://localhost:8000}"
if ! curl -sf "${VIEWER_URL}/feedback/health" >/dev/null; then
  echo "ERROR: orthanc-viewer not reachable at ${VIEWER_URL} -- both suites would skip and exit 0." >&2
  echo "Start the viewer stack, or point ORTHANC_VIEWER_BASE_URL at it." >&2
  exit 1
fi

cd orthanc

if [[ -x ".venv/bin/python" ]] && .venv/bin/python -m pytest --version >/dev/null 2>&1; then
  PYTEST=(.venv/bin/python -m pytest)
elif command -v python3 >/dev/null 2>&1 && python3 -m pytest --version >/dev/null 2>&1; then
  PYTEST=(python3 -m pytest)
else
  echo "ERROR: no interpreter can run pytest (tried .venv/bin/python -m pytest, python3 -m pytest)" >&2
  exit 1
fi

"${PYTEST[@]}" tests/integration/test_model_roster_smoke.py -v -m integration
"${PYTEST[@]}" tests/integration/test_model_roster_roundtrip.py -v -m integration
