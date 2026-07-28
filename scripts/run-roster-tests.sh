#!/usr/bin/env bash
# Bring up the odelia-models profile and run the per-model roster suites.
set -euo pipefail

cd "$(dirname "$0")/.."

sudo docker compose --profile odelia-models up -d

echo "Waiting for roster services to report healthy..."
for port in 5558 5559 5561 5562 5563 5564; do
  until curl -sf "http://localhost:${port}/health" >/dev/null; do
    sleep 5
  done
  echo "  backend ${port} ready"
done

cd orthanc
pytest tests/integration/test_model_roster_smoke.py -v -m integration
pytest tests/integration/test_model_roster_roundtrip.py -v -m integration
