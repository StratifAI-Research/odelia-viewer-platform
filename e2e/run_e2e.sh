#!/usr/bin/env bash
# Regenerate the Odelia OHIF frontend e2e walkthrough PDF, end-to-end.
#
# Targets the LOCAL stack (http://localhost:8081 by default). Bring the stack up
# first from the repo root:   docker compose up -d
# Override the target if needed:   VIEWER_BASE_URL=http://host:port ./run_e2e.sh
#
# Usage:   ./run_e2e.sh
# Output:  ./odelia_frontend_e2e_report.pdf
set -uo pipefail

E2E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PW="/tmp/odelia-e2e-pw-venv"
RP="/tmp/odelia-e2e-report-venv"
VIEWER="${VIEWER_BASE_URL:-http://localhost:8081}"
cd "$E2E"

# 1) venvs (Playwright + reportlab). Browsers use Playwright's default cache.
if [ ! -x "$PW/bin/python" ]; then
  echo "[setup] creating Playwright venv"
  python3 -m venv "$PW"
  "$PW/bin/pip" install -q playwright
  "$PW/bin/python" -m playwright install chromium
fi
[ -x "$RP/bin/python" ] || { echo "[setup] creating report venv"; python3 -m venv "$RP"; "$RP/bin/pip" install -q reportlab Pillow; }

# 2) require the local stack to be reachable.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$VIEWER/" 2>/dev/null || echo 000)
if [ "$code" = "000" ]; then
  echo "ERROR: viewer not reachable at $VIEWER"
  echo "Start the local stack first (from the repo root):"
  echo "  docker compose up -d"
  echo "(or point this run elsewhere with VIEWER_BASE_URL=...)"
  exit 1
fi

# 3) run each area. Headless + CPU AI inference is slow and fragile, so keep
#    going on per-area failure; the report records per-area PASS/NOTE.
export VIEWER_BASE_URL="$VIEWER"
for a in area1_auth area2_worklist area3_viewer \
         area4_5_send_to_ai_and_results area6_chat area7_feedback; do
  echo "=== $a ==="
  timeout 600 "$PW/bin/python" "$a.py" || echo "WARN: $a did not complete cleanly"
done

# 4) compile the sectioned PDF (<=3 stage screenshots per area; PASS/FAIL header).
"$RP/bin/python" make_report.py
echo "Done: $E2E/odelia_frontend_e2e_report.pdf"
