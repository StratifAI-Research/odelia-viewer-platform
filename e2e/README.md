# Frontend e2e walkthrough

Playwright suite that drives the OHIF viewer through its functional areas and
compiles a sectioned PDF (A–G, PASS/FAIL per section, ≤3 stage screenshots
each). Verifies the frontend isn't broken by backend changes.

## Prerequisites
- The stack running locally (from the repo root): `docker compose up -d`
- A study loaded in `orthanc-viewer` (the suite expects the `UKA_1` MR study).
- `python3` + network access on first run (to install Playwright + chromium).

## Run
```bash
./run_e2e.sh
```
Produces `odelia_frontend_e2e_report.pdf` here. Override the target with
`VIEWER_BASE_URL=http://host:port ./run_e2e.sh`.

## Layout
- `_helpers.py` — login / banner-dismiss / screenshot / event capture.
- `area1_auth.py` … `area7_feedback.py` — one functional area each; each writes
  `shots/<area>_summary.json` + screenshots.
- `make_report.py` — compiles the summaries + curated screenshots into the PDF.
- `run_e2e.sh` — venv setup, local-stack check, run all areas, compile.

Screenshots (`shots/`) and the PDF are generated artifacts (git-ignored).
