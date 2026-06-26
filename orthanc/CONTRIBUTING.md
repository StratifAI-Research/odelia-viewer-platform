# Contributing to platform Python services

## Local development

Each service has its own `pyproject.toml` (direct deps) and `requirements.txt` (hand-pinned, no hashes).

### Install dev tools (ruff, mypy)

```
pip install -r requirements-dev.txt
```

### Run linters locally

```
cd orthanc  # from the odelia-viewer-platform repo root
ruff check .
ruff format --check .
mypy --explicit-package-bases viewer router MLIntegration/shared
```

## Adding or updating a dependency

1. Edit `<service>/pyproject.toml` — add or modify an entry in `[project].dependencies`. Always use `==X.Y.Z` (or `pkg @ git+url@<40-char-sha>` for git deps).
2. Edit `<service>/requirements.txt` to match — the two files are kept lockstep by hand.
3. CI's `python-pin-check` job verifies both files are fully pinned and agree on versions.

Package-name comparison is PEP 503-normalized — `Flask`, `flask`, `Flask-Cors`, `flask_cors`, and `flask.cors` are treated as the same name across the two files. Version specs (and everything after the name) are still compared verbatim.

### What `python-pin-check` will reject

- Unpinned entries: `flask` (no operator), `flask>=2.0`, `flask~=2.0`.
- Wildcard pins: `flask==2.*`.
- Git deps with short SHAs — require the full 40-character commit hash.
- Drift between `pyproject.toml` and `requirements.txt` (different version, missing entry, or extra entry on either side).
- Unpinned lines in `requirements-dev.txt` (the dev-tool list is also validated).

### `--extra-index-url` and other pip-only directives

`pyproject.toml` cannot express `--extra-index-url`, `--index-url`, or other pip-level flags. When an index URL matters (e.g. MST's CPU torch wheels), put it in `requirements.txt` only. `python-pin-check` skips `-`-prefixed lines, so the version pins below are still enforced — but `pip install -e .` from pyproject alone will hit PyPI, not the alternate index.

## `requires-python` across services

Different services target different floors deliberately. The pin-check does not enforce a single value:

- `MLIntegration/shared`, `MLIntegration/{medgemma-mri,MST-classification,breast-cancer-classification}` — `>=3.10` (CUDA base images use Ubuntu 22.04 / Python 3.10).
- `viewer`, `router`, `MLIntegration/chat-middleware` — `>=3.11` (Orthanc-base / `python:3.11-slim` images).
- Root `MLIntegration/pyproject.toml` — `>=3.10` (lowest common floor; it's a shim that installs the `shared` module into each service image).

When changing a service's Python floor, update its Dockerfile base image in lockstep.

## `lint-py` is gating

The `lint-py` job fails the build on any ruff, format, or mypy violation. The ODV-195 / ODV-198 / ODV-199 cleanup brought the codebase to zero violations; the job now enforces that — new violations block the PR.

A *tool crash* (non-zero exit with zero parsed violations — e.g. ruff config error, mypy import failure) is distinguished from normal violations in the step summary, and also fails the job.

## Python testing

Tests live under `tests/`:

- `tests/unit/` — fast tests, no external services. CI’s default `pytest -m unit` runs only these.
- `tests/integration/` — tests requiring a live Orthanc viewer. Marked `@pytest.mark.integration`; skip themselves when no viewer is reachable. Not run by default in CI.

Markers are auto-applied by location — no `@pytest.mark.unit` on each unit test needed. The top-level `tests/conftest.py` applies the marker based on file path. Tests outside `tests/unit/` and `tests/integration/` are rejected at collection time unless they carry an explicit marker.

### Running locally

    cd orthanc
    pip install -r requirements-tests.txt
    pytest tests -m unit                    # unit only
    pytest tests -m integration             # integration (needs running orthanc-viewer)
    pytest tests                            # both

### Coverage gate

CI enforces `--cov-fail-under=50` via the `python-test` composite action. The ODV-133 unit suite clears this (~80% as of this PR).
