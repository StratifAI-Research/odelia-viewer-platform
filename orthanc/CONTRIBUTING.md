# Contributing to platform Python services

## Local development

Each service has its own `pyproject.toml` (direct deps) and `requirements.txt` (hand-pinned, no hashes).

### Install dev tools (ruff, mypy)

```
pip install -r requirements-dev.txt
```

### Run linters locally

```
cd custom/deploy/orthanc  # or /orthanc inside the platform repo
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

## Subtree layout — `custom/deploy/` is a git subtree

`custom/deploy/` is mirrored from `odelia-deployment.git` via `git subtree`. Two consequences:

1. The platform-side workflow lives at `custom/deploy/.github/workflows/python-lint.yml` (subtree-internal path). The viewer's own CI runs from `.github/workflows/lint.yml` at the repo root and doesn't pick up the subtree workflow.
2. Edits to `custom/deploy/**` made in this repo must be subtree-split and pushed back to `odelia-deployment` — otherwise the next pull from upstream will overwrite them. If you fix something here that also exists on the platform side, fix it on platform first or be prepared to push the subtree.

## `lint-py` is currently red on purpose

The `lint-py` job is gating (`fail-on-error: true` by default) but the codebase still has 888 ruff + 64 format + 172 mypy violations as of this PR landing. They're tracked in ODV-199, ODV-195, and ODV-198 and will be cleared in follow-up PRs. Until then, the job will go red on every push — that's expected, not a regression you introduced.

Two escape hatches if you need to ship something orthogonal:

- Pass `fail-on-error: false` (or set `PYTHON_LINT_WARN_ONLY=true`) to switch the action to warn-only mode — violations are reported but the job stays green.
- A *tool crash* (non-zero exit with zero parsed violations — e.g. ruff config error, mypy import failure) always fails the job, even in warn-only mode. Warn-only is for accepting existing debt, not for hiding broken tooling.

## Python testing

Tests live under `tests/`:

- `tests/unit/` — fast tests, no external services. CI’s default `pytest -m unit` runs only these.
- `tests/integration/` — tests requiring a live Orthanc viewer. Marked `@pytest.mark.integration`; skip themselves when no viewer is reachable. Not run by default in CI.

Markers are auto-applied by location — no `@pytest.mark.unit` on each unit test needed. The top-level `tests/conftest.py` applies the marker based on file path.

Tests outside `tests/unit/` and `tests/integration/` are rejected at collection time unless they carry an explicit marker.

### Running locally

    cd custom/deploy/orthanc
    pip install -r requirements-tests.txt
    pytest tests -m unit                    # unit only
    pytest tests -m integration             # integration (needs running orthanc-viewer)
    pytest tests                            # both

### Coverage gate

CI enforces `--cov-fail-under=50` via the `python-test` composite action (same pattern as `python-lint`).

**Currently red by design.** This PR (ODV-192) ships only two seed tests covering shared/config.py and shared/timing_utils.py — total coverage is well below the 50% floor. The CI gate exists; the test mass to clear it lives in ODV-133, which rebases onto post-192 main. Same shape as `lint-py` red-by-design until ODV-199/195/198 land. Don’t mis-diagnose `test-py` red on your PR as your regression — check the rollup against this baseline.
