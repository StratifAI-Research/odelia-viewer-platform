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

## `lint-py` is currently yellow (warn-only)

The `lint-py` job runs in warn-only mode (`fail-on-error: false`) — violations are reported in the step summary but don't block the PR. The codebase still has ~888 ruff + 64 format + 172 mypy violations tracked in ODV-199, ODV-195, and ODV-198. Once those land, the job will be flipped to gating mode (`fail-on-error: true`).

A *tool crash* (non-zero exit with zero parsed violations — e.g. ruff config error, mypy import failure) always fails the job, even in warn-only mode. Warn-only is for accepting existing debt, not for hiding broken tooling.
