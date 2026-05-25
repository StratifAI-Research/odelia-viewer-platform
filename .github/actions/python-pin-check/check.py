"""Verify pyproject.toml [project].dependencies and requirements.txt are lockstep pinned.

For each service directory, both files must:
  - Have every entry fully pinned (== for PyPI, @ git+...@<40-char-sha> for git URLs).
  - Contain the same set of direct deps with identical version specs.

Also validates requirements-dev.txt (no pyproject equivalent — just pin enforcement).

Exits 1 if any service fails either check.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

# Services validated for pyproject↔requirements lockstep + pin enforcement.
# Kept static (not auto-discovered) so adding a service is a deliberate edit —
# a directory accidentally containing a stray pyproject.toml won't silently
# enter the policy. When adding a new service, append it here and to any
# orchestration that lists service directories.
SERVICES = [
    "viewer",
    "router",
    "MLIntegration/shared",
    "MLIntegration/medgemma-mri",
    "MLIntegration/MST-classification",
    "MLIntegration/chat-middleware",
    "MLIntegration/breast-cancer-classification",
]

# Stand-alone pinned files that have no pyproject counterpart (dev tooling).
EXTRA_PINNED_FILES = ["requirements-dev.txt"]

# A dep line is "pinned" if it has `==<version>` (no wildcards) OR is a git URL
# with a full 40-character commit SHA: `pkg @ git+url@<40-hex>`.
# Version chars limited to PEP 440 alphabet — no `*`, so `pkg==1.*` is rejected.
PINNED_PYPI = re.compile(
    r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?\s*==\s*[A-Za-z0-9.+!_-]+$"
)
PINNED_GIT = re.compile(r"^[A-Za-z0-9_.\-]+\s*@\s*git\+\S+@[0-9a-f]{40}$")


def is_pinned(entry: str) -> bool:
    entry = entry.strip()
    return bool(PINNED_PYPI.match(entry) or PINNED_GIT.match(entry))


_NAME_HEAD = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)(.*)$", re.DOTALL)


def _normalize_entry(entry: str) -> str:
    """Return entry with the leading package name PEP 503-normalized.

    `Flask==3.1.3` and `flask==3.1.3` both become `flask==3.1.3`; the version
    spec / extras / git URL after the name is preserved verbatim.
    """
    m = _NAME_HEAD.match(entry.strip())
    if not m:
        return entry.strip()
    head, rest = m.group(1), m.group(2)
    canonical = re.sub(r"[-_.]+", "-", head).lower()
    return canonical + rest


def read_pyproject_deps(path: Path) -> list[str]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("dependencies", [])


def read_pyproject_build_requires(path: Path) -> list[str]:
    """Return [build-system].requires entries (setuptools, wheel, etc.).

    PEP 517 build environments resolve these at build time. Unpinned entries
    let Docker builds pull mutable setuptools/wheel/etc. from PyPI on every run,
    defeating the lockstep-pinning story for runtime deps.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("build-system", {}).get("requires", [])


def read_requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith("-")
    ]


def check_service(svc: str) -> bool:
    """Return True on success, False on failure. Prints diagnostics either way."""
    pyproject_path = Path(svc) / "pyproject.toml"
    requirements_path = Path(svc) / "requirements.txt"

    if not pyproject_path.exists():
        print(f"::warning::{svc} has no pyproject.toml, skipping")
        return True
    if not requirements_path.exists():
        print(f"::error file={requirements_path}::missing requirements.txt")
        return False

    py_deps = read_pyproject_deps(pyproject_path)
    py_build_reqs = read_pyproject_build_requires(pyproject_path)
    req_deps = read_requirements(requirements_path)
    svc_failed = False

    for entry in py_deps:
        if not is_pinned(entry):
            print(
                f"::error file={pyproject_path}::unpinned dep in pyproject.toml: {entry!r}"
            )
            svc_failed = True

    for entry in py_build_reqs:
        if not is_pinned(entry):
            print(
                f"::error file={pyproject_path}::unpinned [build-system].requires entry: {entry!r}"
            )
            svc_failed = True

    for entry in req_deps:
        if not is_pinned(entry):
            print(
                f"::error file={requirements_path}::unpinned dep in requirements.txt: {entry!r}"
            )
            svc_failed = True

    py_norm = sorted(_normalize_entry(d) for d in py_deps)
    req_norm = sorted(_normalize_entry(d) for d in req_deps)
    if py_norm != req_norm:
        print(
            f"::error::pyproject.toml and requirements.txt disagree in {svc}\n"
            f"  pyproject only: {sorted(set(py_norm) - set(req_norm))}\n"
            f"  requirements only: {sorted(set(req_norm) - set(py_norm))}"
        )
        svc_failed = True

    if not svc_failed:
        print(f"✓ {svc}")
    return not svc_failed


def check_extra_pinned_file(path_str: str) -> bool:
    path = Path(path_str)
    if not path.exists():
        print(f"::warning::{path} not found, skipping")
        return True
    failed = False
    for entry in read_requirements(path):
        if not is_pinned(entry):
            print(f"::error file={path}::unpinned dep: {entry!r}")
            failed = True
    if not failed:
        print(f"✓ {path}")
    return not failed


def main() -> int:
    failed = False
    for svc in SERVICES:
        if not check_service(svc):
            failed = True
    for extra in EXTRA_PINNED_FILES:
        if not check_extra_pinned_file(extra):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
