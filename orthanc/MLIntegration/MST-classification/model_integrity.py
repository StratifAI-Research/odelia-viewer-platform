"""Verify the approved MST code and weights before executing either."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REQUIRED_FILES = frozenset(
    {"models.py", "predict_attention.py", "model_config.json", "state_dict.pt"}
)


def read_integrity(path: Path) -> dict[str, Any]:
    integrity = json.loads(path.read_text())
    if integrity.get("repo_id") != "ODELIA-AI/MST":
        raise ValueError("Unexpected MST repository")
    if not re.fullmatch(r"[0-9a-f]{40}", integrity.get("revision", "")):
        raise ValueError("MST revision must be a full commit hash")
    digests = integrity.get("sha256", {})
    if set(digests) != REQUIRED_FILES or not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in digests.values()
    ):
        raise ValueError("MST integrity metadata must contain SHA-256 for all required files")
    return integrity


def verify_files(directory: Path, integrity: dict[str, Any]) -> dict[str, str]:
    files = {}
    for filename, expected in integrity["sha256"].items():
        path = directory / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"MST integrity check failed: {filename}")
        files[filename] = str(path.absolute())
    return files
