"""Stage one private HF model release during the image build (ODV-216).

No downloaded Python is imported while the BuildKit token is mounted. The next
build step constructs and validates the model without that credential.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from models import available_models
from weights import read_release, sha256


def fetch(model_name: str, repo: str, revision: str, token_file: Path) -> None:
    """Install a pinned release into the selected vendored subunit directory."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("HF_REVISION must be a full 40-character commit SHA")
    directory_name = available_models()[model_name]
    token = token_file.read_text().strip() if token_file.is_file() else False
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(
            snapshot_download(
                repo_id=repo,
                revision=revision,
                token=token,
                cache_dir=temp,
                allow_patterns=[
                    "*.py",
                    "*.json",
                    "*.safetensors",
                    "README.md",
                    "LICENSE*",
                    "NOTICE*",
                    "requirements.txt",
                    "_dinov2/*",
                ],
            )
        )
        release = read_release(directory, model_name)
        if not release or not release.get("release"):
            raise ValueError("HF package needs release.json with an explicit release")
        digest = sha256(directory / "weights.safetensors")
        if release.get("weights_sha256") != digest:
            raise ValueError("HF weights SHA-256 does not match release.json")
        for filename in ("__init__.py", "loader.py"):
            if not (directory / filename).is_file():
                raise ValueError(f"HF package missing {filename}")
        dest = Path("models") / directory_name
        # Copy out of the ephemeral Hub cache so the final image has no token
        # file, Hub cache, or symlinks pointing outside the installed package.
        staged = Path(temp) / "staged"
        shutil.copytree(directory, staged)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(staged, dest)
        lock = {
            "model": model_name,
            "subunit": directory_name,
            "weights": "weights.safetensors",
            "weights_sha256": digest,
            "release": release["release"],
            "hf_repo": repo,
            "hf_revision": revision,
        }
        Path("models-lock.json").write_text(json.dumps(lock, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("--token-file", type=Path, default=Path("/run/secrets/hf_token"))
    args = parser.parse_args()
    fetch(args.model, args.repo, args.revision, args.token_file)


if __name__ == "__main__":
    main()
