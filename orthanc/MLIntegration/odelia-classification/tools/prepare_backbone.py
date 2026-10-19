"""Vendor pinned DINOv2 source at build/export time; never download weights."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def prepare(directory: Path) -> None:
    """Place the pinned source beside the MST architecture for offline creation."""
    dest = directory / "_dinov2"
    if (dest / "REVISION").is_file():
        if (dest / "REVISION").read_text().strip() != DINOV2_REVISION:
            raise ValueError("Unexpected vendored DINOv2 revision")
        return
    if dest.exists():
        raise ValueError(f"Incomplete DINOv2 source: {dest}")
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp)
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "-q",
                "--depth=1",
                "https://github.com/facebookresearch/dinov2.git",
                DINOV2_REVISION,
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "checkout", "-q", "FETCH_HEAD"], check=True)
        dest.mkdir(parents=True)
        shutil.copytree(source / "dinov2", dest / "dinov2")
        shutil.copyfile(source / "LICENSE", dest / "LICENSE")
        # Only expose the backbone entrypoints we use, avoiding optional heads.
        (dest / "hubconf.py").write_text(
            "from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14\n"
            'dependencies = ["torch"]\n'
        )
        (dest / "REVISION").write_text(DINOV2_REVISION + "\n")


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
