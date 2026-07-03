"""Build helper: bake + check the selected subunit's weights, write provenance.

Run from the service dir during the Docker build, after select_subunit:
    python -m tools.bake_weights "$MODEL" /tmp/weight.in "$WEIGHT_PATH"

``WEIGHT_PATH`` empty -> init-only image (the staged file is ignored). Otherwise
the staged file is validated, copied into the subunit as ``weights.safetensors``,
and its sha256 recorded in ``models-lock.json``. Strict-loading the weights into
the model is a later ticket; here we only place + checksum.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from models.loader_util import available_models


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: python -m tools.bake_weights <MODEL> <staged_file> <weight_path_arg>",
            file=sys.stderr,
        )
        return 2
    model, staged, weight_arg = argv[1], Path(argv[2]), argv[3].strip()

    mapping = available_models()
    if model not in mapping:
        print(f"Unknown MODEL {model!r}; available: {sorted(mapping)}", file=sys.stderr)
        return 1
    keep = mapping[model]

    lock: dict[str, str] = {"model": model, "subunit": keep}
    if weight_arg:
        if not weight_arg.endswith(".safetensors"):
            print(f"WEIGHT_PATH must be a .safetensors file (got {weight_arg})", file=sys.stderr)
            return 1
        if not staged.is_file():
            print(f"WEIGHT_PATH={weight_arg} not found in build context", file=sys.stderr)
            return 1
        dest = Path("models") / keep / "weights.safetensors"
        shutil.copyfile(staged, dest)
        lock["weights"] = "weights.safetensors"
        lock["weights_sha256"] = _sha256(dest)
        print(f"Baked weights for {model}: sha256 {lock['weights_sha256'][:12]}…")
    else:
        lock["weights"] = "init-only"
        print(f"No WEIGHT_PATH; {model} image is init-only")

    Path("models-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
