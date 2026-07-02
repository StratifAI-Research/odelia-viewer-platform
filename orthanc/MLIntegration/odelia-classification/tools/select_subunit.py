"""Build helper: keep only the subunit selected by MODEL, prune the rest.

One image = one model. Run from the service dir during the Docker build:
    python -m tools.select_subunit "$MODEL"
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from models.loader_util import available_models


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m tools.select_subunit <MODEL>", file=sys.stderr)
        return 2
    name = argv[1]
    mapping = available_models()
    if name not in mapping:
        print(f"Unknown MODEL {name!r}; available: {sorted(mapping)}", file=sys.stderr)
        return 1

    keep = mapping[name]
    models_dir = Path("models")
    pruned = []
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and (entry / "loader.py").is_file() and entry.name != keep:
            shutil.rmtree(entry)
            pruned.append(entry.name)
    print(f"Kept subunit '{keep}' for MODEL={name}; pruned {pruned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
