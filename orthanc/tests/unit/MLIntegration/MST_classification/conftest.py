"""Path setup for MST-classification tests + sibling-name eviction."""
import os
import sys
from typing import Iterator

import pytest

from _colliders import ML_SERVICE_COLLIDERS

_HERE = os.path.dirname(os.path.abspath(__file__))
_MST_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', 'MLIntegration', 'MST-classification'))


@pytest.fixture(autouse=True)
def _force_mst_path() -> Iterator[None]:
    """Ensure MST-classification dir is at sys.path[0] and evict colliding sibling names."""
    saved = list(sys.path)
    if _MST_DIR in sys.path:
        sys.path.remove(_MST_DIR)
    sys.path.insert(0, _MST_DIR)
    # Evict names that exist in multiple ML services (collision risk)
    for k in list(sys.modules):
        top = k.split('.', 1)[0]
        if top in ML_SERVICE_COLLIDERS:
            del sys.modules[k]
    try:
        yield
    finally:
        sys.path[:] = saved
