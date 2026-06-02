"""Path setup for viewer-side unit tests."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIEWER_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'viewer'))

if _VIEWER_DIR not in sys.path:
    sys.path.insert(0, _VIEWER_DIR)
