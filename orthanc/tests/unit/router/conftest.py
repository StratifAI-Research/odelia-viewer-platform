"""Path setup for router-side unit tests."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROUTER_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..', 'router'))

if _ROUTER_DIR not in sys.path:
    sys.path.insert(0, _ROUTER_DIR)
