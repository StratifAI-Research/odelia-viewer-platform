"""ODELIA model zoo: the MST built-in, the challenge roster, and create_model.

Reimplements the MediSwarm models the service serves, matching their semantics
and state_dict layout (not a verbatim vendor). The challenge models live under
``challenge/`` and are imported lazily by the factory.
"""

from .base import BasicClassifier, ModelWrapper
from .factory import create_model
from .mst import MST

__all__ = ["MST", "BasicClassifier", "ModelWrapper", "create_model"]
