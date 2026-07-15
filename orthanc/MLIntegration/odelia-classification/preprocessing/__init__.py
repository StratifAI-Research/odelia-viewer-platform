"""Preprocessing package for the ODELIA model service.

Public API:
  - ``preprocess``           the MediSwarm default (subtraction NIfTI -> views)
  - ``resolve_preprocessor`` pick a model-local override, else the default
  - ``VolumeView``           one model-ready view
"""

from preprocessing.dispatch import resolve_preprocessor
from preprocessing.pipeline import preprocess
from preprocessing.types import VolumeView

__all__ = ["VolumeView", "preprocess", "resolve_preprocessor"]
