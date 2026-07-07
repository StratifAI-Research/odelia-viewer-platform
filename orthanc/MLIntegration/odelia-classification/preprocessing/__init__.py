"""Preprocessing package for the ODELIA model service.

Public API:
  - ``preprocess``            the MediSwarm default (subtraction NIfTI -> views)
  - ``resolve_preprocessor``  pick a model-local override, else the default
  - ``VolumeView``            one model-ready view
  - ``prepare_single_channel`` legacy ODV-214 placeholder (removed in the
    model_service rewire task)
"""

from preprocessing._legacy import prepare_single_channel
from preprocessing.dispatch import resolve_preprocessor
from preprocessing.pipeline import preprocess
from preprocessing.types import VolumeView

__all__ = [
    "VolumeView",
    "prepare_single_channel",
    "preprocess",
    "resolve_preprocessor",
]
