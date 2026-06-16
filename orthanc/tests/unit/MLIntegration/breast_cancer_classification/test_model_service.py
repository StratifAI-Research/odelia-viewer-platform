"""Tests for BreastCancerModelService per-side error handling.

model_service imports torch, torchio, monai.networks, and dicom_converter
(-> dicom_utils -> SimpleITK) at module load; stub them. Imports run inside
tests, after the path/stub fixtures.
"""
import sys
import types
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stub_heavy_deps(torch_stub, torchio_stub, sitk_stub, monkeypatch) -> Iterator[None]:
    monai = types.ModuleType("monai")
    monai.networks = types.ModuleType("monai.networks")
    monai.networks.nets = MagicMock()
    monkeypatch.setitem(sys.modules, "monai", monai)
    monkeypatch.setitem(sys.modules, "monai.networks", monai.networks)

    du_stub = types.ModuleType("dicom_utils")
    du_stub.dicom_to_nifti = MagicMock()
    monkeypatch.setitem(sys.modules, "dicom_utils", du_stub)

    sys.modules.pop("model_service", None)
    yield


def test_per_side_failure_returns_generic_error(tmp_path, monkeypatch):
    import model_service as ms
    from shared.config import StorageConfig

    service = ms.BreastCancerModelService(
        MagicMock(), StorageConfig(image_folder=tmp_path, cleanup_on_start=False)
    )
    service.model = MagicMock()  # not None

    strategy = MagicMock()
    strategy.retrieve.return_value = (str(tmp_path), "1.2.3")
    monkeypatch.setattr(service, "_create_retrieval_strategy", lambda req: strategy)
    monkeypatch.setattr(ms, "convert_to_unilateral_nifti", lambda folder: {"dummy": 1})
    monkeypatch.setattr(ms, "get_preprocessing_pipeline", lambda: MagicMock())

    secret = "boom /srv/internal/secret/path"
    monkeypatch.setattr(service, "_process_side", MagicMock(side_effect=RuntimeError(secret)))

    captured = {}
    monkeypatch.setattr(
        ms,
        "build_bilateral_classification",
        lambda left, right: captured.update(left=left, right=right) or {"ok": True},
    )

    service.analyze_mri_series({"wado_rs_retrieval": [{"x": 1}]})

    for side in ("left", "right"):
        assert captured[side] == {"error": f"Processing error for {side} side"}
        assert secret not in str(captured[side])
