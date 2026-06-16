"""Tests for MSTModelService._resolve_within (path-injection barrier).

model_service imports torch + huggingface_hub (model_loader) + dicom_utils
(dicom_converter -> SimpleITK) at module load; stub them, mirroring
test_dicom_converter.py. Imports run inside tests, after the path/stub fixtures.
"""
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stub_heavy_deps(torch_stub, sitk_stub, monkeypatch) -> Iterator[None]:
    hf_stub = types.ModuleType("huggingface_hub")
    hf_stub.hf_hub_download = MagicMock()
    hf_stub.login = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_stub)

    du_stub = types.ModuleType("dicom_utils")
    du_stub.dicom_to_nifti = MagicMock()
    du_stub.dicom_to_nifti_subtraction = MagicMock()
    du_stub.compute_subtraction_from_nifti = MagicMock()
    monkeypatch.setitem(sys.modules, "dicom_utils", du_stub)

    sys.modules.pop("model_service", None)
    yield


def _make_service(image_root: Path):
    import model_service
    from shared.config import StorageConfig

    storage_config = StorageConfig(image_folder=image_root, cleanup_on_start=False)
    return model_service.MSTModelService(MagicMock(), storage_config)


def test_resolve_within_passes_through_folder_inside_root(tmp_path):
    service = _make_service(tmp_path)

    resolved = service._resolve_within(Path("1.2.840.113619"))

    assert resolved == tmp_path / "1.2.840.113619"


def test_resolve_within_rejects_traversal_as_inference_error(tmp_path):
    service = _make_service(tmp_path)
    from exceptions import InferenceError

    with pytest.raises(InferenceError):
        service._resolve_within(Path("../../etc/passwd"))


def test_resolve_within_rejects_absolute_escape_as_inference_error(tmp_path):
    service = _make_service(tmp_path)
    from exceptions import InferenceError

    with pytest.raises(InferenceError):
        service._resolve_within(Path("/etc/passwd"))


def test_resolve_within_rejection_does_not_leak_path(tmp_path):
    # rejected path must not appear in the client-facing message
    service = _make_service(tmp_path)
    from exceptions import InferenceError

    secret = "../../etc/shadow"
    with pytest.raises(InferenceError) as exc_info:
        service._resolve_within(Path(secret))

    assert secret not in str(exc_info.value)
    assert "etc" not in str(exc_info.value)
