"""HTTP-level test for the MST /analyze/mri path-containment contract.

app_refactored imports model_service (torch + huggingface_hub + dicom_utils ->
SimpleITK) at module load; stub them. Imports run inside the test, after the
path/stub fixtures.
"""
import sys
import types
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
    sys.modules.pop("app_refactored", None)
    yield


def test_traversal_derived_folder_returns_generic_500(tmp_path, monkeypatch):
    import app_refactored as appmod
    import model_service as ms
    from shared.config import StorageConfig

    service = ms.MSTModelService(
        MagicMock(), StorageConfig(image_folder=tmp_path, cleanup_on_start=False)
    )
    service.model = MagicMock()  # not None

    strategy = MagicMock()
    strategy.retrieve.return_value = ("../../etc/passwd", "1.2.3")
    monkeypatch.setattr(service, "_create_retrieval_strategy", lambda req: strategy)
    monkeypatch.setattr(appmod, "model_service", service)

    client = appmod.app.test_client()
    resp = client.post("/analyze/mri", json={"wado_rs_retrieval": [{"x": 1}]})

    assert resp.status_code == 500
    assert resp.get_json() == {"error": "Inference failed"}
