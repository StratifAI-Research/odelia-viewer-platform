"""Tests for breast-cancer-classification/dicom_converter.py — convert_to_unilateral_nifti.

dicom_converter.py imports dicom2nfti_onthefly at module level,
which in turn imports torchio and torch at module level.
We stub both plus dicom2nfti_onthefly itself before importing dicom_converter.
"""
import sys
import types
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stub_dicom2nfti(torch_stub, torchio_stub, monkeypatch) -> Iterator[ModuleType]:
    """Stub out dicom2nfti_onthefly before importing dicom_converter."""
    d2n_stub = types.ModuleType("dicom2nfti_onthefly")
    d2n_stub.dicom_to_unilateral_nifti = MagicMock(return_value={
        "Pre_left": MagicMock(),
        "Pre_right": MagicMock(),
        "Post_1_left": MagicMock(),
        "Post_1_right": MagicMock(),
    })
    monkeypatch.setitem(sys.modules, "dicom2nfti_onthefly", d2n_stub)
    sys.modules.pop("dicom_converter", None)
    yield d2n_stub
    sys.modules.pop("dicom_converter", None)


def test_convert_to_unilateral_nifti_calls_dicom_to_unilateral_nifti(tmp_path, _stub_dicom2nfti):
    import dicom_converter
    result = dicom_converter.convert_to_unilateral_nifti(tmp_path)
    _stub_dicom2nfti.dicom_to_unilateral_nifti.assert_called_once_with(tmp_path)


def test_convert_to_unilateral_nifti_returns_dict(tmp_path, _stub_dicom2nfti):
    import dicom_converter
    result = dicom_converter.convert_to_unilateral_nifti(tmp_path)
    assert isinstance(result, dict)


def test_convert_to_unilateral_nifti_has_expected_keys(tmp_path, _stub_dicom2nfti):
    import dicom_converter
    result = dicom_converter.convert_to_unilateral_nifti(tmp_path)
    assert "Pre_left" in result
    assert "Pre_right" in result
    assert "Post_1_left" in result
    assert "Post_1_right" in result


def test_convert_to_unilateral_nifti_raises_on_empty_result(tmp_path, _stub_dicom2nfti):
    _stub_dicom2nfti.dicom_to_unilateral_nifti.return_value = {}
    import dicom_converter
    with pytest.raises(ValueError, match="No NIfTI images were created"):
        dicom_converter.convert_to_unilateral_nifti(tmp_path)


def test_convert_to_unilateral_nifti_wraps_exceptions(tmp_path, _stub_dicom2nfti):
    _stub_dicom2nfti.dicom_to_unilateral_nifti.side_effect = RuntimeError("DICOM read failed")
    import dicom_converter
    with pytest.raises(ValueError, match="Conversion failed"):
        dicom_converter.convert_to_unilateral_nifti(tmp_path)


def test_convert_to_unilateral_nifti_raised_is_value_error(tmp_path, _stub_dicom2nfti):
    """The original exception must be chained as __cause__."""
    _stub_dicom2nfti.dicom_to_unilateral_nifti.side_effect = RuntimeError("boom")
    import dicom_converter
    with pytest.raises(ValueError) as exc_info:
        dicom_converter.convert_to_unilateral_nifti(tmp_path)
    assert exc_info.value.__cause__ is not None
