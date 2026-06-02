"""Tests for MST-classification/dicom_converter.py.

dicom_converter.py imports dicom_utils at module level (not lazy),
and dicom_utils imports SimpleITK at module level.
We stub SimpleITK (via sitk_stub fixture) and dicom_utils before importing
dicom_converter.
"""
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True)
def _stub_dicom_utils(sitk_stub, monkeypatch) -> Iterator[ModuleType]:
    du_stub = types.ModuleType("dicom_utils")
    du_stub.dicom_to_nifti = MagicMock(return_value="/tmp/mri_series.nii.gz")
    du_stub.dicom_to_nifti_subtraction = MagicMock(return_value="/tmp/mri_subtraction.nii.gz")
    du_stub.compute_subtraction_from_nifti = MagicMock(return_value="/tmp/mri_subtraction.nii.gz")
    monkeypatch.setitem(sys.modules, "dicom_utils", du_stub)

    sys.modules.pop("dicom_converter", None)
    yield du_stub


def test_convert_series_to_nifti_calls_dicom_to_nifti(tmp_path, _stub_dicom_utils):
    nifti_path = tmp_path / "mri_series.nii.gz"
    nifti_path.touch()
    _stub_dicom_utils.dicom_to_nifti.return_value = str(nifti_path)

    import dicom_converter
    result = dicom_converter.convert_series_to_nifti(tmp_path)

    _stub_dicom_utils.dicom_to_nifti.assert_called_once_with(str(tmp_path))
    assert result == nifti_path


def test_convert_series_to_nifti_returns_path_object(tmp_path, _stub_dicom_utils):
    nifti_path = tmp_path / "mri_series.nii.gz"
    nifti_path.touch()
    _stub_dicom_utils.dicom_to_nifti.return_value = str(nifti_path)

    import dicom_converter
    result = dicom_converter.convert_series_to_nifti(tmp_path)
    assert isinstance(result, Path)


def test_convert_series_to_nifti_raises_if_file_not_created(tmp_path, _stub_dicom_utils):
    _stub_dicom_utils.dicom_to_nifti.return_value = str(tmp_path / "nonexistent.nii.gz")

    import dicom_converter
    with pytest.raises(ValueError, match="NIfTI file was not created"):
        dicom_converter.convert_series_to_nifti(tmp_path)


def test_convert_multiphase_to_subtraction_nifti_calls_subtraction(tmp_path, _stub_dicom_utils):
    nifti_path = tmp_path / "mri_subtraction.nii.gz"
    nifti_path.touch()
    _stub_dicom_utils.dicom_to_nifti_subtraction.return_value = str(nifti_path)

    import dicom_converter
    result = dicom_converter.convert_multiphase_to_subtraction_nifti(tmp_path)

    _stub_dicom_utils.dicom_to_nifti_subtraction.assert_called_once_with(str(tmp_path))
    assert result == nifti_path


def test_convert_multiphase_to_subtraction_nifti_raises_if_file_missing(tmp_path, _stub_dicom_utils):
    _stub_dicom_utils.dicom_to_nifti_subtraction.return_value = str(tmp_path / "missing.nii.gz")

    import dicom_converter
    with pytest.raises(ValueError, match="Subtraction NIfTI was not created"):
        dicom_converter.convert_multiphase_to_subtraction_nifti(tmp_path)


def test_compute_subtraction_nifti_calls_compute_func(tmp_path, _stub_dicom_utils):
    nifti_path = tmp_path / "mri_subtraction.nii.gz"
    nifti_path.touch()
    _stub_dicom_utils.compute_subtraction_from_nifti.return_value = str(nifti_path)

    pre = tmp_path / "pre.nii.gz"
    post = tmp_path / "post.nii.gz"

    import dicom_converter
    result = dicom_converter.compute_subtraction_nifti(pre, post)

    _stub_dicom_utils.compute_subtraction_from_nifti.assert_called_once_with(str(pre), str(post))
    assert result == nifti_path


def test_compute_subtraction_nifti_raises_if_file_missing(tmp_path, _stub_dicom_utils):
    _stub_dicom_utils.compute_subtraction_from_nifti.return_value = str(tmp_path / "missing.nii.gz")

    import dicom_converter
    with pytest.raises(ValueError, match="Subtraction NIfTI was not created"):
        dicom_converter.compute_subtraction_nifti(tmp_path / "pre.nii.gz", tmp_path / "post.nii.gz")


# ---------------------------------------------------------------------------
# M3: wrapper catches non-ValueError underlying exceptions and re-raises as
# ValueError("...: <orig msg>") — pins the error-translation contract.
# ---------------------------------------------------------------------------

def _raise_runtime(msg):
    def _r(*a, **kw):
        raise RuntimeError(msg)
    return _r


def test_convert_series_to_nifti_translates_runtime_error_to_value_error(monkeypatch, tmp_path):
    import dicom_converter
    monkeypatch.setattr(dicom_converter, "dicom_to_nifti", _raise_runtime("low-level sitk failure"))
    with pytest.raises(ValueError, match="Conversion failed: low-level sitk failure"):
        dicom_converter.convert_series_to_nifti(tmp_path)


def test_convert_multiphase_to_subtraction_translates_runtime_error(monkeypatch, tmp_path):
    import dicom_converter
    monkeypatch.setattr(dicom_converter, "dicom_to_nifti_subtraction", _raise_runtime("phase mismatch"))
    with pytest.raises(ValueError, match="Multi-phase conversion failed: phase mismatch"):
        dicom_converter.convert_multiphase_to_subtraction_nifti(tmp_path)


def test_compute_subtraction_nifti_translates_runtime_error(monkeypatch, tmp_path):
    import dicom_converter
    monkeypatch.setattr(dicom_converter, "compute_subtraction_from_nifti", _raise_runtime("io error"))
    with pytest.raises(ValueError, match="Subtraction failed: io error"):
        dicom_converter.compute_subtraction_nifti(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz")
