"""Tests for MST-classification/dicom_utils.py.

SimpleITK is stubbed via the shared sitk_stub fixture (provided by
tests/unit/MLIntegration/conftest.py) before any import of dicom_utils,
since it imports `import SimpleITK as sitk` at module level and uses
sitk.Image as a type annotation at parse time.

Coverage note: the 2-phase subtraction *success* path in dicom_to_nifti_subtraction
(dicom_utils.py lines 90-101) and compute_subtraction_from_nifti array math
(lines 187-204) are intentionally deferred — they require a full multi-phase
DICOM fixture set.  See _compute_subtraction_array for the array-level test
that covers the arithmetic in isolation.
"""
import sys
from types import ModuleType
from typing import Iterator
from unittest.mock import MagicMock
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _stub_sitk_and_evict(sitk_stub, monkeypatch) -> Iterator[ModuleType]:
    """Wire the shared sitk_stub and evict the cached dicom_utils module."""
    sys.modules.pop("dicom_utils", None)
    yield sitk_stub


def test_dicom_to_nifti_single_phase_calls_write_image(tmp_path, _stub_sitk_and_evict):
    import dicom_utils
    for i in range(3):
        (tmp_path / f"{i}.dcm").write_bytes(b"fake")
    result = dicom_utils.dicom_to_nifti(str(tmp_path))
    import SimpleITK as sitk
    assert sitk.WriteImage.called
    assert result.endswith(".nii.gz")


def test_dicom_to_nifti_returns_string(tmp_path, _stub_sitk_and_evict):
    import dicom_utils
    for i in range(2):
        (tmp_path / f"{i}.dcm").write_bytes(b"fake")
    result = dicom_utils.dicom_to_nifti(str(tmp_path))
    assert isinstance(result, str)


def test_dicom_to_nifti_raises_on_empty_folder(tmp_path, _stub_sitk_and_evict):
    import dicom_utils
    with pytest.raises(ValueError, match="No DICOM files"):
        dicom_utils.dicom_to_nifti(str(tmp_path))


def test_dicom_to_nifti_subtraction_raises_with_single_phase(tmp_path, _stub_sitk_and_evict):
    import dicom_utils
    (tmp_path / "a.dcm").write_bytes(b"fake")
    with pytest.raises(ValueError, match="at least 2 temporal phases"):
        dicom_utils.dicom_to_nifti_subtraction(str(tmp_path))


def test_compute_subtraction_from_nifti_calls_read_image(tmp_path, _stub_sitk_and_evict):
    import dicom_utils
    import SimpleITK as sitk
    pre = str(tmp_path / "pre.nii.gz")
    post = str(tmp_path / "post.nii.gz")
    out = str(tmp_path / "sub.nii.gz")
    result = dicom_utils.compute_subtraction_from_nifti(pre, post, out)
    assert sitk.ReadImage.call_count == 2
    assert sitk.WriteImage.called
    assert result == out


def test_compute_subtraction_from_nifti_default_output_path(tmp_path, _stub_sitk_and_evict):
    import dicom_utils
    pre = str(tmp_path / "pre.nii.gz")
    post = str(tmp_path / "post.nii.gz")
    result = dicom_utils.compute_subtraction_from_nifti(pre, post)
    assert "mri_subtraction.nii.gz" in result
    assert str(tmp_path) in result


def test_compute_subtraction_array_shifts_to_non_negative_origin(_stub_sitk_and_evict):
    """Production does sub - sub.min() (shift), not clipping.
    For pre=[[10,20],[30,40]], post=[[5,25],[25,45]] -> raw=[[-5,5],[-5,5]] -> shifted=[[0,10],[0,10]]."""
    import dicom_utils
    import SimpleITK as sitk
    pre_arr = np.array([[[10, 20], [30, 40]]], dtype=np.float32)
    post_arr = np.array([[[5, 25], [25, 45]]], dtype=np.float32)
    sitk.GetArrayFromImage.side_effect = [pre_arr, post_arr]
    pre_img = MagicMock()
    post_img = MagicMock()
    result = dicom_utils._compute_subtraction_array(pre_img, post_img)
    assert result.min() == 0
    assert result.max() == 10
    assert result.dtype == np.uint16
