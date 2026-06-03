"""M1/M2 — dicom_utils tests using REAL SimpleITK (no _stub_sitk_and_evict).

The peer file test_dicom_utils.py uses an autouse fixture that monkeypatches
SimpleITK -> stub. To exercise the actual 2-phase subtraction math and the
multi-phase dicom_to_nifti branch, those tests live here in a separate module
where the autouse stub does not apply.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# NOTE: no `@pytest.fixture(autouse=True) def _stub_sitk_*` in this file —
# we want the real SimpleITK from the test venv.


def _evict_dicom_utils():
    sys.modules.pop("dicom_utils", None)


def test_dicom_to_nifti_subtraction_success_writes_shifted_uint16_nifti(tmp_path, monkeypatch):
    """2-phase success path: writes the (post - pre) shifted-to-non-negative volume as uint16."""
    import SimpleITK as sitk
    _evict_dicom_utils()
    import dicom_utils

    pre_arr = np.array([[[1.0, 2.0]]], dtype=np.float32)
    post_arr = np.array([[[3.0, 5.0]]], dtype=np.float32)
    pre_img = sitk.GetImageFromArray(pre_arr)
    post_img = sitk.GetImageFromArray(post_arr)

    monkeypatch.setattr(
        dicom_utils, "_gather_temporal_groups",
        lambda folder: (tmp_path, {0: [{"path": "p0.dcm"}], 1: [{"path": "p1.dcm"}]}),
    )
    reads = []
    def _fake_read(meta_list):
        reads.append(meta_list)
        return pre_img if len(reads) == 1 else post_img
    monkeypatch.setattr(dicom_utils, "_read_temporal_group", _fake_read)

    result_path = dicom_utils.dicom_to_nifti_subtraction(str(tmp_path))

    assert Path(result_path).exists()
    sub_arr = sitk.GetArrayFromImage(sitk.ReadImage(result_path))
    # post - pre = [2, 3]; shifted (- min = -2): [0, 1]
    assert sub_arr.dtype == np.uint16
    assert sub_arr.shape == (1, 1, 2)
    assert int(sub_arr[0, 0, 0]) == 0
    assert int(sub_arr[0, 0, 1]) == 1


def test_dicom_to_nifti_subtraction_raises_with_single_phase(tmp_path, monkeypatch):
    """If _gather_temporal_groups returns only 1 phase, subtraction raises ValueError."""
    _evict_dicom_utils()
    import dicom_utils
    monkeypatch.setattr(
        dicom_utils, "_gather_temporal_groups",
        lambda folder: (tmp_path, {0: [{"path": "single.dcm"}]}),
    )
    with pytest.raises(ValueError, match="at least 2 temporal phases"):
        dicom_utils.dicom_to_nifti_subtraction(str(tmp_path))


def test_dicom_to_nifti_multi_phase_extracts_first_phase_only(tmp_path, monkeypatch):
    """Multi-phase dicom_to_nifti path: reads only the lowest temporal_pos group."""
    import SimpleITK as sitk
    _evict_dicom_utils()
    import dicom_utils

    img = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
    monkeypatch.setattr(
        dicom_utils, "_gather_temporal_groups",
        lambda folder: (tmp_path, {0: [{"path": "p0.dcm"}], 1: [{"path": "p1.dcm"}]}),
    )
    captured = []
    def _fake_read(meta_list):
        captured.append(meta_list)
        return img
    monkeypatch.setattr(dicom_utils, "_read_temporal_group", _fake_read)

    result_path = dicom_utils.dicom_to_nifti(str(tmp_path))

    assert Path(result_path).exists()
    assert len(captured) == 1                   # only the first temporal phase was read
    assert captured[0] == [{"path": "p0.dcm"}]
