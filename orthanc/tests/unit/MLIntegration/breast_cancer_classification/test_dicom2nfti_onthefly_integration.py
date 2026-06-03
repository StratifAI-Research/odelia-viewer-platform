"""D3: dicom_to_unilateral_nifti integration with REAL torch + torchio.

The peer file test_dicom2nfti_onthefly.py has an autouse fixture that stubs torch
and torchio. To exercise the real CropOrPad / threshold / split logic of
dicom_to_unilateral_nifti, this test lives in its own module without that stub.
"""
import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _evict_module() -> Iterator[None]:
    """Ensure dicom2nfti_onthefly re-imports with REAL torch/torchio (no stub fixture used)."""
    sys.modules.pop("dicom2nfti_onthefly", None)
    yield
    sys.modules.pop("dicom2nfti_onthefly", None)


def _fake_dicom2nii_from_module(d_module):
    """Build a fake dicom2nii returning a real ScalarImage with a non-degenerate gradient."""
    import torch
    import torchio as tio
    def _impl(item, root):
        series_uid_str, _paths = item
        data = torch.linspace(0.1, 1.0, 1 * 64 * 64 * 8).reshape(1, 64, 64, 8)
        series_name = series_uid_str.split("_", 1)[1]
        return series_name, tio.ScalarImage(tensor=data, affine=torch.eye(4))
    return _impl


def test_dicom_to_unilateral_nifti_returns_left_right_per_sequence(tmp_path, monkeypatch):
    """End-to-end happy path: 2 trigger times (Pre/Post_1) x 2 slices each ->
    4 entries in the returned dict: Pre_left, Pre_right, Post_1_left, Post_1_right."""
    import torchio as tio
    import dicom2nfti_onthefly as d

    for name in ("pre_a.dcm", "pre_b.dcm", "post_a.dcm", "post_b.dcm"):
        (tmp_path / name).write_bytes(b"")

    def fake_read_metadata(args):
        path_dcm, root = args
        trigger = 0.0 if "pre" in path_dcm.name else 1000.0
        return {
            "SeriesInstanceUID": "SE1",
            "TriggerTime": trigger,
            "_Path": str(path_dcm.relative_to(root)),
        }
    monkeypatch.setattr(d, "read_metadata", fake_read_metadata)
    monkeypatch.setattr(d, "dicom2nii", _fake_dicom2nii_from_module(d))

    nifties = d.dicom_to_unilateral_nifti(tmp_path)

    keys = sorted(nifties.keys())
    assert keys == ["Post_1_left", "Post_1_right", "Pre_left", "Pre_right"]
    for img in nifties.values():
        assert isinstance(img, tio.ScalarImage)
        assert img.data.dim() == 4


def test_dicom_to_unilateral_nifti_writes_to_output_folder_when_provided(tmp_path, monkeypatch):
    """When nifti_output_folder is provided, .nii.gz files are written per side per sequence."""
    import dicom2nfti_onthefly as d

    for name in ("pre_a.dcm", "pre_b.dcm", "post_a.dcm", "post_b.dcm"):
        (tmp_path / name).write_bytes(b"")

    monkeypatch.setattr(d, "read_metadata", lambda args: {
        "SeriesInstanceUID": "SE1",
        "TriggerTime": 0.0 if "pre" in args[0].name else 1000.0,
        "_Path": str(args[0].relative_to(args[1])),
    })
    monkeypatch.setattr(d, "dicom2nii", _fake_dicom2nii_from_module(d))

    out = tmp_path / "nifti_out"
    d.dicom_to_unilateral_nifti(tmp_path, nifti_output_folder=out)

    written = sorted(p.name for p in out.iterdir())
    assert written == ["Post_1_left.nii.gz", "Post_1_right.nii.gz",
                       "Pre_left.nii.gz", "Pre_right.nii.gz"]


def test_dicom_to_unilateral_nifti_raises_when_sort_dyn_returns_none(tmp_path, monkeypatch):
    """Unequal slice counts -> sort_dyn returns None; downstream raises (no production guard).

    Pins the failure surface so a future production guard (returning empty / logging)
    surfaces here as a test break rather than silently changing behavior."""
    import dicom2nfti_onthefly as d

    # 2 pre + 1 post -> unequal trigger groups
    for name in ("pre_a.dcm", "pre_b.dcm", "post_a.dcm"):
        (tmp_path / name).write_bytes(b"")

    monkeypatch.setattr(d, "read_metadata", lambda args: {
        "SeriesInstanceUID": "SE1",
        "TriggerTime": 0.0 if "pre" in args[0].name else 1000.0,
        "_Path": str(args[0].relative_to(args[1])),
    })
    with pytest.raises((AttributeError, TypeError)):
        d.dicom_to_unilateral_nifti(tmp_path)
