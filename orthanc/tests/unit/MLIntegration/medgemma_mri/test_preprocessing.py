"""Tests for medgemma-mri/preprocessing.py — extract_central_slices and helpers.

preprocessing.py imports SimpleITK, numpy, PIL, and pathlib at module level.
SimpleITK is stubbed via sitk_stub; PIL and numpy are available in the venv.

extract_central_slices calls read_dicom_volume which first checks for real
.dcm files on disk.  To avoid file-system fixtures, we patch read_dicom_volume
at the preprocessing-module level and supply a numpy array directly.

Uncovered paths (intentional, documented here):
- read_dicom_volume 4D branch: requires per-file temporal-position metadata;
  deferred to integration tests.
- get_dicom_metadata individual tag branches (TriggerTime, ImagePositionPatient):
  smoke-tested via module import.
- read_dicom_volume itself is only smoke-tested here (error-on-missing-dcm);
  full path needs real DICOM files.
"""
from typing import Iterator
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _evict_module(sitk_stub) -> Iterator[None]:
    sys.modules.pop('preprocessing', None)
    yield
    sys.modules.pop('preprocessing', None)


def test_preprocessing_imports_with_stubs(sitk_stub):
    pass  # should not raise


def test_normalize_slice_standard_input():
    import preprocessing
    arr = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)
    result = preprocessing.normalize_slice(arr)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_normalize_slice_constant_returns_zeros():
    import preprocessing
    arr = np.full((10, 10), 42.0, dtype=np.float32)
    result = preprocessing.normalize_slice(arr)
    assert (result == 0).all()


def test_normalize_slice_output_shape_preserved():
    import preprocessing
    arr = np.random.rand(20, 30).astype(np.float32)
    result = preprocessing.normalize_slice(arr)
    assert result.shape == (20, 30)


def test_normalize_slice_uses_percentile_windowing():
    """Outlier at (0,0) must saturate to 255; in-range values normalize to [0, <255]."""
    import preprocessing
    arr = np.zeros(100, dtype=np.float32)
    arr[0] = 1000.0   # massive outlier at position (0,0) after reshape
    arr[1:] = np.linspace(0, 10, 99)
    arr = arr.reshape(10, 10)
    result = preprocessing.normalize_slice(arr)
    assert result.dtype == np.uint8
    assert result[0, 0] == 255              # outlier clipped to p_high -> normalizes to max
    assert (result == 255).sum() == 1       # only the outlier saturates; bulk wasn't all-clipped
    assert result.min() == 0                # lowest in-range value normalizes to 0


def test_read_dicom_volume_raises_on_missing_dcm(tmp_path):
    import preprocessing
    # tmp_path exists but has no .dcm files
    with pytest.raises(ValueError, match='No DICOM files found'):
        preprocessing.read_dicom_volume(tmp_path)


def _patch_read_dicom_volume(volume_array, sitk_stub, monkeypatch):
    """Return a context manager patching read_dicom_volume in the preprocessing module."""
    mock_img = MagicMock()
    mock_img.GetSize.return_value = (volume_array.shape[2], volume_array.shape[1], volume_array.shape[0])
    mock_img.GetSpacing.return_value = (1.0, 1.0, 2.0)
    monkeypatch.setattr(sitk_stub, 'GetArrayFromImage', MagicMock(return_value=volume_array))
    return patch('preprocessing.read_dicom_volume', return_value=mock_img)


def test_extract_central_slices_returns_list_of_pil_images(sitk_stub, monkeypatch):
    from PIL import Image as PILImage
    import preprocessing

    vol = np.ones((10, 64, 64), dtype=np.float32) * 50.0
    mock_img = MagicMock()
    monkeypatch.setattr(sitk_stub, 'GetArrayFromImage', MagicMock(return_value=vol))

    with patch('preprocessing.read_dicom_volume', return_value=mock_img):
        result = preprocessing.extract_central_slices('/fake/dicom', num_slices=3)

    assert isinstance(result, list)
    assert len(result) == 3
    for img in result:
        assert isinstance(img, PILImage.Image)
        assert img.mode == 'RGB'


def test_extract_central_slices_returns_correct_count(sitk_stub, monkeypatch):
    import preprocessing

    vol = np.ones((10, 64, 64), dtype=np.float32)
    monkeypatch.setattr(sitk_stub, 'GetArrayFromImage', MagicMock(return_value=vol))

    with patch('preprocessing.read_dicom_volume', return_value=MagicMock()):
        result = preprocessing.extract_central_slices('/fake/dicom', num_slices=5)
    assert len(result) == 5


def test_extract_central_slices_fewer_slices_than_volume(sitk_stub, monkeypatch):
    """When volume has fewer slices than requested, num_slices is clamped."""
    import preprocessing

    vol = np.ones((10, 64, 64), dtype=np.float32)
    monkeypatch.setattr(sitk_stub, 'GetArrayFromImage', MagicMock(return_value=vol))

    with patch('preprocessing.read_dicom_volume', return_value=MagicMock()):
        result = preprocessing.extract_central_slices('/fake/dicom', num_slices=20)
    assert len(result) == 10


def test_extract_central_slices_single_slice(sitk_stub, monkeypatch):
    import preprocessing

    vol = np.ones((10, 64, 64), dtype=np.float32)
    monkeypatch.setattr(sitk_stub, 'GetArrayFromImage', MagicMock(return_value=vol))

    with patch('preprocessing.read_dicom_volume', return_value=MagicMock()):
        result = preprocessing.extract_central_slices('/fake/dicom', num_slices=1)
    assert len(result) == 1


def test_extract_central_slices_image_size(sitk_stub, monkeypatch):
    """Each PIL image has dimensions matching the volume's Y, X axes."""
    import preprocessing

    vol = np.ones((10, 64, 64), dtype=np.float32)
    monkeypatch.setattr(sitk_stub, 'GetArrayFromImage', MagicMock(return_value=vol))

    with patch('preprocessing.read_dicom_volume', return_value=MagicMock()):
        result = preprocessing.extract_central_slices('/fake/dicom', num_slices=1)
    assert result[0].size == (64, 64)


# ---------------------------------------------------------------------------
# G2: get_dicom_metadata explicit branch coverage + read_dicom_volume 4D path
# ---------------------------------------------------------------------------

def test_get_dicom_metadata_extracts_temporal_position_from_tag_0020_0100(sitk_stub, monkeypatch):
    """When 0020|0100 (TemporalPositionIdentifier) is present, it is read first."""
    import preprocessing
    reader = MagicMock()
    reader.HasMetaDataKey.side_effect = lambda k: k in {"0020|0100", "0020|1041", "0020|0013"}
    reader.GetMetaData.side_effect = lambda k: {
        "0020|0100": "3", "0020|1041": "5.5", "0020|0013": "7",
    }[k]
    monkeypatch.setattr(sitk_stub, "ImageFileReader", MagicMock(return_value=reader))
    temporal, loc, num = preprocessing.get_dicom_metadata("/fake/file.dcm")
    assert temporal == 3
    assert loc == 5.5
    assert num == 7


def test_get_dicom_metadata_falls_back_to_trigger_time_0018_1060(sitk_stub, monkeypatch):
    """When 0020|0100 is missing but 0018|1060 (TriggerTime) is present, it is used."""
    import preprocessing
    reader = MagicMock()
    reader.HasMetaDataKey.side_effect = lambda k: k == "0018|1060"
    reader.GetMetaData.side_effect = lambda k: {"0018|1060": "42.5"}[k]
    monkeypatch.setattr(sitk_stub, "ImageFileReader", MagicMock(return_value=reader))
    temporal, _, _ = preprocessing.get_dicom_metadata("/fake/file.dcm")
    assert temporal == 42  # int(float("42.5"))


def test_get_dicom_metadata_falls_back_to_image_position_patient_for_z(sitk_stub, monkeypatch):
    """When 0020|1041 is missing but 0020|0032 is present, use the Z component of IPP."""
    import preprocessing
    reader = MagicMock()
    reader.HasMetaDataKey.side_effect = lambda k: k == "0020|0032"
    reader.GetMetaData.side_effect = lambda k: {"0020|0032": "1.0\\2.0\\3.5"}[k]
    monkeypatch.setattr(sitk_stub, "ImageFileReader", MagicMock(return_value=reader))
    _, slice_loc, _ = preprocessing.get_dicom_metadata("/fake/file.dcm")
    assert slice_loc == 3.5


def test_get_dicom_metadata_returns_zero_tuple_on_reader_failure(sitk_stub, monkeypatch):
    """If the reader raises (e.g. unreadable file), return (0, 0.0, 0) sentinel."""
    import preprocessing
    reader = MagicMock()
    reader.ReadImageInformation.side_effect = RuntimeError("unreadable")
    monkeypatch.setattr(sitk_stub, "ImageFileReader", MagicMock(return_value=reader))
    out = preprocessing.get_dicom_metadata("/fake/file.dcm")
    assert out == (0, 0.0, 0)


def test_read_dicom_volume_extracts_first_temporal_phase_when_multiple_phases(tmp_path, sitk_stub, monkeypatch):
    """4D branch: two temporal positions present -> select the lowest key, use only its files."""
    import preprocessing
    # Create two empty .dcm files; metadata is supplied by the patched helper.
    f0 = tmp_path / "phase0.dcm"
    f1 = tmp_path / "phase1.dcm"
    f0.write_bytes(b"")
    f1.write_bytes(b"")

    metadata_for = {str(f0): (0, 0.0, 0), str(f1): (1, 0.0, 1)}
    monkeypatch.setattr(preprocessing, "get_dicom_metadata",
                         lambda p: metadata_for[str(p)])

    captured = []
    series_reader = MagicMock()
    def _set_files(files):
        captured.append(list(files))
    series_reader.SetFileNames.side_effect = _set_files
    series_reader.Execute.return_value = MagicMock()
    monkeypatch.setattr(sitk_stub, "ImageSeriesReader", MagicMock(return_value=series_reader))

    preprocessing.read_dicom_volume(tmp_path)

    # Only the lower temporal phase (key=0 = phase0.dcm) should have been read.
    assert len(captured) == 1
    assert captured[0] == [str(f0)]
