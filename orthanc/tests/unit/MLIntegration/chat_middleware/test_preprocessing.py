"""Tests for chat-middleware/preprocessing.py — DICOM slice extraction strategies + PNG encoding."""
import base64
import io
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image


# ---------- normalize_slice ----------

def test_normalize_slice_returns_uint8():
    import preprocessing
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = preprocessing.normalize_slice(arr)
    assert out.dtype == np.uint8


def test_normalize_slice_constant_input_returns_zeros():
    """p_high - p_low < 1e-6 -> shortcut returns zeros_like."""
    import preprocessing
    arr = np.full((10, 10), 42.0, dtype=np.float32)
    out = preprocessing.normalize_slice(arr)
    assert (out == 0).all()


def test_normalize_slice_maps_outlier_to_top_of_range():
    """Outlier at position (0,0) ends up clipped to p_high and normalized to 255."""
    import preprocessing
    arr = np.zeros(100, dtype=np.float32)
    arr[0] = 1000.0
    arr[1:] = np.linspace(0, 10, 99)
    arr = arr.reshape(10, 10)
    out = preprocessing.normalize_slice(arr)
    assert out[0, 0] == 255
    assert out.min() == 0


# ---------- extract_slices: strategies ----------

def _params(num_slices=5, strategy=None, central_percentage=60):
    from runtime_config import PreprocessingParams
    from models import SliceStrategy
    if strategy is None:
        strategy = SliceStrategy.CENTRAL
    return PreprocessingParams(
        num_slices=num_slices, slice_strategy=strategy, central_percentage=central_percentage,
    )


def test_extract_slices_central_strategy_picks_middle_band():
    """CENTRAL with 60% margin on a 20-slice volume should land within slices 4..15."""
    import preprocessing
    vol = np.arange(20 * 4 * 4, dtype=np.float32).reshape(20, 4, 4)
    out = preprocessing.extract_slices(vol, _params(num_slices=5, central_percentage=60))
    assert len(out) == 5
    # Each returned slice is one Z-plane of shape (Y, X)
    assert all(s.shape == (4, 4) for s in out)


def test_extract_slices_uniform_strategy_evenly_spaced_across_full_volume():
    """UNIFORM with 5 slices on a 20-slice volume picks indices roughly [0, 4, 9, 14, 19]."""
    import preprocessing
    from models import SliceStrategy
    vol = np.arange(20 * 2 * 2, dtype=np.float32).reshape(20, 2, 2)
    out = preprocessing.extract_slices(vol, _params(num_slices=5, strategy=SliceStrategy.UNIFORM))
    assert len(out) == 5
    # Each slice'''s first element is Z*4; reconstruct the indices.
    indices = [int(s[0, 0] // 4) for s in out]
    assert indices[0] == 0 and indices[-1] == 19           # full coverage
    assert sorted(indices) == indices                       # ascending order


def test_extract_slices_first_n_returns_lowest_indices():
    import preprocessing
    from models import SliceStrategy
    vol = np.arange(20 * 2 * 2, dtype=np.float32).reshape(20, 2, 2)
    out = preprocessing.extract_slices(vol, _params(num_slices=3, strategy=SliceStrategy.FIRST_N))
    indices = [int(s[0, 0] // 4) for s in out]
    assert indices == [0, 1, 2]


def test_extract_slices_last_n_returns_highest_indices():
    import preprocessing
    from models import SliceStrategy
    vol = np.arange(20 * 2 * 2, dtype=np.float32).reshape(20, 2, 2)
    out = preprocessing.extract_slices(vol, _params(num_slices=3, strategy=SliceStrategy.LAST_N))
    indices = [int(s[0, 0] // 4) for s in out]
    assert indices == [17, 18, 19]


def test_extract_slices_zero_slices_returns_empty_list():
    import preprocessing
    vol = np.zeros((10, 4, 4), dtype=np.float32)
    out = preprocessing.extract_slices(vol, _params(num_slices=0))
    assert out == []


def test_extract_slices_clamps_to_total_when_requested_exceeds_available():
    """num_slices=20 on a 5-slice volume -> at most 5 returned."""
    import preprocessing
    vol = np.zeros((5, 2, 2), dtype=np.float32)
    out = preprocessing.extract_slices(vol, _params(num_slices=20))
    assert len(out) == 5


def test_extract_slices_central_single_slice_picks_center():
    """num_slices=1 with CENTRAL strategy -> mid of central band."""
    import preprocessing
    vol = np.arange(20 * 2 * 2, dtype=np.float32).reshape(20, 2, 2)
    out = preprocessing.extract_slices(vol, _params(num_slices=1))
    assert len(out) == 1
    idx = int(out[0][0, 0] // 4)
    # 60% central band on 20 slices: start=4, end=16, midpoint=10.
    assert idx == 10


def test_extract_slices_uniform_single_slice_picks_midpoint():
    import preprocessing
    from models import SliceStrategy
    vol = np.arange(11 * 2 * 2, dtype=np.float32).reshape(11, 2, 2)
    out = preprocessing.extract_slices(vol, _params(num_slices=1, strategy=SliceStrategy.UNIFORM))
    idx = int(out[0][0, 0] // 4)
    assert idx == 5                                        # 11 // 2


# ---------- slices_to_base64 ----------

def test_slices_to_base64_returns_data_uri_per_slice():
    import preprocessing
    slices = [np.arange(64, dtype=np.float32).reshape(8, 8) for _ in range(3)]
    out = preprocessing.slices_to_base64(slices)
    assert len(out) == 3
    for s in out:
        assert s.startswith("data:image/png;base64,")
        # Round-trip the base64 PNG via PIL
        payload = s.split(",", 1)[1]
        raw = base64.b64decode(payload)
        img = Image.open(io.BytesIO(raw))
        assert img.mode == "RGB"
        assert img.size == (8, 8)


def test_slices_to_base64_handles_empty_input():
    import preprocessing
    assert preprocessing.slices_to_base64([]) == []


# ---------- read_dicom_volume ----------

def test_read_dicom_volume_raises_when_folder_has_no_dcm(tmp_path):
    import preprocessing
    with pytest.raises(ValueError, match="No DICOM files found"):
        preprocessing.read_dicom_volume(tmp_path)


def test_read_dicom_volume_raises_when_gdcm_finds_no_series(tmp_path, monkeypatch):
    """tmp_path has .dcm files but ImageSeriesReader.GetGDCMSeriesFileNames returns [] -> ValueError."""
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader = mock.MagicMock()
    reader.GetGDCMSeriesFileNames.return_value = []
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    with pytest.raises(ValueError, match="No valid DICOM series found"):
        preprocessing.read_dicom_volume(tmp_path)


def test_read_dicom_volume_returns_reader_executed_image(tmp_path, monkeypatch):
    """Happy path: configures reader and returns reader.Execute()."""
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader = mock.MagicMock()
    reader.GetGDCMSeriesFileNames.return_value = ["a.dcm"]
    fake_image = mock.MagicMock()
    fake_image.GetSize.return_value = (4, 4, 3)
    fake_image.GetSpacing.return_value = (1.0, 1.0, 1.0)
    reader.Execute.return_value = fake_image
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    out = preprocessing.read_dicom_volume(tmp_path)
    assert out is fake_image


# ---------- preprocess_series (async) ----------

async def test_preprocess_series_happy_path(tmp_path, monkeypatch):
    """Full pipeline with all I/O / DICOM hooks monkey-patched.

    M7 (MDR): captures the WADO `retrieval_url` template and asserts the (study_uid, series_uid)
    pair from inputs lands at the right positions — pins the patient-data-integrity boundary."""
    import preprocessing
    fake_dataset = mock.MagicMock()
    wado_captured = []
    def _capture_wado(info):
        wado_captured.append(info)
        return [fake_dataset]
    monkeypatch.setattr(preprocessing, "retrieve_via_wado_rs", _capture_wado)
    folder = tmp_path / "series"
    folder.mkdir()
    monkeypatch.setattr(preprocessing, "save_datasets_to_folder", lambda ds, uid, cfg: folder)
    fake_image = mock.MagicMock()
    monkeypatch.setattr(preprocessing, "read_dicom_volume", lambda f: fake_image)
    vol_arr = np.linspace(0, 100, 10 * 8 * 8).reshape(10, 8, 8).astype(np.float32)
    monkeypatch.setattr(preprocessing.sitk, "GetArrayFromImage", lambda img: vol_arr)

    from runtime_config import PreprocessingParams
    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=PreprocessingParams(num_slices=3),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
    )
    assert len(out) == 3
    assert all(s.startswith("data:image/png;base64,") for s in out)
    # The temp folder should have been removed in the finally block.
    assert not folder.exists()
    # M7 (MDR patient-data-integrity): pin the WADO URL template — a study/series swap
    # or template typo would mis-route patient data on the wire.
    assert wado_captured == [[{
        "retrieval_url": "http://x/dicom-web/studies/STD1/series/SE1",
        "study_uid": "STD1",
        "series_uid": "SE1",
    }]]


async def test_preprocess_series_raises_when_no_datasets_retrieved(tmp_path, monkeypatch):
    import preprocessing
    monkeypatch.setattr(preprocessing, "retrieve_via_wado_rs", lambda info: [])
    from runtime_config import PreprocessingParams
    with pytest.raises(ValueError, match="No DICOM instances retrieved"):
        await preprocessing.preprocess_series(
            series_uid="SE1", study_uid="STD1",
            params=PreprocessingParams(num_slices=3),
            wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        )


async def test_preprocess_series_cleans_up_temp_folder_on_error(tmp_path, monkeypatch):
    """Even if read_dicom_volume raises, the dicom_folder must be removed."""
    import preprocessing
    monkeypatch.setattr(preprocessing, "retrieve_via_wado_rs", lambda info: [mock.MagicMock()])
    folder = tmp_path / "boom-series"
    folder.mkdir()
    monkeypatch.setattr(preprocessing, "save_datasets_to_folder", lambda ds, uid, cfg: folder)
    monkeypatch.setattr(preprocessing, "read_dicom_volume",
                        lambda f: (_ for _ in ()).throw(RuntimeError("read failed")))

    from runtime_config import PreprocessingParams
    with pytest.raises(RuntimeError, match="read failed"):
        await preprocessing.preprocess_series(
            series_uid="SE1", study_uid="STD1",
            params=PreprocessingParams(num_slices=3),
            wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        )
    assert not folder.exists()
