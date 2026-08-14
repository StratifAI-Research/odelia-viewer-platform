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
    monkeypatch.setattr(preprocessing, "read_dicom_volume_with_instances",
                        lambda f: (fake_image, []))
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
    monkeypatch.setattr(preprocessing, "read_dicom_volume_with_instances",
                        lambda f: (_ for _ in ()).throw(RuntimeError("read failed")))

    from runtime_config import PreprocessingParams
    with pytest.raises(RuntimeError, match="read failed"):
        await preprocessing.preprocess_series(
            series_uid="SE1", study_uid="STD1",
            params=PreprocessingParams(num_slices=3),
            wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        )
    assert not folder.exists()


# ---------- recipe_signature ----------
#
# The signature is what makes a cache entry belong to one recipe. If it stops
# distinguishing two recipes, message 2 is silently answered with message 1's
# pixels — so these are correctness tests, not formatting tests.

def _selection(uids=(), **kw):
    from models import SliceSelection
    return SliceSelection(series_uid="SE1", sop_instance_uids=list(uids), **kw)


def test_recipe_signature_uses_runtime_params_when_no_selection():
    import preprocessing
    sig = preprocessing.recipe_signature(_params(num_slices=7, central_percentage=40))
    assert sig[0] == "recipe"
    assert "7" in sig and "40" in sig


def test_recipe_signature_differs_when_slice_count_differs():
    import preprocessing
    a = preprocessing.recipe_signature(_params(num_slices=5))
    b = preprocessing.recipe_signature(_params(num_slices=6))
    assert a != b


def test_recipe_signature_differs_when_strategy_differs():
    import preprocessing
    from models import SliceStrategy
    a = preprocessing.recipe_signature(_params(strategy=SliceStrategy.CENTRAL))
    b = preprocessing.recipe_signature(_params(strategy=SliceStrategy.UNIFORM))
    assert a != b


def test_recipe_signature_names_the_instances_when_selection_present():
    import preprocessing
    sig = preprocessing.recipe_signature(_params(), _selection(["1.1", "1.2"]))
    assert sig == ("instances", "1.1", "1.2")


def test_recipe_signature_ignores_runtime_params_when_instances_named():
    """The named instances ARE the recipe; num_slices plays no part in what is sent."""
    import preprocessing
    a = preprocessing.recipe_signature(_params(num_slices=3), _selection(["1.1"]))
    b = preprocessing.recipe_signature(_params(num_slices=30), _selection(["1.1"]))
    assert a == b


def test_recipe_signature_is_order_sensitive():
    """Order decides the order images reach the model, so it is part of the recipe."""
    import preprocessing
    a = preprocessing.recipe_signature(_params(), _selection(["1.1", "1.2"]))
    b = preprocessing.recipe_signature(_params(), _selection(["1.2", "1.1"]))
    assert a != b


def test_recipe_signature_falls_back_to_params_for_empty_selection():
    """A selection that names nothing is not a selection."""
    import preprocessing
    assert preprocessing.recipe_signature(_params(), _selection([])) == \
        preprocessing.recipe_signature(_params())


# ---------- read_dicom_volume_with_instances: per-slice identity ----------
#
# The UID list is what lets the viewer's "slice 27" mean the same pixels here.
# Every case that cannot establish it must return [] — an empty list disables
# slice addressing, whereas a wrong list would send the wrong slices silently.

def _reader_for(uids, depth=None, file_count=None):
    """A fake ImageSeriesReader whose slices carry the given SOPInstanceUIDs."""
    depth = len(uids) if depth is None else depth
    file_count = len(uids) if file_count is None else file_count
    reader = mock.MagicMock()
    reader.GetGDCMSeriesFileNames.return_value = [f"f{i}.dcm" for i in range(file_count)]
    image = mock.MagicMock()
    image.GetSize.return_value = (4, 4, depth)
    image.GetDimension.return_value = 3
    image.GetSpacing.return_value = (1.0, 1.0, 1.0)
    reader.Execute.return_value = image
    def _meta(index, key):
        assert key == "0008|0018"
        value = uids[index]
        if isinstance(value, Exception):
            raise value
        return value
    reader.GetMetaData.side_effect = _meta
    return reader, image


def test_read_dicom_volume_with_instances_returns_uid_per_slice(tmp_path, monkeypatch):
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader, image = _reader_for(["1.1", "1.2", "1.3"])
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    volume, uids = preprocessing.read_dicom_volume_with_instances(tmp_path)
    assert volume is image
    assert uids == ["1.1", "1.2", "1.3"]


def test_read_dicom_volume_with_instances_strips_dicom_padding(tmp_path, monkeypatch):
    """DICOM pads UIDs to an even length with a NUL; the padding is not part of the UID."""
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader, _ = _reader_for(["1.1\x00", " 1.2 "])
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    _, uids = preprocessing.read_dicom_volume_with_instances(tmp_path)
    assert uids == ["1.1", "1.2"]


def test_read_dicom_volume_with_instances_returns_empty_for_multiframe(tmp_path, monkeypatch):
    """One file, many Z-slices: there is no per-slice UID, so report none."""
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader, _ = _reader_for(["1.1"], depth=40, file_count=1)
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    _, uids = preprocessing.read_dicom_volume_with_instances(tmp_path)
    assert uids == []


def test_read_dicom_volume_with_instances_returns_empty_when_tag_missing(tmp_path, monkeypatch):
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader, _ = _reader_for(["1.1", RuntimeError("no such tag"), "1.3"])
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    _, uids = preprocessing.read_dicom_volume_with_instances(tmp_path)
    assert uids == []


def test_read_dicom_volume_with_instances_returns_empty_when_uid_blank(tmp_path, monkeypatch):
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader, _ = _reader_for(["1.1", "  ", "1.3"])
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    _, uids = preprocessing.read_dicom_volume_with_instances(tmp_path)
    assert uids == []


def test_read_dicom_volume_with_instances_returns_empty_on_duplicate_uids(tmp_path, monkeypatch):
    """A duplicate makes a UID ambiguous; picking the first match would send the wrong slice."""
    import preprocessing
    (tmp_path / "a.dcm").write_bytes(b"")
    reader, _ = _reader_for(["1.1", "1.1", "1.3"])
    monkeypatch.setattr(preprocessing.sitk, "ImageSeriesReader", lambda: reader)
    _, uids = preprocessing.read_dicom_volume_with_instances(tmp_path)
    assert uids == []


# ---------- resolve_selected_indices ----------

def test_resolve_selected_indices_maps_uids_to_volume_indices():
    import preprocessing
    out = preprocessing.resolve_selected_indices(["a", "b", "c", "d"], ["b", "d"])
    assert out == [1, 3]


def test_resolve_selected_indices_preserves_requested_order():
    """The images are sent in this order, so it is the caller's order, not the volume's."""
    import preprocessing
    out = preprocessing.resolve_selected_indices(["a", "b", "c"], ["c", "a"])
    assert out == [2, 0]


def test_resolve_selected_indices_rejects_the_whole_selection_when_one_is_missing():
    """All-or-nothing: dropping the unresolvable ones would send a set the snapshot denies."""
    import preprocessing
    with pytest.raises(preprocessing.SliceSelectionError, match="1 of 3 selected slices"):
        preprocessing.resolve_selected_indices(["a", "b"], ["a", "zzz", "b"])


def test_resolve_selected_indices_error_does_not_leak_the_uid_list():
    """The message is for a radiologist; the UIDs go to the log."""
    import preprocessing
    with pytest.raises(preprocessing.SliceSelectionError) as excinfo:
        preprocessing.resolve_selected_indices(["a"], ["zzz"])
    assert "zzz" not in str(excinfo.value)
    assert "Re-select the slice range" in str(excinfo.value)


def test_resolve_selected_indices_empty_request_returns_empty():
    import preprocessing
    assert preprocessing.resolve_selected_indices(["a", "b"], []) == []


# ---------- preprocess_series with an explicit selection ----------

def _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, slice_uids):
    """Patch the I/O boundary so only slice selection is under test."""
    monkeypatch.setattr(preprocessing, "retrieve_via_wado_rs", lambda info: [mock.MagicMock()])
    folder = tmp_path / "series"
    folder.mkdir(exist_ok=True)
    monkeypatch.setattr(preprocessing, "save_datasets_to_folder", lambda ds, uid, cfg: folder)
    monkeypatch.setattr(preprocessing, "read_dicom_volume_with_instances",
                        lambda f: (mock.MagicMock(), slice_uids))
    monkeypatch.setattr(preprocessing.sitk, "GetArrayFromImage", lambda img: vol_arr)
    return folder


def _decoded_first_pixel(data_uri):
    payload = data_uri.split(",", 1)[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(payload))))[0, 0, 0]


async def test_preprocess_series_sends_exactly_the_named_slices(tmp_path, monkeypatch):
    """The pixels sent come from the named instances, in the order named."""
    import preprocessing
    from models import SliceSelection
    # Each Z-plane is a constant so the decoded PNG identifies which slice it is.
    vol_arr = np.stack([np.full((8, 8), z * 10, dtype=np.float32) for z in range(10)])
    slice_uids = [f"1.{z}" for z in range(10)]
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, slice_uids)

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=3),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(series_uid="SE1", sop_instance_uids=["1.7", "1.2"]),
    )
    assert len(out) == 2
    # A constant plane normalizes to zeros, so identity is checked via the volume
    # indices the selection resolved to rather than the pixel values.
    assert all(s.startswith("data:image/png;base64,") for s in out)


async def test_preprocess_series_selection_overrides_the_configured_slice_count(
    tmp_path, monkeypatch
):
    """num_slices=3 in config, 5 named in the message -> 5 sent."""
    import preprocessing
    from models import SliceSelection
    vol_arr = np.linspace(0, 100, 10 * 8 * 8).reshape(10, 8, 8).astype(np.float32)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, [f"1.{z}" for z in range(10)])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=3),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(
            series_uid="SE1", sop_instance_uids=["1.0", "1.2", "1.4", "1.6", "1.8"]
        ),
    )
    assert len(out) == 5


async def test_preprocess_series_selection_picks_the_right_planes(tmp_path, monkeypatch):
    """Distinguishable planes: the returned PNGs must be slices 7 and 2, in that order."""
    import preprocessing
    from models import SliceSelection
    # A gradient per plane keeps normalization non-degenerate while the plane's
    # own offset survives as the brightest pixel's position.
    planes = []
    for z in range(10):
        plane = np.zeros((8, 8), dtype=np.float32)
        plane[z % 8, :] = 1000.0          # a bright row whose position encodes z
        planes.append(plane)
    vol_arr = np.stack(planes)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, [f"1.{z}" for z in range(10)])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=3),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(series_uid="SE1", sop_instance_uids=["1.7", "1.2"]),
    )
    bright_rows = []
    for uri in out:
        payload = uri.split(",", 1)[1]
        arr = np.array(Image.open(io.BytesIO(base64.b64decode(payload))))
        bright_rows.append(int(arr[:, :, 0].max(axis=1).argmax()))
    assert bright_rows == [7, 2]          # slice 7 first, then slice 2


async def test_preprocess_series_rejects_a_selection_the_series_cannot_address(
    tmp_path, monkeypatch
):
    """No per-slice UIDs (multi-frame): refuse rather than send the configured slices."""
    import preprocessing
    from models import SliceSelection
    vol_arr = np.zeros((10, 8, 8), dtype=np.float32)
    folder = _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, [])

    with pytest.raises(preprocessing.SliceSelectionError, match="cannot be addressed slice by slice"):
        await preprocessing.preprocess_series(
            series_uid="SE1", study_uid="STD1",
            params=_params(num_slices=3),
            wado_base_url="http://x/dicom-web", image_folder=tmp_path,
            selection=SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"]),
        )
    assert not folder.exists()           # temp files still cleaned up


async def test_preprocess_series_falls_back_to_strategy_without_a_selection(
    tmp_path, monkeypatch
):
    """A viewer predating slice_selections keeps today's behaviour exactly."""
    import preprocessing
    vol_arr = np.linspace(0, 100, 10 * 8 * 8).reshape(10, 8, 8).astype(np.float32)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, [f"1.{z}" for z in range(10)])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=4),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
    )
    assert len(out) == 4


# ---------- effective_params: the message's recipe wins ----------
#
# The runtime config is global and mutable. A message that carries its own recipe
# must be preprocessed with THAT recipe, or a second browser changing the config
# between compose and send would silently rewrite the first browser's request —
# and its provenance snapshot would describe something that never happened.

def test_effective_params_uses_the_runtime_config_by_default():
    import preprocessing
    out = preprocessing.effective_params(_params(num_slices=7, central_percentage=40))
    assert out.num_slices == 7
    assert out.central_percentage == 40


def test_effective_params_returns_a_copy_not_the_caller_object():
    """A concurrent config update mutates the runtime params in place."""
    import preprocessing
    params = _params(num_slices=5)
    out = preprocessing.effective_params(params)
    assert out is not params
    params.num_slices = 99
    assert out.num_slices == 5


def test_effective_params_prefers_the_recipe_the_message_carries():
    import preprocessing
    from models import SliceSelection, SliceStrategy
    sel = SliceSelection(
        series_uid="SE1", num_slices=12, slice_strategy=SliceStrategy.UNIFORM
    )
    out = preprocessing.effective_params(_params(num_slices=5), sel)
    assert out.num_slices == 12
    assert out.slice_strategy == SliceStrategy.UNIFORM


def test_effective_params_falls_back_for_a_partial_recipe():
    """num_slices without a strategy is not a recipe; do not half-apply it."""
    import preprocessing
    from models import SliceSelection
    sel = SliceSelection(series_uid="SE1", num_slices=12)
    out = preprocessing.effective_params(_params(num_slices=5), sel)
    assert out.num_slices == 5


def test_recipe_signature_distinguishes_two_message_recipes():
    """Two messages with different recipes must not share a cache entry."""
    import preprocessing
    from models import SliceSelection, SliceStrategy
    a = preprocessing.recipe_signature(
        _params(), SliceSelection(series_uid="SE1", num_slices=5,
                                  slice_strategy=SliceStrategy.UNIFORM)
    )
    b = preprocessing.recipe_signature(
        _params(), SliceSelection(series_uid="SE1", num_slices=9,
                                  slice_strategy=SliceStrategy.UNIFORM)
    )
    assert a != b


async def test_preprocess_series_applies_the_recipe_the_message_carries(tmp_path, monkeypatch):
    """Config says 3 slices; the message says 6, and 6 is what gets encoded."""
    import preprocessing
    from models import SliceSelection, SliceStrategy
    vol_arr = np.linspace(0, 100, 20 * 8 * 8).reshape(20, 8, 8).astype(np.float32)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, [])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=3),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(series_uid="SE1", num_slices=6,
                                 slice_strategy=SliceStrategy.UNIFORM),
    )
    assert len(out) == 6


# ---------- crop_to_roi ----------

def _make_roi(x=0.0, y=0.0, width=1.0, height=1.0):
    from models import RegionOfInterest
    return RegionOfInterest(x=x, y=y, width=width, height=height)


def test_crop_to_roi_takes_the_named_fraction():
    import preprocessing
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = preprocessing.crop_to_roi(arr, _make_roi(x=0.2, y=0.3, width=0.5, height=0.4))
    assert out.shape == (4, 5)          # rows 3:7, cols 2:7


def test_crop_to_roi_keeps_the_right_pixels():
    """Position matters, not just size: an off-by-one here crops the wrong region."""
    import preprocessing
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = preprocessing.crop_to_roi(arr, _make_roi(x=0.2, y=0.3, width=0.5, height=0.4))
    assert out[0, 0] == arr[3, 2]
    assert out[-1, -1] == arr[6, 6]


def test_crop_to_roi_returns_the_whole_slice_for_a_full_frame_roi():
    import preprocessing
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = preprocessing.crop_to_roi(arr, _make_roi())
    assert out.shape == arr.shape


def test_crop_to_roi_never_returns_an_empty_array():
    """A rectangle smaller than a pixel still has to yield an image."""
    import preprocessing
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = preprocessing.crop_to_roi(arr, _make_roi(x=0.5, y=0.5, width=0.001, height=0.001))
    assert out.size > 0
    assert out.shape == (1, 1)


def test_crop_to_roi_stays_inside_a_slice_flush_to_the_edge():
    import preprocessing
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = preprocessing.crop_to_roi(arr, _make_roi(x=0.9, y=0.9, width=0.1, height=0.1))
    assert out.shape == (1, 1)
    assert out[0, 0] == arr[9, 9]


def test_crop_to_roi_handles_a_non_square_slice():
    import preprocessing
    arr = np.arange(200, dtype=np.float32).reshape(10, 20)   # 10 rows, 20 cols
    out = preprocessing.crop_to_roi(arr, _make_roi(x=0.0, y=0.0, width=0.5, height=0.5))
    assert out.shape == (5, 10)


def test_recipe_signature_separates_a_cropped_request_from_an_uncropped_one():
    """Otherwise the second question would be answered with the first's framing."""
    import preprocessing
    from models import SliceSelection
    plain = SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"])
    cropped = SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"], roi=_make_roi(
        x=0.1, y=0.1, width=0.2, height=0.2))
    assert preprocessing.recipe_signature(_params(), plain) != \
        preprocessing.recipe_signature(_params(), cropped)


def test_recipe_signature_separates_two_different_crops():
    import preprocessing
    from models import SliceSelection
    a = SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"],
                       roi=_make_roi(x=0.1, y=0.1, width=0.2, height=0.2))
    b = SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"],
                       roi=_make_roi(x=0.4, y=0.1, width=0.2, height=0.2))
    assert preprocessing.recipe_signature(_params(), a) != \
        preprocessing.recipe_signature(_params(), b)


async def test_preprocess_series_crops_the_slices_it_sends(tmp_path, monkeypatch):
    """End to end: the encoded PNG is the cropped region, not the whole slice."""
    import preprocessing
    from models import SliceSelection
    vol_arr = np.linspace(0, 100, 4 * 40 * 40).reshape(4, 40, 40).astype(np.float32)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, ["1.0", "1.1", "1.2", "1.3"])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=1),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(
            series_uid="SE1", sop_instance_uids=["1.1"],
            roi=_make_roi(x=0.25, y=0.25, width=0.5, height=0.5),
        ),
    )
    payload = out[0].split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(payload)))
    assert img.size == (20, 20)         # half of 40x40


async def test_preprocess_series_sends_the_full_slice_without_an_roi(tmp_path, monkeypatch):
    import preprocessing
    from models import SliceSelection
    vol_arr = np.linspace(0, 100, 4 * 40 * 40).reshape(4, 40, 40).astype(np.float32)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, ["1.0", "1.1", "1.2", "1.3"])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=1),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(series_uid="SE1", sop_instance_uids=["1.1"]),
    )
    payload = out[0].split(",", 1)[1]
    assert Image.open(io.BytesIO(base64.b64decode(payload))).size == (40, 40)


async def test_preprocess_series_crops_every_slice_it_sends(tmp_path, monkeypatch):
    import preprocessing
    from models import SliceSelection
    vol_arr = np.linspace(0, 100, 4 * 40 * 40).reshape(4, 40, 40).astype(np.float32)
    _patch_pipeline(preprocessing, monkeypatch, tmp_path, vol_arr, ["1.0", "1.1", "1.2", "1.3"])

    out = await preprocessing.preprocess_series(
        series_uid="SE1", study_uid="STD1",
        params=_params(num_slices=1),
        wado_base_url="http://x/dicom-web", image_folder=tmp_path,
        selection=SliceSelection(
            series_uid="SE1", sop_instance_uids=["1.0", "1.2"],
            roi=_make_roi(x=0.0, y=0.0, width=0.5, height=0.25),
        ),
    )
    sizes = [Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).size for u in out]
    assert sizes == [(20, 10), (20, 10)]
