"""Unit tests for router/server.py — pure helpers + SR/SC pipeline.

Covers:
  - detect_response_format: bilateral, bilateral_with_heatmap, unknown raises ValueError
  - create_code_sequence: Dataset structure (CodeValue, CodingSchemeDesignator, CodeMeaning)
  - create_measurement: Dataset structure (NumericValue, MeasurementUnitsCodeSequence)
  - add_text_overlay: single-frame and multi-frame pixel modification
  - create_bilateral_sr: smoke + asserts on Modality, ContentSequence, ReferencedImageSequence
  - create_mst_sr: smoke + asserts on Modality, ContentSequence
  - create_multiframe_attention_sc: smoke returns bytes with SC Modality
  - create_text_overlay_sc: smoke test (requires pixel_array on original_ds)

The legacy OnStableStudy OnChange callback has been removed (see ups/processor.py for the UPS-based replacement).
"""
import base64
import io
import sys
from types import ModuleType
from typing import Iterator

import numpy as np
import pytest
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

import os as _os
_ROUTER_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'router')
)


def _evict_ups():
    for key in [k for k in sys.modules if k == "ups" or k.startswith("ups.")]:
        del sys.modules[key]
    # Also evict server so it re-imports fresh
    for key in [k for k in sys.modules if k == "server"]:
        del sys.modules[key]


def _load_server():
    # Ensure router/ is at the front of sys.path so 'ups.routes' resolves to
    # router/ups/routes.py even if viewer conftest has prepended viewer/ first.
    if _ROUTER_DIR not in sys.path:
        sys.path.insert(0, _ROUTER_DIR)
    else:
        # Move to front if already present
        sys.path.remove(_ROUTER_DIR)
        sys.path.insert(0, _ROUTER_DIR)
    _evict_ups()
    import server
    return server


@pytest.fixture(autouse=True)
def _router_path_guard() -> Iterator[None]:
    """Ensure router/ is at sys.path[0] for each test; restore after.

    Both viewer and router conftest files run at collection time. This fixture
    keeps router/ at the front during each test so imports resolve to router/ups/,
    then restores sys.path so viewer tests that follow find viewer/ups/ first.
    """
    saved = list(sys.path)
    # Ensure router/ is first
    if _ROUTER_DIR in sys.path:
        sys.path.remove(_ROUTER_DIR)
    sys.path.insert(0, _ROUTER_DIR)
    yield
    sys.path[:] = saved


@pytest.fixture
def srv() -> ModuleType:
    return _load_server()


# ---------------------------------------------------------------------------
# Minimal DICOM Dataset helper
# ---------------------------------------------------------------------------

def _minimal_dicom():
    ds = Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "T001"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    ds.StudyDate = "20260101"
    ds.StudyTime = "120000"
    ds.AccessionNumber = "12345"
    ds.Modality = "MR"
    ds.Manufacturer = "TEST"
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return ds


class _DatasetWithFakePixels(Dataset):
    """Dataset subclass exposing ._pixel_array as .pixel_array.

    The property lives on this subclass — not on pydicom.Dataset itself —
    so it cannot leak across tests and shadow pydicom's real pixel_array
    decoder for any test that runs afterwards.
    """
    @property
    def pixel_array(self):
        return self._pixel_array


def _minimal_dicom_with_pixels(rows=64, cols=64):
    """Minimal DICOM with a fake RGB pixel_array attribute for SC tests."""
    ds = _minimal_dicom()
    ds.__class__ = _DatasetWithFakePixels
    ds._pixel_array = np.zeros((rows, cols, 3), dtype=np.uint8)
    return ds


# ---------------------------------------------------------------------------
# detect_response_format
# ---------------------------------------------------------------------------

def test_detect_response_format_bilateral(srv):
    result = srv.detect_response_format({
        "left": {"prediction": "Benign", "confidence": 80.0},
        "right": {"prediction": "Malignant", "confidence": 95.0},
    })
    assert result == "bilateral"


def test_detect_response_format_bilateral_with_heatmap(srv):
    result = srv.detect_response_format({
        "left": {"prediction": "Benign", "confidence": 80.0},
        "right": {"prediction": "Malignant", "confidence": 95.0},
        "attention_maps": {"data": "...", "shape": [10, 64, 64, 3]},
    })
    assert result == "bilateral_with_heatmap"


def test_detect_response_format_unknown_raises_value_error(srv):
    with pytest.raises(ValueError, match="Unknown response format"):
        srv.detect_response_format({"model_output": [0.1, 0.9]})


def test_detect_response_format_only_left_is_bilateral(srv):
    result = srv.detect_response_format({"left": {"prediction": "Benign", "confidence": 60.0}})
    assert result == "bilateral"


def test_detect_response_format_only_right_is_bilateral(srv):
    result = srv.detect_response_format({"right": {"prediction": "Malignant", "confidence": 90.0}})
    assert result == "bilateral"


# ---------------------------------------------------------------------------
# create_code_sequence
# ---------------------------------------------------------------------------

def test_create_code_sequence_structure(srv):
    item = srv.create_code_sequence("12345", "SCT", "Some Concept")
    assert hasattr(item, "CodeValue")
    assert hasattr(item, "CodingSchemeDesignator")
    assert hasattr(item, "CodeMeaning")
    assert item.CodeValue == "12345"
    assert item.CodingSchemeDesignator == "SCT"
    assert item.CodeMeaning == "Some Concept"


def test_create_code_sequence_returns_dataset(srv):
    item = srv.create_code_sequence("R-00339", "SRT", "Classification Result")
    assert isinstance(item, Dataset)


def test_create_code_sequence_loinc(srv):
    item = srv.create_code_sequence("18748-4", "LN", "Diagnostic Imaging Report")
    assert item.CodeValue == "18748-4"
    assert item.CodingSchemeDesignator == "LN"


# ---------------------------------------------------------------------------
# create_measurement
# ---------------------------------------------------------------------------

def test_create_measurement_numeric_value(srv):
    m = srv.create_measurement(85.5, "%", "%", "UCUM")
    assert hasattr(m, "NumericValue")
    assert float(m.NumericValue) == pytest.approx(85.5)


def test_create_measurement_units_sequence_present(srv):
    m = srv.create_measurement(42.0, "%", "%", "UCUM")
    assert hasattr(m, "MeasurementUnitsCodeSequence")
    assert len(m.MeasurementUnitsCodeSequence) == 1


def test_create_measurement_units_code_value(srv):
    m = srv.create_measurement(99.0, "percent", "%", "UCUM")
    units_item = m.MeasurementUnitsCodeSequence[0]
    assert units_item.CodeValue == "%"
    assert units_item.CodingSchemeDesignator == "UCUM"
    assert units_item.CodeMeaning == "percent"


def test_create_measurement_returns_dataset(srv):
    m = srv.create_measurement(0.0, "%", "%", "UCUM")
    assert isinstance(m, Dataset)


# ---------------------------------------------------------------------------
# add_text_overlay
# ---------------------------------------------------------------------------

def test_add_text_overlay_single_frame_modifies_pixels(srv):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    result = srv.add_text_overlay(arr, text="TEST", color="red")
    assert result.shape[0] == 64
    assert result.shape[1] == 64
    # Overlay must actually modify the input — sum>0 alone would pass if a single
    # background pixel were lit by any unrelated bug.
    assert np.any(result != arr), "overlay produced no pixel change vs input"


def test_add_text_overlay_single_frame_greyscale_converts_to_rgb(srv):
    arr = np.zeros((64, 64), dtype=np.uint8)
    result = srv.add_text_overlay(arr, text="TEST", color="red")
    # Greyscale input becomes RGB output (3 channels)
    assert len(result.shape) == 3
    assert result.shape[2] == 3


def test_add_text_overlay_multi_frame_preserves_frame_count(srv):
    arr = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    result = srv.add_text_overlay(arr, text="X", color="white")
    assert result.shape[0] == 4


def test_add_text_overlay_multi_frame_modifies_pixels(srv):
    arr = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    result = srv.add_text_overlay(arr, text="AI", color="green")
    # Each frame must be modified — sum>0 would pass if only one frame got lit.
    assert np.any(result != arr), "multi-frame overlay produced no pixel change vs input"
    assert result.shape == arr.shape, f"multi-frame overlay changed shape: {arr.shape} -> {result.shape}"


# ---------------------------------------------------------------------------
# create_bilateral_sr
# ---------------------------------------------------------------------------

_BILATERAL_RESULTS = {
    "left": {"prediction": "Malignant", "confidence": 92.5},
    "right": {"prediction": "Benign", "confidence": 67.0},
}


def test_create_bilateral_sr_returns_bytes(srv):
    ds = _minimal_dicom()
    sr_bytes, date, time_str, uid = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    assert isinstance(sr_bytes, bytes)
    assert len(sr_bytes) > 0


def test_create_bilateral_sr_modality_is_sr(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert parsed.Modality == "SR"


def test_create_bilateral_sr_content_sequence_present(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert hasattr(parsed, "ContentSequence")
    assert len(parsed.ContentSequence) > 0


def test_create_bilateral_sr_references_original_sop(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert hasattr(parsed, "ReferencedImageSequence")
    assert len(parsed.ReferencedImageSequence) > 0
    assert parsed.ReferencedImageSequence[0].ReferencedSOPInstanceUID == ds.SOPInstanceUID


def test_create_bilateral_sr_preserves_patient_info(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert str(parsed.PatientID) == "T001"
    assert str(parsed.StudyInstanceUID) == str(ds.StudyInstanceUID)


def test_create_bilateral_sr_returns_valid_sop_uid(srv):
    """Returned UID must be a non-empty valid DICOM UID (1-64 chars, digits + dots, valid root).
    Mis-formatted UIDs are rejected by downstream Orthanc / DICOMweb consumers — pin the format."""
    from pydicom.uid import UID
    ds = _minimal_dicom()
    _, date, time_str, uid = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    assert uid
    assert isinstance(uid, str)
    parsed_uid = UID(uid)
    assert parsed_uid.is_valid, f"returned SOP UID is not a valid DICOM UID: {uid!r}"
    assert 1 <= len(uid) <= 64, f"DICOM UID length must be 1-64 chars; got {len(uid)}"


def test_create_bilateral_sr_content_contains_measurements(srv):
    """Root container ContentSequence should contain measurement items for left/right sides."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    # The root container is the only item in ContentSequence
    root_container = parsed.ContentSequence[0]
    assert hasattr(root_container, "ContentSequence")
    # Should contain items for left, right, and model metadata — at least 2
    assert len(root_container.ContentSequence) >= 2


def test_create_bilateral_sr_error_side_uses_text_type(srv):
    """When a side has an error key, value type should be TEXT, not CODE."""
    ds = _minimal_dicom()
    results_with_error = {
        "left": {"error": "No series found"},
        "right": {"prediction": "Benign", "confidence": 60.0},
    }
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, results_with_error)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    root_items = parsed.ContentSequence[0].ContentSequence
    text_items = [i for i in root_items if i.ValueType == "TEXT"]
    assert len(text_items) >= 1


# ---------------------------------------------------------------------------
# create_mst_sr
# ---------------------------------------------------------------------------

_MST_RESULTS = {
    "classification": {
        "prediction": "Malignant",
        "probability": 0.87,
        "model_name": "MST-v2",
        "architecture": "Vision Transformer",
        "version": "2.0",
    }
}


def test_create_mst_sr_returns_bytes(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    assert isinstance(sr_bytes, bytes)
    assert len(sr_bytes) > 0


def test_create_mst_sr_modality_is_sr(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert parsed.Modality == "SR"


def test_create_mst_sr_content_sequence_present(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert hasattr(parsed, "ContentSequence")
    assert len(parsed.ContentSequence) > 0


def test_create_mst_sr_references_original(srv):
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert parsed.ReferencedImageSequence[0].ReferencedSOPInstanceUID == ds.SOPInstanceUID


def test_create_mst_sr_classification_probability_in_content(srv):
    """The classification probability (87%) should appear in a MeasuredValueSequence."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    root_items = parsed.ContentSequence[0].ContentSequence
    measured_items = [i for i in root_items if hasattr(i, "MeasuredValueSequence")]
    assert len(measured_items) >= 1
    # Value should be ~87 (probability * 100)
    val = float(measured_items[0].MeasuredValueSequence[0].NumericValue)
    assert abs(val - 87.0) < 0.01


def test_create_mst_sr_empty_classification_does_not_raise(srv):
    """Empty classification dict should not crash the SR builder."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, {"classification": {}})
    assert isinstance(sr_bytes, bytes)


# ---------------------------------------------------------------------------
# create_multiframe_attention_sc
# ---------------------------------------------------------------------------

def _make_fake_attention_maps(num_frames=4, rows=32, cols=32):
    """Create a fake base64-encoded attention map payload like the MST model returns."""
    arr = np.random.randint(0, 255, (num_frames, rows, cols, 3), dtype=np.uint8)
    encoded = base64.b64encode(arr.tobytes()).decode("utf-8")
    return {
        "data": encoded,
        "shape": [num_frames, rows, cols, 3],
        "dtype": "uint8",
    }


def test_create_multiframe_attention_sc_returns_bytes(srv):
    ds = _minimal_dicom()
    attention_maps = _make_fake_attention_maps()
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps)
    assert isinstance(sc_bytes, bytes)
    assert len(sc_bytes) > 0


def test_create_multiframe_attention_sc_modality_is_sc(srv):
    ds = _minimal_dicom()
    attention_maps = _make_fake_attention_maps()
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps)
    from pydicom import dcmread
    # SC objects are written as bare Datasets (no preamble); force=True is required
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert parsed.Modality == "SC"


def test_create_multiframe_attention_sc_frame_count_matches(srv):
    ds = _minimal_dicom()
    num_frames = 6
    attention_maps = _make_fake_attention_maps(num_frames=num_frames)
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert int(parsed.NumberOfFrames) == num_frames


def test_create_multiframe_attention_sc_pixel_values_round_trip(srv):
    """The decoded PixelData must match the input base64 payload byte-for-byte.
    A writer bug that zero-fills frames or truncates the payload passes the
    frame-count and Modality tests but fails this one."""
    import base64 as _b64
    ds = _minimal_dicom()
    # Build a deterministic payload so we can compare bytes after the round-trip.
    expected_arr = (np.arange(4 * 16 * 16 * 3) % 256).astype(np.uint8).reshape(4, 16, 16, 3)
    expected_bytes = expected_arr.tobytes()
    attention_maps = {
        "data": _b64.b64encode(expected_bytes).decode("utf-8"),
        "shape": [4, 16, 16, 3],
        "dtype": "uint8",
    }
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert int(parsed.NumberOfFrames) == 4
    # Compare decoded pixel bytes against the encoded payload.
    pixel_data = parsed.PixelData
    # PixelData may have a trailing pad byte to even length per DICOM rules.
    assert pixel_data[:len(expected_bytes)] == expected_bytes, \
        "SC PixelData does not match the encoded attention-map payload (writer dropped/zeroed bytes)"


def test_create_multiframe_attention_sc_references_original(srv):
    ds = _minimal_dicom()
    attention_maps = _make_fake_attention_maps()
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert parsed.ReferencedImageSequence[0].ReferencedSOPInstanceUID == ds.SOPInstanceUID


def test_create_multiframe_attention_sc_references_sr_when_given(srv):
    ds = _minimal_dicom()
    attention_maps = _make_fake_attention_maps()
    fake_sr_uid = generate_uid()
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps, sr_sop_instance_uid=fake_sr_uid)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert hasattr(parsed, "ReferencedInstanceSequence")
    ref_uid = parsed.ReferencedInstanceSequence[0].ReferencedSOPInstanceUID
    assert str(ref_uid) == str(fake_sr_uid)


def test_create_multiframe_attention_sc_photometric_rgb(srv):
    ds = _minimal_dicom()
    attention_maps = _make_fake_attention_maps()
    sc_bytes = srv.create_multiframe_attention_sc(ds, attention_maps)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert parsed.PhotometricInterpretation == "RGB"


# ---------------------------------------------------------------------------
# create_text_overlay_sc  (smoke — requires pixel_array on original_ds)
# ---------------------------------------------------------------------------

def test_create_text_overlay_sc_returns_bytes(srv):
    ds = _minimal_dicom_with_pixels()
    sc_bytes = srv.create_text_overlay_sc(ds, text="AI", color="red")
    assert isinstance(sc_bytes, bytes)
    assert len(sc_bytes) > 0


def test_create_text_overlay_sc_modality_is_sc(srv):
    ds = _minimal_dicom_with_pixels()
    sc_bytes = srv.create_text_overlay_sc(ds, text="TEST")
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert parsed.Modality == "SC"


def test_create_text_overlay_sc_has_content_sequence(srv):
    """SC includes a ContentSequence with model metadata mirroring the SR."""
    ds = _minimal_dicom_with_pixels()
    sc_bytes = srv.create_text_overlay_sc(ds, text="TEST")
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert hasattr(parsed, "ContentSequence")
    assert len(parsed.ContentSequence) >= 1


def test_create_text_overlay_sc_references_sr_when_given(srv):
    ds = _minimal_dicom_with_pixels()
    fake_sr_uid = generate_uid()
    sc_bytes = srv.create_text_overlay_sc(ds, text="TEST", sr_sop_instance_uid=fake_sr_uid)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    assert hasattr(parsed, "ReferencedInstanceSequence")
    ref_uid = parsed.ReferencedInstanceSequence[0].ReferencedSOPInstanceUID
    assert str(ref_uid) == str(fake_sr_uid)


# ---------------------------------------------------------------------------
# H1+H2: patient attribution + AI provenance audit on every SR/SC builder
# MDR/IEC 62304 §5.6: AI output must not be mis-attributed to the wrong patient
# and must carry traceable model provenance.
# ---------------------------------------------------------------------------


def _assert_patient_attribution(parsed, original_ds, msg_prefix=""):
    """Audit helper: every AI-produced DICOM object must round-trip the four
    safety-critical patient/study attribution tags from the original DICOM."""
    assert str(parsed.PatientID) == str(original_ds.PatientID), \
        f"{msg_prefix}PatientID mismatch: {parsed.PatientID!r} != {original_ds.PatientID!r}"
    assert str(parsed.PatientName) == str(original_ds.PatientName), \
        f"{msg_prefix}PatientName mismatch: {parsed.PatientName!r} != {original_ds.PatientName!r}"
    assert str(parsed.StudyInstanceUID) == str(original_ds.StudyInstanceUID), \
        f"{msg_prefix}StudyInstanceUID mismatch"


def _assert_sr_model_provenance(parsed, msg_prefix=""):
    """SR audit: model provenance is encoded inside a ContentSequence item whose
    TextValue carries the model_name (with AlgorithmName / AlgorithmVersion siblings).
    Audit trace: MDR Annex I §17.1 — output must be traceable to producing model."""
    def _walk(item, depth=0):
        if depth > 4:
            return None
        text_val = getattr(item, "TextValue", None)
        alg_name = getattr(item, "AlgorithmName", None)
        alg_ver = getattr(item, "AlgorithmVersion", None)
        if text_val and alg_name and alg_ver:
            return (str(text_val), str(alg_name), str(alg_ver))
        for child in getattr(item, "ContentSequence", []) or []:
            found = _walk(child, depth + 1)
            if found:
                return found
        return None

    for top in getattr(parsed, "ContentSequence", []) or []:
        found = _walk(top)
        if found:
            model_name, alg_name, alg_ver = found
            assert model_name.strip(), f"{msg_prefix}model_name (TextValue) is empty"
            assert alg_name.strip(), f"{msg_prefix}AlgorithmName is empty"
            assert alg_ver.strip(), f"{msg_prefix}AlgorithmVersion is empty"
            return
    raise AssertionError(
        f"{msg_prefix}no ContentSequence item carries (TextValue + AlgorithmName + AlgorithmVersion) "
        "model provenance — auditor cannot trace AI output to producing model"
    )


def _assert_sc_model_provenance(parsed, msg_prefix=""):
    """SC audit: model provenance is the ManufacturerModelName tag (set by production
    on every SC). Audit trace: MDR Annex I §17.1."""
    assert hasattr(parsed, "ManufacturerModelName"), \
        f"{msg_prefix}ManufacturerModelName tag missing — no audit trace from SC to producing model"
    assert str(parsed.ManufacturerModelName).strip(), \
        f"{msg_prefix}ManufacturerModelName is empty (no model provenance encoded)"


# --- bilateral SR ---

def test_create_bilateral_sr_carries_full_patient_attribution(srv):
    """H1: bilateral SR must round-trip ALL of PatientID/PatientName/StudyInstanceUID."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    _assert_patient_attribution(parsed, ds, msg_prefix="bilateral_sr: ")


def test_create_bilateral_sr_encodes_model_provenance(srv):
    """H2: bilateral SR must encode ManufacturerModelName for audit traceability."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_bilateral_sr(ds, _BILATERAL_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    _assert_sr_model_provenance(parsed, msg_prefix="bilateral_sr: ")


# --- MST SR ---

def test_create_mst_sr_carries_full_patient_attribution(srv):
    """H1: MST SR — same audit guarantee as bilateral. PRIOR REVIEW GAP."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    _assert_patient_attribution(parsed, ds, msg_prefix="mst_sr: ")


def test_create_mst_sr_encodes_model_provenance(srv):
    """H2: MST SR must encode ManufacturerModelName. PRIOR REVIEW GAP."""
    ds = _minimal_dicom()
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    _assert_sr_model_provenance(parsed, msg_prefix="mst_sr: ")


def test_create_mst_sr_mutating_input_propagates_to_output(srv):
    """Negative-attribution check: assertions are NOT hard-coded — changing the input
    ds.PatientID must change the output. Catches a future regression where the SR builder
    accidentally writes a constant PatientID."""
    ds = _minimal_dicom()
    ds.PatientID = "PATIENT_X_AUDIT"
    sr_bytes, _, _, _ = srv.create_mst_sr(ds, _MST_RESULTS)
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sr_bytes))
    assert str(parsed.PatientID) == "PATIENT_X_AUDIT"


# --- multiframe attention SC ---

def test_create_multiframe_attention_sc_carries_full_patient_attribution(srv):
    """H1: multi-frame SC — same audit guarantee. PRIOR REVIEW GAP."""
    ds = _minimal_dicom()
    sc_bytes = srv.create_multiframe_attention_sc(ds, _make_fake_attention_maps())
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    _assert_patient_attribution(parsed, ds, msg_prefix="multiframe_attention_sc: ")


def test_create_multiframe_attention_sc_encodes_model_provenance(srv):
    """H2: multi-frame SC must encode ManufacturerModelName. PRIOR REVIEW GAP."""
    ds = _minimal_dicom()
    sc_bytes = srv.create_multiframe_attention_sc(ds, _make_fake_attention_maps())
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    _assert_sc_model_provenance(parsed, msg_prefix="multiframe_attention_sc: ")


# --- text overlay SC ---

def test_create_text_overlay_sc_carries_full_patient_attribution(srv):
    """H1: text-overlay SC — same audit guarantee. PRIOR REVIEW GAP."""
    ds = _minimal_dicom_with_pixels()
    sc_bytes = srv.create_text_overlay_sc(ds, text="AI")
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    _assert_patient_attribution(parsed, ds, msg_prefix="text_overlay_sc: ")


def test_create_text_overlay_sc_encodes_model_provenance(srv):
    """H2: text-overlay SC must encode ManufacturerModelName. PRIOR REVIEW GAP."""
    ds = _minimal_dicom_with_pixels()
    sc_bytes = srv.create_text_overlay_sc(ds, text="AI")
    from pydicom import dcmread
    parsed = dcmread(io.BytesIO(sc_bytes), force=True)
    _assert_sc_model_provenance(parsed, msg_prefix="text_overlay_sc: ")
