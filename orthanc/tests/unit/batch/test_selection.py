"""Unit tests for manifest-aware input selection and the sequence mapping CSV (ODV-221)."""

import pydicom
import pytest

from batch.selection import (
    SeriesInfo,
    is_ai_result,
    load_sequence_mapping,
    resolve_input_selection,
)

pytestmark = pytest.mark.unit


def _infos(*rows):
    return [SeriesInfo(uid, modality, description) for uid, modality, description in rows]


# ---------------------------------------------------------------------------
# resolve_input_selection with a mapping
# ---------------------------------------------------------------------------

def test_mapped_sub_selects_subtraction_configuration() -> None:
    series = _infos(("1.1", "MR", "sub"), ("1.2", "MR", "pre"))
    sel = resolve_input_selection(series, {"Sub_1": "1.1", "Pre": "1.2"})
    assert sel.configuration_id == "subtraction"
    assert sel.mapping == {"sub": "1.1"}
    assert sel.series_uids == ["1.1"]


def test_mapped_pre_post_selects_pre_post_configuration() -> None:
    series = _infos(("1.2", "MR", "pre"), ("1.3", "MR", "post"))
    sel = resolve_input_selection(series, {"Pre": "1.2", "Post_1": "1.3"})
    assert sel.configuration_id == "pre_post"
    assert sel.mapping == {"pre": "1.2", "post": "1.3"}
    assert set(sel.series_uids) == {"1.2", "1.3"}


def test_mapped_series_absent_from_study_is_a_skip_reason() -> None:
    series = _infos(("1.2", "MR", "pre"))
    result = resolve_input_selection(series, {"Pre": "1.2", "Post_1": "1.3"})
    assert isinstance(result, str)
    assert "1.3" in result


def test_mapping_without_usable_roles_is_a_skip_reason() -> None:
    series = _infos(("1.4", "MR", "t2"))
    result = resolve_input_selection(series, {"T2": "1.4"})
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# resolve_input_selection without a mapping (auto-multiphase)
# ---------------------------------------------------------------------------

def test_single_mr_series_auto_selects_multiphase() -> None:
    series = _infos(("1.1", "MR", "dynamic"), ("2.1", "SR", "Automated Diagnostic Findings"))
    sel = resolve_input_selection(series, None)
    assert sel.configuration_id == "multiphase"
    assert sel.mapping == {"multiphase": "1.1"}
    assert sel.series_uids == ["1.1"]


def test_auto_selection_excludes_prior_ai_result_series() -> None:
    series = _infos(("1.1", "MR", "dynamic"), ("2.1", "MR", "MST - Heatmap"))
    sel = resolve_input_selection(series, None)
    assert sel.configuration_id == "multiphase"
    assert sel.mapping == {"multiphase": "1.1"}


def test_multiple_mr_series_without_mapping_is_a_skip_reason() -> None:
    result = resolve_input_selection(_infos(("1.1", "MR", "dyn"), ("1.2", "MR", "t2")), None)
    assert isinstance(result, str)
    assert "2" in result


def test_no_mr_series_is_a_skip_reason() -> None:
    result = resolve_input_selection(_infos(("2.1", "SR", "report")), None)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# is_ai_result
# ---------------------------------------------------------------------------

def test_is_ai_result_matches_viewer_markers() -> None:
    assert is_ai_result("SR", "Automated Diagnostic Findings")
    assert is_ai_result("MR", "MST - Heatmap")
    assert is_ai_result("SC", "AI overlay")
    assert not is_ai_result("MR", "t1_fl3d dynamic")


# ---------------------------------------------------------------------------
# load_sequence_mapping
# ---------------------------------------------------------------------------

def _write_minimal_dicom(path, series_uid):
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = f"{series_uid}.0"
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = pydicom.uid.MRImageStorage
    ds.SOPInstanceUID = f"{series_uid}.0"
    ds.SeriesInstanceUID = series_uid
    ds.save_as(str(path), enforce_file_format=True)


def test_load_sequence_mapping_reads_series_uid_from_dicom(tmp_path) -> None:
    series_dir = tmp_path / "P1" / "9.9.9" / "s1"
    series_dir.mkdir(parents=True)
    _write_minimal_dicom(series_dir / "000.dcm", "1.1.1")
    csv_path = tmp_path / "mapping.csv"
    csv_path.write_text(
        "PatientID,StudyInstanceUID,SequenceName,SeriesPath\n"
        "P1,9.9.9,Pre,P1/9.9.9/s1\n"
    )

    mapping, warnings = load_sequence_mapping(csv_path, tmp_path)

    assert mapping == {"9.9.9": {"Pre": "1.1.1"}}
    assert warnings == []


def test_load_sequence_mapping_warns_on_missing_series_dir(tmp_path) -> None:
    csv_path = tmp_path / "mapping.csv"
    csv_path.write_text(
        "PatientID,StudyInstanceUID,SequenceName,SeriesPath\n"
        "P1,9.9.9,Pre,P1/9.9.9/GONE\n"
    )

    mapping, warnings = load_sequence_mapping(csv_path, tmp_path)

    assert mapping == {}
    assert len(warnings) == 1
    assert "GONE" in warnings[0]


def test_load_sequence_mapping_rejects_missing_columns(tmp_path) -> None:
    csv_path = tmp_path / "mapping.csv"
    csv_path.write_text("StudyInstanceUID,SeriesPath\n9.9.9,foo\n")

    with pytest.raises(SystemExit, match="missing columns"):
        load_sequence_mapping(csv_path, tmp_path)
