"""Unit tests for router/wado_utils.py — DICOMweb metadata retrieval helpers."""
import pytest


def _make_metadata_doc(instance_uid="1.2.4.1", instance_number=1, temporal_position=None):
    """Build a DICOMweb-style metadata dict (tag -> Value)."""
    doc = {
        "00080018": {"Value": [instance_uid]},
        "00200013": {"Value": [instance_number]},
        "00200032": {"Value": [0.0, 0.0, float(instance_number)]},
    }
    if temporal_position is not None:
        doc["00200100"] = {"Value": [temporal_position]}
    return doc


def _retrieval_arg(study="A", series="B"):
    return [{"retrieval_url": f"http://x/studies/{study}", "study_uid": study, "series_uid": series}]


def test_retrieve_series_metadata_sorts_by_instance_number(wado_fake):
    from wado_utils import retrieve_series_metadata_sorted
    wado_fake.metadata_responses[("A", "B")] = [
        _make_metadata_doc(instance_uid="1.2.4.3", instance_number=3),
        _make_metadata_doc(instance_uid="1.2.4.1", instance_number=1),
        _make_metadata_doc(instance_uid="1.2.4.2", instance_number=2),
    ]
    meta, positions, spacing = retrieve_series_metadata_sorted(_retrieval_arg())

    z_values = [p[2] for p in positions]
    assert z_values == sorted(z_values)
    assert z_values == [1.0, 2.0, 3.0]
    assert meta["00200013"]["Value"][0] == 1
    assert isinstance(spacing, float)
    assert spacing >= 0.0


def test_retrieve_series_metadata_empty_list_raises_index_error():
    """Empty wado_rs_retrieval raises IndexError on `first_retrieval = wado_rs_retrieval[0]`.

    Pinned to a single exception class per IEC 62304 §5.5.2 (validation contracts must
    specify one outcome). A refactor that introduces upfront validation raising ValueError
    will fail this test and force a deliberate spec decision."""
    from wado_utils import retrieve_series_metadata_sorted
    with pytest.raises(IndexError):
        retrieve_series_metadata_sorted([])


# ---------------------------------------------------------------------------
# R3: multi-temporal phase grouping + edge cases
# ---------------------------------------------------------------------------

def test_multi_temporal_phases_selects_first_phase_sorted(wado_fake):
    """When two temporal phases are present, only instances from the lowest phase are returned."""
    from wado_utils import retrieve_series_metadata_sorted
    wado_fake.metadata_responses[("A", "B")] = [
        _make_metadata_doc(instance_uid="p2.i1", instance_number=1, temporal_position=2),
        _make_metadata_doc(instance_uid="p1.i2", instance_number=2, temporal_position=1),
        _make_metadata_doc(instance_uid="p1.i1", instance_number=1, temporal_position=1),
    ]
    meta, positions, _ = retrieve_series_metadata_sorted(_retrieval_arg())
    assert len(positions) == 2
    assert meta["00080018"]["Value"][0] == "p1.i1"


def test_instances_missing_ipp_are_silently_skipped(wado_fake):
    """Instances without 00200032 (IPP) are skipped from selection."""
    from wado_utils import retrieve_series_metadata_sorted
    no_ipp = {"00200013": {"Value": [99]}}
    wado_fake.metadata_responses[("A", "B")] = [
        no_ipp,
        _make_metadata_doc(instance_uid="ok.1", instance_number=1),
        _make_metadata_doc(instance_uid="ok.2", instance_number=2),
    ]
    meta, positions, _ = retrieve_series_metadata_sorted(_retrieval_arg())
    assert len(positions) == 2
    assert meta["00080018"]["Value"][0] == "ok.1"


def test_no_well_formed_instances_raises_value_error(wado_fake):
    """If all instances lack IPP or InstanceNumber, raise ValueError."""
    from wado_utils import retrieve_series_metadata_sorted
    wado_fake.metadata_responses[("A", "B")] = [
        {"00200013": {"Value": [1]}},               # no IPP
        {"00200032": {"Value": [0.0, 0.0, 0.0]}},   # no InstanceNumber
    ]
    with pytest.raises(ValueError, match="No instances"):
        retrieve_series_metadata_sorted(_retrieval_arg())


def test_single_instance_spacing_falls_back_to_one(wado_fake):
    """With only one instance, slice_spacing falls back to 1.0 (cannot compute from positions)."""
    from wado_utils import retrieve_series_metadata_sorted
    wado_fake.metadata_responses[("A", "B")] = [
        _make_metadata_doc(instance_uid="only.1", instance_number=1),
    ]
    _, positions, spacing = retrieve_series_metadata_sorted(_retrieval_arg())
    assert len(positions) == 1
    assert spacing == 1.0
