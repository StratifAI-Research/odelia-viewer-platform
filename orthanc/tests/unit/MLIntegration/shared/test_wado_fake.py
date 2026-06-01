"""Self-tests for the wado_fake fixture."""
import pydicom
import pytest


def _make_dataset(sop_uid="1.2.3.1"):
    ds = pydicom.dataset.Dataset()
    ds.SOPInstanceUID = sop_uid
    return ds


def test_wado_fake_retrieve_series_returns_bound_datasets(wado_fake):
    wado_fake.series_responses[("S1", "SE1")] = [_make_dataset("1.2.3.1")]
    from shared.wado_retrieval import retrieve_via_wado_rs

    result = retrieve_via_wado_rs(
        [{"retrieval_url": "http://host/dicom-web", "study_uid": "S1", "series_uid": "SE1"}]
    )
    assert len(result) == 1
    assert result[0].SOPInstanceUID == "1.2.3.1"


def test_wado_fake_tracks_calls(wado_fake):
    wado_fake.series_responses[("S1", "SE1")] = []
    from shared.wado_retrieval import retrieve_via_wado_rs

    retrieve_via_wado_rs(
        [{"retrieval_url": "http://host/dicom-web", "study_uid": "S1", "series_uid": "SE1"}]
    )
    assert wado_fake.calls == [("retrieve_series", "S1", "SE1")]


def test_wado_fake_unbound_key_raises(wado_fake):
    from shared.wado_retrieval import retrieve_via_wado_rs
    from shared.exceptions import DicomRetrievalError

    with pytest.raises(DicomRetrievalError, match="WADO-RS retrieval failed"):
        retrieve_via_wado_rs(
            [{"retrieval_url": "http://host/dicom-web", "study_uid": "X", "series_uid": "Y"}]
        )


def test_wado_fake_metadata_returns_bound_dicts(wado_fake):
    wado_fake.metadata_responses[("S1", "SE1")] = [{"00200032": {"Value": [0, 0, 0]}}]

    from shared.wado_retrieval import DICOMwebClient

    client = DICOMwebClient(url="http://host/dicom-web")
    result = client.retrieve_series_metadata(
        study_instance_uid="S1", series_instance_uid="SE1"
    )
    assert result == [{"00200032": {"Value": [0, 0, 0]}}]
