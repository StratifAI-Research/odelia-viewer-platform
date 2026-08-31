"""Unit tests for the Orthanc + router HTTP client (ODV-221).

The client takes an injected transport (anything with ``get``/``post``), so these
tests drive it with a fake that records calls and returns canned responses -- no
live Orthanc needed. Request shapes mirror the ODV-219 round-trip test.
"""

import pytest
import requests

from batch.client import OrthancRouterClient

pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeHttp:
    def __init__(self):
        self.get_responses: dict[str, FakeResponse] = {}
        self.post_responses: dict[str, FakeResponse] = {}
        self.calls: list[tuple[str, str, dict]] = []

    def _match(self, table, method, url):
        for key, resp in table.items():
            if key in url:
                return resp
        raise AssertionError(f"no fake {method} response for {url}")

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._match(self.get_responses, "GET", url)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._match(self.post_responses, "POST", url)


def _client(http):
    return OrthancRouterClient("http://localhost:8000", roster_host="http://localhost", http=http)


def _dicom_bytes():
    return b"\x00" * 128 + b"DICM" + b"rest-of-file"


def test_upload_instance_posts_dicom_and_returns_parent_study(tmp_path):
    http = FakeHttp()
    http.post_responses["/instances"] = FakeResponse(json_data={"ParentStudy": "orthanc-study-1"})
    dcm = tmp_path / "a.dcm"
    dcm.write_bytes(_dicom_bytes())

    study_id = _client(http).upload_instance(dcm)

    assert study_id == "orthanc-study-1"
    method, url, kwargs = http.calls[0]
    assert method == "POST"
    assert url == "http://localhost:8000/instances"
    assert kwargs["headers"]["Content-Type"] == "application/dicom"
    assert kwargs["data"] == _dicom_bytes()


def test_upload_instance_skips_non_dicom_without_posting(tmp_path):
    http = FakeHttp()
    png = tmp_path / "image.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n not a dicom")

    study_id = _client(http).upload_instance(png)

    assert study_id is None
    assert http.calls == []


def test_study_series_returns_uid_modality_description(tmp_path):
    http = FakeHttp()
    http.get_responses["/studies/S1/series"] = FakeResponse(
        json_data=[
            {
                "MainDicomTags": {
                    "Modality": "MR",
                    "SeriesInstanceUID": "mr-1",
                    "SeriesDescription": "t1 dynamic",
                },
                "Instances": ["i1"],
            },
            # SeriesDescription may be absent entirely
            {"MainDicomTags": {"Modality": "SR", "SeriesInstanceUID": "sr-1"}, "Instances": ["i2"]},
        ]
    )

    series = _client(http).study_series("S1")

    assert [(s.uid, s.modality, s.description) for s in series] == [
        ("mr-1", "MR", "t1 dynamic"),
        ("sr-1", "SR", ""),
    ]


def test_study_instance_uid_reads_main_dicom_tags(tmp_path):
    http = FakeHttp()
    http.get_responses["/studies/S1"] = FakeResponse(
        json_data={"MainDicomTags": {"StudyInstanceUID": "9.9.9"}}
    )

    assert _client(http).study_instance_uid("S1") == "9.9.9"


def test_study_instance_uid_returns_none_when_absent(tmp_path):
    http = FakeHttp()
    http.get_responses["/studies/S1"] = FakeResponse(json_data={"MainDicomTags": {}})

    assert _client(http).study_instance_uid("S1") is None


def test_sr_instance_ids_collects_only_sr_series():
    http = FakeHttp()
    http.get_responses["/studies/S1/series"] = FakeResponse(
        json_data=[
            {"MainDicomTags": {"Modality": "MR"}, "Instances": ["mr0"]},
            {"MainDicomTags": {"Modality": "SR"}, "Instances": ["sr0", "sr1"]},
            {"MainDicomTags": {"Modality": "SC"}, "Instances": ["sc0"]},
        ]
    )

    assert _client(http).sr_instance_ids("S1") == {"sr0", "sr1"}


def test_send_to_ai_posts_payload_and_returns_workitem_uid():
    http = FakeHttp()
    http.post_responses["/send-to-ai"] = FakeResponse(json_data={"workitem_uid": "wi-42"})

    uid = _client(http).send_to_ai(
        "S1",
        "ODELIA MST init weights preview",
        "http://router:8042/dicom-web",
        ["series-1"],
        input_configuration_id="multiphase",
        input_mapping={"multiphase": "series-1"},
    )

    assert uid == "wi-42"
    _, url, kwargs = http.calls[0]
    assert url == "http://localhost:8000/send-to-ai"
    assert kwargs["json"] == {
        "study_id": "S1",
        "target": "ODELIA MST init weights preview",
        "target_url": "http://router:8042/dicom-web",
        "series_uids": ["series-1"],
        "input_configuration_id": "multiphase",
        "input_mapping": {"multiphase": "series-1"},
    }


def test_send_to_ai_raises_when_no_workitem_uid():
    http = FakeHttp()
    http.post_responses["/send-to-ai"] = FakeResponse(json_data={"status": "success"}, text="{}")

    with pytest.raises(Exception, match="workitem_uid"):
        _client(http).send_to_ai(
            "S1",
            "MST",
            "http://router:8042/dicom-web",
            ["series-1"],
            input_configuration_id="multiphase",
            input_mapping={"multiphase": "series-1"},
        )


def test_workitem_state_reads_procedure_step_state():
    http = FakeHttp()
    http.get_responses["/ups-rs/workitems/wi-1"] = FakeResponse(
        json_data={"00741000": {"Value": ["COMPLETED"]}}
    )

    state = _client(http).workitem_state(8049, "wi-1")

    assert state == "COMPLETED"
    _, url, _ = http.calls[0]
    assert url == "http://localhost:8049/ups-rs/workitems/wi-1"


def test_workitem_state_returns_none_on_404():
    http = FakeHttp()
    http.get_responses["/ups-rs/workitems/wi-1"] = FakeResponse(status_code=404)

    assert _client(http).workitem_state(8049, "wi-1") is None
