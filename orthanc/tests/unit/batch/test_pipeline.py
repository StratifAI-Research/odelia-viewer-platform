"""Unit tests for the batch send-to-AI orchestration (ODV-221).

The pipeline is pure orchestration over an injected client, so these tests use a
hand-written fake instead of mocking HTTP. Each test names the client behaviour
it drives and asserts on the resulting report.
"""

from pathlib import Path

import pytest

from batch.pipeline import ModelSpec, run_batch
from batch.selection import SeriesInfo

pytestmark = pytest.mark.unit


MST = ModelSpec(
    model_name="MST",
    ai_name="ODELIA MST init weights preview",
    router_host="orthanc-router-odelia-mst",
    router_port=8049,
)

_SINGLE_MR = [SeriesInfo("series-1", "MR", "dynamic")]


class FakeClient:
    """Records calls and returns scripted results for the pipeline to consume.

    ``sr_before`` is what ``sr_instance_ids`` returns until a workitem is sent for
    a study; afterwards it returns ``sr_before | sr_after``. ``states`` is the
    sequence of workitem states handed out on successive polls (last value
    repeats once exhausted).
    """

    def __init__(
        self,
        *,
        study_of_file: dict[str, str] | None = None,
        series: list[SeriesInfo] | None = None,
        study_uid: str = "9.9.9",
        states: list[str] | None = None,
        sr_after: set[str] | None = None,
    ) -> None:
        self._study_of_file = study_of_file or {}
        self._series = series if series is not None else list(_SINGLE_MR)
        self._study_uid = study_uid
        self._states = states or ["COMPLETED"]
        self._sr_after = sr_after if sr_after is not None else {"sr-1"}
        self.sent: list[tuple[str, str, str, list[str], str, dict[str, str]]] = []
        self._poll_count = 0
        self._sent_studies: set[str] = set()

    def upload_instance(self, path: Path) -> str | None:
        return self._study_of_file.get(path.name, "S1")

    def study_instance_uid(self, study_id: str) -> str | None:
        return self._study_uid

    def study_series(self, study_id: str) -> list[SeriesInfo]:
        return list(self._series)

    def sr_instance_ids(self, study_id: str) -> set[str]:
        base = {"pre-existing-sr"}
        if study_id in self._sent_studies:
            return base | self._sr_after
        return base

    def send_to_ai(
        self,
        study_id: str,
        target: str,
        target_url: str,
        series_uids: list[str],
        *,
        input_configuration_id: str,
        input_mapping: dict[str, str],
    ) -> str:
        self.sent.append(
            (study_id, target, target_url, series_uids, input_configuration_id, dict(input_mapping))
        )
        self._sent_studies.add(study_id)
        return "wi-1"

    def workitem_state(self, router_port: int, workitem_uid: str) -> str | None:
        state = self._states[min(self._poll_count, len(self._states) - 1)]
        self._poll_count += 1
        return state


def _no_sleep(_seconds: float) -> None:
    return None


def test_completed_workitem_with_new_sr_marks_pair_created() -> None:
    client = FakeClient()
    files = [Path("a.dcm"), Path("b.dcm")]

    report = run_batch(client, files, [MST], poll_interval_s=0, sleep=_no_sleep)

    assert report.ok
    assert len(report.studies) == 1
    study = report.studies[0]
    assert study.orthanc_study_id == "S1"
    assert study.study_uid == "9.9.9"
    assert study.configuration_id == "multiphase"
    assert study.input_series_uids == ["series-1"]

    assert len(study.model_results) == 1
    result = study.model_results[0]
    assert result.model_name == "MST"
    assert result.final_state == "COMPLETED"
    assert result.created is True
    assert result.new_sr_ids == ["sr-1"]

    # The single MR series auto-selects the multiphase input configuration.
    assert client.sent == [
        (
            "S1",
            "ODELIA MST init weights preview",
            "http://orthanc-router-odelia-mst:8042/dicom-web",
            ["series-1"],
            "multiphase",
            {"multiphase": "series-1"},
        )
    ]


def test_mapped_pre_post_study_dispatches_pre_post_configuration() -> None:
    client = FakeClient(
        series=[SeriesInfo("s-pre", "MR", "t1 pre"), SeriesInfo("s-post", "MR", "t1 post")]
    )
    mapping = {"9.9.9": {"Pre": "s-pre", "Post_1": "s-post"}}

    report = run_batch(
        client, [Path("a.dcm")], [MST], sequence_mapping=mapping,
        poll_interval_s=0, sleep=_no_sleep,
    )

    assert report.ok
    study = report.studies[0]
    assert study.configuration_id == "pre_post"
    (_, _, _, series_uids, config_id, input_mapping) = client.sent[0]
    assert config_id == "pre_post"
    assert input_mapping == {"pre": "s-pre", "post": "s-post"}
    assert set(series_uids) == {"s-pre", "s-post"}


def test_mapped_sub_study_dispatches_subtraction_configuration() -> None:
    client = FakeClient(series=[SeriesInfo("s-sub", "MR", "scanner sub")])
    mapping = {"9.9.9": {"Sub_1": "s-sub"}}

    report = run_batch(
        client, [Path("a.dcm")], [MST], sequence_mapping=mapping,
        poll_interval_s=0, sleep=_no_sleep,
    )

    assert report.ok
    (_, _, _, series_uids, config_id, input_mapping) = client.sent[0]
    assert config_id == "subtraction"
    assert input_mapping == {"sub": "s-sub"}
    assert series_uids == ["s-sub"]


def test_ambiguous_multi_series_study_is_skipped_not_guessed() -> None:
    client = FakeClient(
        series=[SeriesInfo("s-1", "MR", "dyn"), SeriesInfo("s-2", "MR", "also dyn")]
    )

    report = run_batch(client, [Path("a.dcm")], [MST], poll_interval_s=0, sleep=_no_sleep)

    result = report.studies[0].model_results[0]
    assert result.created is False
    assert result.error is not None and "mapping" in result.error
    assert client.sent == []
    assert report.ok is False


def test_prior_ai_heatmap_series_does_not_block_auto_selection() -> None:
    client = FakeClient(
        series=[SeriesInfo("series-1", "MR", "dynamic"), SeriesInfo("s-heat", "MR", "MST - Heatmap")]
    )

    report = run_batch(client, [Path("a.dcm")], [MST], poll_interval_s=0, sleep=_no_sleep)

    assert report.ok
    assert client.sent[0][5] == {"multiphase": "series-1"}


def test_canceled_workitem_is_not_created_and_reports_reason() -> None:
    client = FakeClient(states=["IN_PROGRESS", "CANCELED"], sr_after=set())

    report = run_batch(client, [Path("a.dcm")], [MST], poll_interval_s=0, sleep=_no_sleep)

    result = report.studies[0].model_results[0]
    assert result.final_state == "CANCELED"
    assert result.created is False
    assert result.new_sr_ids == []
    assert result.error is not None and "CANCELED" in result.error
    assert report.ok is False


def test_completed_without_new_sr_is_not_created_and_reports_reason() -> None:
    client = FakeClient(states=["COMPLETED"], sr_after=set())

    report = run_batch(client, [Path("a.dcm")], [MST], poll_interval_s=0, sleep=_no_sleep)

    result = report.studies[0].model_results[0]
    assert result.final_state == "COMPLETED"
    assert result.created is False
    assert result.error is not None and "SR" in result.error
    assert report.ok is False


def test_timeout_while_in_flight_is_not_created_and_reports_reason() -> None:
    # poll_timeout_s=0 means the deadline is already past after the first poll, so
    # the workitem never leaves its in-flight state.
    client = FakeClient(states=["IN_PROGRESS"], sr_after=set())

    report = run_batch(
        client, [Path("a.dcm")], [MST], poll_timeout_s=0, poll_interval_s=0, sleep=_no_sleep
    )

    result = report.studies[0].model_results[0]
    assert result.final_state == "IN_PROGRESS"
    assert result.created is False
    assert result.error is not None and "terminal" in result.error
    assert report.ok is False


def test_non_mr_study_reports_skip_reason_without_triggering() -> None:
    client = FakeClient(series=[SeriesInfo("sr-series", "SR", "report")])

    report = run_batch(client, [Path("a.dcm")], [MST], poll_interval_s=0, sleep=_no_sleep)

    study = report.studies[0]
    assert study.configuration_id is None
    result = study.model_results[0]
    assert result.created is False
    assert result.error is not None and "MR" in result.error
    # A study with no resolvable input must never be sent to a model.
    assert client.sent == []
    assert report.ok is False


def test_two_studies_each_produce_a_created_result() -> None:
    client = FakeClient(
        study_of_file={"a.dcm": "S1", "b.dcm": "S2"}, states=["COMPLETED"], sr_after={"sr-x"}
    )

    report = run_batch(
        client, [Path("a.dcm"), Path("b.dcm")], [MST], poll_interval_s=0, sleep=_no_sleep
    )

    assert [s.orthanc_study_id for s in report.studies] == ["S1", "S2"]
    assert all(s.model_results[0].created for s in report.studies)
    assert report.uploaded_files == 2
    assert report.ok is True


def test_skipped_non_dicom_file_is_counted_and_not_a_study() -> None:
    client = FakeClient(study_of_file={"a.dcm": "S1", "note.txt": None})

    report = run_batch(
        client, [Path("a.dcm"), Path("note.txt")], [MST], poll_interval_s=0, sleep=_no_sleep
    )

    assert report.uploaded_files == 1
    assert report.skipped_files == 1
    assert [s.orthanc_study_id for s in report.studies] == ["S1"]


def test_no_studies_discovered_is_not_ok() -> None:
    client = FakeClient(study_of_file={"x.txt": None})

    report = run_batch(client, [Path("x.txt")], [MST], poll_interval_s=0, sleep=_no_sleep)

    assert report.studies == []
    assert report.skipped_files == 1
    assert report.ok is False
