"""Pure orchestration for batch send-to-AI over an injected client.

For every study discovered in an input folder and every requested model, this
resolves the study's manifest input configuration (see :mod:`batch.selection`),
triggers the viewer's send-to-AI flow, polls the resulting UPS-RS workitem to a
terminal state, and confirms a new SR was written back to the study. All I/O is
delegated to a :class:`BatchClient`, so the orchestration is tested without a
live Orthanc (see ``tests/unit/batch/test_pipeline.py``).
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from batch.selection import InputSelection, SeriesInfo, resolve_input_selection

_IN_FLIGHT = frozenset({"SCHEDULED", "IN PROGRESS", "IN_PROGRESS"})
_COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class ModelSpec:
    """A model routing target: its display name and where its router lives."""

    model_name: str
    ai_name: str
    router_host: str
    router_port: int

    @property
    def target_url(self) -> str:
        # The viewer container reaches the router over the compose network on the
        # fixed internal port 8042; router_port is the host-mapped polling port.
        return f"http://{self.router_host}:8042/dicom-web"


class BatchClient(Protocol):
    """The I/O boundary the pipeline drives (Orthanc + router HTTP)."""

    def upload_instance(self, path: Path) -> str | None: ...

    def study_instance_uid(self, study_id: str) -> str | None: ...

    def study_series(self, study_id: str) -> list[SeriesInfo]: ...

    def sr_instance_ids(self, study_id: str) -> set[str]: ...

    def send_to_ai(
        self,
        study_id: str,
        target: str,
        target_url: str,
        series_uids: list[str],
        *,
        input_configuration_id: str,
        input_mapping: dict[str, str],
    ) -> str: ...

    def workitem_state(self, router_port: int, workitem_uid: str) -> str | None: ...


@dataclass
class ModelResult:
    """Outcome of one (study, model) send-to-AI round-trip."""

    model_name: str
    ai_name: str
    workitem_uid: str | None
    final_state: str | None
    created: bool
    new_sr_ids: list[str]
    elapsed_s: float
    error: str | None = None


@dataclass
class StudyResult:
    """One uploaded study and its per-model results."""

    orthanc_study_id: str
    study_uid: str | None
    configuration_id: str | None
    input_series_uids: list[str] = field(default_factory=list)
    model_results: list[ModelResult] = field(default_factory=list)


@dataclass
class BatchReport:
    """The full manifest of a batch run."""

    studies: list[StudyResult] = field(default_factory=list)
    uploaded_files: int = 0
    skipped_files: int = 0

    @property
    def ok(self) -> bool:
        """True only if every study produced a created result for every model."""
        if not self.studies:
            return False
        for study in self.studies:
            if not study.model_results:
                return False
            if not all(result.created for result in study.model_results):
                return False
        return True


def run_batch(
    client: BatchClient,
    files: Sequence[Path],
    models: Sequence[ModelSpec],
    *,
    sequence_mapping: dict[str, dict[str, str]] | None = None,
    poll_timeout_s: float = 900.0,
    poll_interval_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BatchReport:
    """Upload ``files``, then run each discovered study against each model."""
    report = BatchReport()

    study_ids: list[str] = []
    for path in files:
        study_id = client.upload_instance(path)
        if study_id is None:
            report.skipped_files += 1
            continue
        report.uploaded_files += 1
        if study_id not in study_ids:
            study_ids.append(study_id)

    for study_id in study_ids:
        study_uid = client.study_instance_uid(study_id)
        series = client.study_series(study_id)
        study_mapping = (sequence_mapping or {}).get(study_uid) if study_uid else None
        selection = resolve_input_selection(series, study_mapping)

        if isinstance(selection, str):
            study_result = StudyResult(
                orthanc_study_id=study_id, study_uid=study_uid, configuration_id=None
            )
            study_result.model_results = [_skipped_result(model, selection) for model in models]
            report.studies.append(study_result)
            continue

        study_result = StudyResult(
            orthanc_study_id=study_id,
            study_uid=study_uid,
            configuration_id=selection.configuration_id,
            input_series_uids=list(selection.series_uids),
        )
        for model in models:
            study_result.model_results.append(
                _run_pair(
                    client,
                    study_id,
                    selection,
                    model,
                    poll_timeout_s=poll_timeout_s,
                    poll_interval_s=poll_interval_s,
                    sleep=sleep,
                    clock=clock,
                )
            )
        report.studies.append(study_result)

    return report


def _skipped_result(model: ModelSpec, reason: str) -> ModelResult:
    """A pair that was never dispatched because the study's input could not be resolved."""
    return ModelResult(
        model_name=model.model_name,
        ai_name=model.ai_name,
        workitem_uid=None,
        final_state=None,
        created=False,
        new_sr_ids=[],
        elapsed_s=0.0,
        error=reason,
    )


def _run_pair(
    client: BatchClient,
    study_id: str,
    selection: InputSelection,
    model: ModelSpec,
    *,
    poll_timeout_s: float,
    poll_interval_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> ModelResult:
    started = clock()
    before = client.sr_instance_ids(study_id)
    workitem_uid = client.send_to_ai(
        study_id,
        model.ai_name,
        model.target_url,
        list(selection.series_uids),
        input_configuration_id=selection.configuration_id,
        input_mapping=selection.mapping,
    )
    final_state = _poll_workitem(
        client,
        model.router_port,
        workitem_uid,
        poll_timeout_s=poll_timeout_s,
        poll_interval_s=poll_interval_s,
        sleep=sleep,
        clock=clock,
    )
    new_sr_ids = sorted(client.sr_instance_ids(study_id) - before)
    created, error = _classify(final_state, new_sr_ids, poll_timeout_s)
    return ModelResult(
        model_name=model.model_name,
        ai_name=model.ai_name,
        workitem_uid=workitem_uid,
        final_state=final_state,
        created=created,
        new_sr_ids=new_sr_ids,
        elapsed_s=clock() - started,
        error=error,
    )


def _classify(
    final_state: str | None, new_sr_ids: list[str], poll_timeout_s: float
) -> tuple[bool, str | None]:
    """Turn a terminal (or timed-out) workitem state into (created, error)."""
    if final_state == _COMPLETED:
        if new_sr_ids:
            return True, None
        return False, "workitem COMPLETED but no SR was written back to the viewer"
    if final_state is None or final_state in _IN_FLIGHT:
        return False, (
            f"workitem did not reach a terminal state within {poll_timeout_s:g}s "
            f"(last state: {final_state})"
        )
    return False, f"workitem ended {final_state}, not COMPLETED"


def _poll_workitem(
    client: BatchClient,
    router_port: int,
    workitem_uid: str,
    *,
    poll_timeout_s: float,
    poll_interval_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> str | None:
    deadline = clock() + poll_timeout_s
    state = client.workitem_state(router_port, workitem_uid)
    while (state is None or state in _IN_FLIGHT) and clock() < deadline:
        sleep(poll_interval_s)
        state = client.workitem_state(router_port, workitem_uid)
    return state
