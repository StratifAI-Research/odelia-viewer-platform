"""HTTP client for the viewer's Orthanc and the model routers.

Wraps exactly the endpoints the pipeline needs: upload DICOM to the viewer's
Orthanc, describe a study's series, list its SR instances, trigger send-to-AI
with a manifest input configuration, and read a workitem's state. The transport
is injectable so the client is unit-testable without a live server; it defaults
to ``requests``.

Request shapes intentionally match ``tests/integration/test_model_roster_roundtrip.py``
(ODV-219): send-to-AI ``target_url`` uses the router's internal port 8042, while
workitem polling uses the host-mapped ``router_port``.
"""

from pathlib import Path
from typing import Any, Protocol, cast

import requests

from batch.selection import SeriesInfo

_DICOM_MAGIC_OFFSET = 128
_DICOM_MAGIC = b"DICM"
_WORKITEM_STATE_TAG = "00741000"  # Procedure Step State
_SEND_TIMEOUT_S = 120.0


class HttpTransport(Protocol):
    """The subset of ``requests`` the client uses."""

    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...


# requests satisfies HttpTransport at runtime; its stubs are stricter than the
# small surface we use, so cast rather than widen the Protocol.
_DEFAULT_HTTP: HttpTransport = cast("HttpTransport", requests)


class BatchError(RuntimeError):
    """A router/Orthanc response was malformed enough to abort a pair."""


def _looks_like_dicom(data: bytes) -> bool:
    """True if ``data`` has the DICOM part-10 preamble magic at offset 128."""
    end = _DICOM_MAGIC_OFFSET + len(_DICOM_MAGIC)
    return len(data) >= end and data[_DICOM_MAGIC_OFFSET:end] == _DICOM_MAGIC


class OrthancRouterClient:
    """Talks to the viewer Orthanc (``base_url``) and routers (``roster_host``)."""

    def __init__(
        self,
        base_url: str,
        *,
        roster_host: str = "http://localhost",
        http: HttpTransport = _DEFAULT_HTTP,
        timeout: float = 30.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._roster_host = roster_host.rstrip("/")
        self._http = http
        self._timeout = timeout

    def upload_instance(self, path: Path) -> str | None:
        """Upload one DICOM file; return its Orthanc study id, or None if skipped."""
        data = path.read_bytes()
        if not _looks_like_dicom(data):
            return None
        resp = self._http.post(
            f"{self._base}/instances",
            data=data,
            headers={"Content-Type": "application/dicom"},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            return None
        parent: str | None = resp.json().get("ParentStudy")
        return parent

    def study_instance_uid(self, study_id: str) -> str | None:
        """The study's DICOM StudyInstanceUID, or None if Orthanc does not report one."""
        resp = self._http.get(f"{self._base}/studies/{study_id}", timeout=self._timeout)
        resp.raise_for_status()
        uid: str | None = resp.json().get("MainDicomTags", {}).get("StudyInstanceUID")
        return uid

    def study_series(self, study_id: str) -> list[SeriesInfo]:
        """Every series in the study as (uid, modality, description)."""
        return [
            SeriesInfo(
                uid=s["MainDicomTags"].get("SeriesInstanceUID", ""),
                modality=s["MainDicomTags"].get("Modality", ""),
                description=s["MainDicomTags"].get("SeriesDescription", ""),
            )
            for s in self._series(study_id)
        ]

    def sr_instance_ids(self, study_id: str) -> set[str]:
        """Instance ids of every SR series in the study."""
        return {
            inst
            for s in self._series(study_id)
            if s["MainDicomTags"].get("Modality") == "SR"
            for inst in s.get("Instances", [])
        }

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
        """Trigger send-to-AI with an explicit input configuration; return the workitem UID."""
        resp = self._http.post(
            f"{self._base}/send-to-ai",
            json={
                "study_id": study_id,
                "target": target,
                "target_url": target_url,
                "series_uids": series_uids,
                "input_configuration_id": input_configuration_id,
                "input_mapping": input_mapping,
            },
            timeout=_SEND_TIMEOUT_S,
        )
        resp.raise_for_status()
        workitem_uid: str | None = resp.json().get("workitem_uid")
        if not workitem_uid:
            raise BatchError(f"send-to-ai returned no workitem_uid: {resp.text[:300]}")
        return workitem_uid

    def workitem_state(self, router_port: int, workitem_uid: str) -> str | None:
        """Read one workitem's Procedure Step State; None if the router has no such item."""
        resp = self._http.get(
            f"{self._roster_host}:{router_port}/ups-rs/workitems/{workitem_uid}",
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        state: str | None = resp.json().get(_WORKITEM_STATE_TAG, {}).get("Value", [None])[0]
        return state

    def _series(self, study_id: str) -> list[dict[str, Any]]:
        resp = self._http.get(f"{self._base}/studies/{study_id}/series", timeout=self._timeout)
        resp.raise_for_status()
        series: list[dict[str, Any]] = resp.json()
        return series
