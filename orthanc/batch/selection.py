"""Manifest-aware input selection for batch send-to-AI (ODV-221).

The model services declare three input configurations in their manifest.json
(multiphase / pre_post / subtraction). This module decides, per study, which
configuration to dispatch and which series fill its input roles:

  - with a sequence mapping (CSV): a mapped ``Sub_1`` selects ``subtraction``,
    else ``Pre`` + ``Post_1`` selects ``pre_post``
  - without: a study with exactly one non-AI MR series selects ``multiphase``
  - anything ambiguous becomes a skip reason string, never a guess
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class SeriesInfo(NamedTuple):
    """One series as the viewer's Orthanc reports it."""

    uid: str
    modality: str
    description: str


@dataclass(frozen=True)
class InputSelection:
    """A resolved dispatch: manifest configuration id plus role -> series UID."""

    configuration_id: str
    mapping: dict[str, str]
    series_uids: list[str]


MAPPING_COLUMNS = ("PatientID", "StudyInstanceUID", "SequenceName", "SeriesPath")

# AI result series markers, mirroring viewer/router.py FilterAIResultSeries.
_AI_MARKERS = (
    "Automated Diagnostic Findings",
    "- Heatmap",
    "AI Analysis Result",
    "AI Generated",
    "Secondary Capture AI",
    "AI Structured Report",
)


def is_ai_result(modality: str, description: str) -> bool:
    """True if a series looks like an AI result rather than acquired data."""
    return (
        any(marker in description for marker in _AI_MARKERS)
        or (modality in {"SC", "SR"} and "AI" in description.upper())
        or description.startswith("AI_")
        or description.endswith("_AI")
    )


def resolve_input_selection(
    series: list[SeriesInfo], study_mapping: dict[str, str] | None
) -> InputSelection | str:
    """Pick an input configuration for a study; a string return is a skip reason."""
    present = {s.uid for s in series}

    if study_mapping:
        sub_uid = study_mapping.get("Sub_1")
        if sub_uid:
            if sub_uid not in present:
                return f"mapped Sub_1 series not found in study: {sub_uid}"
            return InputSelection("subtraction", {"sub": sub_uid}, [sub_uid])
        pre_uid = study_mapping.get("Pre")
        post_uid = study_mapping.get("Post_1")
        if pre_uid and post_uid:
            missing = [uid for uid in (pre_uid, post_uid) if uid not in present]
            if missing:
                return f"mapped series not found in study: {', '.join(missing)}"
            return InputSelection(
                "pre_post", {"pre": pre_uid, "post": post_uid}, [pre_uid, post_uid]
            )
        return "mapping provides neither Sub_1 nor Pre+Post_1"

    candidates = [
        s for s in series if s.modality == "MR" and not is_ai_result(s.modality, s.description)
    ]
    if len(candidates) == 1:
        uid = candidates[0].uid
        return InputSelection("multiphase", {"multiphase": uid}, [uid])
    if not candidates:
        return "no MR series in study"
    return f"{len(candidates)} MR series and no mapping; cannot choose"


def _read_series_uid(series_dir: Path) -> str | None:
    """SeriesInstanceUID from the first readable DICOM in a series directory."""
    # pydicom is only needed when a mapping CSV is used; keep the import local so
    # mapping-less runs need no extra dependency.
    import pydicom

    if not series_dir.is_dir():
        return None
    for path in sorted(series_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            return str(ds.SeriesInstanceUID)
        except Exception:
            continue
    return None


def load_sequence_mapping(
    csv_path: Path, data_raw: Path
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Read the mapping CSV into {StudyInstanceUID: {SequenceName: SeriesInstanceUID}}.

    Series UIDs are read from the DICOM files themselves (folder names are not
    trusted). Unreadable rows are dropped and reported as warnings.
    """
    mapping: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in MAPPING_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"mapping is missing columns: {missing}")
        for row in reader:
            series_dir = data_raw / row["SeriesPath"]
            uid = _read_series_uid(series_dir)
            if uid is None:
                warnings.append(
                    f"{row['StudyInstanceUID']}/{row['SequenceName']}: "
                    f"no readable DICOM in {series_dir}"
                )
                continue
            mapping.setdefault(row["StudyInstanceUID"], {})[row["SequenceName"]] = uid
    return mapping, warnings
