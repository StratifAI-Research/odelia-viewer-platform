"""Command-line entry point for the batch send-to-AI tool (ODV-221).

    python -m batch --input <dicom-folder> [--models MST,agaldran | all]
                    [--mapping sequence_mapping.csv [--data-raw <root>]]

Uploads every DICOM under ``--input`` to the viewer's Orthanc, resolves each
study's manifest input configuration (multiphase / pre_post / subtraction, see
``batch.selection``), runs send-to-AI for each (study, model) pair, and writes
a report. Exits non-zero if any pair failed to produce an AI result.
"""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from batch.client import OrthancRouterClient
from batch.pipeline import BatchReport, ModelSpec, run_batch
from batch.roster import ROSTER, ROSTER_IDS
from batch.selection import load_sequence_mapping

_DEFAULT_BASE_URL = os.environ.get("ORTHANC_VIEWER_BASE_URL", "http://localhost:8000")
_DEFAULT_ROSTER_HOST = os.environ.get("ROSTER_HOST", "http://localhost")


def resolve_models(names: Sequence[str] | None) -> list[ModelSpec]:
    """Map model names (or None = all) to ModelSpecs, preserving the given order."""
    if not names:
        selected = list(ROSTER)
    else:
        by_name = {m.model_name: m for m in ROSTER}
        selected = []
        for name in names:
            if name not in by_name:
                raise ValueError(f"unknown model {name!r}; known models: {ROSTER_IDS}")
            selected.append(by_name[name])
    return [ModelSpec(m.model_name, m.ai_name, m.router_host, m.router_port) for m in selected]


def discover_files(input_dir: Path) -> list[Path]:
    """Every file under ``input_dir`` (recursive), sorted for deterministic order."""
    return sorted((p for p in input_dir.rglob("*") if p.is_file()), key=str)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="batch", description="Batch send-to-AI preloader.")
    parser.add_argument("--input", type=Path, required=True, help="Folder of DICOM files.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names, or 'all' (default). E.g. MST,agaldran",
    )
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="Viewer Orthanc base URL.")
    parser.add_argument(
        "--roster-host", default=_DEFAULT_ROSTER_HOST, help="Host for router workitem polling."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Sequence mapping CSV (PatientID,StudyInstanceUID,SequenceName,SeriesPath).",
    )
    parser.add_argument(
        "--data-raw",
        type=Path,
        default=None,
        help="Root that mapping SeriesPath entries are relative to (default: --input).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the JSON report here.")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="Seconds to wait per workitem."
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Seconds between workitem polls."
    )
    args = parser.parse_args(argv)
    args.models = _normalize_models(args.models)
    if args.data_raw is None:
        args.data_raw = args.input
    return args


def _normalize_models(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip().lower() == "all":
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def _report_to_dict(report: BatchReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "uploaded_files": report.uploaded_files,
        "skipped_files": report.skipped_files,
        "studies": [
            {
                "orthanc_study_id": study.orthanc_study_id,
                "study_uid": study.study_uid,
                "input_configuration_id": study.configuration_id,
                "input_series_uids": study.input_series_uids,
                "models": [asdict(result) for result in study.model_results],
            }
            for study in report.studies
        ],
    }


def _print_summary(report: BatchReport) -> None:
    for study in report.studies:
        config = study.configuration_id or "unresolved"
        print(f"Study {study.study_uid or study.orthanc_study_id} ({config}):")
        for result in study.model_results:
            tag = "OK  " if result.created else "FAIL"
            detail = f"sr:{len(result.new_sr_ids)}" if result.created else (result.error or "")
            print(f"  [{tag}] {result.model_name:<18} {result.final_state or '-':<12} {detail}")
    created = sum(r.created for s in report.studies for r in s.model_results)
    total = sum(len(s.model_results) for s in report.studies)
    print(f"{created}/{total} pairs created across {len(report.studies)} study(ies).")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models = resolve_models(args.models)

    sequence_mapping = None
    if args.mapping is not None:
        sequence_mapping, warnings = load_sequence_mapping(args.mapping, args.data_raw)
        for warning in warnings:
            print(f"mapping warning: {warning}")

    client = OrthancRouterClient(args.base_url, roster_host=args.roster_host)
    files = discover_files(args.input)
    report = run_batch(
        client,
        files,
        models,
        sequence_mapping=sequence_mapping,
        poll_timeout_s=args.timeout,
        poll_interval_s=args.poll_interval,
    )
    if args.out is not None:
        args.out.write_text(json.dumps(_report_to_dict(report), indent=2))
    _print_summary(report)
    return 0 if report.ok else 1
