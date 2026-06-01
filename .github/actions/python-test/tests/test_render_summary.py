"""Tests for the python-test composite action's summary renderer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ACTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTION_DIR))

import render_summary  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_passing_suite(work_dir: Path) -> None:
    outputs = work_dir / "outputs"
    (outputs / "pytest.xml").write_text((FIXTURES / "pytest-mixed.xml").read_text())
    (outputs / "pytest-coverage.xml").write_text((FIXTURES / "coverage.xml").read_text())

    render_summary.main()

    summary = (outputs / "summary.md").read_text()
    metadata = json.loads((outputs / "metadata.json").read_text())
    assert "## Test PY (pytest)" in summary
    assert "Collected: **10**" in summary
    assert "Passed: **8**" in summary
    assert "Failed: **1**" in summary
    assert "Skipped: **1**" in summary
    assert "80/200" in summary
    assert metadata["tool"] == "pytest+coverage"


def test_zero_tests_collected(work_dir: Path) -> None:
    outputs = work_dir / "outputs"
    (outputs / "pytest.xml").write_text((FIXTURES / "pytest-empty.xml").read_text())

    with pytest.raises(SystemExit) as exc_info:
        render_summary.main()
    assert exc_info.value.code == 1

    summary = (outputs / "summary.md").read_text()
    assert "zero tests collected" in summary


def test_missing_junit_xml(work_dir: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        render_summary.main()
    assert exc_info.value.code == 1

    summary = (work_dir / "outputs" / "summary.md").read_text()
    assert "pytest did not produce" in summary


def test_malformed_junit_xml(work_dir: Path) -> None:
    (work_dir / "outputs" / "pytest.xml").write_text("this is not xml at all <<<>>>")

    with pytest.raises(SystemExit) as exc_info:
        render_summary.main()
    assert exc_info.value.code == 1

    summary = (work_dir / "outputs" / "summary.md").read_text()
    assert "not valid XML" in summary


def test_missing_coverage_xml(work_dir: Path) -> None:
    outputs = work_dir / "outputs"
    (outputs / "pytest.xml").write_text((FIXTURES / "pytest-mixed.xml").read_text())

    render_summary.main()

    summary = (outputs / "summary.md").read_text()
    assert "0/0" in summary
    assert "0.0%" in summary
