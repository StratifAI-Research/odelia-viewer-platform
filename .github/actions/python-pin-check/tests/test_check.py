"""Tests for the python-pin-check action."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ACTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTION_DIR))

import check  # noqa: E402

FORTY_SHA = "0123456789abcdef0123456789abcdef01234567"
SHORT_SHA = "0123456"


def _write_service(
    root: Path,
    name: str,
    pyproject_deps: list[str],
    req_lines: list[str],
    build_requires: list[str] | None = None,
) -> None:
    svc = root / name
    svc.mkdir(parents=True, exist_ok=True)
    deps_block = ",\n  ".join(f'"{d}"' for d in pyproject_deps)
    build_block = ""
    if build_requires is not None:
        br = ", ".join(f'"{b}"' for b in build_requires)
        build_block = f"[build-system]\nrequires = [{br}]\n\n"
    (svc / "pyproject.toml").write_text(
        build_block
        + "[project]\n"
        + f'name = "{name.replace("/", "-")}"\n'
        + 'version = "0.1.0"\n'
        + "dependencies = [\n  " + deps_block + ("\n]\n" if pyproject_deps else "]\n")
    )
    (svc / "requirements.txt").write_text("\n".join(req_lines) + "\n")


@pytest.fixture
def fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Service tree with one passing service per SERVICES entry."""
    for svc in check.SERVICES:
        _write_service(tmp_path, svc, ["pydicom==3.0.2"], ["pydicom==3.0.2"])
    (tmp_path / "requirements-dev.txt").write_text("ruff==0.8.4\nmypy==1.13.0\n")
    (tmp_path / "requirements-tests.txt").write_text("pytest==9.0.2\nrequests==2.32.3\njq==1.6.0\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_clean_tree_returns_zero(fake_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert check.main() == 0
    out = capsys.readouterr().out
    for svc in check.SERVICES:
        assert f"✓ {svc}" in out
    assert "✓ requirements-dev.txt" in out


def test_unpinned_pyproject_dep_fails(fake_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (fake_tree / "viewer" / "pyproject.toml").write_text(
        '[project]\nname = "v"\nversion = "0"\ndependencies = ["flask>=2.0.0"]\n'
    )
    (fake_tree / "viewer" / "requirements.txt").write_text("flask>=2.0.0\n")
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "unpinned dep in pyproject.toml" in out


def test_unpinned_requirements_dep_fails(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (fake_tree / "viewer" / "requirements.txt").write_text("pydicom\n")
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "unpinned dep in requirements.txt" in out


def test_wildcard_pin_rejected(fake_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_service(fake_tree, "viewer", ["pkg==1.*"], ["pkg==1.*"])
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "unpinned dep" in out


def test_short_git_sha_rejected(fake_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dep = f"transformers @ git+https://github.com/huggingface/transformers.git@{SHORT_SHA}"
    _write_service(fake_tree, "viewer", [dep], [dep])
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "unpinned dep" in out


def test_full_git_sha_accepted(fake_tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dep = f"transformers @ git+https://github.com/huggingface/transformers.git@{FORTY_SHA}"
    _write_service(fake_tree, "viewer", [dep], [dep])
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "✓ viewer" in out


def test_pyproject_requirements_drift_fails(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_service(fake_tree, "viewer", ["pydicom==3.0.2"], ["pydicom==3.0.1"])
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "disagree" in out


def test_per_service_success_printed_after_earlier_failure(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If an earlier service fails, later passing services must still print ✓."""
    _write_service(fake_tree, "viewer", ["pydicom>=3.0"], ["pydicom>=3.0"])
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "✓ router" in out
    assert "✓ MLIntegration/shared" in out


def test_requirements_dev_unpinned_fails(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (fake_tree / "requirements-dev.txt").write_text("ruff\nmypy==1.13.0\n")
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "requirements-dev.txt" in out
    assert "unpinned dep" in out


def test_case_insensitive_name_comparison(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`Flask==X` in pyproject vs `flask==X` in requirements must NOT be drift."""
    _write_service(fake_tree, "viewer", ["Flask==3.1.3"], ["flask==3.1.3"])
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "disagree" not in out
    assert "✓ viewer" in out


def test_dash_underscore_dot_equivalence(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PEP 503: `flask-cors`, `flask_cors`, `flask.cors` are the same name."""
    _write_service(fake_tree, "viewer", ["flask_cors==6.0.2"], ["flask-cors==6.0.2"])
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "disagree" not in out


def test_unpinned_build_system_requires_fails(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`build-system.requires` entries (setuptools, wheel) must be pinned too.

    Unpinned build-system deps let PEP 517 build envs pull the latest from PyPI
    at build time, which makes Docker `pip install -e .` non-reproducible even
    when [project].dependencies are fully pinned.
    """
    _write_service(
        fake_tree,
        "viewer",
        ["pydicom==3.0.2"],
        ["pydicom==3.0.2"],
        build_requires=["setuptools>=61"],
    )
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "[build-system].requires" in out
    assert "setuptools>=61" in out


def test_pinned_build_system_requires_passes(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_service(
        fake_tree,
        "viewer",
        ["pydicom==3.0.2"],
        ["pydicom==3.0.2"],
        build_requires=["setuptools==82.0.1", "wheel==0.47.0"],
    )
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "✓ viewer" in out


def test_requirements_tests_unpinned_fails(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (fake_tree / "requirements-tests.txt").write_text("pytest\nrequests==2.32.3\n")
    assert check.main() == 1
    out = capsys.readouterr().out
    assert "requirements-tests.txt" in out
    assert "unpinned dep" in out
