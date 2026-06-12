"""Auto-mark tests by location; fail loudly on orphan tests.

Tests under tests/unit/   -> @pytest.mark.unit
Tests under tests/integration/ -> @pytest.mark.integration
Tests anywhere else that do not carry an explicit marker -> collection error.

Rationale: orphan tests would silently skip under `-m unit`, hiding regressions.
"""
from pathlib import Path

import pytest

# Real anonymized DICOM committed under orthanc/sample_data/mri (155-slice MR series).
_SAMPLE_MRI_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "mri"


@pytest.fixture(scope="session")
def mri_sample_dir() -> Path:
    """Directory of the committed MRI sample series; skips if not present."""
    if not _SAMPLE_MRI_DIR.is_dir() or not any(_SAMPLE_MRI_DIR.glob("*.dcm")):
        pytest.skip("MRI sample data (orthanc/sample_data/mri) not available")
    return _SAMPLE_MRI_DIR


@pytest.fixture
def mri_sample_file(mri_sample_dir: Path) -> Path:
    """A single representative .dcm from the MRI sample series."""
    return sorted(mri_sample_dir.glob("*.dcm"))[0]


def pytest_collection_modifyitems(config, items):
    orphans = []
    for item in items:
        p = str(item.path).replace("\\", "/")
        marks = {m.name for m in item.iter_markers()}
        if "unit" in marks or "integration" in marks:
            continue
        if "/tests/unit/" in p:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in p:
            item.add_marker(pytest.mark.integration)
        else:
            orphans.append(p)
    if orphans:
        raise pytest.UsageError(
            "Tests outside tests/unit/ and tests/integration/ must carry an explicit "
            "@pytest.mark.unit or @pytest.mark.integration. Offenders:\n  "
            + "\n  ".join(orphans)
        )
