"""Auto-mark tests by location; fail loudly on orphan tests.

Tests under tests/unit/   -> @pytest.mark.unit
Tests under tests/integration/ -> @pytest.mark.integration
Tests anywhere else that do not carry an explicit marker -> collection error.

Rationale: orphan tests would silently skip under `-m unit`, hiding regressions.
"""
import pytest


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
