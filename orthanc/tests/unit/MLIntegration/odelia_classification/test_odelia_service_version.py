"""SERVICE_VERSION reaches DICOM as AlgorithmVersion; pyproject.toml is the authority.

The service package is never pip-installed in its image (it runs from copied
source), so the two version strings cannot share one runtime source; this test
enforces the lockstep instead. Parsed lexically rather than with tomllib: the
smoke workflow runs this directory on Python 3.10, the service image's floor.
"""

import re
from pathlib import Path


def test_service_version_matches_pyproject():
    import config

    text = Path(config.__file__).with_name("pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match, "no version line in pyproject.toml"
    assert config.SERVICE_VERSION == match.group(1)
