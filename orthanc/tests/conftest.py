import os
import time
from typing import Dict

import pytest
import requests


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("ORTHANC_VIEWER_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session", autouse=True)
def wait_for_orthanc(base_url: str):
    # Wait up to 30s for health endpoint
    deadline = time.time() + 30
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/feedback/health", timeout=2)
            if r.status_code == 200:
                return
            last_error = f"HTTP {r.status_code}"
        except Exception as e:
            last_error = str(e)
        time.sleep(1)
    pytest.skip(f"orthanc-viewer not ready at {base_url}: {last_error}")


@pytest.fixture()
def unique_payload() -> Dict[str, str]:
    # Use time-based unique values
    ts = int(time.time() * 1000)
    return {
        "study_uid": f"1.2.826.0.1.3680043.2.1125.{ts}",
        "model_name": "TumorSeg",
        "model_version": "1.4.0",
        "result_ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "user_id": f"user-{ts}",
    }
