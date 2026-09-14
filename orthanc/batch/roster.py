"""Roster of odelia-classification model pairs and their host ports.

Mirrors the ``odelia-models`` compose profile (ODV-218). Backend 5560 belongs to
chat-middleware, hence the gap. Promoted out of the test tree (ODV-221) so it is
a single source shared by the integration suites and the batch tool.
"""

from typing import NamedTuple


class RosterModel(NamedTuple):
    model_name: str
    ai_name: str
    router_host: str
    router_port: int
    backend_port: int


_PREVIEW = "init weights preview"

ROSTER = [
    RosterModel(
        "agaldran", f"ODELIA agaldran {_PREVIEW}", "orthanc-router-odelia-agaldran", 8045, 5558
    ),
    RosterModel(
        "BCN_AIM", f"ODELIA BCN_AIM {_PREVIEW}", "orthanc-router-odelia-bcn-aim", 8046, 5559
    ),
    RosterModel(
        "DivideAndConquer",
        f"ODELIA DivideAndConquer {_PREVIEW}",
        "orthanc-router-odelia-divide-and-conquer",
        8047,
        5561,
    ),
    RosterModel(
        "LME_ABMIL", f"ODELIA LME_ABMIL {_PREVIEW}", "orthanc-router-odelia-lme-abmil", 8048, 5562
    ),
    RosterModel("MST", f"ODELIA MST {_PREVIEW}", "orthanc-router-odelia-mst", 8049, 5563),
    RosterModel("Pimed", f"ODELIA Pimed {_PREVIEW}", "orthanc-router-odelia-pimed", 8050, 5564),
]

ROSTER_IDS = [m.model_name for m in ROSTER]
