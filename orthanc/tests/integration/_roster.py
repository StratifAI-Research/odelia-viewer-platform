"""Roster of odelia-classification model pairs and their host ports.

Mirrors the default trained pairs plus the optional Pimed preview. Backend 5560 belongs to
chat-middleware, hence the gap.
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
        "agaldran", "agaldran swarm May 2026", "orthanc-router-odelia-agaldran", 8045, 5558
    ),
    RosterModel("BCN_AIM", "SwinUNETR swarm May 2026", "orthanc-router-odelia-bcn-aim", 8046, 5559),
    RosterModel(
        "DivideAndConquer",
        "MeisenMeister swarm August 2026",
        "orthanc-router-odelia-divide-and-conquer",
        8047,
        5561,
    ),
    RosterModel("LME_ABMIL", "ABMIL swarm May 2026", "orthanc-router-odelia-lme-abmil", 8048, 5562),
    RosterModel("MST", "MST swarm May 2026", "orthanc-router-mst", 8043, 5556),
    RosterModel("Pimed", f"ODELIA Pimed {_PREVIEW}", "orthanc-router-odelia-pimed", 8050, 5564),
]

ROSTER_IDS = [m.model_name for m in ROSTER]
