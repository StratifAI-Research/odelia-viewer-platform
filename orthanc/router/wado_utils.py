"""WADO-RS metadata retrieval utilities"""
import numpy as np
from dicomweb_client.api import DICOMwebClient
from typing import Tuple, Dict, List
from collections import defaultdict

def retrieve_series_metadata_sorted(wado_rs_retrieval: List[dict]) -> Tuple[Dict, List[List[float]], float]:
    """
    Retrieve series metadata only (no pixel data) and return sorted by temporal phase and InstanceNumber

    Args:
        wado_rs_retrieval: List of dicts with retrieval_url, study_uid, series_uid

    Returns:
        (first_instance_metadata, list_of_positions, slice_spacing)
        - first_instance_metadata: metadata dict for first sorted instance
        - list_of_positions: list of ImagePositionPatient [x,y,z] for each frame in order
        - slice_spacing: calculated spacing as fallback (float)
    """
    first_retrieval = wado_rs_retrieval[0]
    base_url = first_retrieval["retrieval_url"].split("/studies/")[0]

    client = DICOMwebClient(url=base_url)
    instances_metadata = client.retrieve_series_metadata(
        study_instance_uid=first_retrieval["study_uid"],
        series_instance_uid=first_retrieval["series_uid"]
    )

    print(f"Retrieved {len(instances_metadata)} instances")

    # Extract instances with required tags
    # Tags: 00200032=ImagePositionPatient, 00200013=InstanceNumber, 00200100=TemporalPositionIdentifier
    instance_data = []
    for inst_meta in instances_metadata:
        ipp_tag = inst_meta.get("00200032")  # ImagePositionPatient
        instance_num_tag = inst_meta.get("00200013")  # InstanceNumber
        temporal_tag = inst_meta.get("00200100")  # TemporalPositionIdentifier

        if ipp_tag and ipp_tag.get("Value") and instance_num_tag and instance_num_tag.get("Value"):
            ipp_values = ipp_tag["Value"]
            if len(ipp_values) >= 3:
                position = [float(v) for v in ipp_values]
                instance_number = int(instance_num_tag["Value"][0])

                # Get temporal position (default to 1 if not present = single temporal phase)
                temporal_position = 1
                if temporal_tag and temporal_tag.get("Value"):
                    temporal_position = int(temporal_tag["Value"][0])

                instance_data.append({
                    "metadata": inst_meta,
                    "position": position,
                    "instance_number": instance_number,
                    "temporal_position": temporal_position
                })

    if not instance_data:
        raise ValueError("No instances with ImagePositionPatient (00200032) and InstanceNumber (00200013) found")

    # Group by temporal position (matches model behavior)
    temporal_groups = defaultdict(list)
    for item in instance_data:
        temporal_groups[item['temporal_position']].append(item)

    num_temporal_phases = len(temporal_groups)
    print(f"Detected {num_temporal_phases} temporal phase(s): {sorted(temporal_groups.keys())}")

    # Select first temporal phase (matches model: dicom_utils.py line 110)
    sorted_temporal_keys = sorted(temporal_groups.keys())
    first_temporal_key = sorted_temporal_keys[0]
    selected_instances = temporal_groups[first_temporal_key]

    print(f"Using temporal phase {first_temporal_key} with {len(selected_instances)} instances")

    # Sort by InstanceNumber ascending within temporal group
    selected_instances.sort(key=lambda x: x['instance_number'])

    print(f"Sorted {len(selected_instances)} instances by InstanceNumber: {selected_instances[0]['instance_number']} to {selected_instances[-1]['instance_number']}")

    # Extract positions list for all frames
    positions_list = [item["position"] for item in selected_instances]

    # Calculate spacing from first two instances as fallback
    slice_spacing = 1.0  # Default fallback
    if len(selected_instances) >= 2:
        pos_diff = np.array(selected_instances[1]["position"]) - np.array(selected_instances[0]["position"])
        slice_spacing = float(np.linalg.norm(pos_diff))

    print(f"Using slice spacing: {slice_spacing:.2f}mm (fallback)")

    return selected_instances[0]["metadata"], positions_list, slice_spacing
