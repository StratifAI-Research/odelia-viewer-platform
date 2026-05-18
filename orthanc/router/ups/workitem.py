"""
UPS Workitem management with DICOM JSON format
Based on DICOM PS3.4 Section CC and PS3.18 Section 11
"""

import json
from datetime import datetime
from pydicom.uid import generate_uid


class UPSWorkitem:
    """
    UPS workitem using DICOM JSON format (application/dicom+json)
    """

    def __init__(self, study_uid, series_uids, wado_rs_retrieval, priority="MEDIUM",
                 workitem_uid=None, input_mapping=None, input_configuration_id=None):
        """
        Create a new UPS workitem

        Args:
            study_uid: StudyInstanceUID
            series_uids: List of SeriesInstanceUIDs
            wado_rs_retrieval: List of dicts with retrieval_url, study_uid, series_uid
            priority: MEDIUM, HIGH, or LOW
            workitem_uid: Optional existing workitem UID (for deserialization)
            input_mapping: Optional dict {role_key: series_uid} for structured input
            input_configuration_id: Optional string identifying the input configuration
        """
        self.workitem_uid = workitem_uid or generate_uid()
        self.data = self._create_dicom_json(
            study_uid, series_uids, wado_rs_retrieval, priority,
            input_mapping, input_configuration_id
        )

    def _create_dicom_json(self, study_uid, series_uids, wado_rs_retrieval, priority,
                           input_mapping=None, input_configuration_id=None):
        """Create DICOM JSON structure per DICOMweb standard"""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        dicom_json = {
            "00080016": {"vr": "UI", "Value": ["1.2.840.10008.5.1.4.34.6.1"]},  # SOPClassUID: UPS Push
            "00080018": {"vr": "UI", "Value": [self.workitem_uid]},  # SOPInstanceUID
            "0020000D": {"vr": "UI", "Value": [study_uid]},  # StudyInstanceUID
            "00741000": {"vr": "CS", "Value": ["SCHEDULED"]},  # ProcedureStepState
            "00741200": {"vr": "CS", "Value": [priority]},  # ScheduledProcedureStepPriority
            "00741202": {"vr": "LO", "Value": ["AI-INFERENCE"]},  # WorklistLabel
            "00741204": {"vr": "LO", "Value": ["AI Model Inference"]},  # ProcedureStepLabel
            "00404005": {"vr": "DT", "Value": [date_str + time_str]},  # ScheduledProcedureStepStartDateTime
            "00404041": {"vr": "CS", "Value": ["READY"]},  # InputReadinessState

            # Input Information Sequence with WADO-RS retrieval
            "00404021": {"vr": "SQ", "Value": self._build_input_sequence(wado_rs_retrieval)},

            # Scheduled Workitem Code Sequence
            "00404018": {"vr": "SQ", "Value": [{
                "00080100": {"vr": "SH", "Value": ["110004"]},  # Computer Aided Detection
                "00080102": {"vr": "SH", "Value": ["DCM"]},
                "00080104": {"vr": "LO", "Value": ["Computer Aided Detection"]}
            }]},
        }

        # Scheduled Processing Parameters Sequence (0074,1210) for structured input mapping
        if input_mapping:
            processing_params = self._build_processing_parameters(input_mapping, input_configuration_id)
            if processing_params:
                dicom_json["00741210"] = {"vr": "SQ", "Value": processing_params}

        return dicom_json

    def _build_input_sequence(self, wado_rs_retrieval):
        """Build Input Information Sequence with WADO-RS URLs"""
        input_items = []
        for item in wado_rs_retrieval:
            input_items.append({
                "0040E020": {"vr": "CS", "Value": ["DICOM"]},  # TypeOfInstances
                "0020000D": {"vr": "UI", "Value": [item["study_uid"]]},
                "0020000E": {"vr": "UI", "Value": [item["series_uid"]]},
                "0040E025": {"vr": "SQ", "Value": [{  # WADO-RS Retrieval Sequence
                    "00081190": {"vr": "UR", "Value": [item["retrieval_url"]]},
                    "0040E011": {"vr": "UI", "Value": [generate_uid()]}  # Retrieve Location UID
                }]}
            })
        return input_items

    def _build_processing_parameters(self, input_mapping, input_configuration_id=None):
        """
        Build Scheduled Processing Parameters Sequence (0074,1210) using
        Content Item Macro (DICOM PS3.3 Table 10-2).

        Each role mapping is a UIDREF content item with:
          - Concept Name Code Sequence: role key under private scheme 99ODELIA
          - UID value: the mapped SeriesInstanceUID

        The input configuration ID is stored as a TEXT content item.
        """
        items = []

        for role_key, series_uid in input_mapping.items():
            items.append({
                "0040A040": {"vr": "CS", "Value": ["UIDREF"]},
                "0040A043": {"vr": "SQ", "Value": [{
                    "00080100": {"vr": "SH", "Value": [role_key]},
                    "00080102": {"vr": "SH", "Value": ["99ODELIA"]},
                    "00080104": {"vr": "LO", "Value": [f"{role_key} Input Series"]}
                }]},
                "0040A124": {"vr": "UI", "Value": [series_uid]}
            })

        if input_configuration_id:
            items.append({
                "0040A040": {"vr": "CS", "Value": ["TEXT"]},
                "0040A043": {"vr": "SQ", "Value": [{
                    "00080100": {"vr": "SH", "Value": ["inputConfigId"]},
                    "00080102": {"vr": "SH", "Value": ["99ODELIA"]},
                    "00080104": {"vr": "LO", "Value": ["Input Configuration Identifier"]}
                }]},
                "0040A160": {"vr": "LO", "Value": [input_configuration_id]}
            })

        return items

    def get_input_mapping(self):
        """
        Parse Scheduled Processing Parameters (0074,1210) back into a
        structured mapping: {role_key: series_uid}.
        Returns None if no structured mapping is present (legacy workitem).
        """
        proc_params = self.data.get("00741210", {}).get("Value", [])
        if not proc_params:
            return None

        mapping = {}
        config_id = None

        for item in proc_params:
            value_type = item.get("0040A040", {}).get("Value", [None])[0]
            concept_seq = item.get("0040A043", {}).get("Value", [{}])
            if not concept_seq:
                continue
            concept = concept_seq[0]
            coding_scheme = concept.get("00080102", {}).get("Value", [None])[0]

            if coding_scheme != "99ODELIA":
                continue

            code_value = concept.get("00080100", {}).get("Value", [None])[0]
            if not code_value:
                continue

            if value_type == "UIDREF":
                uid = item.get("0040A124", {}).get("Value", [None])[0]
                if uid:
                    mapping[code_value] = uid
            elif value_type == "TEXT" and code_value == "inputConfigId":
                config_id = item.get("0040A160", {}).get("Value", [None])[0]

        if not mapping:
            return None

        return {"mapping": mapping, "input_configuration_id": config_id}

    def update_state(self, new_state, progress_percent=None, progress_description=None, cancellation_reason=None):
        """
        Update ProcedureStepState and/or progress information

        Args:
            new_state: SCHEDULED, IN_PROGRESS, COMPLETED, or CANCELED
            progress_percent: Optional progress percentage (0-100) for IN_PROGRESS state
            progress_description: Optional textual description of progress
            cancellation_reason: Optional reason for cancellation (used when state is CANCELED)
        """
        # Always update state if provided
        if new_state:
            self.data["00741000"] = {"vr": "CS", "Value": [new_state]}

        # Handle progress information (for IN_PROGRESS state OR when updating existing IN_PROGRESS)
        current_state = self.data["00741000"]["Value"][0]
        if current_state == "IN_PROGRESS" and (progress_percent is not None or progress_description is not None):
            progress_item = {}
            if progress_percent is not None:
                progress_item["00741004"] = {"vr": "DS", "Value": [str(progress_percent)]}  # Procedure Step Progress
            if progress_description:
                progress_item["00741006"] = {"vr": "ST", "Value": [progress_description]}  # Procedure Step Progress Description

            if progress_item:
                self.data["00741002"] = {"vr": "SQ", "Value": [progress_item]}  # Progress Information Sequence

        # Handle cancellation information for CANCELED state
        if new_state == "CANCELED":
            now = datetime.now()
            datetime_str = now.strftime("%Y%m%d%H%M%S")
            self.data["00404052"] = {"vr": "DT", "Value": [datetime_str]}  # Procedure Step Cancellation DateTime

            if cancellation_reason:
                self.data["00741238"] = {"vr": "LO", "Value": [cancellation_reason]}  # Reason For Cancellation

    def add_output_reference(self, series_uid, study_uid):
        """Add to Output Information Sequence (0040,4033)"""
        if "00404033" not in self.data:
            self.data["00404033"] = {"vr": "SQ", "Value": []}

        self.data["00404033"]["Value"].append({
            "0020000D": {"vr": "UI", "Value": [study_uid]},
            "0020000E": {"vr": "UI", "Value": [series_uid]}
        })

    def to_json(self):
        """Serialize to JSON string for K-V storage"""
        return json.dumps(self.data)

    @classmethod
    def from_json(cls, json_str, workitem_uid):
        """
        Deserialize from K-V storage

        Args:
            json_str: JSON string from storage
            workitem_uid: The workitem UID

        Returns:
            UPSWorkitem instance
        """
        instance = cls.__new__(cls)
        instance.workitem_uid = workitem_uid
        instance.data = json.loads(json_str)
        return instance

    def get_state(self):
        """Get current ProcedureStepState"""
        return self.data["00741000"]["Value"][0]

    def get_wado_rs_urls(self):
        """
        Extract WADO-RS retrieval URLs from Input Information Sequence

        Returns:
            List of dicts with retrieval_url, study_uid, series_uid
        """
        urls = []
        input_seq = self.data.get("00404021", {}).get("Value", [])
        for item in input_seq:
            retrieval_seq = item.get("0040E025", {}).get("Value", [])
            for ret_item in retrieval_seq:
                url = ret_item.get("00081190", {}).get("Value", [None])[0]
                if url:
                    urls.append({
                        "retrieval_url": url,
                        "study_uid": item.get("0020000D", {}).get("Value", [None])[0],
                        "series_uid": item.get("0020000E", {}).get("Value", [None])[0]
                    })
        return urls

    def get_study_uid(self):
        """Get StudyInstanceUID"""
        return self.data.get("0020000D", {}).get("Value", [None])[0]
