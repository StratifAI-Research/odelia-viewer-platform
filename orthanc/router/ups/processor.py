"""
UPS Workitem processor - executes AI inference tasks
"""

import os
import io
import json
import time
import requests
from datetime import datetime

import orthanc
from pydicom import dcmread
from pydicom.uid import generate_uid

from ups.storage import ups_storage
from wado_utils import retrieve_series_metadata_sorted


# Get MODEL_BACKEND_URL from environment (configured per router instance in docker-compose)
MODEL_BACKEND_URL = os.environ.get("MODEL_BACKEND_URL", "http://breast-cancer-classification:5555")
AI_TEXT = os.environ.get("AI_TEXT", "PROCESSED BY AI")
AI_COLOR = os.environ.get("AI_COLOR", "red")
AI_NAME = os.environ.get("AI_NAME", "Breast Cancer Classification Model")


def notify_subscriber(workitem, subscriber_url):
    """
    Send UPS notification to a single subscriber (RAD-87)

    Args:
        workitem: UPSWorkitem instance
        subscriber_url: Subscriber's callback URL
    """
    try:
        response = requests.post(
            f"{subscriber_url}/ups-rs/workitems/{workitem.workitem_uid}",
            data=workitem.to_json(),
            headers={"Content-Type": "application/dicom+json"},
            timeout=5
        )
        if response.status_code == 200:
            print(f"Notified subscriber {subscriber_url}: workitem {workitem.workitem_uid} state={workitem.get_state()}")
        else:
            print(f"Notification failed for {subscriber_url}: {response.status_code}")
    except Exception as e:
        print(f"Error notifying {subscriber_url}: {str(e)}")


def notify_all_subscribers(workitem):
    """
    Send UPS notifications to all registered subscribers (RAD-87)

    Args:
        workitem: UPSWorkitem instance
    """
    from ups.subscription_storage import subscription_storage

    subscribers = subscription_storage.get_subscribers(workitem.workitem_uid)

    if not subscribers:
        print(f"No subscribers for workitem {workitem.workitem_uid}")
        return

    print(f"Notifying {len(subscribers)} subscriber(s) for workitem {workitem.workitem_uid}")

    for subscriber_url in subscribers:
        notify_subscriber(workitem, subscriber_url)


def process_workitem(workitem):
    """
    Process a UPS workitem immediately (similar to OnStableStudy pattern)

    Args:
        workitem: UPSWorkitem instance
    """
    print(f"Processing workitem {workitem.workitem_uid}")
    overall_start = time.time()

    try:
        # Step 1: Update state to IN_PROGRESS with initial progress
        workitem.update_state(
            "IN_PROGRESS",
            progress_percent=10,
            progress_description="Starting AI inference"
        )
        ups_storage.store_workitem(workitem)
        notify_all_subscribers(workitem)

        # Step 2: Extract WADO-RS retrieval URLs from workitem
        wado_rs_urls = workitem.get_wado_rs_urls()
        study_uid = workitem.get_study_uid()

        print(f"Workitem has {len(wado_rs_urls)} WADO-RS retrieval URLs")

        # Update: Retrieved metadata
        workitem.update_state(
            "IN_PROGRESS",
            progress_percent=20,
            progress_description="Retrieved study metadata"
        )
        ups_storage.store_workitem(workitem)
        notify_all_subscribers(workitem)

        # Step 3: Call AI model with WADO-RS URLs (and structured input mapping if present)
        try:
            workitem.update_state(
                "IN_PROGRESS",
                progress_percent=30,
                progress_description="Sending data to AI model"
            )
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)

            # Build request body for the model backend
            model_request_body = {
                "wado_rs_retrieval": wado_rs_urls,
                "study_uid": study_uid
            }

            # Extract structured input mapping from workitem (if present)
            structured_input = workitem.get_input_mapping()
            if structured_input:
                role_mapping = structured_input["mapping"]
                config_id = structured_input.get("input_configuration_id")
                print(f"Structured input mapping found: config={config_id}, roles={list(role_mapping.keys())}")

                wado_rs_by_series = {}
                for item in wado_rs_urls:
                    wado_rs_by_series[item["series_uid"]] = item["retrieval_url"]

                input_mapping_for_model = {}
                for role_key, series_uid in role_mapping.items():
                    input_mapping_for_model[role_key] = {
                        "series_uid": series_uid,
                        "study_uid": study_uid,
                        "wado_rs_url": wado_rs_by_series.get(series_uid)
                    }

                model_request_body["input_mapping"] = input_mapping_for_model
                if config_id:
                    model_request_body["input_configuration_id"] = config_id
            else:
                print("No structured input mapping in workitem, using flat WADO-RS URLs")

            step_start = time.time()
            model_response = requests.post(
                f"{MODEL_BACKEND_URL}/analyze/mri",
                json=model_request_body,
                timeout=1000,
            )
            step_duration = (time.time() - step_start) * 1000
            print(f"TIMING: model_backend_request: {step_duration:.2f}ms")

            # Update: Model processing
            workitem.update_state(
                "IN_PROGRESS",
                progress_percent=50,
                progress_description="AI model analyzing data"
            )
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)

            if model_response.status_code != 200:
                error_msg = f"Model error: {model_response.status_code} - {model_response.text}"
                print(error_msg)
                workitem.update_state("CANCELED", cancellation_reason=error_msg)
                ups_storage.store_workitem(workitem)
                notify_all_subscribers(workitem)
                return

            model_results = model_response.json()

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error calling model: {str(e)}"
            print(error_msg)
            workitem.update_state("CANCELED", cancellation_reason=error_msg)
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)
            return

        # Step 4: Process results and upload to viewer (import existing SR/SC creation logic)
        try:
            # Import the existing result processing functions from server.py
            from server import (
                detect_response_format,
                create_bilateral_sr,
                create_multiframe_attention_sc
            )

            # Update: Retrieving source metadata
            workitem.update_state(
                "IN_PROGRESS",
                progress_percent=70,
                progress_description="Retrieving source metadata"
            )
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)

            # Get spatial metadata (metadata-only, no pixel data)
            first_instance_meta, positions_list, slice_spacing = retrieve_series_metadata_sorted(wado_rs_urls)

            # Create minimal Dataset with spatial tags only
            from pydicom import Dataset
            original_dicom = Dataset()

            # Extract from DICOM JSON format (tag->Value structure)
            ipp_tag = first_instance_meta.get("00200032")  # ImagePositionPatient
            if ipp_tag and ipp_tag.get("Value"):
                original_dicom.ImagePositionPatient = [float(v) for v in ipp_tag["Value"]]

            iop_tag = first_instance_meta.get("00200037")  # ImageOrientationPatient
            if iop_tag and iop_tag.get("Value"):
                original_dicom.ImageOrientationPatient = [float(v) for v in iop_tag["Value"]]

            for_tag = first_instance_meta.get("00200052")  # FrameOfReferenceUID
            if for_tag and for_tag.get("Value"):
                original_dicom.FrameOfReferenceUID = for_tag["Value"][0]

            # Copy required study/patient tags
            tag_mapping = {
                "00100010": "PatientName",      # Patient Name (PN - special handling)
                "00100020": "PatientID",        # Patient ID
                "0020000D": "StudyInstanceUID", # Study Instance UID
                "00080016": "SOPClassUID",      # SOP Class UID
                "00080018": "SOPInstanceUID"    # SOP Instance UID
            }

            for hex_tag, attr_name in tag_mapping.items():
                tag_data = first_instance_meta.get(hex_tag)
                if tag_data and tag_data.get("Value"):
                    value = tag_data["Value"][0] if isinstance(tag_data["Value"], list) else tag_data["Value"]

                    # Special handling for PersonName (PN) VR - extract Alphabetic component
                    if attr_name == "PatientName" and isinstance(value, dict):
                        value = value.get("Alphabetic", "")

                    setattr(original_dicom, attr_name, value)

            print(f"Using spatially first instance with position {original_dicom.ImagePositionPatient}")

            # Update: Creating DICOM results
            workitem.update_state(
                "IN_PROGRESS",
                progress_percent=85,
                progress_description="Creating DICOM results"
            )
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)

            # Detect response format and create DICOM objects
            response_format = detect_response_format(model_results)
            print(f"Detected response format: {response_format}")

            dicom_objects_to_upload = []
            current_date = datetime.now().strftime("%Y%m%d")
            current_time = datetime.now().strftime("%H%M%S.%f")[:-3]

            if response_format == "bilateral":
                sr_bytes, current_date, current_time, sr_sop_instance_uid = (
                    create_bilateral_sr(original_dicom, model_results)
                )
                dicom_objects_to_upload = [(sr_bytes, "SR-Bilateral")]

            elif response_format == "bilateral_with_heatmap":
                sr_bytes, current_date, current_time, sr_sop_instance_uid = (
                    create_bilateral_sr(original_dicom, model_results)
                )
                dicom_objects_to_upload.append((sr_bytes, "SR-Bilateral-MST"))

                # Create multi-frame SC with attention maps
                attention_maps = model_results.get("attention_maps", {})
                if attention_maps and attention_maps.get("data"):
                    # Need first instance with proper spatial metadata
                    sc_bytes = create_multiframe_attention_sc(
                        original_dicom,
                        attention_maps,
                        creation_date=current_date,
                        creation_time=current_time,
                        sr_sop_instance_uid=sr_sop_instance_uid,
                        slice_spacing=slice_spacing,  # Use calculated spacing as fallback
                        positions_list=positions_list  # Use actual positions from sorted instances
                    )
                    dicom_objects_to_upload.append((sc_bytes, "SC-MultiFrame"))

            # Step 5: Upload results to viewer
            # Update: Uploading results
            workitem.update_state(
                "IN_PROGRESS",
                progress_percent=95,
                progress_description="Uploading results to viewer"
            )
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)

            upload_start = time.time()
            for dicom_bytes, desc in dicom_objects_to_upload:
                upload_item_start = time.time()
                response = requests.post(
                    "http://orthanc-viewer:8042/instances",
                    data=dicom_bytes,
                    headers={"Content-Type": "application/dicom"},
                    timeout=10,
                )
                upload_item_duration = (time.time() - upload_item_start) * 1000
                print(f"TIMING: upload_{desc}: {upload_item_duration:.2f}ms")

                if response.status_code == 200:
                    print(f"AI {desc} uploaded to viewer")
                else:
                    print(f"Failed to upload {desc}: {response.status_code}")

            upload_duration = (time.time() - upload_start) * 1000
            print(f"TIMING: upload_all_to_viewer: {upload_duration:.2f}ms")

        except Exception as e:
            error_msg = f"Error processing results: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            workitem.update_state("CANCELED", cancellation_reason=error_msg)
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)
            return

        # Step 6: Complete workitem
        workitem.update_state("COMPLETED", "AI inference completed successfully")
        ups_storage.store_workitem(workitem)
        notify_all_subscribers(workitem)

        overall_duration = (time.time() - overall_start) * 1000
        print(f"TIMING: total_workitem_processing: {overall_duration:.2f}ms")
        print(f"Successfully processed workitem {workitem.workitem_uid}")

    except Exception as e:
        error_msg = f"Unexpected error processing workitem: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        try:
            workitem.update_state("CANCELED", cancellation_reason=error_msg)
            ups_storage.store_workitem(workitem)
            notify_all_subscribers(workitem)
        except:
            pass  # Best effort state update
