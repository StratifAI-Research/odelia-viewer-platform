import json
import os
import sys

import orthanc
import requests

# Ensure the directory of this script is importable for sibling modules
try:
    current_dir = os.path.dirname(__file__)
    if current_dir and current_dir not in sys.path:
        sys.path.insert(0, current_dir)
except Exception:
    pass

# Feedback endpoints
try:
    import feedback_routes  # type: ignore

    register_feedback_endpoints = feedback_routes.register_feedback_endpoints
except Exception:
    register_feedback_endpoints = None

# UPS storage for workitem persistence
try:
    from ups.storage import UPSStorage
    from ups.workitem import UPSWorkitem

    ups_storage = UPSStorage()
except Exception as e:
    print(f"Warning: Could not initialize UPS storage: {e}")
    ups_storage = None


def FilterAIResultSeries(study_id):
    """
    Get all non-AI series from a study for AI processing.
    Returns a list of series IDs that should be sent to AI models.
    Filters out any series that appear to be AI-generated results.
    """
    try:
        # Get all series in the study
        series_list = json.loads(orthanc.RestApiGet(f"/studies/{study_id}/series"))

        original_series = []
        ai_series_count = 0

        for series in series_list:
            series_id = series["ID"]

            # Get series tags to check if it's an AI result
            try:
                series_tags = json.loads(
                    orthanc.RestApiGet(f"/series/{series_id}/tags?simplify")
                )

                series_description = series_tags.get("SeriesDescription", "").strip()
                modality = series_tags.get("Modality", "").strip()

                # Check for AI result markers (based on server.py analysis)
                ai_markers = [
                    "Automated Diagnostic Findings",  # Exact SR match from server.py
                    "- Heatmap",  # SC pattern match from server.py
                    "AI Analysis Result",  # Generic fallback
                    "AI Generated",  # Generic fallback
                    "Secondary Capture AI",  # Generic fallback
                    "AI Structured Report",  # Generic fallback
                ]

                is_ai_result = (
                    any(marker in series_description for marker in ai_markers)
                    or (modality in ["SC", "SR"] and "AI" in series_description.upper())
                    or series_description.startswith("AI_")
                    or series_description.endswith("_AI")
                )

                if is_ai_result:
                    ai_series_count += 1
                    print(
                        f"Filtering out AI result series: {series_id} ({series_description}, {modality})"
                    )
                else:
                    original_series.append(series_id)

            except Exception as e:
                print(f"Warning: Could not check series {series_id}: {str(e)}")
                # If we can't check, assume it's original data and include it
                original_series.append(series_id)

        print(
            f"Study {study_id}: Found {len(original_series)} original series, {ai_series_count} AI result series"
        )
        return original_series

    except Exception as e:
        print(f"Error filtering AI result series for study {study_id}: {str(e)}")
        # Return empty list on error to prevent sending anything
        return []


def HasProcessableContent(study_id):
    """
    Check if study has any non-AI series that can be processed.
    Returns True if there are original series available for AI processing.
    """
    original_series = FilterAIResultSeries(study_id)
    return len(original_series) > 0


def GetStudyInstanceUID(study_id):
    """Get the DICOM StudyInstanceUID from an Orthanc study ID"""
    try:
        # Get the study information
        study_info = json.loads(orthanc.RestApiGet(f"/studies/{study_id}"))
        # Get the MainDicomTags which contains the StudyInstanceUID
        main_tags = study_info.get("MainDicomTags", {})
        study_instance_uid = main_tags.get("StudyInstanceUID")
        if not study_instance_uid:
            print(f"Warning: StudyInstanceUID not found for study {study_id}")
            return None
        return study_instance_uid
    except Exception as e:
        print(f"Error getting StudyInstanceUID: {str(e)}")
        return None


def ListModalities():
    """List all configured DICOM modalities"""
    try:
        modalities = json.loads(orthanc.RestApiGet("/modalities"))
        print("Configured DICOM modalities:")
        for modality in modalities:
            modality_info = json.loads(orthanc.RestApiGet(f"/modalities/{modality}"))
            print(
                f"  - {modality}: {modality_info.get('Host', 'unknown')}:{modality_info.get('Port', 'unknown')} (AET: {modality_info.get('AET', 'unknown')})"
            )
        return modalities
    except Exception as e:
        print(f"Error listing modalities: {str(e)}")
        return []


def SendToAiDicom(output, uri, **request):
    """REST endpoint to send a study to target server using DICOM protocol"""
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        # Parse the POST body
        body = json.loads(request["body"])
        study_id = body.get("study_id")
        target = body.get("target")
        target_url = body.get("target_url")
        series_uids = body.get("series_uids")  # Optional: filter by specific series

        if not study_id or not target:
            output.SendHttpStatus(400, "Missing study_id or target in request body")
            return

        # If series_uids not provided, check if study has processable content
        if not series_uids and not HasProcessableContent(study_id):
            output.SendHttpStatus(
                400, "Study contains no processable content (only AI results or empty)"
            )
            return

        # List all configured modalities before proceeding
        ListModalities()

        # Configure DICOM modality if target_url is provided
        if target_url:
            try:
                # Parse the target URL to extract host, port, and AE Title
                # Expected format: host:port/AET
                print(f"Parsing target URL: {target_url}")

                # First, check if the modality already exists
                try:
                    existing_modality = json.loads(
                        orthanc.RestApiGet(f"/modalities/{target}")
                    )
                    print(
                        f"Modality {target} already exists with configuration: {existing_modality}"
                    )

                    # Delete the existing modality to ensure a clean configuration
                    orthanc.RestApiDelete(f"/modalities/{target}")
                    print(f"Deleted existing modality {target}")
                except:
                    # Modality doesn't exist, which is fine
                    pass

                # Parse the URL parts
                url_parts = target_url.split("/")
                if len(url_parts) >= 2:
                    host_port = url_parts[0].split(":")
                    host = host_port[0]
                    port = (
                        int(host_port[1]) if len(host_port) > 1 else 104
                    )  # Default DICOM port
                    aet = (
                        url_parts[1] if len(url_parts) > 1 else target
                    )  # Use target name as AE Title if not specified

                    print(f"Extracted host: {host}, port: {port}, AET: {aet}")

                    # Configure the DICOM modality with more detailed settings
                    modality_config = {
                        "AET": aet,
                        "Host": host,
                        "Port": port,
                        "Manufacturer": "Generic",
                        "AllowEcho": True,
                        "AllowFind": True,
                        "AllowGet": True,
                        "AllowMove": True,
                        "AllowStore": True,
                        "CheckCalledAet": False,
                        "DicomAet": "ORTHANC",  # Your Orthanc's AE Title
                        "DicomCheckCalledAet": False,
                        "DicomPort": 4242,  # Your Orthanc's DICOM port
                        "DicomWeb": {
                            "Enable": False,
                            "Root": "/dicom-web/",
                            "Ssl": False,
                            "Studies": True,
                            "EnableWado": False,
                            "WadoRoot": "/wado",
                            "WadoMetadata": {"Enable": False, "MaxResults": 100},
                        },
                        "Timeout": 60,  # Increase timeout to 60 seconds
                        "ConcurrentOperations": 1,  # Limit to 1 concurrent operation
                        "RetryCount": 3,  # Retry up to 3 times
                        "RetryDelay": 5,  # Wait 5 seconds between retries
                        "TransferSyntaxes": [
                            "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
                            "1.2.840.10008.1.2",  # Implicit VR Little Endian
                            "1.2.840.10008.1.2.2",  # Explicit VR Big Endian
                        ],
                    }

                    # Add the modality configuration
                    orthanc.RestApiPut(
                        f"/modalities/{target}", json.dumps(modality_config)
                    )
                    print(f"Successfully configured DICOM modality: {target}")

                    # Verify the configuration
                    try:
                        configured_modality = json.loads(
                            orthanc.RestApiGet(f"/modalities/{target}")
                        )
                        print(f"Verified modality configuration: {configured_modality}")
                    except Exception as e:
                        print(
                            f"Warning: Failed to verify modality configuration: {str(e)}"
                        )
                else:
                    print(f"Invalid target URL format: {target_url}")
            except Exception as e:
                print(f"Warning: Failed to configure DICOM modality: {str(e)}")
                # Continue anyway as the modality might already be configured

        # Get series to send (either by series_uids or filter AI results)
        if series_uids:
            # Convert DICOM SeriesInstanceUIDs to Orthanc series IDs
            print(f"Filtering by {len(series_uids)} specific series UIDs")
            original_series = []
            for series_uid in series_uids:
                try:
                    lookup_result = json.loads(
                        orthanc.RestApiPost("/tools/lookup", series_uid)
                    )
                    series_result = [r for r in lookup_result if r["Type"] == "Series"]
                    if series_result:
                        original_series.append(series_result[0]["ID"])
                        print(
                            f"Found series {series_result[0]['ID']} for UID {series_uid}"
                        )
                    else:
                        print(f"Warning: Series UID {series_uid} not found")
                except Exception as e:
                    print(
                        f"Warning: Could not lookup series UID {series_uid}: {str(e)}"
                    )
        else:
            # Use existing filter (exclude AI results)
            original_series = FilterAIResultSeries(study_id)

        if not original_series:
            output.SendHttpStatus(
                400,
                "No series found to send",
            )
            return

        # Collect all instances from filtered series
        instance_ids = []
        for series_id in original_series:
            try:
                series_instances = json.loads(
                    orthanc.RestApiGet(f"/series/{series_id}/instances")
                )
                series_instance_ids = [instance["ID"] for instance in series_instances]
                instance_ids.extend(series_instance_ids)
                print(f"Series {series_id} has {len(series_instance_ids)} instances")
            except Exception as e:
                print(
                    f"Warning: Could not get instances for series {series_id}: {str(e)}"
                )

        print(
            f"Collected {len(instance_ids)} instances from {len(original_series)} series"
        )

        # Try to send the filtered instances using DICOM modality
        try:
            print(
                f"Attempting to send {len(instance_ids)} instances from study {study_id} to DICOM modality {target}"
            )
            orthanc.RestApiPost(f"/modalities/{target}/store", json.dumps(instance_ids))
            print(
                f"Successfully sent {len(instance_ids)} instances from study {study_id} to DICOM modality {target}"
            )

            response_data = {
                "status": "success",
                "message": f"Successfully sent study {study_id} to {target} using DICOM protocol",
                "study_id": study_id,
                "target": target,
            }
            output.AnswerBuffer(json.dumps(response_data), "application/json")
        except Exception as e:
            error_message = f"Failed to send study using DICOM protocol: {str(e)}"
            print(error_message)
            error_response = {"status": "error", "message": error_message}
            output.AnswerBuffer(json.dumps(error_response), "application/json")

    except Exception as e:
        error_message = str(e)
        error_response = {
            "status": "error",
            "message": f"Error sending study: {error_message}",
        }
        output.AnswerBuffer(json.dumps(error_response), "application/json")


def SendToAiDicomWeb(output, uri, **request):
    """REST endpoint to send a study to target server using DICOMweb protocol"""
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        print("SendToAiDicomWeb: Starting processing of request")

        # Parse the POST body
        body = json.loads(request["body"])
        study_id = body.get("study_id")
        target = body.get("target")
        target_url = body.get("target_url")
        series_uids = body.get("series_uids")  # Optional: filter by specific series
        input_mapping = body.get("input_mapping")  # Structured role-to-series mapping
        input_configuration_id = body.get("input_configuration_id")

        print(
            f"SendToAiDicomWeb: Request parameters - study_id: {study_id}, target: {target}, target_url: {target_url}, series_uids: {len(series_uids) if series_uids else 0}, input_mapping: {input_mapping is not None}"
        )

        if not study_id or not target:
            print("SendToAiDicomWeb: Missing required parameters")
            output.SendHttpStatus(400, "Missing study_id or target in request body")
            return

        if not target_url:
            print("SendToAiDicomWeb: Missing target_url parameter")
            output.SendHttpStatus(400, "Missing target_url in request body")
            return

        # Verify study_id exists in Orthanc
        try:
            study_info = json.loads(orthanc.RestApiGet(f"/studies/{study_id}"))
            print(f"SendToAiDicomWeb: Valid study found with ID {study_id}")
            print(
                f"SendToAiDicomWeb: Study contains {len(study_info['Series'])} series and {study_info['PatientMainDicomTags'].get('PatientName', 'Unknown')} patient"
            )
        except Exception as e:
            print(f"SendToAiDicomWeb: Error verifying study existence: {str(e)}")
            output.SendHttpStatus(404, f"Study with ID {study_id} not found: {str(e)}")
            return

        # If series_uids not provided, check if study has processable content
        if not series_uids and not HasProcessableContent(study_id):
            print(
                "SendToAiDicomWeb: Study contains no processable content (only AI results or empty)"
            )
            output.SendHttpStatus(
                400, "Study contains no processable content (only AI results or empty)"
            )
            return

        try:
            # Configure the DICOMweb server
            print(
                f"SendToAiDicomWeb: Configuring DICOMweb server {target} with URL {target_url}"
            )

            # Create server configuration
            server_config = {
                "Url": target_url,
                "Username": body.get("username", ""),
                "Password": body.get("password", ""),
                "HttpHeaders": {},
            }

            # Configure the server using direct HTTP request
            config_response = requests.put(
                f"http://localhost:8042/dicom-web/servers/{target}", json=server_config
            )

            if config_response.status_code not in [200, 201, 204]:
                error_message = (
                    f"Error configuring DICOMweb server: {config_response.text}"
                )
                print(f"SendToAiDicomWeb: {error_message}")
                output.SendHttpStatus(500, error_message)
                return

            print(
                f"SendToAiDicomWeb: Successfully configured DICOMweb server: {target}"
            )

            # Get series to send (either by series_uids or filter AI results)
            if series_uids:
                # Convert DICOM SeriesInstanceUIDs to Orthanc series IDs
                print(
                    f"SendToAiDicomWeb: Filtering by {len(series_uids)} specific series UIDs"
                )
                original_series = []
                for series_uid in series_uids:
                    try:
                        lookup_result = json.loads(
                            orthanc.RestApiPost("/tools/lookup", series_uid)
                        )
                        series_result = [
                            r for r in lookup_result if r["Type"] == "Series"
                        ]
                        if series_result:
                            original_series.append(series_result[0]["ID"])
                            print(
                                f"SendToAiDicomWeb: Found series {series_result[0]['ID']} for UID {series_uid}"
                            )
                        else:
                            print(
                                f"SendToAiDicomWeb: Warning - Series UID {series_uid} not found"
                            )
                    except Exception as e:
                        print(
                            f"SendToAiDicomWeb: Warning - Could not lookup series UID {series_uid}: {str(e)}"
                        )
            else:
                # Use existing filter (exclude AI results)
                original_series = FilterAIResultSeries(study_id)

            if not original_series:
                error_message = "No series found to send"
                print(f"SendToAiDicomWeb: {error_message}")
                output.SendHttpStatus(400, error_message)
                return

            # Collect all instances from filtered series
            instance_ids = []
            for series_id in original_series:
                try:
                    series_instances = json.loads(
                        orthanc.RestApiGet(f"/series/{series_id}/instances")
                    )
                    series_instance_ids = [
                        instance["ID"] for instance in series_instances
                    ]
                    instance_ids.extend(series_instance_ids)
                    print(
                        f"SendToAiDicomWeb: Series {series_id} has {len(series_instance_ids)} instances"
                    )
                except Exception as e:
                    print(
                        f"SendToAiDicomWeb: Warning - Could not get instances for series {series_id}: {str(e)}"
                    )

            print(
                f"SendToAiDicomWeb: Collected {len(instance_ids)} instances from {len(original_series)} series"
            )

            # NEW: Create UPS workitem on router before sending data
            # Extract router URL from target_url (remove /dicom-web suffix if present)
            router_base_url = target_url.replace("/dicom-web", "").rstrip("/")

            # Get study UID
            study_uid = GetStudyInstanceUID(study_id)
            if not study_uid:
                print("SendToAiDicomWeb: Could not get StudyInstanceUID")
                output.SendHttpStatus(500, "Could not get StudyInstanceUID")
                return

            # Get series UIDs (DICOM UIDs, not Orthanc IDs)
            dicom_series_uids = []
            for series_id in original_series:
                try:
                    series_info = json.loads(orthanc.RestApiGet(f"/series/{series_id}"))
                    series_dicom_uid = series_info.get("MainDicomTags", {}).get("SeriesInstanceUID")
                    if series_dicom_uid:
                        dicom_series_uids.append(series_dicom_uid)
                except Exception as e:
                    print(f"Warning: Could not get SeriesInstanceUID for {series_id}: {str(e)}")

            # Create UPS workitem on router
            try:
                ups_workitem_request = {
                    "study_uid": study_uid,
                    "series_uids": dicom_series_uids,
                    "wado_rs_base": "http://orthanc-viewer:8042/dicom-web",
                    "priority": "MEDIUM"
                }

                if input_mapping:
                    ups_workitem_request["input_mapping"] = input_mapping
                if input_configuration_id:
                    ups_workitem_request["input_configuration_id"] = input_configuration_id

                post_url = f"{router_base_url}/ups-rs/workitems"
                print(f"SendToAiDicomWeb: Creating UPS workitem on router at {post_url}")
                print(f"SendToAiDicomWeb: Request body: {json.dumps(ups_workitem_request)}")

                ups_response = requests.post(
                    post_url,
                    json=ups_workitem_request,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )

                print(f"SendToAiDicomWeb: POST response status: {ups_response.status_code}")
                if ups_response.status_code in [200, 201]:
                    ups_workitem_data = ups_response.json()
                    workitem_uid = ups_workitem_data.get("00080018", {}).get("Value", [None])[0]
                    print(f"SendToAiDicomWeb: Created UPS workitem on router: {workitem_uid}")

                    # Subscribe to workitem notifications (RAD-86)
                    try:
                        subscribe_url = f"{router_base_url}/ups-rs/workitems/{workitem_uid}/subscribers"
                        subscribe_body = {
                            "subscriber_url": "http://orthanc-viewer:8042",
                            "deletion_lock": False
                        }
                        subscribe_response = requests.post(
                            subscribe_url,
                            json=subscribe_body,
                            timeout=5
                        )
                        if subscribe_response.status_code == 200:
                            print(f"SendToAiDicomWeb: Successfully subscribed to workitem {workitem_uid}")
                        else:
                            print(f"SendToAiDicomWeb: Subscription failed: {subscribe_response.status_code}")
                    except Exception as e:
                        print(f"SendToAiDicomWeb: Error subscribing to workitem: {str(e)}")
                else:
                    print(f"SendToAiDicomWeb: Failed to create UPS workitem: {ups_response.status_code} - {ups_response.text}")
                    workitem_uid = None
            except Exception as e:
                print(f"SendToAiDicomWeb: Error creating UPS workitem: {str(e)}")
                import traceback
                traceback.print_exc()
                workitem_uid = None

            # Check if workitem creation succeeded
            if workitem_uid is None:
                error_message = "Failed to create UPS workitem on router"
                print(f"SendToAiDicomWeb: {error_message}")
                error_response = {
                    "status": "error",
                    "message": error_message
                }
                output.AnswerBuffer(json.dumps(error_response), "application/json")
                return

            # UPS-RS: No data transfer to router
            # Data stays in viewer, router retrieves via WADO-RS when processing
            print("SendToAiDicomWeb: UPS workitem created, no data transfer to router")

            # Return success response with workitem UID for tracking
            success_response = {
                "status": "success",
                "message": f"UPS workitem created for study {study_id}",
                "study_id": study_id,
                "target": target,
                "workitem_uid": workitem_uid,
                "series_count": len(original_series),
            }
            print(f"SendToAiDicomWeb: Returning success response with workitem_uid={workitem_uid}")
            print(f"SendToAiDicomWeb: Full response: {json.dumps(success_response)}")
            output.AnswerBuffer(
                json.dumps(success_response), "application/json"
            )

        except Exception as e:
            error_message = f"Error during STOW-RS request: {str(e)}"
            print(f"SendToAiDicomWeb: {error_message}")
            output.SendHttpStatus(500, error_message)

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        print(f"SendToAiDicomWeb: {error_message}")
        output.SendHttpStatus(500, error_message)


def SendToAi(output, uri, **request):
    """REST endpoint to send a study to target server using DICOMWeb protocol"""
    # This is now just a wrapper around SendToAiDicomWeb for backward compatibility
    print("giving control to SendToAiDicomWeb")
    SendToAiDicomWeb(output, uri, **request)


# UPS-RS endpoints for receiving workitem updates from router
def UPSUpdateWorkitem(output, uri, **request):
    """
    POST /ups-rs/workitems/{uid}
    Receive workitem state updates from router
    """
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        workitem_uid = uri.split('/')[-1]
        body = json.loads(request["body"])

        # Store workitem using UPS storage
        if ups_storage:
            # Use from_json method with JSON string
            workitem = UPSWorkitem.from_json(request["body"], workitem_uid)
            ups_storage.store_workitem(workitem)
            state = workitem.get_state()
        else:
            # Fallback: just log if storage not available
            state = body.get('00741000', {}).get('Value', ['UNKNOWN'])[0]

        print(f"Received workitem update: {workitem_uid}, state: {state}")

        output.AnswerBuffer(json.dumps({"status": "updated"}), "application/json")
    except Exception as e:
        print(f"Error updating workitem: {str(e)}")
        output.SendHttpStatus(500, str(e))


def UPSGetWorkitem(output, uri, **request):
    """
    GET /ups-rs/workitems/{uid}
    Retrieve workitem from local storage (updated via router notifications)
    """
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        workitem_uid = uri.split('/')[-1]
        print(f"UPSGetWorkitem: Retrieving workitem {workitem_uid} from local storage")

        if not ups_storage:
            output.SendHttpStatus(500, "UPS storage not initialized")
            return

        workitem = ups_storage.get_workitem(workitem_uid)
        if workitem:
            output.AnswerBuffer(workitem.to_json(), "application/dicom+json")
        else:
            print(f"UPSGetWorkitem: Workitem {workitem_uid} not found")
            output.SendHttpStatus(404, f"Workitem {workitem_uid} not found")

    except Exception as e:
        print(f"Error retrieving workitem: {str(e)}")
        import traceback
        traceback.print_exc()
        output.SendHttpStatus(500, str(e))


def UPSWorkitemHandler(output, uri, **request):
    """
    Unified handler for UPS-RS workitem endpoints
    Routes to appropriate handler based on HTTP method
    """
    if request["method"] == "POST":
        UPSUpdateWorkitem(output, uri, **request)
    elif request["method"] == "GET":
        UPSGetWorkitem(output, uri, **request)
    else:
        output.SendMethodNotAllowed("GET, POST")


def GetAIManifest(output, uri, **request):
    """
    GET /ai-manifest?target_url=http://orthanc-router:8042/dicom-web
    Proxy the model input manifest from the target router.
    Returns {"manifest": null} when the router has no manifest (404 / unreachable).
    """
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        get_params = request.get("get", {})
        target_url = get_params.get("target_url", [None])
        if isinstance(target_url, list):
            target_url = target_url[0]

        if not target_url:
            output.SendHttpStatus(400, "Missing target_url query parameter")
            return

        router_base_url = target_url.replace("/dicom-web", "").rstrip("/")
        manifest_url = f"{router_base_url}/manifest"
        print(f"GetAIManifest: Fetching manifest from {manifest_url}")

        resp = requests.get(manifest_url, timeout=5)
        if resp.status_code == 200:
            output.AnswerBuffer(resp.text, "application/json")
        else:
            print(f"GetAIManifest: Router returned {resp.status_code}, returning null manifest")
            output.AnswerBuffer(json.dumps({"manifest": None}), "application/json")

    except requests.exceptions.RequestException as e:
        print(f"GetAIManifest: Connection error to router: {e}")
        output.AnswerBuffer(json.dumps({"manifest": None}), "application/json")
    except Exception as e:
        print(f"GetAIManifest: Unexpected error: {e}")
        output.AnswerBuffer(json.dumps({"manifest": None}), "application/json")


# Register the REST endpoints
orthanc.RegisterRestCallback("/send-to-ai", SendToAi)
orthanc.RegisterRestCallback("/send-to-ai-dicom", SendToAiDicom)
orthanc.RegisterRestCallback("/send-to-ai-dicomweb", SendToAiDicomWeb)
orthanc.RegisterRestCallback("/ai-manifest", GetAIManifest)

# Register UPS-RS endpoints for receiving workitem updates
orthanc.RegisterRestCallback("/ups-rs/workitems/(.*)", UPSWorkitemHandler)

# Register feedback routes
if register_feedback_endpoints is not None:
    register_feedback_endpoints()
