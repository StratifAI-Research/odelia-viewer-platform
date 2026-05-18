"""
UPS-RS REST API endpoints
Implements DICOM PS3.18 Section 11 (UPS-RS)
"""

import json
import os
import orthanc
import threading

from ups.workitem import UPSWorkitem
from ups.storage import ups_storage
from ups.processor import process_workitem

MANIFEST_PATH = os.environ.get("AI_MANIFEST_PATH", "/etc/orthanc/manifest.json")


def CreateWorkitem(output, uri, **request):
    """
    POST /ups-rs/workitems
    Create new UPS workitem and immediately process it (RAD-80)

    Request body:
    {
        "study_uid": "1.2.3...",
        "series_uids": ["1.2.3..."],
        "wado_rs_base": "http://orthanc-viewer:8042/dicom-web",
        "priority": "MEDIUM"
    }

    Response: DICOM JSON workitem with Content-Type: application/dicom+json
    """
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        body = json.loads(request["body"])

        # Extract parameters
        study_uid = body.get("study_uid")
        series_uids = body.get("series_uids", [])
        wado_rs_base = body.get("wado_rs_base", "http://orthanc-viewer:8042/dicom-web")
        priority = body.get("priority", "MEDIUM")
        input_mapping = body.get("input_mapping")
        input_configuration_id = body.get("input_configuration_id")

        if not study_uid:
            output.SendHttpStatus(400, "Missing study_uid in request body")
            return

        # Build WADO-RS retrieval URLs
        wado_rs_retrieval = []
        for series_uid in series_uids:
            retrieval_url = f"{wado_rs_base}/studies/{study_uid}/series/{series_uid}"
            wado_rs_retrieval.append({
                "retrieval_url": retrieval_url,
                "study_uid": study_uid,
                "series_uid": series_uid
            })

        # Create workitem (with optional structured input mapping)
        workitem = UPSWorkitem(
            study_uid=study_uid,
            series_uids=series_uids,
            wado_rs_retrieval=wado_rs_retrieval,
            priority=priority,
            input_mapping=input_mapping,
            input_configuration_id=input_configuration_id
        )

        print(f"CreateWorkitem: Created workitem with UID: {workitem.workitem_uid}")

        # Store workitem
        ups_storage.store_workitem(workitem)
        print(f"CreateWorkitem: Stored workitem {workitem.workitem_uid}")

        # Verify storage
        verify = ups_storage.get_workitem(workitem.workitem_uid)
        if verify:
            print(f"CreateWorkitem: Verification successful - workitem {workitem.workitem_uid} can be retrieved")
        else:
            print(f"CreateWorkitem: WARNING - workitem {workitem.workitem_uid} was NOT stored properly!")

        print(f"Created workitem {workitem.workitem_uid} for study {study_uid}")

        # Process workitem immediately in background thread
        # (similar to OnStableStudy pattern - immediate execution, not polling)
        def process_in_background():
            try:
                process_workitem(workitem)
            except Exception as e:
                print(f"Error processing workitem in background: {str(e)}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()

        # Return created workitem as DICOM JSON
        output.AnswerBuffer(
            json.dumps(workitem.data),
            "application/dicom+json"
        )

    except Exception as e:
        error_message = f"Error creating workitem: {str(e)}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def GetWorkitem(output, uri, **request):
    """
    GET /ups-rs/workitems/{uid}
    Retrieve workitem (RAD-83)

    Response: DICOM JSON workitem
    """
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        # Extract workitem UID from URI
        # URI format: /ups-rs/workitems/{uid}
        workitem_uid = request["groups"][0] if request.get("groups") else None
        print(f"GetWorkitem: URI={uri}, groups={request.get('groups')}, extracted UID={workitem_uid}")

        if not workitem_uid:
            output.SendHttpStatus(400, "Missing workitem UID in URL")
            return

        # Retrieve workitem
        print(f"GetWorkitem: Attempting to retrieve workitem {workitem_uid}")
        workitem = ups_storage.get_workitem(workitem_uid)

        if not workitem:
            print(f"GetWorkitem: Workitem {workitem_uid} not found in storage")
            # List all workitems for debugging
            all_workitems = ups_storage.list_workitems()
            print(f"GetWorkitem: Available workitems: {[w.workitem_uid for w in all_workitems]}")
            output.SendHttpStatus(404, f"Workitem {workitem_uid} not found")
            return

        print(f"GetWorkitem: Successfully retrieved workitem {workitem_uid}")
        # Return workitem as DICOM JSON
        output.AnswerBuffer(
            json.dumps(workitem.data),
            "application/dicom+json"
        )

    except Exception as e:
        error_message = f"Error retrieving workitem: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        output.SendHttpStatus(500, error_message)


def UpdateWorkitemState(output, uri, **request):
    """
    PUT /ups-rs/workitems/{uid}/state
    Update workitem state (RAD-84/85/86)

    Request body:
    {
        "state": "IN_PROGRESS"|"COMPLETED"|"CANCELED",
        "progress_info": "Processing..."
    }
    """
    if request["method"] != "PUT":
        output.SendMethodNotAllowed("PUT")
        return

    try:
        # Extract workitem UID from URI
        workitem_uid = request["groups"][0] if request.get("groups") else None

        if not workitem_uid:
            output.SendHttpStatus(400, "Missing workitem UID in URL")
            return

        body = json.loads(request["body"])
        new_state = body.get("state")
        progress_info = body.get("progress_info")

        if not new_state:
            output.SendHttpStatus(400, "Missing state in request body")
            return

        # Retrieve workitem
        workitem = ups_storage.get_workitem(workitem_uid)

        if not workitem:
            output.SendHttpStatus(404, f"Workitem {workitem_uid} not found")
            return

        # Update state
        workitem.update_state(new_state, progress_info)
        ups_storage.store_workitem(workitem)

        print(f"Updated workitem {workitem_uid} state to {new_state}")

        # Return updated workitem
        output.AnswerBuffer(
            json.dumps(workitem.data),
            "application/dicom+json"
        )

    except Exception as e:
        error_message = f"Error updating workitem state: {str(e)}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def QueryWorkitems(output, uri, **request):
    """
    GET /ups-rs/workitems?state=SCHEDULED
    Query workitems (RAD-81)

    Query parameters:
        state: Optional state filter

    Response: Array of DICOM JSON workitems
    """
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        # Parse query parameters
        get_params = request.get("get", {})
        state_filter = get_params.get("state", [None])[0] if "state" in get_params else None

        # Query workitems
        workitems = ups_storage.list_workitems(state=state_filter)

        # Convert to DICOM JSON array
        result = [workitem.data for workitem in workitems]

        print(f"Query returned {len(result)} workitems (state filter: {state_filter})")

        output.AnswerBuffer(
            json.dumps(result),
            "application/dicom+json"
        )

    except Exception as e:
        error_message = f"Error querying workitems: {str(e)}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def SubscribeToWorkitem(output, uri, **request):
    """
    POST /ups-rs/workitems/{uid}/subscribers
    Subscribe to notifications for a specific workitem (RAD-86)

    Request body:
    {
        "subscriber_url": "http://orthanc-viewer:8042"
    }
    """
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        workitem_uid = request["groups"][0] if request.get("groups") else None

        if not workitem_uid:
            output.SendHttpStatus(400, "Missing workitem UID in URL")
            return

        body = json.loads(request["body"])
        subscriber_url = body.get("subscriber_url")
        deletion_lock = body.get("deletion_lock", False)

        if not subscriber_url:
            output.SendHttpStatus(400, "Missing subscriber_url in request body")
            return

        # Verify workitem exists
        workitem = ups_storage.get_workitem(workitem_uid)
        if not workitem:
            output.SendHttpStatus(404, f"Workitem {workitem_uid} not found")
            return

        # Add subscription
        from ups.subscription_storage import subscription_storage
        subscription_storage.add_subscription(workitem_uid, subscriber_url, deletion_lock)

        # Send initial notification to new subscriber
        from ups.processor import notify_subscriber
        notify_subscriber(workitem, subscriber_url)

        print(f"Subscriber {subscriber_url} subscribed to workitem {workitem_uid}")
        output.AnswerBuffer(json.dumps({"status": "subscribed"}), "application/json")

    except Exception as e:
        error_message = f"Error creating subscription: {str(e)}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def UnsubscribeFromWorkitem(output, uri, **request):
    """
    DELETE /ups-rs/workitems/{uid}/subscribers/{subscriber_url}
    Unsubscribe from workitem notifications (RAD-86)
    """
    if request["method"] != "DELETE":
        output.SendMethodNotAllowed("DELETE")
        return

    try:
        workitem_uid = request["groups"][0] if request.get("groups") else None
        subscriber_url = request["groups"][1] if len(request.get("groups", [])) > 1 else None

        if not workitem_uid or not subscriber_url:
            output.SendHttpStatus(400, "Missing workitem UID or subscriber URL")
            return

        from ups.subscription_storage import subscription_storage
        subscription_storage.remove_subscription(workitem_uid, subscriber_url)

        output.AnswerBuffer(json.dumps({"status": "unsubscribed"}), "application/json")

    except Exception as e:
        error_message = f"Error removing subscription: {str(e)}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def ServeManifest(output, uri, **request):
    """
    GET /manifest
    Serve the model input manifest (if mounted).
    Returns 404 when no manifest is available, allowing the viewer to fall back
    to flat series selection.
    """
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return

    if not os.path.isfile(MANIFEST_PATH):
        output.SendHttpStatus(404, "No manifest available")
        return

    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest_data = f.read()
        json.loads(manifest_data)  # validate JSON
        output.AnswerBuffer(manifest_data, "application/json")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading manifest: {e}")
        output.SendHttpStatus(500, f"Error reading manifest: {e}")


# Helper to register all UPS routes
def register_ups_routes():
    """Register all UPS-RS REST endpoints"""
    orthanc.RegisterRestCallback('/ups-rs/workitems$', CreateWorkitem)
    orthanc.RegisterRestCallback('/ups-rs/workitems/([0-9.]+)$', GetWorkitem)
    orthanc.RegisterRestCallback('/ups-rs/workitems/([0-9.]+)/state$', UpdateWorkitemState)
    orthanc.RegisterRestCallback('/ups-rs/workitems/([0-9.]+)/subscribers$', SubscribeToWorkitem)
    orthanc.RegisterRestCallback('/ups-rs/workitems/([0-9.]+)/subscribers/(.+)$', UnsubscribeFromWorkitem)
    orthanc.RegisterRestCallback('/manifest$', ServeManifest)

    print("UPS-RS REST endpoints registered (including /manifest)")
