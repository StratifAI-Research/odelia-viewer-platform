"""
UPS-RS REST API endpoints
Implements DICOM PS3.18 Section 11 (UPS-RS)
"""

import json
import os
import threading
from pathlib import Path
from typing import Any

import orthanc

from ups.processor import process_workitem
from ups.storage import ups_storage
from ups.workitem import UPSWorkitem

# Optional outbound-host allowlist (ROUTER_HOST_ALLOWLIST env var). When
# unset/empty, host_is_allowed() returns True — preserves current research
# behaviour. See docs/production-hardening.md.
try:
    from host_allowlist import host_is_allowed
except Exception:

    def host_is_allowed(_url: str) -> bool:
        return True


MANIFEST_PATH = os.environ.get("AI_MANIFEST_PATH", "/etc/orthanc/manifest.json")


def CreateWorkitem(output: Any, uri: str, **request: Any) -> None:
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

        if not series_uids:
            output.SendHttpStatus(400, "Missing or empty series_uids in request body")
            return

        if not host_is_allowed(wado_rs_base):
            print(f"CreateWorkitem: wado_rs_base host not in ROUTER_HOST_ALLOWLIST: {wado_rs_base}")
            output.SendHttpStatus(403, "wado_rs_base host not allowed")
            return

        # Build WADO-RS retrieval URLs
        wado_rs_retrieval = []
        for series_uid in series_uids:
            retrieval_url = f"{wado_rs_base}/studies/{study_uid}/series/{series_uid}"
            wado_rs_retrieval.append(
                {"retrieval_url": retrieval_url, "study_uid": study_uid, "series_uid": series_uid}
            )

        # Create workitem (with optional structured input mapping)
        workitem = UPSWorkitem(
            study_uid=study_uid,
            series_uids=series_uids,
            wado_rs_retrieval=wado_rs_retrieval,
            priority=priority,
            input_mapping=input_mapping,
            input_configuration_id=input_configuration_id,
        )

        print(f"CreateWorkitem: Created workitem with UID: {workitem.workitem_uid}")

        # Store workitem
        ups_storage.store_workitem(workitem)
        print(f"CreateWorkitem: Stored workitem {workitem.workitem_uid}")

        # Verify storage
        verify = ups_storage.get_workitem(workitem.workitem_uid)
        if verify:
            print(
                f"CreateWorkitem: Verification successful - workitem {workitem.workitem_uid} can be retrieved"
            )
        else:
            print(
                f"CreateWorkitem: WARNING - workitem {workitem.workitem_uid} was NOT stored properly!"
            )

        print(f"Created workitem {workitem.workitem_uid} for study {study_uid}")

        # Process workitem immediately in background thread
        # (similar to OnStableStudy pattern - immediate execution, not polling)
        def process_in_background() -> None:
            try:
                process_workitem(workitem)
            except Exception as e:
                print(f"Error processing workitem in background: {e!s}")
                import traceback

                traceback.print_exc()

        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()

        # Return created workitem as DICOM JSON
        output.AnswerBuffer(json.dumps(workitem.data), "application/dicom+json")

    except Exception as e:
        error_message = f"Error creating workitem: {e!s}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def GetWorkitem(output: Any, uri: str, **request: Any) -> None:
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
        print(
            f"GetWorkitem: URI={uri}, groups={request.get('groups')}, extracted UID={workitem_uid}"
        )

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
        output.AnswerBuffer(json.dumps(workitem.data), "application/dicom+json")

    except Exception as e:
        error_message = f"Error retrieving workitem: {e!s}"
        print(error_message)
        import traceback

        traceback.print_exc()
        output.SendHttpStatus(500, error_message)


def UpdateWorkitemState(output: Any, uri: str, **request: Any) -> None:
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

        # Update state. progress_info is free text, so route it to the progress
        # *description* (ST) — passing it positionally would land it in the numeric
        # Procedure Step Progress DS tag, which only accepts a decimal string.
        workitem.update_state(new_state, progress_description=progress_info)
        ups_storage.store_workitem(workitem)

        print(f"Updated workitem {workitem_uid} state to {new_state}")

        # Return updated workitem
        output.AnswerBuffer(json.dumps(workitem.data), "application/dicom+json")

    except Exception as e:
        error_message = f"Error updating workitem state: {e!s}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def QueryWorkitems(output: Any, uri: str, **request: Any) -> None:
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
        # Parse query parameters. Orthanc may hand a query param as either a bare
        # string or a list of strings; normalize both so a value like "COMPLETED"
        # is never indexed down to its first character ("C").
        get_params = request.get("get", {})
        state_param = get_params.get("state")
        if isinstance(state_param, list | tuple):
            state_filter = state_param[0] if state_param else None
        else:
            state_filter = state_param

        # Query workitems
        workitems = ups_storage.list_workitems(state=state_filter)

        # Convert to DICOM JSON array
        result = [workitem.data for workitem in workitems]

        print(f"Query returned {len(result)} workitems (state filter: {state_filter})")

        output.AnswerBuffer(json.dumps(result), "application/dicom+json")

    except Exception as e:
        error_message = f"Error querying workitems: {e!s}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def SubscribeToWorkitem(output: Any, uri: str, **request: Any) -> None:
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

        if not host_is_allowed(subscriber_url):
            print(
                f"SubscribeToWorkitem: subscriber_url host not in ROUTER_HOST_ALLOWLIST: {subscriber_url}"
            )
            output.SendHttpStatus(403, "subscriber_url host not allowed")
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
        error_message = f"Error creating subscription: {e!s}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def UnsubscribeFromWorkitem(output: Any, uri: str, **request: Any) -> None:
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
        error_message = f"Error removing subscription: {e!s}"
        print(error_message)
        output.SendHttpStatus(500, error_message)


def ServeManifest(output: Any, uri: str, **request: Any) -> None:
    """
    GET /manifest
    Serve the model input manifest (if mounted).
    Returns 404 when no manifest is available, allowing the viewer to fall back
    to flat series selection.
    """
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return

    if not Path(MANIFEST_PATH).is_file():
        output.SendHttpStatus(404, "No manifest available")
        return

    try:
        with Path(MANIFEST_PATH).open() as f:
            manifest_data = f.read()
        json.loads(manifest_data)  # validate JSON
        output.AnswerBuffer(manifest_data, "application/json")
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading manifest: {e}")
        output.SendHttpStatus(500, f"Error reading manifest: {e}")


# Helper to register all UPS routes
def register_ups_routes() -> None:
    """Register all UPS-RS REST endpoints"""
    orthanc.RegisterRestCallback("/ups-rs/workitems$", CreateWorkitem)
    orthanc.RegisterRestCallback("/ups-rs/workitems/([0-9.]+)$", GetWorkitem)
    orthanc.RegisterRestCallback("/ups-rs/workitems/([0-9.]+)/state$", UpdateWorkitemState)
    orthanc.RegisterRestCallback("/ups-rs/workitems/([0-9.]+)/subscribers$", SubscribeToWorkitem)
    orthanc.RegisterRestCallback(
        "/ups-rs/workitems/([0-9.]+)/subscribers/(.+)$", UnsubscribeFromWorkitem
    )
    orthanc.RegisterRestCallback("/manifest$", ServeManifest)

    print("UPS-RS REST endpoints registered (including /manifest)")
