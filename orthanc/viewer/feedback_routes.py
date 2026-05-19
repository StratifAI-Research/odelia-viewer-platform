import json
from typing import Any, Dict

import orthanc

# Import sibling module when this file is loaded as a top-level module
try:
    import feedback_db  # type: ignore
except Exception as _e:
    raise


def _json(output, obj: Dict[str, Any], status_ok: bool = True):
    body = json.dumps(obj)
    if status_ok:
        output.AnswerBuffer(body, "application/json")
    else:
        # For non-200 (e.g., 201/409), SendHttpStatus is the only way to set code.
        # Content-Type may be text/plain, but body is JSON string.
        # Callers should still be able to parse.
        # The caller must set the exact status via output.SendHttpStatus before returning.
        return body


def _bad_request(output, message: str):
    output.SendHttpStatus(400, message)


def _validate_submit_payload(p: Dict[str, Any]) -> str:
    required = [
        "study_uid",
        "model_name",
        "model_version",
        "result_ts",
        "user_id",
        "verdict_L",
        "verdict_R",
    ]
    missing = [k for k in required if k not in p]
    if missing:
        return f"Missing fields: {', '.join(missing)}"
    try:
        vL = int(p["verdict_L"])
        vR = int(p["verdict_R"])
    except Exception:
        return "verdict_L and verdict_R must be integers in (-1,0,1)"
    if vL not in (-1, 0, 1) or vR not in (-1, 0, 1):
        return "verdict_L and verdict_R must be in (-1,0,1)"
    # optional edited flag must be boolean if present
    if "edited" in p and not isinstance(p["edited"], (bool, int)):
        return "edited must be boolean if provided"
    return ""


def FeedbackSubmit(output, uri, **request):
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return
    try:
        p = json.loads(request.get("body", "{}"))
    except Exception as e:
        _bad_request(output, f"Invalid JSON body: {str(e)}")
        return
    err = _validate_submit_payload(p)
    if err:
        _bad_request(output, err)
        return
    try:
        saved = feedback_db.submit_feedback(p)
        # 201 Created
        body = json.dumps(saved)
        output.SendHttpStatus(201, body)
    except feedback_db.ConflictError as e:
        output.SendHttpStatus(
            409,
            json.dumps({"code": 409, "message": str(e)}),
        )
    except Exception as e:
        output.SendHttpStatus(500, json.dumps({"code": 500, "message": str(e)}))


def FeedbackRead(output, uri, **request):
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return
    q = request.get("get", {}) or {}
    study_uid = q.get("study_uid")
    model_name = q.get("model_name")
    model_version = q.get("model_version")
    result_ts = q.get("result_ts")
    include_users = str(q.get("includeUsers", "false")).lower() in ("1", "true", "yes")
    include_history = str(q.get("includeHistory", "false")).lower() in (
        "1",
        "true",
        "yes",
    )
    if not all([study_uid, model_name, model_version, result_ts]):
        _bad_request(
            output,
            "Missing one of required query params: study_uid, model_name, model_version, result_ts",
        )
        return
    try:
        data = feedback_db.read_feedback(
            study_uid,
            model_name,
            model_version,
            result_ts,
            include_users,
            include_history,
        )
        _json(output, data)
    except Exception as e:
        output.SendHttpStatus(500, json.dumps({"code": 500, "message": str(e)}))


def FeedbackRegisterResult(output, uri, **request):
    if request["method"] != "POST":
        output.SendMethodNotAllowed("POST")
        return
    try:
        p = json.loads(request.get("body", "{}"))
    except Exception as e:
        _bad_request(output, f"Invalid JSON body: {str(e)}")
        return
    required = ["study_uid", "model_name", "model_version", "result_ts"]
    if not all(k in p for k in required):
        _bad_request(
            output, f"Missing fields: {', '.join([k for k in required if k not in p])}"
        )
        return
    try:
        res = feedback_db.register_result(
            p["study_uid"],
            p["model_name"],
            p["model_version"],
            p["result_ts"],
            p.get("meta_json"),
        )
        body = json.dumps(res)
        output.SendHttpStatus(201 if res.get("created") else 200, body)
    except Exception as e:
        output.SendHttpStatus(500, json.dumps({"code": 500, "message": str(e)}))


def FeedbackExportNdjson(output, uri, **request):
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return
    q = request.get("get", {}) or {}
    since = q.get("since")
    until = q.get("until")
    model_name = q.get("model_name")
    model_version = q.get("model_version")
    scope = q.get("scope", "history")
    try:
        chunks = []
        for obj in feedback_db.export_rows_ndjson(
            since, until, model_name, model_version, scope
        ):
            chunks.append(json.dumps(obj))
        ndjson = "\n".join(chunks)
        output.AnswerBuffer(ndjson, "application/x-ndjson")
    except Exception as e:
        output.SendHttpStatus(500, json.dumps({"code": 500, "message": str(e)}))


def FeedbackExportCsv(output, uri, **request):
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return
    q = request.get("get", {}) or {}
    since = q.get("since")
    until = q.get("until")
    model_name = q.get("model_name")
    model_version = q.get("model_version")
    scope = q.get("scope", "history")
    try:
        header, rows_iter = feedback_db.export_rows_csv(
            since, until, model_name, model_version, scope
        )
        lines = [header]
        for r in rows_iter:
            # Ensure proper CSV escaping by replacing potential commas/newlines in fields if any
            # Our schema contains simple fields; keep minimal
            fields = [
                str(r[0]),
                str(r[1]),
                str(r[2]),
                str(r[3]),
                str(r[4]),
                str(int(r[5])),
                str(int(r[6])),
                str(r[7]),
                str(r[8]),
            ]
            lines.append(",".join(fields) + "\n")
        output.AnswerBuffer("".join(lines), "text/csv")
    except Exception as e:
        output.SendHttpStatus(500, json.dumps({"code": 500, "message": str(e)}))


def FeedbackHealth(output, uri, **request):
    if request["method"] != "GET":
        output.SendMethodNotAllowed("GET")
        return
    try:
        info = feedback_db.health()
        _json(output, info)
    except Exception as e:
        output.SendHttpStatus(500, json.dumps({"code": 500, "message": str(e)}))


def register_feedback_endpoints():
    orthanc.RegisterRestCallback("/feedback/submit", FeedbackSubmit)
    orthanc.RegisterRestCallback("/feedback", FeedbackRead)
    orthanc.RegisterRestCallback("/feedback/register-result", FeedbackRegisterResult)
    orthanc.RegisterRestCallback("/feedback/export.ndjson", FeedbackExportNdjson)
    orthanc.RegisterRestCallback("/feedback/export.csv", FeedbackExportCsv)
    orthanc.RegisterRestCallback("/feedback/health", FeedbackHealth)
