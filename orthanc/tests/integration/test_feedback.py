import json

import requests


def test_health(base_url):
    r = requests.get(f"{base_url}/feedback/health")
    assert r.status_code == 200
    j = r.json()
    assert "db_ready" in j and j["db_ready"] is True
    assert "sqlite_version" in j


def test_register_and_submit_flow(base_url, unique_payload):
    # 1) register-result idempotent
    reg1 = requests.post(
        f"{base_url}/feedback/register-result",
        json={
            "study_uid": unique_payload["study_uid"],
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "result_ts": unique_payload["result_ts"],
        },
    )
    assert reg1.status_code in (200, 201)

    reg2 = requests.post(
        f"{base_url}/feedback/register-result",
        json={
            "study_uid": unique_payload["study_uid"],
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "result_ts": unique_payload["result_ts"],
        },
    )
    assert reg2.status_code in (200, 201)

    # 2) submit
    sub = requests.post(
        f"{base_url}/feedback/submit",
        json={
            **unique_payload,
            "verdict_L": 1,
            "verdict_R": -1,
        },
    )
    assert sub.status_code == 201
    j = json.loads(sub.text)
    assert j["study_uid"] == unique_payload["study_uid"]
    assert j["verdict_L"] == 1 and j["verdict_R"] == -1

    # 3) duplicate submit
    dup = requests.post(
        f"{base_url}/feedback/submit",
        json={
            **unique_payload,
            "verdict_L": 1,
            "verdict_R": -1,
        },
    )
    assert dup.status_code == 409

    # 4) aggregates
    agg = requests.get(
        f"{base_url}/feedback",
        params={
            "study_uid": unique_payload["study_uid"],
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "result_ts": unique_payload["result_ts"],
            "includeUsers": "true",
        },
    )
    assert agg.status_code == 200
    aj = agg.json()
    assert aj["n_submissions"] >= 1
    assert aj["aggregate"]["L"]["agree"] >= 1
    assert isinstance(aj.get("users", []), list)

    # 5) exports
    nd = requests.get(
        f"{base_url}/feedback/export.ndjson",
        params={
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
        },
    )
    assert nd.status_code == 200
    assert len(nd.text.strip()) > 0

    csv = requests.get(
        f"{base_url}/feedback/export.csv",
        params={
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
        },
    )
    assert csv.status_code == 200
    assert csv.text.splitlines()[0].startswith("study_uid,model_name,model_version")


def test_edit_flow_and_exports(base_url, unique_payload):
    # Initial submit
    sub1 = requests.post(
        f"{base_url}/feedback/submit",
        json={
            **unique_payload,
            "verdict_L": 1,
            "verdict_R": 0,
        },
    )
    assert sub1.status_code == 201
    j1 = json.loads(sub1.text)
    assert j1["submission_kind"] == "initial"

    # Duplicate without edited flag should 409
    dup = requests.post(
        f"{base_url}/feedback/submit",
        json={
            **unique_payload,
            "verdict_L": 1,
            "verdict_R": 0,
        },
    )
    assert dup.status_code == 409

    # Edit with edited=true
    sub2 = requests.post(
        f"{base_url}/feedback/submit",
        json={
            **unique_payload,
            "verdict_L": -1,
            "verdict_R": 1,
            "edited": True,
        },
    )
    assert sub2.status_code == 201
    j2 = json.loads(sub2.text)
    assert j2["submission_kind"] == "edit"

    # Read current aggregates and users
    agg = requests.get(
        f"{base_url}/feedback",
        params={
            "study_uid": unique_payload["study_uid"],
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "result_ts": unique_payload["result_ts"],
            "includeUsers": "true",
            "includeHistory": "true",
        },
    )
    assert agg.status_code == 200
    aj = agg.json()
    # Current should reflect the latest edit values
    assert aj["aggregate"]["L"]["disagree"] >= 1
    assert aj["aggregate"]["R"]["agree"] >= 1
    assert isinstance(aj.get("users", []), list)
    # History should contain at least 2 events for this user/result
    history = [
        h for h in aj.get("history", []) if h["user_id"] == unique_payload["user_id"]
    ]
    assert len(history) >= 2
    assert set(h["submission_kind"] for h in history) >= {"initial", "edit"}

    # NDJSON export history: filter lines for our study/result
    nd_hist = requests.get(
        f"{base_url}/feedback/export.ndjson",
        params={
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "scope": "history",
        },
    )
    assert nd_hist.status_code == 200
    lines = [ln for ln in nd_hist.text.splitlines() if ln.strip()]
    parsed = [json.loads(ln) for ln in lines]
    mine = [
        p
        for p in parsed
        if p["study_uid"] == unique_payload["study_uid"]
        and p["result_ts"] == unique_payload["result_ts"]
        and p["user_id"] == unique_payload["user_id"]
    ]
    assert len(mine) >= 2
    assert set(p["submission_kind"] for p in mine) >= {"initial", "edit"}

    # NDJSON export current: exactly one current row per user/result
    nd_curr = requests.get(
        f"{base_url}/feedback/export.ndjson",
        params={
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "scope": "current",
        },
    )
    assert nd_curr.status_code == 200
    curr_lines = [ln for ln in nd_curr.text.splitlines() if ln.strip()]
    curr_parsed = [json.loads(ln) for ln in curr_lines]
    curr_mine = [
        p
        for p in curr_parsed
        if p["study_uid"] == unique_payload["study_uid"]
        and p["result_ts"] == unique_payload["result_ts"]
        and p["user_id"] == unique_payload["user_id"]
    ]
    assert len(curr_mine) == 1
    assert curr_mine[0]["submission_kind"] == "edit"

    # CSV export current/history headers include submission_kind
    csv_hist = requests.get(
        f"{base_url}/feedback/export.csv",
        params={
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "scope": "history",
        },
    )
    assert csv_hist.status_code == 200
    assert csv_hist.text.splitlines()[0].endswith(",submission_kind")

    csv_curr = requests.get(
        f"{base_url}/feedback/export.csv",
        params={
            "model_name": unique_payload["model_name"],
            "model_version": unique_payload["model_version"],
            "scope": "current",
        },
    )
    assert csv_curr.status_code == 200
    assert csv_curr.text.splitlines()[0].endswith(",submission_kind")
