"""Public deployment behaviour: access control, demo sessions, rate limits, Lambda entry point."""
import base64
import copy
import json

import pytest
from conftest import TELMI, photo

from pillcheck.config import settings
from pillcheck.lambda_handler import handler

TOKEN = "test-admin-token"


@pytest.fixture()
def locked():
    """Deployed configuration: admin token set, tight limits."""
    original_token, original_limits = settings.api_token, settings.limits
    tight = copy.copy(original_limits)
    object.__setattr__(tight, "checks_per_ip_hour", 3)
    object.__setattr__(tight, "demo_sessions_per_ip_hour", 2)
    object.__setattr__(settings, "api_token", TOKEN)
    object.__setattr__(settings, "limits", tight)
    yield {"Authorization": f"Bearer {TOKEN}"}
    object.__setattr__(settings, "api_token", original_token)
    object.__setattr__(settings, "limits", original_limits)


def visitor(ip):
    return {"X-Forwarded-For": ip}


def test_real_patients_need_the_token(client, locked):
    assert client.post("/patients", json={"name": "Real"}).status_code == 401
    real = client.post("/patients", json={"name": "Real"}, headers=locked).json()
    assert client.get(f"/patients/{real['id']}").status_code == 401
    assert client.get(f"/patients/{real['id']}/today").status_code == 401
    assert client.get(f"/patients/{real['id']}", headers=locked).status_code == 200
    assert client.get("/patients").status_code == 401
    assert client.post("/agent/tick", json={}).status_code == 401
    assert client.get("/health").json()["auth"] == "token"


def test_demo_session_end_to_end(client, locked):
    ip = visitor("203.0.113.7")
    r = client.post("/demo/session", json={"language": "en"}, headers=ip)
    assert r.status_code == 201, r.text
    session = r.json()
    pid = session["patient"]["id"]
    telmi = next(m for m in session["medications"] if m["name"] == "Telmisartan")
    assert {s["id"] for s in session["samples"]} >= {"telmisartan_dose1", "telmisartan_extra", "metformin_dose1"}
    assert client.get("/demo/assets/telmisartan_dose1.jpg").status_code == 200

    today = client.get(f"/patients/{pid}/today", headers=ip).json()
    assert all(m["has_strip"] for m in today["medications"])

    day = today["local_date"]
    chk = client.post(f"/patients/{pid}/checks", headers=ip,
                      data={"sample_id": "telmisartan_dose1", "medication_id": telmi["id"], "at": f"{day}T09:10:00+05:30"})
    assert chk.status_code == 201, chk.text
    body = chk.json()
    assert body["status"] == "ok" and body["perception"]["newly_popped"] == [0]
    assert any(s["tool"] == "record_dose" and s["status"] == "executed" for s in body["agent_run"]["steps"])
    assert client.get(f"/checks/{body['id']}/evidence.jpg", headers=ip).status_code == 200

    # Fast-forward the clock: Metformin 08:30 was never taken -> missed -> caregiver alerted (in the app only).
    run = client.post(f"/patients/{pid}/tick", json={"at": f"{day}T12:00:00+05:30"}, headers=ip).json()["run"]
    assert any(s["tool"] == "alert_caregiver" for s in run["steps"])
    notes = client.get(f"/patients/{pid}/notifications", params={"to": "caregiver"}, headers=ip).json()
    assert notes and all(n["channels"] == ["app"] for n in notes)

    # Sample photos are refused for real patients; uploads still work for demo patients.
    up = client.post(f"/patients/{pid}/checks", headers=ip, files={"file": ("p.jpg", photo(TELMI, {0}, seed=91), "image/jpeg")},
                     data={"medication_id": telmi["id"]})
    assert up.status_code == 201


def test_rate_limits(client, locked):
    ip = visitor("198.51.100.23")
    pid = client.post("/demo/session", json={}, headers=ip).json()["patient"]["id"]
    assert client.post("/demo/session", json={}, headers=ip).status_code == 201
    assert client.post("/demo/session", json={}, headers=ip).status_code == 429  # 2 per hour in this test
    codes = [client.post(f"/patients/{pid}/checks", headers=ip, data={"sample_id": "telmisartan_blurry"}).status_code
             for _ in range(4)]
    assert codes == [201, 201, 201, 429]
    # Another visitor is unaffected; the admin is never limited.
    assert client.post("/demo/session", json={}, headers=visitor("192.0.2.1")).status_code == 201
    assert client.post(f"/patients/{pid}/checks", headers=locked, data={"sample_id": "telmisartan_blurry"}).status_code == 201


def _url_event(method, path, body=None, headers=None):
    """A Lambda Function URL (payload v2) event."""
    hdrs = {"host": "abc.lambda-url.ap-south-1.on.aws", "x-forwarded-for": "203.0.113.50", **(headers or {})}
    ev = {"version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "", "headers": hdrs,
          "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1", "sourceIp": "203.0.113.50",
                                      "userAgent": "test"}, "domainName": hdrs["host"], "stage": "$default"},
          "isBase64Encoded": False}
    if body is not None:
        ev["body"] = json.dumps(body)
        ev["headers"]["content-type"] = "application/json"
    return ev


def test_lambda_handler_http_and_scheduled_events():
    r = handler(_url_event("GET", "/health"), None)
    assert r["statusCode"] == 200 and json.loads(r["body"])["ok"]
    r = handler(_url_event("POST", "/demo/session", {"language": "hi"}), None)
    assert r["statusCode"] == 201
    pid = json.loads(r["body"])["patient"]["id"]
    # Binary responses come back base64-encoded for the Function URL.
    r = handler(_url_event("GET", "/demo/assets/telmisartan_reference.jpg"), None)
    assert r["statusCode"] == 200 and r["isBase64Encoded"] and base64.b64decode(r["body"])[:2] == b"\xff\xd8"
    assert handler({"pillcheck": "tick"}, None)["task"] == "tick"
    assert handler({"pillcheck": "warm"}, None)["engine"] == "aliked+lightglue"
    assert pid
