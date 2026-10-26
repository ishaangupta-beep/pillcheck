"""Expiry date: read from the setup photo with OpenCV text models, corrected by the caregiver, acted on by the agent."""
import pytest
from conftest import TELMI, photo

from pillcheck.vision import expiry
from pillcheck.vision.expiry import normalise, parse_expiry, status

from datetime import date as _date, timedelta as _td
# Tomorrow: doses before a medication was created are ignored, so the test day must be after "now".
DAY = (_date.today() + _td(days=1)).isoformat() + "T{}:00+05:30"


@pytest.mark.parametrize("words,date", [
    (["bno1001mfdo11202gexp1212027"], "2027-12"),  # what the CRNN reads off "B.No. 1001 MFD 01/2026 EXP 12/2027"
    (["mfgdt", "01", "2025", "expdt", "dec2026"], "2026-12"),
    (["exp", "1227"], "2027-12"),
    (["expiry", "o8", "2o26"], "2026-08"),
    (["usebefore", "jun", "2029"], "2029-06"),
    (["mfg", "0324", "0926"], "2026-09"),  # no EXP keyword: the date after MFG's is the expiry
])
def test_parse_expiry(words, date):
    assert parse_expiry(words)["date"] == date


def test_parse_nothing():
    assert parse_expiry(["telmisartan", "40mg"]) is None
    assert "exp 12" in normalise("EXP1212027")


def test_status():
    from datetime import date
    assert status("2026-09", date(2026, 10, 22)) == "expired"
    assert status("2026-10", date(2026, 10, 22)) == "expiring_soon"  # valid through 31 Oct
    assert status("2026-10", date(2026, 10, 31)) == "expiring_soon"
    assert status("2027-12", date(2026, 10, 22)) == "ok"
    assert status(None, date(2026, 10, 22)) is None


@pytest.mark.skipif(not expiry.available(), reason="OpenCV text models not downloaded")
def test_read_from_strip_photo():
    import cv2 as cv
    import numpy as np
    img = cv.imdecode(np.frombuffer(photo(TELMI, set(), seed=3, difficulty=0.2), np.uint8), cv.IMREAD_COLOR)
    r = expiry.read_expiry(img)
    assert r["date"] == "2027-12" and r["confidence"] == "high"


def test_expired_strip_alerts_caregiver_and_patient(client):
    pid = client.post("/patients", json={"name": "Kamla"}).json()["id"]
    med = client.post(f"/patients/{pid}/medications", json={
        "name": "Telmisartan", "strength": "40 mg", "schedule": [{"time": "09:00"}]}).json()
    ref = client.post(f"/medications/{med['id']}/reference",
                      files={"file": ("r.jpg", photo(TELMI, set(), seed=11, difficulty=0.1), "image/jpeg")}).json()
    # The caregiver corrects the date (here: to one already past).
    r = client.put(f"/references/{ref['id']}/confirm", json={"expiry": "2026-09"})
    assert r.status_code == 200, r.text
    today = client.get(f"/patients/{pid}/today", params={"at": DAY.format("08:00")}).json()
    assert today["medications"][0]["expiry"] == {"date": "2026-09", "source": "caregiver", "confidence": "confirmed", "status": "expired"}

    check = client.post(f"/patients/{pid}/checks", data={"medication_id": med["id"], "at": DAY.format("09:05")},
                        files={"file": ("p.jpg", photo(TELMI, {0}, seed=12, difficulty=0.3), "image/jpeg")}).json()
    steps = {(s["tool"], s["input"].get("reason")) for s in check["agent_run"]["steps"] if s["status"] == "executed"}
    assert ("record_dose", None) in steps  # the dose still counts
    assert ("alert_caregiver", "expired") in steps
    notes = client.get(f"/patients/{pid}/notifications").json()
    assert any(n["kind"] == "expired" and "Sep 2026" in n["message"] for n in notes)
    assert any(n["to"] == "patient" and "expiry date" in n["message"] for n in notes)

    # Once a day, not on every photo.
    check2 = client.post(f"/patients/{pid}/checks", data={"medication_id": med["id"], "at": DAY.format("10:00")},
                         files={"file": ("p.jpg", photo(TELMI, {0}, seed=13, difficulty=0.3), "image/jpeg")}).json()
    assert not any(s["tool"] == "alert_caregiver" and s["input"].get("reason") == "expired" and s["status"] == "executed"
                   for s in check2["agent_run"]["steps"])

    # Fixing the date clears the state.
    r = client.put(f"/medications/{med['id']}/expiry", json={"expiry": "2028-03"})
    assert r.status_code == 200 and r.json()["expiry"]["date"] == "2028-03"
    assert client.put(f"/medications/{med['id']}/expiry", json={"expiry": "March"}).status_code == 422
