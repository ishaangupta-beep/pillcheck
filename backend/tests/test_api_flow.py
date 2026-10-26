"""End-to-end through the HTTP API: setup, daily checks, agent decisions, scheduler.

The story (patient in Asia/Kolkata, 21 Oct 2026):
  Telmisartan 09:00 + 21:00, Metformin 08:30 + 20:30.
"""
from conftest import METF, OTHER, TELMI, photo

from datetime import date as _date, timedelta as _td
# Tomorrow: doses before a medication was created are ignored, so the test day must be after "now".
DAY = (_date.today() + _td(days=1)).isoformat() + "T{}:00+05:30"


def at(hhmm: str) -> str:
    return DAY.format(hhmm)


def tools(check: dict) -> list[tuple[str, str]]:
    return [(s["tool"], s["status"]) for s in check["agent_run"]["steps"]]


def executed(check: dict) -> set[str]:
    return {s["tool"] for s in check["agent_run"]["steps"] if s["status"] == "executed"}


def post_check(client, pid, data, med=None, when=None):
    r = client.post(f"/patients/{pid}/checks", files={"file": ("p.jpg", data, "image/jpeg")},
                    data={k: v for k, v in {"medication_id": med, "at": when}.items() if v})
    assert r.status_code == 201, r.text
    return r.json()


def setup_strip(client, med_id, spec):
    r = client.post(f"/medications/{med_id}/reference",
                    files={"file": ("ref.jpg", photo(spec, seed=11, difficulty=0.1), "image/jpeg")})
    assert r.status_code == 201, r.text
    ref = r.json()
    assert len(ref["pockets"]) == spec.rows * spec.cols
    assert client.get(f"/references/{ref['id']}/preview.jpg").status_code == 200
    r = client.put(f"/references/{ref['id']}/confirm", json={})
    assert r.status_code == 200 and r.json()["confirmed"]
    return ref


def test_full_day(client):
    assert client.get("/health").json()["vision_engine"] == "aliked+lightglue"

    pid = client.post("/patients", json={"name": "Asha", "language": "en",
                                         "caregiver": {"name": "Ishaan", "phone": "+91-0000000000"}}).json()["id"]
    telmi = client.post(f"/patients/{pid}/medications", json={
        "name": "Telmisartan", "strength": "40 mg", "schedule": [{"time": "09:00"}, {"time": "21:00"}]}).json()
    metf = client.post(f"/patients/{pid}/medications", json={
        "name": "Metformin", "strength": "500 mg", "schedule": [{"time": "08:30"}, {"time": "20:30"}]}).json()

    # A check before any strip is registered is refused clearly.
    r = client.post(f"/patients/{pid}/checks", files={"file": ("p.jpg", photo(TELMI, seed=1), "image/jpeg")})
    assert r.status_code == 409

    setup_strip(client, telmi["id"], TELMI)
    setup_strip(client, metf["id"], METF)

    # 09:10 Telmisartan taken: one pocket popped -> dose recorded, patient thanked.
    c1 = post_check(client, pid, photo(TELMI, {0}, seed=21, difficulty=0.6), telmi["id"], at("09:10"))
    assert c1["status"] == "ok" and c1["perception"]["newly_popped"] == [0]
    assert ("record_dose", "executed") in tools(c1)
    assert client.get(f"/checks/{c1['id']}/evidence.jpg").status_code == 200

    # Same photo sent again -> rejected as duplicate, retake requested, nothing recorded.
    dup = post_check(client, pid, photo(TELMI, {0}, seed=21, difficulty=0.6), telmi["id"], at("09:12"))
    assert dup["status"] == "duplicate"
    assert executed(dup) == {"request_retake"}

    # 09:20 another Telmisartan removed: next slot is 21:00 (outside the early window) -> extra -> urgent alert.
    c2 = post_check(client, pid, photo(TELMI, {0, 1}, seed=22, difficulty=0.6), telmi["id"], at("09:20"))
    assert c2["perception"]["newly_popped"] == [1]
    assert {"record_dose", "alert_caregiver"} <= executed(c2), [(x["tool"],x["status"],x["result"],x["reason"]) for x in c2["agent_run"]["steps"]]
    alert = next(s for s in c2["agent_run"]["steps"] if s["tool"] == "alert_caregiver")
    assert alert["input"]["severity"] == "urgent" and alert["input"]["reason"] == "extra_removed"

    # Blurry photo -> retake with capture guidance.
    blur = post_check(client, pid, photo(TELMI, {0, 1}, seed=23, difficulty=0.5, blur_sigma=4), telmi["id"], at("09:25"))
    assert blur["status"] == "retake" and executed(blur) == {"request_retake"}

    # Unregistered strip -> unrecognised.
    unk = post_check(client, pid, photo(OTHER, seed=24), telmi["id"], at("09:30"))
    assert unk["status"] == "unrecognised"

    # 12:00 scheduler: Metformin 08:30 is 210 min late -> missed -> caregiver alerted (once).
    mine = lambda t: [r for r in t["runs"] if r["patient_id"] == pid]  # other tests share the store
    t = mine(client.post("/agent/tick", json={"at": at("12:00")}).json())
    assert len(t) == 1
    steps = t[0]["steps"]
    assert any(s["tool"] == "alert_caregiver" and s["input"]["reason"] == "missed" for s in steps)
    assert mine(client.post("/agent/tick", json={"at": at("12:15")}).json()) == []  # no repeat

    # 20:40 expecting Telmisartan, but the Metformin strip is photographed with a pill removed:
    # identified as Metformin, recorded against the 20:30 dose (08:30 stays missed), patient told which strip it is.
    wrong = post_check(client, pid, photo(METF, {0}, seed=25, difficulty=0.6), telmi["id"], at("20:40"))
    assert wrong["status"] == "wrong_medication" and wrong["medication_id"] == metf["id"], (wrong["perception"].get("identity"), wrong["perception"].get("registration"))
    assert ("record_dose", "executed") in tools(wrong)

    # 20:45 Telmisartan photo with nothing new: 21:00 is within the early window -> reminder.
    nothing = post_check(client, pid, photo(TELMI, {0, 1}, seed=26, difficulty=0.6), telmi["id"], at("20:45"))
    assert nothing["status"] == "ok" and nothing["perception"]["newly_popped"] == []
    msg = next(s for s in nothing["agent_run"]["steps"] if s["tool"] == "send_patient_message")
    assert msg["input"]["kind"] == "reminder"

    today = client.get(f"/patients/{pid}/today", params={"at": at("21:00")}).json()
    statuses = {m["medication"]["name"]: [s["status"] for s in m["slots"]] for m in today["medications"]}
    assert statuses["Telmisartan"] == ["taken", "pending"]  # the 09:20 pill was extra, not an early 21:00 dose
    assert statuses["Metformin"] == ["missed", "taken"]

    # The timeline and notifications reflect all of it.
    tl = client.get(f"/patients/{pid}/timeline").json()
    assert {i["entry"] for i in tl} >= {"check", "notification", "dose", "action"}
    carer = client.get(f"/patients/{pid}/notifications", params={"to": "caregiver"}).json()
    assert {n["kind"] for n in carer} >= {"extra_removed", "missed"}
