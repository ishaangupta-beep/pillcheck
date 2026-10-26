"""Agent behaviour: refill with approval, human review, Bedrock planner + safety floor, policy guards."""
import copy
from datetime import datetime, timezone

import pytest
from conftest import TELMI, photo

from pillcheck import domain, service
from pillcheck.agent import policy, runner
from pillcheck.agent.context import AgentContext
from pillcheck.config import settings
from pillcheck.storage import get_store

from datetime import date as _date, timedelta as _td
# Tomorrow: doses before a medication was created are ignored, so the test day must be after "now".
DAY = (_date.today() + _td(days=1)).isoformat() + "T{}:00+05:30"


def at(hhmm):
    return DAY.format(hhmm)


@pytest.fixture()
def patient_with_strip(client):
    """Telmisartan twice daily; the strip was registered with 7 of 10 pockets already empty."""
    pid = client.post("/patients", json={"name": "Ramesh", "language": "hi"}).json()["id"]
    med = client.post(f"/patients/{pid}/medications", json={
        "name": "Telmisartan", "strength": "40 mg", "schedule": [{"time": "09:00"}, {"time": "21:00"}]}).json()
    empty = {0, 1, 2, 3, 4, 5, 6}
    ref = client.post(f"/medications/{med['id']}/reference",
                      files={"file": ("r.jpg", photo(TELMI, empty, seed=41, difficulty=0.1), "image/jpeg")}).json()
    pockets = [{**p, "initially_empty": p["id"] in empty} for p in ref["pockets"]]
    r = client.put(f"/references/{ref['id']}/confirm", json={"pockets": pockets})
    assert r.status_code == 200, r.text
    return pid, med


def test_refill_drafted_and_approved(client, patient_with_strip):
    pid, med = patient_with_strip
    r = client.post(f"/patients/{pid}/checks", files={"file": ("p.jpg", photo(TELMI, {0, 1, 2, 3, 4, 5, 6, 7}, seed=42, difficulty=0.5), "image/jpeg")},
                    data={"medication_id": med["id"], "at": at("09:05")})
    check = r.json()
    assert check["perception"]["newly_popped"] == [7]
    done = {s["tool"] for s in check["agent_run"]["steps"] if s["status"] == "executed"}
    assert {"record_dose", "draft_refill_request"} <= done
    # Hindi patient: the confirmation is in Hindi.
    msg = next(s for s in check["agent_run"]["steps"] if s["tool"] == "send_patient_message")
    assert "धन्यवाद" in msg["input"]["message"]

    pending = client.get(f"/patients/{pid}/actions", params={"status": "pending_approval"}).json()
    refill = next(a for a in pending if a["type"] == "draft_refill_request")
    assert refill["payload"]["pills_left"] == 2 and refill["payload"]["days_left"] == 1.0
    decided = client.post(f"/actions/{refill['id']}/decision", json={"approve": True}).json()
    assert decided["status"] == "approved"
    assert client.post(f"/actions/{refill['id']}/decision", json={"approve": True}).status_code == 409


def _ctx(store, patient, med, perception, check_extra=None, now=None):
    now = now or datetime(2026, 10, 22, 3, 40, tzinfo=timezone.utc)  # 09:10 IST
    check = {"id": "chk_test", "recent_unclear": 0, "recent_duplicates": 0, **(check_extra or {})}
    ctx = AgentContext(store=store, patient=patient, medications={med["id"]: med}, now=now,
                       event="photo_check", check=check, perception=perception)
    ctx.reconciliation[med["id"]] = domain.reconcile(store, patient, med, len(perception.get("newly_popped", [])), now)
    return ctx


def _perception(med_id, status="ok", newly=(), unclear=(), inconsistent=()):
    pockets = [{"id": i, "state": "uncertain" if i in unclear else ("popped" if i in newly else "intact")} for i in range(10)]
    return {"status": status, "medication_id": med_id, "identity": {"expected_medication_id": med_id},
            "newly_popped": list(newly), "inconsistent": list(inconsistent), "pockets": pockets,
            "counts": {}, "quality": {"issues": []}, "guidance": []}


@pytest.fixture()
def plain_patient():
    store = get_store()
    patient = service.create_patient(store, {"name": "Meera"})
    med = service.create_medication(store, patient["id"], {"name": "Amlodipine", "strength": "5 mg", "schedule": [{"time": "09:00"}]})
    med["created_at"] = "2026-01-01T00:00:00+00:00"
    store.put("medications", med)
    return store, patient, med


def test_unclear_twice_escalates_to_caregiver_review(plain_patient):
    store, patient, med = plain_patient
    first = runner.run(_ctx(store, patient, med, _perception(med["id"], unclear=(3,))))
    assert [s["tool"] for s in first["steps"]] == ["request_retake"]
    second = runner.run(_ctx(store, patient, med, _perception(med["id"], unclear=(3,)), {"recent_unclear": 1}))
    assert second["steps"][0]["tool"] == "request_human_review"
    review = store.find("actions", patient["id"], type="request_human_review")[-1]
    assert review["status"] == "pending_approval" and review["payload"]["kind"] == "confirm_dose"
    # Caregiver confirms -> the 09:00 dose is recorded.
    service.decide_action(store, review["id"], True, "saw her take it", at=at("09:30"))
    slot = store.get("doses", domain.slot_id(med["id"], datetime.fromisoformat(at("09:30")).date(), "09:00"))
    assert slot["status"] == "taken"


def test_new_strip_is_not_recorded(plain_patient):
    store, patient, med = plain_patient
    run = runner.run(_ctx(store, patient, med, _perception(med["id"], newly=(), inconsistent=(0, 1, 2))))
    assert [s["tool"] for s in run["steps"]] == ["request_human_review"]


def test_policy_blocks_unsafe_calls(plain_patient):
    store, patient, med = plain_patient
    ctx = _ctx(store, patient, med, _perception(med["id"], status="retake"))
    ok, why = policy.validate(ctx, "record_dose", {"medication_id": med["id"]})
    assert not ok and "not trusted" in why
    ok, why = policy.validate(ctx, "send_patient_message", {"message": "You can take another tablet now.", "kind": "information"})
    assert not ok and "dosing advice" in why
    ok, _ = policy.validate(ctx, "alert_caregiver", {"message": "Possible double dose of Amlodipine.", "severity": "urgent", "reason": "extra_removed"})
    assert ok
    ok, why = policy.validate(ctx, "delete_everything", {})
    assert not ok


class FakeBedrock:
    """Scripted Converse responses: the 'model' confirms the dose but forgets the caregiver alert."""

    def __init__(self, med_id):
        self.calls = 0
        self.med_id = med_id

    def converse(self, **kw):
        self.calls += 1
        assert kw["toolConfig"]["tools"] and "dosing advice" in kw["system"][0]["text"]
        if self.calls == 1:
            content = [{"text": "Two tablets were removed; I'll record the dose."},
                       {"toolUse": {"toolUseId": "t1", "name": "record_dose", "input": {"medication_id": self.med_id}}},
                       {"toolUse": {"toolUseId": "t2", "name": "send_patient_message",
                                    "input": {"message": "Please take another tablet tomorrow.", "kind": "information"}}}]
        else:
            content = [{"text": "Recorded the 9 AM dose."}]
        return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": "end_turn"}


def test_bedrock_planner_with_safety_floor(plain_patient, monkeypatch):
    store, patient, med = plain_patient
    fake = FakeBedrock(med["id"])
    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
    original = settings.agent
    patched = copy.copy(original)
    object.__setattr__(patched, "bedrock_model_id", "test-model")  # settings are frozen dataclasses
    object.__setattr__(settings, "agent", patched)
    try:
        run = runner.run(_ctx(store, patient, med, _perception(med["id"], newly=(0, 1))))
    finally:
        object.__setattr__(settings, "agent", original)
    assert run["mode"] == "bedrock"
    by_tool = {(s["tool"], s["source"]): s for s in run["steps"]}
    assert by_tool[("record_dose", "bedrock")]["status"] == "executed"
    # The model's dosing-advice message was blocked...
    assert by_tool[("send_patient_message", "bedrock")]["status"] == "blocked"
    # ...and the urgent alert it forgot was added by the safety floor.
    floor = by_tool[("alert_caregiver", "safety_floor")]
    assert floor["status"] == "executed" and floor["input"]["severity"] == "urgent"
