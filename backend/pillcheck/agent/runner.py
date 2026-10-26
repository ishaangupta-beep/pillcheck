"""Agent entry points: one run per photo check, one per scheduler tick.

Perception -> decision -> action, with a full trace:
  1. facts: the perception record and computed reconciliation
  2. plan: Bedrock (if configured) or the rule planner
  3. every call is validated by the policy before it executes
  4. safety floor: mandatory actions the plan skipped are executed
"""
from __future__ import annotations

import json
import logging
import time
from datetime import timedelta

from .. import domain
from ..config import settings
from ..storage import new_id
from . import bedrock, policy, rules
from .context import AgentContext
from .tools import execute

log = logging.getLogger("pillcheck.agent")


def _run_rules(ctx: AgentContext, steps: list[dict], source: str) -> None:
    for s in steps:
        ok, why = policy.validate(ctx, s["tool"], s["args"])
        entry = {"source": source, "tool": s["tool"], "input": s["args"], "reason": s["reason"]}
        if ok:
            entry.update(status="executed", result=execute(ctx, s["tool"], s["args"]))
        else:
            entry.update(status="blocked", result={"blocked_by_policy": why})
        ctx.trace.append(entry)


def run(ctx: AgentContext) -> dict:
    t0 = time.perf_counter()
    planned = rules.plan(ctx)
    mandatory = [s for s in planned if s["mandatory"]]
    mode, summary, fallback_reason = "rules", "", None

    use_bedrock = bool(settings.agent.bedrock_model_id)
    if use_bedrock:
        # Daily budget: past the cap the agent quietly uses the free rule planner.
        day = ctx.now.strftime("%Y-%m-%d")
        if ctx.store.incr(f"bedrock_runs_{day}", 3 * 86400) > settings.limits.bedrock_runs_per_day:
            use_bedrock = False
            fallback_reason = "daily AI budget reached; rule planner used"
    if use_bedrock:
        try:
            summary = bedrock.run(ctx)
            mode = "bedrock"
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            log.warning("Bedrock planner failed, using rules: %s", fallback_reason)
    if mode == "rules":
        _run_rules(ctx, planned, "rules")
    else:
        missing = policy.missing_mandatory(ctx, mandatory)
        if missing:
            _run_rules(ctx, missing, "safety_floor")

    if not summary:
        done = [e for e in ctx.trace if e["status"] == "executed"]
        summary = "; ".join(f"{e['tool']}: {e['reason']}" for e in done) or "No action needed."
    run_doc = {
        "mode": mode,
        "model_id": settings.agent.bedrock_model_id or None,
        "fallback_reason": fallback_reason,
        "facts": ctx.facts(),
        "steps": ctx.trace,
        "summary": summary,
        "action_ids": [a["id"] for a in ctx.actions],
        "elapsed_ms": round(1000 * (time.perf_counter() - t0)),
    }
    log.info(json.dumps({"event": "agent_run", "patient_id": ctx.patient["id"], "trigger": ctx.event, "mode": mode,
                         "tools": [(e["tool"], e["status"]) for e in ctx.trace]}))
    return run_doc


def run_tick(store, patient: dict, now) -> dict | None:
    """Scheduler tick: mark overdue/missed slots, then let the agent act."""
    meds = {m["id"]: m for m in store.find("medications", patient["id"], status="active")}
    overdue, missed = [], []
    remind_after = timedelta(minutes=settings.agent.reminder_after_min)
    alert_after = timedelta(minutes=settings.agent.caregiver_after_min)
    for med in meds.values():
        name = f"{med['name']} {med.get('strength', '')}".strip()
        for s in domain.day_slots(store, patient, med, now):
            rec = s["record"]
            if rec["status"] not in ("pending", "partial"):
                continue
            late = now - s["at"]
            mins = int(late.total_seconds() // 60)
            if late >= alert_after and not rec.get("caregiver_alerted_at"):
                rec["status"] = "missed"
                rec["caregiver_alerted_at"] = domain.iso(now)
                store.put("doses", rec)
                missed.append({"medication_id": med["id"], "med": name, "time": s["time"], "date": s["date"], "minutes_late": mins})
            elif remind_after <= late < alert_after and not rec.get("reminded_at"):
                rec["reminded_at"] = domain.iso(now)
                store.put("doses", rec)
                overdue.append({"medication_id": med["id"], "med": name, "time": s["time"], "date": s["date"], "minutes_late": mins})

    ctx = AgentContext(store=store, patient=patient, medications=meds, now=now, event="schedule_tick",
                       tick_findings={"overdue": overdue, "missed": missed})
    if not overdue and not missed and not rules._refill_steps(ctx, list(meds)) and not rules._expiry_steps(ctx, list(meds)):
        return None
    run_doc = run(ctx)
    run_doc.update(id=new_id("run"), patient_id=patient["id"], trigger="schedule_tick", created_at=domain.iso(now))
    store.put("runs", run_doc)
    return run_doc
