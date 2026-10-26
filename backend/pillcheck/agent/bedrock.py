"""Bedrock planner: a foundation model chooses tools via the Converse API.

The model sees the facts (perception already summarised, arithmetic already
done) and the tool list. Each tool call it makes goes through the policy
before running, and the result goes back to the model, so it can adapt: for
example, when record_dose returns extra_pills > 0, it alerts the caregiver.
"""
from __future__ import annotations

import json
import logging

from ..config import settings
from . import policy
from .context import AgentContext
from .tools import TOOL_SPECS, execute

log = logging.getLogger("pillcheck.agent")

SYSTEM_PROMPT = """You are PillCheck's care agent. An elderly patient photographs their medicine strip after taking a dose; computer vision has already analysed the photo. You decide what happens next by calling tools.

Principles:
- Act on the facts provided. Never invent pill counts, times or medicines.
- Record a dose only when photo.trusted_for_dose_record is true and tablets were newly removed.
- If the photo cannot be used (duplicate, capture problem, unrecognised strip, unclear pockets), ask for a retake with the concrete capture guidance. If unclear photos keep coming (recent_unclear_photos >= 1), ask the caregiver to review instead.
- Alert the caregiver when tablets beyond the schedule were removed (possible double dose), when a medicine that was not due was taken, or when a dose is badly overdue. Alerts must be specific and calm.
- Pockets that went from empty back to full mean the strip was probably replaced: ask the caregiver to review, do not record.
- When supply is low and no refill is pending, draft a refill request.
- Write patient messages in the patient's language (en = English, hi = Hindi), short and kind.
- You are not a doctor. Never give dosing advice: never tell anyone to take, skip, double or change a dose. Confirm, remind, inform and escalate only.
When you have done everything needed, reply with one sentence summarising what you did and why."""


def _tool_config() -> dict:
    return {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                    "inputSchema": {"json": t["input_schema"]}}} for t in TOOL_SPECS],
            "toolChoice": {"auto": {}}}


def run(ctx: AgentContext) -> str:
    """Run the Converse tool loop. Raises on any Bedrock error (caller falls back)."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=settings.agent.aws_region)
    messages = [{"role": "user", "content": [{"text": "Situation:\n" + json.dumps(ctx.facts(), ensure_ascii=False, indent=1)}]}]
    summary = ""
    for step in range(settings.agent.max_steps):
        resp = client.converse(modelId=settings.agent.bedrock_model_id, system=[{"text": SYSTEM_PROMPT}],
                               messages=messages, toolConfig=_tool_config(),
                               inferenceConfig={"maxTokens": 1024, "temperature": 0.0})
        msg = resp["output"]["message"]
        messages.append(msg)
        thought = " ".join(c["text"] for c in msg["content"] if "text" in c).strip()
        uses = [c["toolUse"] for c in msg["content"] if "toolUse" in c]
        if not uses:
            summary = thought
            break
        results = []
        for use in uses:
            name, args = use["name"], use.get("input") or {}
            ok, why = policy.validate(ctx, name, args)
            entry = {"source": "bedrock", "tool": name, "input": args, "reason": thought}
            if ok:
                try:
                    out = execute(ctx, name, args)
                    entry.update(status="executed", result=out)
                    results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": out}], "status": "success"}})
                except Exception as exc:  # keep the loop alive; the model sees the error
                    log.exception("tool %s failed", name)
                    entry.update(status="error", result={"error": str(exc)})
                    results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"text": f"error: {exc}"}], "status": "error"}})
            else:
                entry.update(status="blocked", result={"blocked_by_policy": why})
                results.append({"toolResult": {"toolUseId": use["toolUseId"],
                                               "content": [{"text": f"Blocked by safety policy: {why}"}], "status": "error"}})
            ctx.trace.append(entry)
            thought = ""  # attach the reasoning text to the first call of the turn only
        messages.append({"role": "user", "content": results})
    return summary
