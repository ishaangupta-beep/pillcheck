"""Outbound messages to the patient and caregiver.

Every message is stored (the app shows it in the timeline) and logged. When
PILLCHECK_SNS_TOPIC_ARN is set, caregiver alerts are also published to SNS,
which fans out to SMS / email subscriptions.
"""
from __future__ import annotations

import json
import logging

from ..config import settings
from ..storage import Store, new_id

log = logging.getLogger("pillcheck.notify")


def send(store: Store, patient_id: str, to: str, message: str, *, severity: str = "info",
         kind: str = "message", related: dict | None = None, at: str | None = None) -> dict:
    """to: 'patient' | 'caregiver'."""
    note = {
        "id": new_id("ntf"),
        "patient_id": patient_id,
        "to": to,
        "kind": kind,
        "severity": severity,
        "message": message,
        "related": related or {},
        "channels": ["app"],
    }
    if at:
        note["created_at"] = at
    topic = settings.storage.sns_topic_arn
    # Demo sandboxes never send real email/SMS: their alerts stay in the app.
    is_demo = bool((store.get("patients", patient_id) or {}).get("demo"))
    if topic and to == "caregiver" and not is_demo:
        try:
            import boto3
            boto3.client("sns", region_name=settings.agent.aws_region).publish(
                TopicArn=topic, Subject=f"PillCheck: {kind.replace('_', ' ')}"[:100], Message=message,
                MessageAttributes={"severity": {"DataType": "String", "StringValue": severity}})
            note["channels"].append("sns")
        except Exception as exc:  # delivery failure must never break the agent loop
            log.warning("SNS publish failed: %s", exc)
            note["delivery_error"] = str(exc)
    store.put("notifications", note)
    log.info(json.dumps({"event": "notification", "to": to, "kind": kind, "severity": severity, "patient_id": patient_id}))
    return note
