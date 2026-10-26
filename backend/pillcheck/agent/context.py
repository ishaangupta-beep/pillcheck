"""Everything one agent run needs: the situation, the facts, and the trace."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .. import domain
from ..storage import Store
from .messages import fmt_time


@dataclass
class AgentContext:
    store: Store
    patient: dict
    medications: dict[str, dict]
    now: datetime
    event: str  # photo_check | schedule_tick
    check: dict | None = None
    perception: dict | None = None
    reconciliation: dict[str, dict] = field(default_factory=dict)  # medication_id -> reconcile()
    tick_findings: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    recorded: set[str] = field(default_factory=set)

    @property
    def lang(self) -> str:
        return self.patient.get("language", "en")

    def med_name(self, medication_id: str | None) -> str:
        m = self.medications.get(medication_id or "")
        if not m:
            return "medicine"
        return f"{m['name']} {m.get('strength', '')}".strip()

    @property
    def trusted(self) -> bool:
        """True when the perception is reliable enough to change the dose record."""
        p = self.perception or {}
        return p.get("status") in ("ok", "wrong_medication") and not p.get("inconsistent")

    def facts(self) -> dict:
        """Compact, numbers-already-computed view of the situation for the planner."""
        local = self.now.astimezone(domain.tz_of(self.patient))
        out: dict = {
            "event": self.event,
            "now_local": local.strftime("%Y-%m-%d %H:%M"),
            "patient": {"name": self.patient["name"], "language": self.lang},
            "medications": [],
        }
        for mid, med in self.medications.items():
            slots = domain.day_slots(self.store, self.patient, med, self.now, back_days=0)
            out["medications"].append({
                "medication_id": mid,
                "name": self.med_name(mid),
                "today": [{"time": fmt_time(s["time"]), "status": s["record"]["status"]} for s in slots],
                "supply": domain.supply(self.store, self.patient, med),
                "expiry": {"date": (med.get("expiry") or {}).get("date"),
                           "status": domain.expiry_status(self.patient, med, self.now)},
                "refill_pending": bool(self.store.find("actions", self.patient["id"], type="draft_refill_request",
                                                       status="pending_approval", medication_id=mid)),
            })
        if self.perception is not None:
            p = self.perception
            out["photo"] = {
                "status": p.get("status"),
                "expected_medication": self.med_name(p.get("identity", {}).get("expected_medication_id")) if p.get("identity", {}).get("expected_medication_id") else None,
                "recognised_medication": self.med_name(p.get("medication_id")) if p.get("medication_id") else None,
                "recognised_medication_id": p.get("medication_id"),
                "newly_removed_pills": len(p.get("newly_popped", [])),
                "pocket_counts": p.get("counts"),
                "pockets_back_to_intact": len(p.get("inconsistent", [])),
                "unclear_pockets": [x["id"] + 1 for x in p.get("pockets", []) if x["state"] in ("uncertain", "unknown")],
                "quality_issues": p.get("quality", {}).get("issues", []),
                "capture_guidance": p.get("guidance", []),
                "duplicate_of_earlier_photo": bool(p.get("duplicate")),
                "trusted_for_dose_record": self.trusted,
            }
            out["reconciliation"] = {self.med_name(k): v for k, v in self.reconciliation.items()}
            out["recent_unclear_photos"] = self.check.get("recent_unclear", 0) if self.check else 0
            out["recent_duplicate_photos"] = self.check.get("recent_duplicates", 0) if self.check else 0
        if self.tick_findings:
            out["schedule"] = self.tick_findings
        return out
