import { useEffect, useState } from "react";
import { api, imageUrl, type AgentStep, type Check, type Medication } from "../api";
import { Chip, Evidence, Icon, Skeleton } from "../components/ui";
import { clockOf } from "../i18n";
import { useApp } from "../state";

const TOOL: Record<string, { icon: string; color: string; label: (s: AgentStep) => string }> = {
  record_dose: { icon: "check", color: "var(--taken)", label: (s) => {
    const rec = (s.result.recorded as { time: string }[] | undefined) ?? [];
    return rec.length ? `Recorded the ${rec.map((r) => r.time).join(", ")} dose` : "Recorded the removal";
  } },
  send_patient_message: { icon: "message", color: "var(--brand)", label: () => "Messaged the patient" },
  request_retake: { icon: "camera", color: "var(--due)", label: () => "Asked for a new photo" },
  alert_caregiver: { icon: "bell", color: "var(--missed)", label: (s) => `Alerted you (${String(s.input.severity)})` },
  request_human_review: { icon: "help", color: "var(--brand)", label: () => "Asked you to decide" },
  draft_refill_request: { icon: "pill", color: "var(--brand)", label: () => "Drafted a refill request" },
};

const SOURCE: Record<string, [string, string]> = { bedrock: ["Agent · Bedrock", ""], rules: ["Agent · rules", ""], safety_floor: ["Safety floor", "floor"] };

export default function CheckDetail({ go, checkId }: { go: (to: string) => void; checkId: string }) {
  const { session, notify } = useApp();
  const [check, setCheck] = useState<Check | null>(null);
  const [meds, setMeds] = useState<Medication[]>([]);
  useEffect(() => {
    api.check(checkId).then(setCheck).catch((e) => notify(e.message));
    if (session) api.medications(session.patientId).then(setMeds).catch(() => undefined);
  }, [checkId, session, notify]);

  if (!check) return <main className="screen"><Skeleton height={60} /><Skeleton height={200} /><Skeleton height={260} /></main>;
  const p = check.perception;
  const name = (id: string | null | undefined) => { const m = meds.find((x) => x.id === id); return m ? `${m.name} ${m.strength}` : "unknown strip"; };
  const extra = check.agent_run.steps.some((s) => s.tool === "record_dose" && Number(s.result.extra_pills) > 0);
  const statusChip = extra ? <Chip kind="missed">Extra tablet</Chip>
    : check.status === "ok" ? <Chip kind="taken">Read OK</Chip>
    : check.status === "wrong_medication" ? <Chip kind="due">Wrong strip</Chip>
    : <Chip kind="due">{check.status.replace("_", " ")}</Chip>;
  const reg = p.registration;
  const facts: [string, string][] = [
    ["Strip", p.identity?.matched_medication_id ? `${name(p.identity.matched_medication_id)} (match ${p.identity.appearance?.toFixed(2)})` : "not recognised"],
    ...(check.expected_medication_id && check.expected_medication_id !== p.identity?.matched_medication_id ? [["Expected", name(check.expected_medication_id)] as [string, string]] : []),
    ...(reg ? [["Alignment", `${reg.inliers} matched points · ${reg.reproj_rmse_px} px error`] as [string, string]] : []),
    ...(p.counts ? [["Pockets", `${p.counts.popped} empty · ${p.counts.intact} full · ${p.counts.uncertain + p.counts.unknown} unclear`] as [string, string]] : []),
    ...(p.newly_popped ? [["New since last check", p.newly_popped.length ? `Pocket ${p.newly_popped.map((i) => i + 1).join(", ")}` : "none"] as [string, string]] : []),
    ...(p.quality?.issues?.length ? [["Photo quality", p.quality.issues.join(", ").replace(/_/g, " ")] as [string, string]] : []),
    ...(p.duplicate ? [["Duplicate of", "an earlier photo"] as [string, string]] : []),
    ["Vision engine", `OpenCV 5 · ${p.engine ?? reg?.engine ?? ""}`],
    ...(p.timings_ms ? [["Time", `${Object.values(p.timings_ms).reduce((a, b) => a + b, 0)} ms`] as [string, string]] : []),
  ];

  return (
    <main className="screen" style={{ gap: 14 }}>
      <button className="link" onClick={() => go("/care")}><Icon name="back" size={20} /> Activity</button>
      <div className="row between"><h1 className="t-title">{clockOf(check.created_at)} check</h1>{statusChip}</div>

      {check.evidence_key
        ? <Evidence src={imageUrl(`/checks/${check.id}/evidence.jpg`)} legend={["Still in strip", "Taken before", "Taken in this photo"]} />
        : <img className="evidence" src={imageUrl(`/checks/${check.id}/photo.jpg`)} alt="The photo as sent" style={{ maxHeight: 260, objectFit: "contain" }} />}

      <p className="t-label muted">What the camera saw</p>
      <section className="card" style={{ gap: 8 }}>
        {facts.map(([k, v]) => <div className="kv" key={k}><span>{k}</span><span>{v}</span></div>)}
      </section>

      <p className="t-label muted">What the agent did</p>
      <section className="card" style={{ gap: 14 }}>
        {check.agent_run.steps.length === 0 && <p className="t-caption muted">No action was needed.</p>}
        {check.agent_run.steps.map((s, i) => {
          const t = TOOL[s.tool] ?? { icon: "sparkle", color: "var(--brand)", label: () => s.tool };
          const blocked = s.status !== "executed";
          const [srcLabel, srcClass] = blocked ? ["Policy", "policy"] : SOURCE[s.source] ?? [s.source, ""];
          const text = (s.input.message ?? s.input.question) as string | undefined;
          return (
            <div className="step" key={i}>
              <div className="ic"><Icon name={blocked ? "ban" : t.icon} size={18} color={blocked ? "var(--later)" : t.color} /></div>
              <div className="stack grow" style={{ gap: 2 }}>
                <div className="row between" style={{ alignItems: "flex-start" }}>
                  <p className="t-strong">{blocked ? `Blocked: ${t.label(s).toLowerCase()}` : t.label(s)}</p>
                  <span className={`tag ${srcClass}`}>{srcLabel}</span>
                </div>
                <p className="t-caption muted">
                  {blocked ? String((s.result as { blocked_by_policy?: string }).blocked_by_policy ?? "") : s.reason}
                </p>
                {text && <p className="t-caption" style={{ background: "var(--bg-sunken)", padding: "8px 10px", borderRadius: 10 }}>“{text}”</p>}
              </div>
            </div>
          );
        })}
        <p className="t-caption muted" style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          Planner: {check.agent_run.mode === "bedrock" ? "Amazon Bedrock" : "built-in rules"}
          {check.agent_run.fallback_reason ? ` (${check.agent_run.fallback_reason})` : ""} · every action passed the safety policy first.
        </p>
      </section>
    </main>
  );
}
