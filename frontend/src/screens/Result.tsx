import { useEffect, useState } from "react";
import { api, imageUrl, type Check, type Medication } from "../api";
import { Evidence, Icon, Skeleton } from "../components/ui";
import { clockOf, strings } from "../i18n";
import { useApp } from "../state";

type Tone = "ok" | "warn" | "bad";

export default function Result({ go, checkId }: { go: (to: string) => void; checkId: string }) {
  const { session, lang, notify } = useApp();
  const s = strings(lang);
  const [check, setCheck] = useState<Check | null>(null);
  const [meds, setMeds] = useState<Medication[]>([]);

  useEffect(() => {
    api.check(checkId).then(setCheck).catch((e) => notify(e.message));
    if (session) api.medications(session.patientId).then(setMeds).catch(() => undefined);
  }, [checkId, session, notify]);

  if (!check) return <main className="screen"><Skeleton height={84} /><Skeleton height={260} /></main>;

  const steps = check.agent_run?.steps ?? [];
  const done = steps.filter((st) => st.status === "executed");
  const record = done.find((st) => st.tool === "record_dose");
  const recorded = (record?.result.recorded as { time: string; status: string }[] | undefined) ?? [];
  const extra = Number(record?.result.extra_pills ?? 0);
  const msgStep = [...done].reverse().find((st) => st.tool === "send_patient_message" || st.tool === "request_retake");
  const message = (msgStep?.input.message as string | undefined) ?? check.perception.guidance?.[0];
  const med = meds.find((m) => m.id === check.medication_id);
  const medName = med ? `${med.name} ${med.strength}` : "";

  let tone: Tone = "ok", title = s.recorded, icon = "check";
  if (check.status === "duplicate" || check.status === "retake") { tone = "warn"; title = s.retake; icon = "camera"; }
  else if (check.status === "unrecognised") { tone = "warn"; title = s.notRecognised; icon = "help"; }
  else if (extra > 0) { tone = "bad"; title = s.extra; icon = "alert"; }
  else if (check.status === "wrong_medication") { tone = "warn"; title = s.wrongStrip; icon = "alert"; }
  else if (recorded.length) { title = recorded.some((r) => r.status === "late") ? s.recordedLate : s.recorded; }
  else { tone = "warn"; title = s.noneNew; icon = "clock"; }

  const subtitle = [
    medName && recorded.length ? `${medName} · ${recorded.map((r) => r.time).join(", ")}` : medName,
    s.checkedAt(clockOf(check.created_at)),
  ].filter(Boolean).join("\n");
  const retry = tone === "warn" && (check.status === "duplicate" || check.status === "retake" || check.status === "unrecognised");

  return (
    <main className="screen center" style={{ gap: 16 }}>
      <div className={`badge ${tone}`}><Icon name={icon} size={44} /></div>
      <h1 className="t-display" style={{ textAlign: "center" }}>{title}</h1>
      <p className="t-body muted" style={{ whiteSpace: "pre-line", textAlign: "center" }}>{subtitle}</p>

      {check.evidence_key && (
        <div style={{ width: "100%" }}>
          <Evidence src={imageUrl(`/checks/${check.id}/evidence.jpg`)} legend={s.legend} />
        </div>
      )}

      {message && (
        <section className={`card ${tone === "bad" ? "alert" : "soft"}`} style={{ width: "100%", textAlign: "left" }}>
          <p className="t-label brand">{s.from}</p>
          <p className="t-body">{message}</p>
        </section>
      )}

      <div className="grow" />
      {retry && (
        <button className="btn secondary" onClick={() => go(`/capture${check.expected_medication_id ? `?med=${check.expected_medication_id}` : ""}`)}>
          <Icon name="camera" size={22} /> {s.tryAgain}
        </button>
      )}
      <button className="btn primary" onClick={() => go("/today")}>{s.done}</button>
    </main>
  );
}
