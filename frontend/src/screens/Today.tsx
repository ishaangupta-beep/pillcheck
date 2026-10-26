import { useCallback, useEffect, useState } from "react";
import { api, type Today as TodayData } from "../api";
import { DoseRow, Icon, Skeleton, slotKind } from "../components/ui";
import { fmtTime, strings } from "../i18n";
import { useApp } from "../state";

export default function Today({ go }: { go: (to: string) => void }) {
  const { session, lang, setLang, at, simTime, notify } = useApp();
  const s = strings(lang);
  const [data, setData] = useState<TodayData | null>(null);

  const load = useCallback(async () => {
    if (!session) return;
    try { setData(await api.today(session.patientId, at())); } catch (e) { notify((e as Error).message); }
  }, [session, at, notify]);
  useEffect(() => { load(); }, [load, simTime]);

  if (!session) return null;
  if (!data) return <main className="screen"><Skeleton height={80} /><Skeleton height={220} /><Skeleton height={300} /></main>;

  const now = data.now;
  const rows = data.medications.flatMap((m) =>
    m.slots.map((sl) => ({ med: m.medication, slot: sl, kind: slotKind(sl.status, sl.slot_at, now) })),
  ).sort((a, b) => a.slot.slot_at.localeCompare(b.slot.slot_at));
  const taken = rows.filter((r) => r.kind === "taken").length;
  const next = rows.find((r) => r.kind === "due") ?? rows.find((r) => r.kind === "later");
  const hour = Number(new Date(now).toLocaleString("en-GB", { hour: "2-digit", hour12: false, timeZone: data.patient.timezone }));
  const greet = hour < 12 ? s.morning : hour < 17 ? s.afternoon : s.evening;
  const dateStr = new Date(now).toLocaleDateString(lang === "hi" ? "hi-IN" : "en-GB", { weekday: "long", day: "numeric", month: "long", timeZone: data.patient.timezone });
  const noStrip = data.medications.every((m) => !m.has_strip);

  return (
    <main className="screen">
      <header className="row between" style={{ alignItems: "flex-start" }}>
        <div className="stack" style={{ gap: 4 }}>
          <p className="t-caption muted">{dateStr}</p>
          <h1 className="t-display">{greet},<br />{session.patientName}</h1>
        </div>
        <button className="pill-toggle" onClick={() => setLang(lang === "en" ? "hi" : "en")} lang={lang === "en" ? "hi" : "en"}>
          {lang === "en" ? "हिंदी" : "English"}
        </button>
      </header>

      {noStrip ? (
        <section className="card sunken"><p className="t-heading">{s.noStrip}</p><p className="t-body muted">{s.noStripSub}</p></section>
      ) : next ? (
        <section className="hero" aria-label={s.nextDose}>
          <p className="t-label">{s.nextDose} · {next.kind === "due" ? s.now : s.laterToday}</p>
          <h2 className="t-title">{next.med.name} {next.med.strength}</h2>
          <p className="t-body">
            {fmtTime(next.slot.time).time} {fmtTime(next.slot.time).ampm} · {s.tablets(next.slot.pills)}
            {next.med.instructions ? ` · ${next.med.instructions.toLowerCase()}` : ""}
          </p>
          <div style={{ height: 10 }} />
          <button className="btn secondary" style={{ border: 0 }} onClick={() => go(`/capture?med=${next.med.id}`)}>
            <Icon name="camera" size={22} /> {s.photograph}
          </button>
          <p className="t-caption">{s.takeFirst}</p>
        </section>
      ) : (
        <section className="hero">
          <p className="t-label">{s.today}</p>
          <h2 className="t-title">{s.allDone}</h2>
          <p className="t-body">{s.allDoneSub}</p>
          <div style={{ height: 10 }} />
          <button className="btn secondary" style={{ border: 0 }} onClick={() => go("/capture")}><Icon name="camera" size={22} /> {s.photograph}</button>
        </section>
      )}

      <div className="row between">
        <p className="t-label muted">{s.today}</p>
        <p className="t-caption muted">{s.ofTaken(taken, rows.length)}</p>
      </div>
      <section className="stack" aria-label={s.today}>
        {rows.map((r) => (
          <DoseRow key={`${r.med.id}-${r.slot.time}`} time={r.slot.time} name={r.med.name}
            detail={[r.med.strength, r.med.instructions].filter(Boolean).join(" · ")}
            kind={r.kind} label={s.status[r.slot.status === "late" ? "late" : r.kind]} />
        ))}
      </section>
      <p className="row t-caption muted"><Icon name="shield" size={18} /> {s.reassure(data.patient.caregiver?.name && data.patient.caregiver.name !== "You" ? data.patient.caregiver.name : lang === "hi" ? "आपके परिजन" : "Your caregiver")}</p>
    </main>
  );
}
