import { useState } from "react";
import { api } from "../api";
import { Icon, Spinner } from "../components/ui";
import { useApp } from "../state";

export default function Landing({ go }: { go: (to: string) => void }) {
  const { session, setSession, setLang, notify, setSimTime } = useApp();
  const [busy, setBusy] = useState<"en" | "hi" | null>(null);

  async function start(language: "en" | "hi") {
    setBusy(language);
    try {
      const s = await api.startDemo(language);
      setSession({ patientId: s.patient.id, patientName: s.patient.name.replace(" (demo)", ""), demo: true });
      setLang(language);
      setSimTime("09:10");
      go("/today");
    } catch (e) {
      notify((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="screen">
      <div className="row" style={{ gap: 12 }}>
        <div className="badge ok" style={{ width: 52, height: 52 }}><Icon name="pill" size={26} /></div>
        <p className="t-label brand">OpenCV AI Competition 2026</p>
      </div>
      <h1 className="t-display">Know Mum took her medicine, without asking.</h1>
      <p className="t-body muted">
        After each dose, she photographs her medicine strip. PillCheck uses OpenCV&nbsp;5 to see which pocket was emptied,
        records the dose, and tells you only when something needs attention: a missed dose, a double dose or the wrong medicine.
      </p>

      <div className="card">
        <p className="t-heading">Try the live demo</p>
        <p className="t-caption muted">
          You get a sandbox patient with two medicines and sample strip photos, plus a demo clock to jump through the day.
          Nothing is sent to anyone.
        </p>
        <button className="btn primary" disabled={!!busy} onClick={() => start("en")}>
          {busy === "en" ? <Spinner /> : null} Start demo in English
        </button>
        <button className="btn secondary" disabled={!!busy} onClick={() => start("hi")} lang="hi">
          {busy === "hi" ? <Spinner dark /> : null} हिंदी में डेमो शुरू करें
        </button>
        {session && (
          <button className="btn ghost small" onClick={() => go("/today")}>Continue where you left off</button>
        )}
      </div>

      <div className="stack">
        {[
          ["camera", "One photo after each dose", "Big buttons, Hindi or English, no typing."],
          ["sparkle", "OpenCV 5 checks every pocket", "ALIKED + LightGlue align the photo to the full strip; each pocket is compared."],
          ["bell", "An agent decides what to do", "Record, remind, ask for a retake, alert you, or ask you to decide."],
        ].map(([ic, title, sub]) => (
          <div className="row" key={title} style={{ alignItems: "flex-start", gap: 14 }}>
            <div className="badge ok" style={{ width: 40, height: 40, flex: "none" }}><Icon name={ic} size={20} /></div>
            <div><p className="t-strong">{title}</p><p className="t-caption muted">{sub}</p></div>
          </div>
        ))}
      </div>
      <p className="t-caption muted" style={{ marginTop: "auto" }}>
        PillCheck supports adherence. It never gives medical or dosing advice.
      </p>
    </main>
  );
}
