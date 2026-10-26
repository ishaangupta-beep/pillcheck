import { useEffect, useState } from "react";
import { api, type Medication } from "../api";
import { useApp } from "../state";
import { Icon, Spinner } from "./ui";

/**
 * Guided demo for judges: five one-tap steps that tell the PillCheck story,
 * each with a plain explanation of what OpenCV 5 and the agent just did.
 */
interface Step {
  key: string;
  time: string;
  title: string;
  does: string;
  look: string;
  run: (ctx: RunCtx) => Promise<string>; // returns the route to open
}
interface RunCtx { pid: string; meds: Record<string, string>; at: string; checks: Record<string, string> }

const STEPS: Step[] = [
  {
    key: "dose", time: "09:10", title: "Patient takes the 9 AM tablet",
    does: "A photo of the Telmisartan strip with one tablet removed.",
    look: "OpenCV 5 (ALIKED + LightGlue) aligns the photo to the full strip and finds pocket 1 newly empty. The agent records the 9:00 AM dose. No one is bothered.",
    run: async ({ pid, meds, at }) => `/result/${(await api.submitCheck(pid, { sampleId: "telmisartan_dose1", medicationId: meds.telmi, at })).id}`,
  },
  {
    key: "extra", time: "09:40", title: "Two more tablets are removed",
    does: "Half an hour later, two more pockets are empty.",
    look: "Only one tablet was due. The agent records them as extra and sends an urgent “possible double dose” alert to the caregiver.",
    run: async ({ pid, meds, at }) => `/result/${(await api.submitCheck(pid, { sampleId: "telmisartan_extra", medicationId: meds.telmi, at })).id}`,
  },
  {
    key: "blurry", time: "09:45", title: "A blurry photo",
    does: "The patient's hand shakes.",
    look: "The quality check rejects it before any analysis and asks for a retake in plain words. A photo it is unsure about never becomes a dose.",
    run: async ({ pid, meds, at }) => `/result/${(await api.submitCheck(pid, { sampleId: "telmisartan_blurry", medicationId: meds.telmi, at })).id}`,
  },
  {
    key: "missed", time: "12:00", title: "Noon: Metformin was never taken",
    does: "The 15-minute scheduler (EventBridge on AWS) runs the agent.",
    look: "Metformin 8:30 AM is 3.5 hours late. The agent marks it missed and alerts the caregiver.",
    run: async ({ pid, at }) => { await api.tick(pid, at); return "/care"; },
  },
  {
    key: "why", time: "12:00", title: "The caregiver opens the evidence",
    does: "The caregiver sees only what needs attention.",
    look: "The double-dose evidence: the photo aligned to the full strip, every pocket's verdict, and each agent step with its reason.",
    run: async ({ checks }) => (checks.extra ? `/care/check/${checks.extra}` : "/care"),
  },
];

export default function DemoGuide({ go }: { go: (to: string) => void }) {
  const { session, setSimTime, notify } = useApp();
  const [meds, setMeds] = useState<Record<string, string>>({});
  const [done, setDone] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const [checks, setChecks] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!session) return;
    setDone([]);
    api.medications(session.patientId).then((ms: (Medication & { demo_key?: string })[]) =>
      setMeds(Object.fromEntries(ms.filter((m) => m.demo_key).map((m) => [m.demo_key!, m.id])))).catch(() => {});
  }, [session]);

  if (!session?.demo) return null;
  const next = STEPS.find((s) => !done.includes(s.key));

  async function run(s: Step) {
    if (!session) return;
    setBusy(s.key);
    try {
      setSimTime(s.time);
      const date = new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });
      const route = await s.run({ pid: session.patientId, meds, checks, at: `${date}T${s.time}:00+05:30` });
      const m = route.match(/^\/result\/(.+)$/);
      if (m) setChecks((c) => ({ ...c, [s.key]: m[1] }));
      setDone((d) => (d.includes(s.key) ? d : [...d, s.key]));
      go(route);
      if (window.innerWidth < 1100) setOpen(false); // let the result show on phones
    } catch (e) { notify((e as Error).message); } finally { setBusy(null); }
  }

  return (
    <aside className={`tour ${open ? "open" : ""}`} aria-label="Guided demo">
      <button className="tour-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="tour-badge">Guided demo</span>
        <span className="t-strong">{done.length}/{STEPS.length} · {next ? next.title : "Done. Try your own strip in Caregiver → Medicines"}</span>
        <Icon name={open ? "x" : "sparkle"} size={18} />
      </button>
      {open && (
        <ol className="tour-steps">
          {STEPS.map((s, i) => {
            const isDone = done.includes(s.key);
            const isNext = next?.key === s.key;
            const isLast = done[done.length - 1] === s.key;
            return (
              <li key={s.key} className={`${isDone ? "done" : ""} ${isNext ? "next" : ""}`}>
                <div className="tour-num">{isDone ? <Icon name="check" size={16} /> : i + 1}</div>
                <div className="tour-body">
                  <p className="t-strong">{s.title}</p>
                  {(isNext || isLast) && <p className="t-caption muted">{s.does}</p>}
                  {(isNext || isLast) && <p className="t-caption tour-look"><b>What happens:</b> {s.look}</p>}
                  {(isNext || isDone) && (
                    <button className={`btn ${isNext ? "primary" : "secondary"} small`} disabled={!!busy || !meds.telmi} onClick={() => run(s)}>
                      {busy === s.key ? <Spinner dark={!isNext} /> : isDone ? "Show again" : `Run at ${s.time === "12:00" ? "12:00 PM" : s.time.replace(/^0/, "") + " AM"}`}
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </aside>
  );
}
