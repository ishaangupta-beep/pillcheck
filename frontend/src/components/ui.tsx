import type { ReactNode } from "react";
import { fmtTime } from "../i18n";

// Icons: Lucide paths (ISC licence), same set as the Figma file.
const PATHS: Record<string, ReactNode> = {
  camera: <><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" /><circle cx="12" cy="13" r="3" /></>,
  calendar: <><rect x="3" y="4" width="18" height="18" rx="2" /><path d="M16 2v4" /><path d="M8 2v4" /><path d="M3 10h18" /></>,
  check: <path d="M20 6 9 17l-5-5" />,
  x: <><path d="M18 6 6 18" /><path d="m6 6 12 12" /></>,
  shield: <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" />,
  image: <><rect width="18" height="18" x="3" y="3" rx="2" /><circle cx="9" cy="9" r="2" /><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21" /></>,
  sparkle: <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" />,
  alert: <><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" /><path d="M12 9v4" /><path d="M12 17h.01" /></>,
  back: <path d="m15 18-6-6 6-6" />,
  bell: <><path d="M10.27 21a2 2 0 0 0 3.46 0" /><path d="M3.26 15.33A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.67C19.41 13.96 18 12.5 18 8A6 6 0 0 0 6 8c0 4.5-1.41 5.96-2.74 7.33" /></>,
  message: <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />,
  ban: <><circle cx="12" cy="12" r="10" /><path d="m4.9 4.9 14.2 14.2" /></>,
  refresh: <><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M8 16H3v5" /></>,
  plus: <><path d="M5 12h14" /><path d="M12 5v14" /></>,
  clock: <><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></>,
  pill: <><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z" /><path d="m8.5 8.5 7 7" /></>,
  help: <><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><path d="M12 17h.01" /></>,
};

export function Icon({ name, size = 24, color = "currentColor", strokeWidth = 2 }: { name: keyof typeof PATHS | string; size?: number; color?: string; strokeWidth?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{PATHS[name]}</svg>
  );
}

export type ChipKind = "taken" | "due" | "missed" | "later" | "review";
export function Chip({ kind, children }: { kind: ChipKind; children: ReactNode }) {
  return <span className={`chip ${kind}`}>{children}</span>;
}

export function DoseRow({ time, name, detail, kind, label }: { time: string; name: string; detail: string; kind: ChipKind; label: string }) {
  const t = fmtTime(time);
  return (
    <div className="dose">
      <div className="time"><b>{t.time}</b><span>{t.ampm}</span></div>
      <div className="med"><p className="t-strong">{name}</p><p className="t-caption muted">{detail}</p></div>
      <Chip kind={kind}>{label}</Chip>
    </div>
  );
}

export function Spinner({ dark = false }: { dark?: boolean }) {
  return <span className={`spinner ${dark ? "dark" : ""}`} role="status" aria-label="Loading" />;
}

export function Skeleton({ height }: { height: number }) {
  return <div className="skeleton" style={{ height }} />;
}

export function Evidence({ src, legend }: { src: string; legend: string[] }) {
  return (
    <div className="card center">
      <img className="evidence" src={src} alt="Your strip today next to the full strip, with each pocket marked" />
      <div className="evidence-caps t-caption muted"><span>Full strip</span><span>This photo</span></div>
      <div className="legend">
        <span><i style={{ background: "#3CB45A" }} />{legend[0]}</span>
        <span><i style={{ background: "#DC3C3C" }} />{legend[1]}</span>
        <span><i style={{ background: "#FF8C00" }} />{legend[2]}</span>
      </div>
    </div>
  );
}

const TOOL_LABEL: Record<string, string> = {
  record_dose: "Dose recorded",
  send_patient_message: "Patient messaged",
  request_retake: "Asked for a new photo",
  alert_caregiver: "Caregiver alerted",
  request_human_review: "Sent for review",
  draft_refill_request: "Refill drafted",
};

/** Turn an agent tool name into words a caregiver reads. */
export const toolLabel = (t: string) => TOOL_LABEL[t] ?? t.replace(/_/g, " ");

/** "record_dose: 1 tablet newly removed; ..." -> "Dose recorded: 1 tablet newly removed" */
export function humanSummary(summary: string): string {
  const first = summary.split("; ")[0];
  const m = first.match(/^([a-z_]+): (.*)$/);
  return m ? `${toolLabel(m[1])}: ${m[2]}` : first;
}

/** Dose-slot status as the patient sees it, from the stored status and the clock. */
export function slotKind(status: string, slotAt: string, nowIso: string): ChipKind {
  if (status === "taken" || status === "late") return "taken";
  if (status === "missed") return "missed";
  const diffMin = (new Date(nowIso).getTime() - new Date(slotAt).getTime()) / 60000;
  if (diffMin >= -120 && diffMin <= 360) return "due";
  if (diffMin > 360) return "missed";
  return "later";
}
