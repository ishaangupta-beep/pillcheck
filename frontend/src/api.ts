// Typed client for the PillCheck API. Same-origin in production; proxied by Vite in development.

export type SlotStatus = "pending" | "taken" | "late" | "partial" | "missed";

export interface Medication {
  id: string;
  patient_id: string;
  name: string;
  strength: string;
  instructions: string;
  schedule: { time: string; pills: number }[];
  active_reference_id: string | null;
  pending_reference_id?: string | null;
  expiry?: Expiry | null;
}

/** Expiry printed on the strip: read from the setup photo or typed by the caregiver. */
export interface Expiry { date?: string; source?: "photo" | "caregiver"; confidence?: string; status?: "expired" | "expiring_soon" | "ok" | null }

export interface Patient {
  id: string;
  name: string;
  language: "en" | "hi";
  timezone: string;
  demo?: boolean;
  caregiver: { name?: string; email?: string; phone?: string };
}

export interface Slot { time: string; slot_at: string; pills: number; status: SlotStatus; pills_taken: number }
export interface Supply { pills_left: number | null; pills_per_day?: number; days_left: number | null }

export interface Today {
  patient: Patient;
  now: string;
  local_date: string;
  medications: { medication: Medication; slots: Slot[]; supply: Supply; has_strip: boolean; expiry: Expiry }[];
  pending_actions: Action[];
}

export interface PocketReading { id: number; state: "intact" | "popped" | "uncertain" | "unknown"; score: number; confidence: number; reason: string }

export interface AgentStep {
  source: "rules" | "bedrock" | "safety_floor";
  tool: string;
  input: Record<string, unknown>;
  reason: string;
  status: "executed" | "blocked" | "error";
  result: Record<string, unknown>;
}

export interface Check {
  id: string;
  patient_id: string;
  created_at: string;
  status: "ok" | "retake" | "duplicate" | "unrecognised" | "wrong_medication";
  medication_id: string | null;
  expected_medication_id: string | null;
  evidence_key: string | null;
  perception: {
    status: string;
    guidance: string[];
    quality?: { issues: string[]; blur_score: number };
    identity?: { matched_medication_id: string | null; appearance: number | null; candidates: { medication_id: string; inliers: number; appearance: number | null }[] };
    registration?: { inliers: number; matches: number; reproj_rmse_px: number; confidence: number; engine: string } | null;
    pockets?: PocketReading[];
    counts?: { intact: number; popped: number; uncertain: number; unknown: number; total: number };
    newly_popped?: number[];
    duplicate?: { check_id: string } | null;
    timings_ms?: Record<string, number>;
    engine?: string;
  };
  agent_run: { mode: "rules" | "bedrock"; summary: string; steps: AgentStep[]; fallback_reason: string | null; elapsed_ms: number };
}

export interface Action {
  id: string;
  type: string;
  status: "done" | "pending_approval" | "approved" | "rejected";
  payload: Record<string, unknown> & { message?: string; question?: string; kind?: string; severity?: string; reason?: string };
  medication_id: string | null;
  created_at: string;
  source: { event: string; check_id: string | null };
}

export interface Notification {
  id: string;
  to: "patient" | "caregiver";
  kind: string;
  severity: string;
  message: string;
  created_at: string;
  related: { check_id?: string; action_id?: string };
}

export interface Sample { id: string; file: string; medication: string | null; title: string; hint: string }
export interface Session { patient: Patient; medications: (Medication & { demo_key?: string })[]; samples: Sample[] }

export interface Reference {
  id: string;
  medication_id: string;
  pockets: { id: number; cx: number; cy: number; rx: number; ry: number; angle: number; initially_empty: boolean }[];
  pocket_confidence: number;
  image_size: [number, number];
  quality: { issues: string[] };
  confirmed: boolean;
  expiry_read?: { date: string | null; confidence: "high" | "medium" | "low" | "none"; raw: string } | null;
}

const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

function token(): string | null {
  try { return localStorage.getItem("pillcheck.token"); } catch { return null; }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  const t = token();
  if (t) headers.Authorization = `Bearer ${t}`;
  let payload: BodyInit | undefined;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(BASE + path, { method, headers, body: payload });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export const imageUrl = (path: string) => {
  const t = token();
  return BASE + path + (t ? `${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(t)}` : "");
};

export const api = {
  health: () => call<{ ok: boolean; vision_engine: string; agent: string; storage: string }>("GET", "/health"),
  startDemo: (language: "en" | "hi") => call<Session>("POST", "/demo/session", { language }),
  samples: () => call<Sample[]>("GET", "/demo/samples"),
  today: (pid: string, at?: string | null) => call<Today>("GET", `/patients/${pid}/today${at ? `?at=${encodeURIComponent(at)}` : ""}`),
  medications: (pid: string) => call<Medication[]>("GET", `/patients/${pid}/medications`),
  addMedication: (pid: string, body: { name: string; strength: string; instructions: string; schedule: { time: string; pills: number }[] }) =>
    call<Medication>("POST", `/patients/${pid}/medications`, body),
  uploadReference: (mid: string, file: Blob) => { const f = new FormData(); f.append("file", file, "reference.jpg"); return call<Reference>("POST", `/medications/${mid}/reference`, f); },
  confirmReference: (rid: string, pockets?: Reference["pockets"], expiry?: string) =>
    call<Reference>("PUT", `/references/${rid}/confirm`, { pockets: pockets ?? null, ...(expiry !== undefined ? { expiry } : {}) }),
  setExpiry: (mid: string, expiry: string) => call<Medication>("PUT", `/medications/${mid}/expiry`, { expiry }),
  submitCheck: (pid: string, opts: { file?: Blob; sampleId?: string; medicationId?: string; at?: string | null }) => {
    const f = new FormData();
    if (opts.file) f.append("file", opts.file, "photo.jpg");
    if (opts.sampleId) f.append("sample_id", opts.sampleId);
    if (opts.medicationId) f.append("medication_id", opts.medicationId);
    if (opts.at) f.append("at", opts.at);
    return call<Check>("POST", `/patients/${pid}/checks`, f);
  },
  check: (cid: string) => call<Check>("GET", `/checks/${cid}`),
  timeline: (pid: string) => call<(Record<string, unknown> & { entry: string; at: string })[]>("GET", `/patients/${pid}/timeline?limit=60`),
  notifications: (pid: string, to?: "patient" | "caregiver") => call<Notification[]>("GET", `/patients/${pid}/notifications${to ? `?to=${to}` : ""}`),
  actions: (pid: string, status?: string) => call<Action[]>("GET", `/patients/${pid}/actions${status ? `?status=${status}` : ""}`),
  decide: (aid: string, approve: boolean, at?: string | null) => call<Action>("POST", `/actions/${aid}/decision${at ? `?at=${encodeURIComponent(at)}` : ""}`, { approve }),
  tick: (pid: string, at?: string | null) => call<{ run: { steps: AgentStep[]; summary: string } | null }>("POST", `/patients/${pid}/tick`, { at: at ?? null }),
};

/** Downscale a photo before upload: phone photos are 3-12 MB; the API accepts 4 MB and needs ~1600 px. */
export async function compressImage(file: Blob, maxSide = 1600, quality = 0.88): Promise<Blob> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" }).catch(() => null);
  if (!bitmap) return file;
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b ?? file), "image/jpeg", quality));
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2027-12" -> "Dec 2027" */
export const fmtExpiry = (d?: string | null) => (d ? `${MONTHS[Number(d.slice(5, 7)) - 1]} ${d.slice(0, 4)}` : "");
