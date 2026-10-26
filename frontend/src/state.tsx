import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Lang } from "./i18n";

// ---------------------------------------------------------------- hash router
export function useRoute(): [string, (to: string) => void] {
  const read = () => window.location.hash.replace(/^#/, "") || "/";
  const [path, setPath] = useState(read);
  useEffect(() => {
    const on = () => { setPath(read()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((to: string) => { window.location.hash = to; }, []);
  return [path, go];
}

// ---------------------------------------------------------------- persisted session
interface SessionState { patientId: string; patientName: string; demo: boolean }

function load<T>(key: string): T | null {
  try { const v = localStorage.getItem(key); return v ? (JSON.parse(v) as T) : null; } catch { return null; }
}
function save(key: string, value: unknown) {
  try { if (value == null) localStorage.removeItem(key); else localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

interface Ctx {
  session: SessionState | null;
  setSession: (s: SessionState | null) => void;
  lang: Lang;
  setLang: (l: Lang) => void;
  /** Simulated clock for the demo (HH:MM on the patient's day), null = real time. */
  simTime: string | null;
  setSimTime: (t: string | null) => void;
  /** ISO timestamp to send as `at`, or null for real time. */
  at: (localDate?: string) => string | null;
  toast: string | null;
  notify: (msg: string) => void;
}

const AppContext = createContext<Ctx | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [session, setSessionState] = useState<SessionState | null>(() => load("pillcheck.session"));
  const [lang, setLangState] = useState<Lang>(() => load<Lang>("pillcheck.lang") ?? "en");
  const [simTime, setSimTimeState] = useState<string | null>(() => load("pillcheck.sim"));
  const [toast, setToast] = useState<string | null>(null);

  const setSession = useCallback((s: SessionState | null) => { setSessionState(s); save("pillcheck.session", s); }, []);
  const setLang = useCallback((l: Lang) => { setLangState(l); save("pillcheck.lang", l); }, []);
  const setSimTime = useCallback((t: string | null) => { setSimTimeState(t); save("pillcheck.sim", t); }, []);
  const notify = useCallback((msg: string) => { setToast(msg); window.setTimeout(() => setToast(null), 3800); }, []);
  const at = useCallback((localDate?: string) => {
    if (!simTime) return null;
    const date = localDate ?? new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Kolkata" });
    return `${date}T${simTime}:00+05:30`;
  }, [simTime]);

  const value = useMemo(() => ({ session, setSession, lang, setLang, simTime, setSimTime, at, toast, notify }),
    [session, setSession, lang, setLang, simTime, setSimTime, at, toast, notify]);
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): Ctx {
  const c = useContext(AppContext);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}
