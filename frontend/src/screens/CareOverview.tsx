import { useCallback, useEffect, useState } from "react";
import { api, fmtExpiry, type Action, type Notification, type Today } from "../api";
import { Chip, Icon, Skeleton, Spinner, humanSummary, slotKind, toolLabel } from "../components/ui";
import { clockOf, fmtTime } from "../i18n";
import { useApp } from "../state";

type Entry = Record<string, unknown> & { entry: string; at: string };

export default function CareOverview({ go }: { go: (to: string) => void }) {
  const { session, at, simTime, notify } = useApp();
  const [today, setToday] = useState<Today | null>(null);
  const [alerts, setAlerts] = useState<Notification[]>([]);
  const [pending, setPending] = useState<Action[]>([]);
  const [timeline, setTimeline] = useState<Entry[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!session) return;
    try {
      const [t, n, a, tl] = await Promise.all([
        api.today(session.patientId, at()), api.notifications(session.patientId, "caregiver"),
        api.actions(session.patientId, "pending_approval"), api.timeline(session.patientId),
      ]);
      setToday(t); setPending(a); setTimeline(tl);
      setAlerts(n.filter((x) => (x.severity === "urgent" || x.severity === "warning") && !["review_needed", "refill"].includes(x.kind)));
    } catch (e) { notify((e as Error).message); }
  }, [session, at, notify]);
  useEffect(() => { load(); }, [load, simTime]);

  async function decide(a: Action, approve: boolean) {
    setBusy(a.id);
    try { await api.decide(a.id, approve, at()); notify(approve ? "Approved." : "Dismissed."); await load(); }
    catch (e) { notify((e as Error).message); } finally { setBusy(null); }
  }

  async function runScheduler() {
    if (!session) return;
    setBusy("tick");
    try {
      const r = await api.tick(session.patientId, at());
      const did = r.run?.steps.filter((s) => s.status === "executed").map((s) => toolLabel(s.tool).toLowerCase()) ?? [];
      notify(did.length ? `Scheduler: ${did.join(", ")}` : "Scheduler: nothing needed right now.");
      await load();
    } catch (e) { notify((e as Error).message); } finally { setBusy(null); }
  }

  if (!session) return null;
  if (!today) return <main className="screen"><Skeleton height={90} /><Skeleton height={160} /><Skeleton height={220} /></main>;

  const rows = today.medications.flatMap((m) => m.slots.map((sl) => ({ med: m.medication, sl, kind: slotKind(sl.status, sl.slot_at, today.now) })))
    .sort((a, b) => a.sl.slot_at.localeCompare(b.sl.slot_at));
  const takenCount = rows.filter((r) => r.kind === "taken").length;
  const latest = alerts[0];
  const lowSupply = today.medications.filter((m) => m.supply.days_left != null && m.supply.days_left <= 3);

  return (
    <main className="screen">
      <header className="stack" style={{ gap: 4 }}>
        <p className="t-label brand">Caregiver</p>
        <h1 className="t-display">{session.patientName}'s medicines</h1>
        <p className="t-caption muted">Everything PillCheck saw and did, newest first.</p>
      </header>

      {latest && (
        <section className={`card ${latest.severity === "urgent" ? "alert" : "warn"}`} role="alert">
          <div className="row">
            <Icon name="alert" size={20} color={latest.severity === "urgent" ? "var(--missed)" : "var(--due)"} />
            <p className="t-heading" style={{ color: latest.severity === "urgent" ? "var(--missed)" : "var(--due)" }}>
              {latest.kind === "extra_removed" ? "Possible double dose" : latest.kind === "missed" ? "Missed dose" :
                latest.kind === "wrong_medication" ? "Wrong medicine taken" : latest.kind === "photo_reuse" ? "Old photo re-sent" :
                latest.kind === "expired" ? "Strip has expired" : latest.kind === "expiring_soon" ? "Strip expires soon" : "Needs attention"}
            </p>
          </div>
          <p className="t-body">{latest.message}</p>
          <p className="t-caption muted">{clockOf(latest.created_at)}</p>
          {latest.related?.check_id && (
            <button className="btn secondary small" onClick={() => go(`/care/check/${latest.related.check_id}`)}>View evidence</button>
          )}
        </section>
      )}

      <section className="card">
        <div className="row between"><p className="t-heading">Today</p><p className="t-strong muted">{takenCount} of {rows.length} doses</p></div>
        <div className="progress">{rows.map((r) => <span key={`${r.med.id}${r.sl.time}`} className={r.kind === "taken" ? "taken" : r.kind === "missed" ? "missed" : ""} />)}</div>
        {rows.map((r) => {
          const t = fmtTime(r.sl.time);
          return (
            <div className="row between" key={`${r.med.id}-${r.sl.time}`}>
              <p className="t-body">{t.time} {t.ampm} · {r.med.name}</p>
              <Chip kind={r.kind}>{r.sl.status === "late" ? "Taken late" : r.kind[0].toUpperCase() + r.kind.slice(1)}</Chip>
            </div>
          );
        })}
        {today.medications.filter((m) => m.expiry?.status === "expired" || m.expiry?.status === "expiring_soon").map((m) => (
          <p className="t-caption" key={`exp-${m.medication.id}`} style={{ color: m.expiry.status === "expired" ? "var(--missed)" : "var(--due)" }}>
            {m.medication.name}: strip {m.expiry.status === "expired" ? "expired" : "expires"} {fmtExpiry(m.expiry.date)}
          </p>
        ))}
        {lowSupply.map((m) => (
          <p className="t-caption muted" key={m.medication.id}>{m.medication.name}: {m.supply.pills_left} tablets left (~{m.supply.days_left} days)</p>
        ))}
      </section>

      {pending.length > 0 && <p className="t-label muted">Needs your decision</p>}
      {pending.map((a) => (
        <section className="card" key={a.id}>
          <div className="row">
            <Chip kind="review">{a.type === "draft_refill_request" ? "Refill" : a.payload.kind === "new_strip" ? "New strip?" : "Confirm dose"}</Chip>
            <p className="t-caption muted">Drafted by PillCheck · {clockOf(a.created_at)}</p>
          </div>
          <p className="t-body">{a.payload.message ?? a.payload.question}</p>
          {a.source?.check_id && <button className="link" onClick={() => go(`/care/check/${a.source.check_id}`)}>See the photo</button>}
          <div className="row">
            <button className="btn primary small" disabled={busy === a.id} onClick={() => decide(a, true)}>{busy === a.id ? <Spinner /> : "Approve"}</button>
            <button className="btn ghost small" disabled={busy === a.id} onClick={() => decide(a, false)}>Dismiss</button>
          </div>
        </section>
      ))}

      <div className="row between">
        <p className="t-label muted">Activity</p>
        <button className="link" disabled={busy === "tick"} onClick={runScheduler}>
          {busy === "tick" ? <Spinner dark /> : <Icon name="refresh" size={16} />} Run reminder check
        </button>
      </div>
      <section className="card" style={{ gap: 0, paddingTop: 6, paddingBottom: 6 }}>
        {timeline.filter((e) => e.entry === "check" || (e.entry === "notification" && e.to === "caregiver")).slice(0, 15).map((e) => (
          <button className="activity" key={`${e.entry}-${e.id as string}`} disabled={e.entry !== "check"}
            onClick={() => e.entry === "check" && go(`/care/check/${e.id as string}`)}>
            <time>{clockOf(e.at)}</time>
            <span className="t-caption">
              {e.entry === "check"
                ? <><b>Photo: {String(e.status).replace("_", " ")}</b>{e.agent_summary ? ` · ${humanSummary(String(e.agent_summary))}` : ""}</>
                : <>Alert sent: {String(e.message)}</>}
            </span>
          </button>
        ))}
        {timeline.length === 0 && <p className="t-caption muted" style={{ padding: "10px 0" }}>No activity yet. Photos from the patient will appear here.</p>}
      </section>
      <button className="btn secondary" onClick={() => go("/care/setup")}><Icon name="pill" size={20} /> Medicines & strips</button>
    </main>
  );
}
