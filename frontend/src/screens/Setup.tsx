import { useCallback, useEffect, useRef, useState } from "react";
import { api, compressImage, fmtExpiry, imageUrl, type Medication, type Reference } from "../api";
import { Chip, Icon, Spinner } from "../components/ui";
import { useApp } from "../state";

type EditMode = "empty" | "add" | "remove";
type Pk = Reference["pockets"][number];

function PocketEditor({ ref_, mode, onChange }: { ref_: Reference; mode: EditMode; onChange: (pockets: Pk[]) => void }) {
  const [w, h] = ref_.image_size;
  const svg = useRef<SVGSVGElement>(null);
  const renumber = (ps: Pk[]) => [...ps].sort((a, b) => (Math.abs(a.cy - b.cy) < Math.max(a.ry, b.ry) ? a.cx - b.cx : a.cy - b.cy)).map((p, i) => ({ ...p, id: i }));

  function tapPocket(e: React.MouseEvent, id: number) {
    e.stopPropagation();
    if (mode === "remove") onChange(renumber(ref_.pockets.filter((p) => p.id !== id)));
    else if (mode === "empty") onChange(ref_.pockets.map((p) => (p.id === id ? { ...p, initially_empty: !p.initially_empty } : p)));
  }
  function tapImage(e: React.MouseEvent<SVGSVGElement>) {
    if (mode !== "add" || !svg.current) return;
    const pt = svg.current.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.current.getScreenCTM()!.inverse());
    const med = (k: "rx" | "ry") => { const v = ref_.pockets.map((q) => q[k]).sort((a, b) => a - b); return v.length ? v[v.length >> 1] : Math.min(w, h) / 12; };
    onChange(renumber([...ref_.pockets, { id: -1, cx: p.x, cy: p.y, rx: med("rx"), ry: med("ry"), angle: 0, initially_empty: false }]));
  }

  return (
    <div className="pocket-editor" style={{ aspectRatio: `${w} / ${h}` }}>
      <img src={imageUrl(`/references/${ref_.id}/image`)} alt="The full strip you photographed" style={{ width: "100%" }} />
      <svg ref={svg} viewBox={`0 0 ${w} ${h}`} role="group" aria-label="Pockets on the strip" onClick={tapImage}
        style={{ cursor: mode === "add" ? "crosshair" : "pointer" }}>
        {ref_.pockets.map((p) => (
          <g key={`${p.id}-${Math.round(p.cx)}`} onClick={(e) => tapPocket(e, p.id)}>
            <ellipse cx={p.cx} cy={p.cy} rx={p.rx} ry={p.ry} transform={`rotate(${p.angle} ${p.cx} ${p.cy})`}
              fill={p.initially_empty ? "rgba(23,35,43,0.55)" : "rgba(14,107,90,0.12)"} stroke={p.initially_empty ? "#B3261E" : "#0E6B5A"}
              strokeWidth={Math.max(3, w / 300)} />
            <text x={p.cx} y={p.cy} textAnchor="middle" dominantBaseline="central" fontSize={Math.max(18, p.ry * 0.7)}
              fontWeight={700} fill={p.initially_empty ? "#fff" : "#17232B"} style={{ pointerEvents: "none" }}>
              {p.initially_empty ? "✕" : p.id + 1}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function Setup({ go }: { go: (to: string) => void }) {
  const { session, notify } = useApp();
  const [meds, setMeds] = useState<Medication[]>([]);
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ name: "", strength: "", instructions: "", times: "09:00, 21:00" });
  const [target, setTarget] = useState<Medication | null>(null);
  const [ref, setRef] = useState<Reference | null>(null);
  const [busy, setBusy] = useState(false);
  const [expiry, setExpiry] = useState("");
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(null);
  const [mode, setMode] = useState<EditMode>("empty");
  const file = useRef<HTMLInputElement>(null);

  const load = useCallback(() => { if (session) api.medications(session.patientId).then(setMeds).catch((e) => notify(e.message)); }, [session, notify]);
  useEffect(load, [load]);

  async function addMedication() {
    if (!session) return;
    const times = form.times.split(/[,\s]+/).filter(Boolean);
    if (!form.name.trim() || !times.length || times.some((t) => !/^([01]\d|2[0-3]):[0-5]\d$/.test(t))) {
      notify("Enter a name and times like 09:00, 21:00"); return;
    }
    setBusy(true);
    try {
      const m = await api.addMedication(session.patientId, { name: form.name.trim(), strength: form.strength.trim(), instructions: form.instructions.trim(),
        schedule: times.map((t) => ({ time: t, pills: 1 })) });
      setAdding(false); setForm({ name: "", strength: "", instructions: "", times: "09:00, 21:00" }); load();
      setTarget(m); notify("Added. Now photograph a full strip.");
    } catch (e) { notify((e as Error).message); } finally { setBusy(false); }
  }

  async function upload(f: File) {
    if (!target) return;
    setBusy(true);
    try {
      const r = await api.uploadReference(target.id, await compressImage(f));
      const read = r.expiry_read;
      setExpiry(read?.date && (read.confidence === "high" || read.confidence === "medium") ? read.date : "");
      setRef(r);
    }
    catch (e) { notify((e as Error).message); } finally { setBusy(false); }
  }

  async function saveExpiry(m: Medication, value: string) {
    setBusy(true);
    try { await api.setExpiry(m.id, value); setEditing(null); notify(value ? `Expiry set to ${fmtExpiry(value)}.` : "Expiry cleared."); load(); }
    catch (e) { notify((e as Error).message); } finally { setBusy(false); }
  }

  async function confirm() {
    if (!ref) return;
    if (!ref.pockets.length) { notify("Add at least one pocket."); return; }
    setBusy(true);
    const read = ref.expiry_read;
    const unchanged = !!read?.date && expiry === read.date && read.confidence !== "low";
    try { await api.confirmReference(ref.id, ref.pockets, unchanged ? undefined : expiry); notify("Strip saved. Daily photos will now be checked against it."); setRef(null); setTarget(null); load(); }
    catch (e) { notify((e as Error).message); } finally { setBusy(false); }
  }

  if (ref && target) {
    const empty = ref.pockets.filter((p) => p.initially_empty).length;
    return (
      <main className="screen">
        <div className="stack" style={{ gap: 6 }}>
          <p className="t-label brand">Step 2 of 2 · {target.name} {target.strength}</p>
          <h1 className="t-display">Check the pockets</h1>
          <p className="t-body muted">We found {ref.pockets.length} pockets. Each pocket should have exactly one circle: add any we missed.</p>
        </div>
        <section className="card center">
          <div className="segmented" role="radiogroup" aria-label="What a tap does">
            {([["empty", "Mark empty"], ["add", "Add pocket"], ["remove", "Remove"]] as const).map(([k, label]) => (
              <button key={k} role="radio" aria-checked={mode === k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>{label}</button>
            ))}
          </div>
          <PocketEditor ref_={ref} mode={mode} onChange={(pockets) => setRef({ ...ref, pockets })} />
          <p className="row t-caption" style={{ color: ref.pocket_confidence >= 0.5 ? "var(--taken)" : "var(--due)" }}>
            <Icon name={ref.pocket_confidence >= 0.5 ? "check" : "alert"} size={18} />
            {ref.pockets.length} pockets · confidence {ref.pocket_confidence >= 0.8 ? "high" : ref.pocket_confidence >= 0.5 ? "medium" : "low"}
            {empty ? ` · ${empty} marked empty` : ""}
          </p>
        </section>
        <section className="card">
          <label className="t-strong" htmlFor="exp">Expiry date on the strip</label>
          <p className="t-caption muted">
            {ref.expiry_read?.date && ref.expiry_read.confidence !== "low" && ref.expiry_read.confidence !== "none"
              ? <>Read from the photo: <b>{fmtExpiry(ref.expiry_read.date)}</b>. Check it matches the “EXP” printed on the strip.</>
              : <>We couldn't read it clearly. Type the month and year printed after “EXP”.</>}
          </p>
          <input id="exp" className="input" type="month" min="2020-01" max="2045-12" value={expiry} onChange={(e) => setExpiry(e.target.value)} />
        </section>
        <section className="card sunken">
          <p className="t-strong">Strip already partly used?</p>
          <p className="t-caption muted">Tap the empty pockets so PillCheck doesn't count them as taken today.</p>
        </section>
        <div className="grow" />
        <button className="btn primary" disabled={busy} onClick={confirm}>{busy ? <Spinner /> : "Looks right"}</button>
        <button className="btn ghost" disabled={busy} onClick={() => { setRef(null); file.current?.click(); }}>Retake photo</button>
        <input ref={file} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) upload(f); }} />
      </main>
    );
  }

  return (
    <main className="screen">
      <button className="link" onClick={() => go("/care")}><Icon name="back" size={20} /> Overview</button>
      <h1 className="t-display">Medicines & strips</h1>
      <p className="t-body muted">Each medicine needs one photo of a <b>full</b> strip. Every later photo is compared with it.</p>

      {meds.map((m) => (
        <section className="card" key={m.id}>
          <div className="row between">
            <div><p className="t-strong">{m.name} {m.strength}</p><p className="t-caption muted">{m.schedule.map((s) => s.time).join(" · ")}{m.instructions ? ` · ${m.instructions}` : ""}</p></div>
            {m.active_reference_id ? <Chip kind="taken">Strip set</Chip> : <Chip kind="due">No strip</Chip>}
          </div>
          {m.active_reference_id && (editing?.id === m.id ? (
            <div className="row">
              <input className="input" type="month" aria-label={`Expiry date for ${m.name}`} value={editing.value}
                onChange={(e) => setEditing({ id: m.id, value: e.target.value })} />
              <button className="btn primary small" style={{ width: "auto" }} disabled={busy} onClick={() => saveExpiry(m, editing.value)}>Save</button>
            </div>
          ) : (
            <button className="link" onClick={() => setEditing({ id: m.id, value: m.expiry?.date ?? "" })}>
              <Icon name="calendar" size={18} />
              {m.expiry?.date ? `Expires ${fmtExpiry(m.expiry.date)}${m.expiry.source === "photo" ? " (read from strip)" : ""}` : "Add expiry date"}
            </button>
          ))}
          {m.active_reference_id && (
            <img className="evidence" src={imageUrl(`/references/${m.active_reference_id}/preview.jpg`)} alt={`Registered strip for ${m.name}`} loading="lazy" />
          )}
          <button className="btn secondary small" disabled={busy} onClick={() => { setTarget(m); file.current?.click(); }}>
            {busy && target?.id === m.id ? <Spinner dark /> : <Icon name="camera" size={20} />} {m.active_reference_id ? "Register a new strip" : "Photograph a full strip"}
          </button>
        </section>
      ))}

      {adding ? (
        <section className="card">
          <p className="t-heading">Add a medicine</p>
          {([["name", "Name", "Telmisartan"], ["strength", "Strength", "40 mg"], ["instructions", "When / how", "After breakfast"], ["times", "Times (24 h)", "09:00, 21:00"]] as const).map(([k, label, ph]) => (
            <div className="field" key={k}>
              <label htmlFor={`f-${k}`}>{label}</label>
              <input id={`f-${k}`} className="input" placeholder={ph} value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} />
            </div>
          ))}
          <button className="btn primary" disabled={busy} onClick={addMedication}>{busy ? <Spinner /> : "Add medicine"}</button>
          <button className="btn ghost small" onClick={() => setAdding(false)}>Cancel</button>
        </section>
      ) : (
        <button className="btn secondary" onClick={() => setAdding(true)}><Icon name="plus" size={20} /> Add a medicine</button>
      )}
      <input ref={file} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) upload(f); }} />
    </main>
  );
}
