import { useEffect, useRef, useState } from "react";
import { api, compressImage, imageUrl, type Medication, type Sample } from "../api";
import { Icon, Spinner } from "../components/ui";
import { strings } from "../i18n";
import { useApp } from "../state";

export default function Capture({ go, medId }: { go: (to: string) => void; medId: string | null }) {
  const { session, lang, at, notify } = useApp();
  const s = strings(lang);
  const [meds, setMeds] = useState<(Medication & { demo_key?: string })[]>([]);
  const [selected, setSelected] = useState<string | null>(medId);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const camera = useRef<HTMLInputElement>(null);
  const gallery = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!session) return;
    api.medications(session.patientId).then((m) => { setMeds(m); setSelected((cur) => cur ?? m[0]?.id ?? null); }).catch((e) => notify(e.message));
    if (session.demo) api.samples().then(setSamples).catch(() => setSamples([]));
  }, [session, notify]);

  const med = meds.find((m) => m.id === selected);

  async function submit(opts: { file?: File; sampleId?: string; previewUrl?: string }) {
    if (!session) return;
    setBusy(true);
    setPreview(opts.previewUrl ?? (opts.file ? URL.createObjectURL(opts.file) : null));
    try {
      const file = opts.file ? await compressImage(opts.file) : undefined;
      const check = await api.submitCheck(session.patientId, { file, sampleId: opts.sampleId, medicationId: selected ?? undefined, at: at() });
      go(`/result/${check.id}`);
    } catch (e) {
      notify((e as Error).message);
      setBusy(false);
      setPreview(null);
    }
  }

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (f) submit({ file: f });
  };

  // Samples for the selected medicine first, then the "something went wrong" ones.
  const key = med && "demo_key" in med ? med.demo_key : undefined;
  const sorted = [...samples].sort((a, b) => Number(b.medication === key) - Number(a.medication === key));

  return (
    <main className="screen dark">
      <header className="row between">
        <div>
          <p className="t-label" style={{ opacity: 0.6 }}>{s.capture}</p>
          <h1 className="t-heading">{med ? `${med.name} ${med.strength}` : s.which}</h1>
        </div>
        <button className="round" aria-label="Close" onClick={() => go("/today")}><Icon name="x" size={20} /></button>
      </header>

      {meds.length > 1 && (
        <div className="row" role="radiogroup" aria-label={s.which} style={{ flexWrap: "wrap", gap: 8 }}>
          {meds.map((m) => (
            <button key={m.id} role="radio" aria-checked={m.id === selected} className="tip"
              style={{ border: 0, color: "#fff", cursor: "pointer", background: m.id === selected ? "#0E6B5A" : undefined }}
              onClick={() => setSelected(m.id)}>{m.name}</button>
          ))}
        </div>
      )}

      <div className="viewfinder">
        {preview ? <img src={preview} alt="" /> : (
          <>
            <div className="guide" />
            <p className="guide-text">{s.fitStrip}</p>
          </>
        )}
        {busy && (
          <div style={{ position: "absolute", inset: 0, background: "rgba(14,19,22,0.65)", display: "grid", placeItems: "center" }}>
            <div className="stack center"><Spinner /><p className="t-strong">{s.checking}</p></div>
          </div>
        )}
      </div>

      <div className="row" style={{ justifyContent: "center", gap: 8 }}>
        {s.tips.map((t) => <span className="tip" key={t}>{t}</span>)}
      </div>

      <div className="row between">
        <label className="side">
          <span className="round" onClick={() => gallery.current?.click()}><Icon name="image" size={22} /></span>{s.gallery}
        </label>
        <button className="shutter" aria-label={s.photograph} disabled={busy || !selected} onClick={() => camera.current?.click()}><span /></button>
        <label className="side">
          <span className="round" onClick={() => document.getElementById("samples")?.scrollIntoView({ behavior: "smooth" })} style={{ opacity: samples.length ? 1 : 0.3 }}>
            <Icon name="sparkle" size={22} />
          </span>{s.sample}
        </label>
      </div>
      <input ref={camera} type="file" accept="image/*" capture="environment" hidden onChange={onFile} />
      <input ref={gallery} type="file" accept="image/*" hidden onChange={onFile} />

      {samples.length > 0 && (
        <section id="samples" className="stack" style={{ marginTop: 12 }}>
          <p className="t-heading">{s.samplesTitle}</p>
          <p className="t-caption" style={{ opacity: 0.7 }}>{s.samplesHint}</p>
          <div className="samples">
            {sorted.map((sm) => (
              <button key={sm.id} className="sample" disabled={busy}
                onClick={() => submit({ sampleId: sm.id, previewUrl: imageUrl(`/demo/assets/${sm.file}`) })}>
                <img src={imageUrl(`/demo/assets/${sm.file}`)} alt="" loading="lazy" />
                <span className="t-caption" style={{ fontWeight: 600 }}>{sm.title}</span>
                <span style={{ fontSize: 12, lineHeight: "16px", opacity: 0.7 }}>{sm.hint}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
