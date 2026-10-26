import { AppProvider, useApp, useRoute } from "./state";
import CareOverview from "./screens/CareOverview";
import Capture from "./screens/Capture";
import CheckDetail from "./screens/CheckDetail";
import Landing from "./screens/Landing";
import Result from "./screens/Result";
import Setup from "./screens/Setup";
import Today from "./screens/Today";
import { fmtTime } from "./i18n";

const CLOCK = ["08:00", "09:10", "09:40", "12:00", "20:40", "21:30"];

function DemoClock() {
  const { simTime, setSimTime } = useApp();
  return (
    <div className="clock" aria-label="Demo clock">
      <div className="inner">
        <span style={{ opacity: 0.7, whiteSpace: "nowrap" }}>Demo clock</span>
        {CLOCK.map((t) => {
          const f = fmtTime(t);
          return <button key={t} className={simTime === t ? "on" : ""} onClick={() => setSimTime(t)}>{f.time} {f.ampm}</button>;
        })}
        <button className={simTime === null ? "on" : ""} onClick={() => setSimTime(null)}>Real time</button>
      </div>
    </div>
  );
}

function Shell() {
  const [path, go] = useRoute();
  const { session, setSession, toast } = useApp();
  const [route, query] = path.split("?");
  const params = new URLSearchParams(query ?? "");
  const parts = route.split("/").filter(Boolean);

  if (!session || parts.length === 0) {
    return <div className="app"><Landing go={go} />{toast && <div className="toast" role="status">{toast}</div>}</div>;
  }
  const care = parts[0] === "care";
  const camera = parts[0] === "capture";

  let screen;
  if (parts[0] === "today") screen = <Today go={go} />;
  else if (camera) screen = <Capture go={go} medId={params.get("med")} />;
  else if (parts[0] === "result" && parts[1]) screen = <Result go={go} checkId={parts[1]} />;
  else if (care && parts[1] === "check" && parts[2]) screen = <CheckDetail go={go} checkId={parts[2]} />;
  else if (care && parts[1] === "setup") screen = <Setup go={go} />;
  else if (care) screen = <CareOverview go={go} />;
  else screen = <Today go={go} />;

  return (
    <div className="app" style={camera ? { background: "var(--bg-camera)" } : undefined}>
      {!camera && (
        <nav className="topnav" aria-label="View">
          <div className="inner">
            <button className={`tab ${!care ? "on" : ""}`} onClick={() => go("/today")}>Patient</button>
            <button className={`tab ${care ? "on" : ""}`} onClick={() => go("/care")}>Caregiver</button>
            <span className="grow" />
            {session.demo && <button className="tab" onClick={() => { setSession(null); go("/"); }}>New demo</button>}
          </div>
        </nav>
      )}
      {session.demo && !camera && <DemoClock />}
      {screen}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}

export default function App() {
  return <AppProvider><Shell /></AppProvider>;
}
