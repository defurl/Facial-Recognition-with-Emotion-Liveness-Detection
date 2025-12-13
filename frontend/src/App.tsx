import { useMemo } from "react";
import "./App.css";
import { useBackendData } from "./hooks/useBackendData";
import { VerificationPanel } from "./components/VerificationPanel";
import { RegistrationPanel } from "./components/RegistrationPanel";
import { AttendancePanel } from "./components/AttendancePanel";
import { VerificationProvider } from "./context/verification";

function App() {
  const { health, threshold, employees } = useBackendData();

  const healthLabel = useMemo(() => {
    if (health === "ok") return "Backend healthy";
    if (health === "unavailable") return "Backend unreachable";
    return "Checking backend";
  }, [health]);

  return (
    <VerificationProvider>
      <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Face Recognition Attendance · Fullstack</p>
          <h1>Verification dashboard</h1>
          <p className="subtitle">Connect the camera preview, liveness signals, and attendance log through the FastAPI backend.</p>
        </div>
        <span className={`status-pill ${health}`}>{healthLabel}</span>
      </header>

      <section className="status-grid">
        <article className="card">
          <h2>Services</h2>
          <dl>
            <dt>Threshold</dt>
            <dd>{threshold !== null ? `${threshold.toFixed(3)} (GUI)` : "loading"}</dd>
            <dt>Registered identities</dt>
            <dd>{employees.length}</dd>
            <dt>Last sync</dt>
            <dd>{health === "ok" ? "now" : "waiting"}</dd>
          </dl>
        </article>

        <article className="card">
          <h2>Quick actions</h2>
          <ul>
            <li>Webcam preview + snapshot to `/verify`</li>
            <li>Registration modal → `/register`</li>
            <li>Attendance table + `/attendance/export`</li>
          </ul>
        </article>

        <article className="card">
          <h2>Employees</h2>
          {employees.length ? (
            <div className="chip-list">
              {employees.map((name) => (
                <span key={name} className="chip">
                  {name}
                </span>
              ))}
            </div>
          ) : (
            <p className="muted">No employees loaded yet.</p>
          )}
        </article>
      </section>

      <section className="panels">
        <VerificationPanel />
        <RegistrationPanel />
        <AttendancePanel />
      </section>
      </div>
    </VerificationProvider>
  );
}

export default App;
