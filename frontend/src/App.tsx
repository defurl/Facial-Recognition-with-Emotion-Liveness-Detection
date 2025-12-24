import { useMemo, useState, useEffect } from "react";
import "./App.css";
import { useBackendData } from "./hooks/useBackendData";
import { VerificationPanel } from "./components/VerificationPanel";
import { RegistrationPanel } from "./components/RegistrationPanel";
import { AttendancePanel } from "./components/AttendancePanel";
import { VerificationProvider } from "./context/verification";
import { updateThreshold } from "./services/apiClient";
import { Modal } from "./components/Modal";


function App() {
  const { health, threshold: backendThreshold, employees, refresh } = useBackendData();
  const [localThreshold, setLocalThreshold] = useState<number | null>(null);
  const [isRegistrationOpen, setIsRegistrationOpen] = useState(false);
  const [showStats, setShowStats] = useState(true);

  useEffect(() => {
    if (backendThreshold !== null) {
      setLocalThreshold(backendThreshold);
    }
  }, [backendThreshold]);

  const handleThresholdChange = async () => {
    if (localThreshold !== null) {
      await updateThreshold({ threshold: localThreshold });
      refresh();
    }
  };

  const healthLabel = useMemo(() => {
    if (health === "ok") return "Online";
    if (health === "unavailable") return "Offline";
    return "Connecting...";
  }, [health]);

  return (
    <VerificationProvider>
      <div className="layout-container">

        {/* Main Content Area */}
        <main className="main-content">
          <header className="top-bar">
            <div>
              <h1>DeepFaceLive</h1>
              <p className="subtitle">Real-time Facial Verification & Liveness</p>
            </div>
            <div className={`status-badge ${health}`}>
              <span className="dot"></span> {healthLabel}
            </div>
          </header>

          <div className="verification-stage">
            <VerificationPanel />
          </div>

          <footer className="action-bar">
            <button
              className="action-btn primary"
              onClick={() => setIsRegistrationOpen(true)}
            >
              <span className="icon">+</span> Register New User
            </button>
            <button
              className="action-btn secondary"
              onClick={() => setShowStats(!showStats)}
            >
              {showStats ? "Hide Sidebar" : "Show Sidebar"}
            </button>
            <button className="action-btn secondary" onClick={() => refresh()}>
              Refresh Data
            </button>
          </footer>
        </main>

        {/* Right Sidebar */}
        {showStats && (
          <aside className="sidebar">
            <div className="sidebar-section">
              <h3>System Status</h3>
              <div className="status-card">
                <div className="field-row">
                  <label>Threshold</label>
                  <span className="value">{localThreshold?.toFixed(2) ?? "..."}</span>
                </div>
                {localThreshold !== null && (
                  <input
                    type="range"
                    min="0.3" max="0.9" step="0.05"
                    value={localThreshold}
                    className="slider"
                    onChange={(e) => setLocalThreshold(parseFloat(e.target.value))}
                    onMouseUp={handleThresholdChange}
                  />
                )}
                <div className="field-row">
                  <label>Identities</label>
                  <span className="value">{employees.length}</span>
                </div>
              </div>
            </div>

            <div className="sidebar-section">
              <h3>Registered Users</h3>
              <div className="user-list">
                {employees.length ? (
                  employees.map((name) => (
                    <div key={name} className="user-chip">
                      <div className="avatar">{name[0].toUpperCase()}</div>
                      <span>{name}</span>
                    </div>
                  ))
                ) : (
                  <p className="muted">No users found.</p>
                )}
              </div>
            </div>

            <div className="sidebar-section flex-grow">

              <div className="attendance-wrapper">
                <AttendancePanel />
              </div>
            </div>
          </aside>
        )}

        {/* Modals */}
        <Modal
          isOpen={isRegistrationOpen}
          onClose={() => setIsRegistrationOpen(false)}
          title="Register New Employee"
        >
          <RegistrationPanel />
        </Modal>

      </div>
    </VerificationProvider>
  );
}

export default App;
