import { useMemo, useState, useEffect } from "react";
import "./App.css";
import { useBackendData } from "./hooks/useBackendData";
import { VerificationPanel } from "./components/VerificationPanel";
import { RegistrationPanel } from "./components/RegistrationPanel";
import { AttendancePanel } from "./components/AttendancePanel";
import { EARDebugPanel } from "./components/EARDebugPanel";
import { CameraSelector } from "./components/CameraSelector";
import { VerificationProvider } from "./context/verification";
import { updateThreshold, deleteEmployee } from "./services/apiClient";
import { Modal } from "./components/Modal";


function App() {
  const { health, threshold: backendThreshold, employees, refresh } = useBackendData();
  const [localThreshold, setLocalThreshold] = useState<number | null>(null);
  const [isRegistrationOpen, setIsRegistrationOpen] = useState(false);
  const [showStats, setShowStats] = useState(true);

  // Delete confirmation modal state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

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

  const handleDeleteEmployee = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    try {
      await deleteEmployee(deleteTarget);
      refresh();
    } catch (err) {
      console.error("Failed to delete employee:", err);
    } finally {
      setIsDeleting(false);
      setDeleteTarget(null);
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
            <VerificationPanel isPaused={isRegistrationOpen} />
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
                <CameraSelector />
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
                      <button
                        className="delete-btn"
                        onClick={() => setDeleteTarget(name)}
                        title="Delete user"
                      >
                        ×
                      </button>
                    </div>
                  ))
                ) : (
                  <p className="muted">No users found.</p>
                )}
              </div>
            </div>

            <div className="sidebar-section">
              <EARDebugPanel />
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

        {/* Delete Confirmation Modal */}
        {deleteTarget && (
          <div className="confirm-modal-backdrop" onClick={() => setDeleteTarget(null)}>
            <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
              <h3>Delete Employee</h3>
              <p>Are you sure you want to remove <strong>{deleteTarget}</strong> from the system? This action cannot be undone.</p>
              <div className="confirm-modal-actions">
                <button className="cancel-btn" onClick={() => setDeleteTarget(null)}>
                  Cancel
                </button>
                <button
                  className="danger-btn"
                  onClick={handleDeleteEmployee}
                  disabled={isDeleting}
                >
                  {isDeleting ? "Deleting..." : "Delete"}
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </VerificationProvider>
  );
}

export default App;

