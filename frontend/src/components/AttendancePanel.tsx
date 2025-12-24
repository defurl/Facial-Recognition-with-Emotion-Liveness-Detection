import { FormEvent, useEffect, useState } from "react";
import { logAttendance, fetchAttendanceToday } from "../services/apiClient";
import { DailyAttendanceRecord } from "../types/api";

export function AttendancePanel() {
  const [identity, setIdentity] = useState("");
  const [logs, setLogs] = useState<DailyAttendanceRecord[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const loadLogs = () => {
    setIsLoading(true);
    fetchAttendanceToday()
      .then(setLogs)
      .catch((err) => console.error("Failed to load logs", err))
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    loadLogs();
  }, []);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setMessage(null);
    try {
      await logAttendance({
        name: identity,
        distance: 0,
        emotion: "Manual",
        liveness: "Manual",
      });
      setMessage("Attendance logged successfully");
      setIdentity("");
      loadLogs();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to submit");
    }
  };

  return (
    <article className="panel attendance-panel">
      <div className="panel-header">
        <h3>Attendance log</h3>
        <button className="ghost-button" onClick={loadLogs} disabled={isLoading}>
          {isLoading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      <form onSubmit={handleSubmit} className="attendance-form">
        <div className="input-group">
          <input
            value={identity}
            onChange={(event) => setIdentity(event.target.value)}
            required
            placeholder="Manual entry name"
            className="compact-input"
          />
          <button type="submit" disabled={!identity}>Log Presence</button>
        </div>
      </form>

      {message && <p className={`message ${message.includes("success") ? "success" : "error"}`}>{message}</p>}

      <div className="table-wrapper">
        <table className="attendance-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Name</th>
              <th>Status</th>
              <th>Distance</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr>
                <td colSpan={4} className="muted center">No records for today</td>
              </tr>
            ) : (
              logs.map((record, i) => (
                <tr key={i}>
                  <td title={record.timestamp as string}>
                    {typeof record.timestamp === "string" ? record.timestamp.split(" ")[1]?.slice(0, 5) ?? record.timestamp : "??:??"}
                  </td>
                  <td>{record.employee_name as string ?? record.name ?? "Unknown"}</td>
                  <td>
                    <span className={`status-pill ${((record.liveness_status as string) ?? "").toLowerCase() === "real" ? "ok" : "warn"}`}>
                      {record.liveness_status as string ?? "Unknown"}
                    </span>
                  </td>
                  <td>{(record.confidence_distance as string | number) ?? (record.distance as number)?.toFixed(3) ?? "-"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </article>
  );
}
