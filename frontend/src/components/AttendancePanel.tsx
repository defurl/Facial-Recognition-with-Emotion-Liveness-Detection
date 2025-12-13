import { FormEvent, useState } from "react";
import { logAttendance } from "../services/apiClient";

export function AttendancePanel() {
  const [identity, setIdentity] = useState("");
  const [status, setStatus] = useState("present");
  const [timestamp, setTimestamp] = useState(new Date().toISOString().slice(0, 16));
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      await logAttendance({ identity, status, timestamp });
      setMessage("Attendance logged");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to submit");
    }
  };

  const handleNow = () => {
    setTimestamp(new Date().toISOString().slice(0, 16));
  };

  return (
    <article className="panel attendance-panel">
      <h3>Attendance log</h3>
      <p className="muted">Submit presence events or pull `/attendance/export` once ready.</p>
      <form onSubmit={handleSubmit} className="attendance-form">
        <label>
          Identity
          <input value={identity} onChange={(event) => setIdentity(event.target.value)} required placeholder="Employee name" />
        </label>
        <label>
          Status
          <select value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="present">Present</option>
            <option value="absent">Absent</option>
            <option value="late">Late</option>
          </select>
        </label>
        <label>
          Timestamp
          <input type="datetime-local" value={timestamp} onChange={(event) => setTimestamp(event.target.value)} />
        </label>
        <div className="attendance-actions">
          <button type="button" onClick={handleNow}>Now</button>
          <button type="submit">Log attendance</button>
        </div>
      </form>
      {message && <p className="muted info">{message}</p>}
      <p className="muted">When the backend is ready, wire `/attendance/export` to download logs.</p>
    </article>
  );
}
