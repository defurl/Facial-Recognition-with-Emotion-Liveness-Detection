## Backend

Purpose: FastAPI service that wraps the verification, blink-based liveness, and attendance logging pipelines. It reads pretrained artifacts from `artifacts/outputs/` and answers requests from the upcoming frontend.

### Getting started

```bash
pip install -r requirements.txt
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

The server keeps `artifacts/outputs/attendance_log.csv`, `gui_threshold.json`, and the employee database in sync with requests.

### API overview

- **GET /health** – reports whether the verification model is loaded, how many employees are registered, and where the attendance log lives.
- **POST /verify** – send `{ image_b64, threshold?, mark_attendance?, blink_sequence? }`. The server crops the primary face, runs the embedding model, compares it against the cached employee index, and optionally logs attendance. `blink_sequence` accepts a list of base64 frames to confirm a blink-derived liveness signal.
- **POST /register** – register new employees by supplying `{ name, images_b64, replace_existing? }`. Each supplied image is embedded and stored; the index is rebuilt automatically.
- **GET /employees** – lists registered identities.
- **GET /threshold** / **POST /threshold** – read or update the GUI threshold (`artifacts/outputs/gui_threshold.json`).
- **GET /attendance/today** – returns today’s attendance rows as JSON.
- **GET /attendance/summary** – returns a summary (total marks, unique employees, avg confidence, live/spoof counts).
- **POST /attendance/mark** – manually append a row: `{ name, distance, emotion, liveness }`.
