# Frontend Application (React + TypeScript)

This React/TypeScript workspace will host the user-facing UI for registration, verification, and attendance tracking. It talks to the FastAPI backend (`backend/server.py`) and reads the shared artifacts under `artifacts/` (models, DB, thresholds).

## Environment

- `REACT_APP_API_BASE_URL` (default: `http://localhost:8000`): base URL for all backend HTTP requests. Override by creating a `.env.local` file if you need to point at a remote service.
- `PUBLIC_URL` is managed by CRA; do not set it unless you are deploying to a nested path.

## Getting started

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) to see the running UI. The dev server supports hot reload.

## Available scripts

- `npm start` – start the development server with fast refresh and overlay diagnostics.
- `npm test` – launch the Jest test runner (use `npm test -- --watchAll=false` for a single pass).
- `npm run build` – compile the production bundle to `frontend/build/`.
- `npm run eject` – expose the CRA toolchain (permanent; only do this if you need deep customization).

## Backend API contracts

The frontend will call these FastAPI routes. All requests are relative to `process.env.REACT_APP_API_BASE_URL`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Sanity check for the backend service |
| `/verify` | POST | Send a face or embedding payload, receive `identity`, `distance`, `confidence`, `liveness` and metadata |
| `/register` | POST | Register a new employee (`name`, `embedding`, optional tags/metadata) |
| `/threshold` | GET / PUT | Read or update the GUI decision threshold |
| `/attendance/log` | POST | Append a new attendance event (`identity`, `status`, `timestamp`) |
| `/attendance/export` | GET | Download the current attendance log CSV |
| `/employees` | GET | List all registered employees and their metadata |

Use typed DTOs in `src/types/` (create the folder) and an HTTP client such as `axios` or `fetch` to call these endpoints.

## Registration workflow

- The React registration panel mirrors the Python GUI: you can choose between quick (three poses) or full (six poses) guidance, capture frames via the verification panel, upload an image, or paste a data URL.
- Use the uploaded image (<kbd>Choose file</kbd>) to automatically convert a photo into the base64 string the backend expects; you can also edit the text box if you already duplicated a data URL.
- After capturing each pose (forward, left, right, etc.), click the register button to send the current capture list to `/register`—the UI currently submits one capture but the instructions tell you which poses the desktop app would cycle through.

## Frontend TODOs

1. Replace the default CRA view (`src/App.tsx`) with the dashboard components for preview, registration, and history.
2. Capture the webcam stream (e.g., using `navigator.mediaDevices.getUserMedia`) and post frames or embeddings to `/verify`.
3. Build a registration modal that saves names + captures and calls `/register`.
4. Display attendance logs and allow export via `/attendance/export`.
5. Add state for liveness/verification thresholds and surface `GET /threshold`/`PUT /threshold` in the UI.

## Next steps

- Wire the frontend components to `backend/server.py` through the documented API.
- Add shared styling/animation using your chosen design system (Material UI, Chakra, Tailwind, etc.).
- Once integration is stable, run `npm run build` and serve the static output alongside the backend for production.

## System process note (web implementation)

- **What we have done:** tightened API payloads, enhanced `RegistrationPanel` with pose instructions, mode selection, file upload that converts to base64, and friendly error handling; the frontend now builds cleanly and matches the Python flow guidance.  
- **What we have to do:** connect the capture-based registration/verification UI to real webcam frames, persist pose history, and replicate the backend’s liveness/attendance workflows (threshold tuning, attendance export) before claiming feature parity with the Python desktop app.  
- **Layout overview:** the layout centers around a dashboard-style panel set—verification view (camera stream and history), registration form (name, mode, upload, instructions), and supporting hints; each panel publishes to the shared API client in `src/services/apiClient.ts`.  
- **Feature comparison to Python version:** the web UI mirrors the registration steps (quick vs full pose guidance, upload conversions, replace toggle) and reads the same endpoints (`/verify`, `/register`, `/threshold`, `/attendance`). Still pending: the richer Python attendance logs/liveness widgets and multi-pose capture automation, which are roadmapped next.
