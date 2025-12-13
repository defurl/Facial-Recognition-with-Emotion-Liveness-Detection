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
