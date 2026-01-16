---
title: Face Recognition Attendance API
emoji: 🧑‍💼
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Face Recognition Attendance API

Real-time face recognition and attendance system with liveness detection.

## Features

- Face verification using deep learning embeddings
- Blink-based liveness detection (anti-spoofing)
- Multi-pose registration for robust recognition
- Attendance logging with timestamps

## API Endpoints

| Endpoint            | Method   | Description               |
| ------------------- | -------- | ------------------------- |
| `/health`           | GET      | Health check              |
| `/verify`           | POST     | Verify face + liveness    |
| `/register`         | POST     | Register new employee     |
| `/employees`        | GET      | List registered employees |
| `/attendance/today` | GET      | Today's attendance log    |
| `/threshold`        | GET/POST | Recognition threshold     |

## Usage

Point your frontend to this Space:

```javascript
const API_URL = "https://YOUR_USERNAME-face-recognition.hf.space";

// Verify a face
const response = await fetch(`${API_URL}/verify`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image_b64: "base64_encoded_image",
    blink_sequence: ["frame1_b64", "frame2_b64", ...]
  })
});
```

## Built With

- FastAPI + Uvicorn
- PyTorch (face embeddings)
- MediaPipe (facial landmarks)
- OpenCV (image processing)
