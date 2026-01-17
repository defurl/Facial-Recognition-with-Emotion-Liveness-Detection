# SSH Server Backend Deployment Guide

## Overview

Deploy a FastAPI face recognition backend on an SSH server with:

- 40 vCPUs
- 2× RTX 4090 (48GB VRAM)
- 98GB RAM
- Ubuntu 22.04, CUDA 12.5

**Known Issue:** The QEMU Virtual CPU may NOT support AVX/AVX2 instructions, which are required by MediaPipe and TensorFlow. This guide includes fallback solutions.

---

## Step 0: Verify AVX Support

```bash
grep -o 'avx\|avx2' /proc/cpuinfo | head -2
```

- **If output shows `avx` and `avx2`** → Proceed to Standard Deployment
- **If output is EMPTY** → Use Fallback Deployment with dlib

---

## Standard Deployment (If AVX Available)

### 1. Setup Environment

```bash
mkdir -p ~/face-recognition && cd ~/face-recognition
python3 -m venv venv && source venv/bin/activate

# CUDA-enabled PyTorch for RTX 4090
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install fastapi uvicorn mediapipe==0.10.15 opencv-python numpy pandas Pillow scipy scikit-learn
```

### 2. Upload Project Files

On your local machine, run:

```bash
scp -r backend src artifacts requirements.txt best_face_embedding_model.pth USER@SERVER:~/face-recognition/
```

### 3. Start Server

```bash
source ~/face-recognition/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=~/face-recognition
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Test

```bash
curl http://localhost:8000/health
```

---

## Fallback Deployment (If AVX NOT Available)

If MediaPipe fails with "Illegal instruction" or TensorFlow crashes, use this approach.

### 1. Setup Environment (No TensorFlow/MediaPipe)

```bash
mkdir -p ~/face-recognition && cd ~/face-recognition
python3 -m venv venv && source venv/bin/activate

# CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies (NO mediapipe, NO tensorflow)
pip install fastapi uvicorn opencv-python numpy pandas Pillow scipy scikit-learn

# dlib for blink detection (replaces MediaPipe)
pip install dlib

# Download dlib's face landmark model
cd ~/face-recognition
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### 2. Modify server.py for dlib

Create a new file `~/face-recognition/backend/blink_dlib.py`:

```python
"""Blink detection using dlib instead of MediaPipe (no AVX required)."""
import numpy as np
import cv2
import dlib

# Initialize dlib
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = None

def init_dlib():
    global predictor
    if predictor is None:
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return predictor

def compute_ear(eye_points):
    """Compute Eye Aspect Ratio from 6 landmark points."""
    eye_points = np.array(eye_points)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def process_blink_sequence_dlib(frames):
    """
    Process blink sequence using dlib instead of MediaPipe.
    Args:
        frames: List of BGR numpy arrays
    Returns:
        Dict with blink detection results
    """
    predictor = init_dlib()
    ear_values = []

    # Sample frames for efficiency
    num_samples = min(8, len(frames))
    step = max(1, len(frames) // num_samples)
    sampled_frames = [frames[i] for i in range(0, len(frames), step)][:num_samples]

    for frame in sampled_frames:
        if frame is None:
            continue
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if not faces:
                continue

            shape = predictor(gray, faces[0])

            # Left eye: landmarks 36-41, Right eye: landmarks 42-47
            left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)

        except Exception as e:
            print(f"[BLINK-DLIB] Error: {e}")
            continue

    if not ear_values:
        return {
            "has_blinked": False,
            "blink_score": 0.0,
            "frames_processed": 0,
            "min_ear": None,
            "max_ear": None,
            "current_ear": None,
            "blinks_needed": 1,
            "threshold": 0.25
        }

    min_ear = min(ear_values)
    max_ear = max(ear_values)
    current_ear = ear_values[-1]

    # Blink detection thresholds
    EAR_CLOSED = 0.22
    EAR_OPEN = 0.28

    has_blinked = min_ear < EAR_CLOSED and max_ear > EAR_OPEN

    if has_blinked:
        blink_score = 1.0
    elif min_ear < 0.25:
        blink_score = 0.7
    else:
        blink_score = 0.0

    print(f"[BLINK-DLIB] {len(ear_values)} frames. EAR: {min_ear:.3f}-{max_ear:.3f}, Blinked: {has_blinked}")

    return {
        "has_blinked": has_blinked,
        "blink_score": blink_score,
        "frames_processed": len(ear_values),
        "min_ear": min_ear,
        "max_ear": max_ear,
        "current_ear": current_ear,
        "blinks_needed": 0 if has_blinked else 1,
        "threshold": EAR_CLOSED
    }
```

### 3. Modify server.py to use dlib

In `server.py`, find the import section and add:

```python
# At the top of server.py, add conditional import
try:
    from src.utils import face_mesh_detector
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"[WARNING] MediaPipe not available: {e}")
    print("[WARNING] Using dlib fallback for blink detection")
    MEDIAPIPE_AVAILABLE = False
    from backend.blink_dlib import process_blink_sequence_dlib
```

Then modify `_process_blink_sequence` function:

```python
def _process_blink_sequence(frames):
    """Blink detection with MediaPipe or dlib fallback."""
    if not MEDIAPIPE_AVAILABLE:
        return process_blink_sequence_dlib(frames)

    # Original MediaPipe code here...
```

### 4. Modify src/utils.py

Add a check at the top of `utils.py`:

```python
import cv2
import numpy as np

# Conditional MediaPipe import
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_detector = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] MediaPipe initialization failed: {e}")
    MEDIAPIPE_AVAILABLE = False
    face_mesh_detector = None
```

### 5. Start Server

```bash
source ~/face-recognition/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=~/face-recognition
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Systemd Service (Auto-start)

Create `/etc/systemd/system/face-recognition.service`:

```ini
[Unit]
Description=Face Recognition API
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/face-recognition
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTHONPATH=/home/YOUR_USERNAME/face-recognition
ExecStart=/home/YOUR_USERNAME/face-recognition/venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable face-recognition
sudo systemctl start face-recognition
sudo systemctl status face-recognition
```

---

## Verification

```bash
# GPU availability
nvidia-smi

# PyTorch sees GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Health check
curl http://localhost:8000/health
```

Expected health response:

```json
{ "model_loaded": true, "employees": 0, "threshold": 1.1 }
```

---

## Firewall (if needed)

```bash
sudo ufw allow 8000/tcp
```

---

## Frontend Configuration

Update frontend `.env.production`:

```
REACT_APP_API_BASE_URL=http://YOUR_SERVER_IP:8000
```
