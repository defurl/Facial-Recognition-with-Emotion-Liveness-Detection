"""FastAPI backend for the face recognition attendance service."""
from __future__ import annotations

import base64
import re
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attendance import AttendanceLogger
from src.blink_detector import BlinkDetector
from src.config import (
    BLINK_THRESHOLD,
    DEVICE,
    EMPLOYEE_DB_PATH,
    MODEL_METRIC_PATH,
    OUTPUT_DIR,
    OPTIMAL_THRESHOLD_GUI,
)
from src.data_loader import get_transforms
from src.pipeline.verification import set_embedding_index, verify_embedding_fast
from src.runtime.db import build_embedding_index, load_employee_db, save_employee_db
from src.runtime.models_loader import load_verification_model
from src.runtime.settings import load_gui_threshold, save_gui_threshold
from src.utils import crop_face_with_padding, detect_faces, face_mesh_detector

app = FastAPI(
    title="Face Recognition Backend",
    description="Backend API for verification, threshold tuning, and attendance logging",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VAL_TRANSFORM: Optional[torch.nn.Module] = None
MODEL: Optional[torch.nn.Module] = None
EMPLOYEE_DB: Dict[str, list] = {}
EMPLOYEE_INDEX: Optional[torch.Tensor] = None
EMPLOYEE_OFFSETS: List = []
THRESHOLD_VALUE: float = OPTIMAL_THRESHOLD_GUI

FACE_MESH_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()

# Attendance logger ensures we continue writing to the artifacts log
attendance_logger = AttendanceLogger(csv_path=OUTPUT_DIR / "attendance_log.csv")


def refresh_employee_index() -> None:
    global EMPLOYEE_INDEX, EMPLOYEE_OFFSETS
    if not EMPLOYEE_DB:
        EMPLOYEE_INDEX = None
        EMPLOYEE_OFFSETS = []
        set_embedding_index(None, [])
        return

    EMPLOYEE_INDEX, EMPLOYEE_OFFSETS = build_embedding_index(EMPLOYEE_DB)
    set_embedding_index(EMPLOYEE_INDEX, EMPLOYEE_OFFSETS)


class VerifyRequest(BaseModel):
    image_b64: str
    threshold: Optional[float] = None
    mark_attendance: bool = False
    blink_sequence: Optional[List[str]] = None


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    images_b64: List[str] = Field(..., min_items=1)
    replace_existing: bool = False


class ThresholdPayload(BaseModel):
    threshold: float


class AttendanceRecordRequest(BaseModel):
    name: str = Field(..., min_length=1)
    distance: float
    emotion: str = "Unknown"
    liveness: str = "Unknown"


def _decode_base64_image(b64data: str) -> np.ndarray:
    data = re.sub(r"^data:image/[^;]+;base64,", "", b64data)
    try:
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Decoded image is empty")
        return frame
    except Exception as exc:
        raise HTTPException(400, detail=f"Invalid image data: {exc}")


def _get_primary_face(frame: np.ndarray) -> np.ndarray:
    faces = detect_faces(frame)
    if not faces:
        raise HTTPException(400, detail="No faces detected in the provided image")
    primary = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    cropped = crop_face_with_padding(frame, *primary, padding_ratio=0.2)
    if cropped.size == 0:
        raise HTTPException(400, detail="Failed to crop detected face")
    return cropped


def _prepare_embedding(face_crop: np.ndarray) -> torch.Tensor:
    if VAL_TRANSFORM is None or MODEL is None:
        raise HTTPException(503, detail="Verification model is not initialized yet")

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with MODEL_LOCK:
        with torch.no_grad():
            embedding = MODEL(tensor, mode="metric")
    return embedding


def _process_blink_sequence(frames: List[np.ndarray]) -> Dict[str, Any]:
    detector = BlinkDetector(ear_threshold=BLINK_THRESHOLD)
    start_time = time.time()
    succeeded = 0
    min_ear = float("inf")

    for frame in frames:
        if frame is None:
            continue
        with FACE_MESH_LOCK:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh_detector.process(rgb)
        if not result or not result.multi_face_landmarks:
            continue
        landmarks = result.multi_face_landmarks[0]
        blink_detected, avg_ear, _ = detector.detect_blink(landmarks, timestamp=time.time())
        succeeded += 1
        min_ear = min(min_ear, avg_ear if avg_ear > 0 else min_ear)

    has_blinked, blinks_needed = detector.requires_blink(start_time, time.time(), min_blinks=1)
    blink_score = 1.0 if has_blinked else (0.5 if succeeded > 0 else 0.0)
    return {
        "has_blinked": has_blinked,
        "blink_score": blink_score,
        "frames_processed": succeeded,
        "min_ear": min_ear if min_ear != float("inf") else None,
        "blinks_needed": blinks_needed,
    }


def _serialize_dataframe(df) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    serializable = df.copy()
    serializable["timestamp"] = serializable["timestamp"].astype(str)
    return serializable.to_dict("records")


@app.on_event("startup")
def _startup() -> None:
    global MODEL, VAL_TRANSFORM, EMPLOYEE_DB, THRESHOLD_VALUE
    _, VAL_TRANSFORM = get_transforms()
    MODEL = load_verification_model(MODEL_METRIC_PATH, DEVICE)
    EMPLOYEE_DB = load_employee_db(EMPLOYEE_DB_PATH)
    THRESHOLD_VALUE = load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
    refresh_employee_index()


@app.get("/health")
def health() -> Dict[str, Any]:
    employees = len(EMPLOYEE_DB)
    return {
        "model_loaded": MODEL is not None,
        "employees": employees,
        "threshold": THRESHOLD_VALUE,
        "attendance_log": str(OUTPUT_DIR / "attendance_log.csv"),
    }


@app.post("/verify")
def verify(request: VerifyRequest) -> Dict[str, Any]:
    frame = _decode_base64_image(request.image_b64)
    face_crop = _get_primary_face(frame)
    embedding = _prepare_embedding(face_crop)

    if EMPLOYEE_INDEX is None:
        raise HTTPException(400, detail="No employees registered yet")

    best_name, min_distance, all_distances = verify_embedding_fast(embedding)
    threshold = round(request.threshold or THRESHOLD_VALUE, 4)
    confidence = max(0.0, min(100.0, (1.0 - (min_distance / threshold)) * 100.0)) if threshold > 0 else 0.0
    matched = best_name if min_distance < threshold else None

    blink_details = None
    if request.blink_sequence:
        frames = [_decode_base64_image(b64) for b64 in request.blink_sequence]
        blink_details = _process_blink_sequence(frames)
    else:
        blink_details = {"has_blinked": False, "blink_score": 0.0, "frames_processed": 0, "blinks_needed": 1}

    liveness_status = "Real" if blink_details["blink_score"] >= 0.5 else "Spoof"

    if request.mark_attendance and matched:
        attendance_logger.mark_attendance(matched, min_distance, "Unknown", liveness_status)

    return {
        "identity": matched or "Not Registered",
        "distance": float(min_distance),
        "confidence": float(confidence),
        "threshold": float(threshold),
        "liveness": liveness_status,
        "blink": blink_details,
        "all_distances": [
            {"name": name, "distance": float(dist)} for name, dist in sorted(all_distances, key=lambda x: x[1])[:5]
        ],
    }


@app.post("/register")
def register(request: RegisterRequest) -> Dict[str, Any]:
    name = request.name.strip()
    if name in EMPLOYEE_DB and not request.replace_existing:
        raise HTTPException(400, detail="Employee already exists (set replace_existing to true to override)")

    embeddings = []
    for image_b64 in request.images_b64:
        frame = _decode_base64_image(image_b64)
        face_crop = _get_primary_face(frame)
        embedding = _prepare_embedding(face_crop)
        embeddings.append(embedding.squeeze(0).detach().cpu())

    EMPLOYEE_DB[name] = embeddings
    save_employee_db(EMPLOYEE_DB, EMPLOYEE_DB_PATH)
    refresh_employee_index()

    return {"success": True, "name": name, "poses": len(embeddings)}


@app.get("/employees")
def list_employees() -> Dict[str, Any]:
    return {"count": len(EMPLOYEE_DB), "employees": sorted(EMPLOYEE_DB.keys())}


@app.get("/attendance/today")
def attendance_today() -> List[Dict[str, Any]]:
    return _serialize_dataframe(attendance_logger.get_today_records())


@app.get("/attendance/summary")
def attendance_summary() -> Dict[str, Any]:
    return attendance_logger.generate_daily_summary()


@app.post("/attendance/mark")
def mark_attendance(payload: AttendanceRecordRequest) -> Dict[str, Any]:
    success, message = attendance_logger.mark_attendance(
        payload.name, payload.distance, payload.emotion, payload.liveness
    )
    return {"success": success, "message": message}


@app.get("/threshold")
def get_threshold() -> Dict[str, float]:
    return {"threshold": THRESHOLD_VALUE}


@app.post("/threshold")
def update_threshold(payload: ThresholdPayload) -> Dict[str, float]:
    global THRESHOLD_VALUE
    THRESHOLD_VALUE = payload.threshold
    save_gui_threshold(THRESHOLD_VALUE)
    return {"threshold": THRESHOLD_VALUE}
