"""FastAPI backend for the face recognition attendance service."""
from __future__ import annotations

import base64
import re
import threading
import time
from typing import Any, Dict, List, Optional

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
LAST_DB_MTIME: float = 0.0  # Track database modification time

# Face change detection: Store last verified embedding to detect when a different person appears
LAST_VERIFIED_EMBEDDING: Optional[torch.Tensor] = None
LAST_VERIFIED_IDENTITY: Optional[str] = None
FACE_CHANGE_THRESHOLD: float = 0.6  # If embedding distance > this, it's a different face

# Rejection settings for unregistered users (ported from desktop)
CONFIDENCE_REJECTION_THRESHOLD: float = 0.05  # 5% minimum confidence to accept
SMALL_DB_THRESHOLD_DIVISOR: float = 1.2  # Make threshold stricter for DBs with ≤2 users (0.90/1.2=0.75)
SMALL_DB_MAX_SIZE: int = 2  # Apply stricter threshold when DB has this many or fewer users

# Identity smoothing: Track recent recognitions to stabilize identity (like desktop)
RECOGNITION_HISTORY: List[Dict[str, Any]] = []  # [{identity, confidence, timestamp}, ...]
RECOGNITION_HISTORY_SIZE: int = 10  # Keep last N recognitions
SMOOTHING_MIN_COUNT: int = 3  # Require at least 3 consistent frames to accept identity
SMOOTHING_WEIGHT_BOOST: float = 1.3  # Boost weight for registered identities

# Verification lock: Prevent flickering after successful check-in
VERIFICATION_LOCK: Dict[str, Any] = {
    "identity": None,
    "locked_at": 0.0,
    "confidence": 0.0,
    "thumbnail_b64": None,
}
LOCK_DURATION: float = 10.0  # seconds to hold lock (extended to reduce flicker)
MAJORITY_THRESHOLD: float = 0.6  # 60% of votes required
VOTING_WINDOW: float = 2.0  # seconds of history to consider
LOCK_STATE_LOCK = threading.Lock()  # Lock for verification lock state

FACE_MESH_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
EMBEDDING_LOCK = threading.Lock()  # Lock for last embedding state
HISTORY_LOCK = threading.Lock()  # Lock for recognition history

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


def check_and_reload_db_if_changed() -> bool:
    """Check if the database file has changed and reload if necessary."""
    global EMPLOYEE_DB, LAST_DB_MTIME
    
    if not EMPLOYEE_DB_PATH.exists():
        return False
        
    try:
        current_mtime = EMPLOYEE_DB_PATH.stat().st_mtime
        if current_mtime > LAST_DB_MTIME:
            print(f"[DB] Detected database change. Reloading...")
            EMPLOYEE_DB = load_employee_db(EMPLOYEE_DB_PATH)
            LAST_DB_MTIME = current_mtime
            refresh_employee_index()
            print(f"[DB] Reloaded {len(EMPLOYEE_DB)} employees. Index updated.")
            return True
    except Exception as e:
        print(f"[DB] Error checking for reload: {e}")
    
    return False


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


class PoseValidateRequest(BaseModel):
    image_b64: str
    target_pose: str = Field(..., description="Target pose: 'center', 'left', or 'right'")


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


def _select_primary_face(frame: np.ndarray, faces=None):
    candidates = faces if faces is not None else detect_faces(frame)
    if not candidates:
        raise HTTPException(400, detail="No faces detected in the provided image")
    primary_idx = max(range(len(candidates)), key=lambda idx: candidates[idx][2] * candidates[idx][3])
    primary_bbox = candidates[primary_idx]
    cropped = crop_face_with_padding(frame, *primary_bbox, padding_ratio=0.2)
    if cropped.size == 0:
        raise HTTPException(400, detail="Failed to crop detected face")
    return cropped, candidates, primary_idx


def _get_primary_face(frame: np.ndarray) -> np.ndarray:
    cropped, _, _ = _select_primary_face(frame)
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
    """
    Fast blink detection optimized for real-time performance.
    Uses simple EAR variance detection with lenient natural blink thresholds.
    """
    if not frames:
        return {"has_blinked": False, "blink_score": 0.0, "frames_processed": 0, "min_ear": None, "blinks_needed": 1}
    
    # Subsample: Take 8 frames to better capture quick blinks
    num_samples = min(8, len(frames))
    step = max(1, len(frames) // num_samples)
    sampled_frames = [frames[i] for i in range(0, len(frames), step)][:num_samples]
    
    ear_values = []
    
    for frame in sampled_frames:
        if frame is None:
            continue
        try:
            with FACE_MESH_LOCK:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh_detector.process(rgb)
            if not result or not result.multi_face_landmarks:
                continue
            
            # Fast inline EAR calculation (avoids BlinkDetector overhead)
            landmarks = result.multi_face_landmarks[0].landmark
            
            # Left eye EAR (vertical / horizontal ratio)
            left_v = (abs(landmarks[159].y - landmarks[145].y) + abs(landmarks[158].y - landmarks[153].y)) / 2
            left_h = abs(landmarks[33].x - landmarks[133].x)
            left_ear = left_v / left_h if left_h > 0.001 else 0
            
            # Right eye EAR
            right_v = (abs(landmarks[386].y - landmarks[374].y) + abs(landmarks[385].y - landmarks[380].y)) / 2
            right_h = abs(landmarks[362].x - landmarks[263].x)
            right_ear = right_v / right_h if right_h > 0.001 else 0
            
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)
                
        except Exception as e:
            print(f"[BLINK] Frame processing error: {e}")
            continue
    
    if not ear_values:
        return {"has_blinked": False, "blink_score": 0.0, "frames_processed": 0, "min_ear": None, "max_ear": None, "current_ear": None, "blinks_needed": 1}
    
    min_ear = min(ear_values)
    max_ear = max(ear_values)
    current_ear = ear_values[-1]
    
    # More lenient thresholds for natural blinks
    # User's typical open eyes: ~0.35-0.45
    # A natural blink should drop EAR by at least 0.08-0.10
    EAR_THRESHOLD = 0.30  # Eye considered closed below this (raised from 0.22)
    EAR_OPEN = 0.36  # Eye considered open above this (raised from 0.28)
    
    # Detect blink: eyes must have closed (min < threshold) AND been open (max > open threshold)
    has_blinked = min_ear < EAR_THRESHOLD and max_ear > EAR_OPEN
    
    # Blink score for progressive feedback (more lenient)
    if has_blinked:
        blink_score = 1.0
    elif min_ear < 0.32 and max_ear > 0.34:
        blink_score = 0.7  # Very close to blink
    elif min_ear < 0.34:
        blink_score = 0.4  # Partial closure
    else:
        blink_score = 0.0
    
    print(f"[BLINK] {len(ear_values)} frames. EAR: {min_ear:.3f}-{max_ear:.3f}, Blinked: {has_blinked}")
    
    return {
        "has_blinked": bool(has_blinked),
        "blink_score": float(blink_score),
        "frames_processed": len(ear_values),
        "min_ear": float(min_ear),
        "max_ear": float(max_ear),
        "current_ear": float(current_ear),
        "blinks_needed": 0 if has_blinked else 1,
        "threshold": float(EAR_THRESHOLD),
    }


def _serialize_dataframe(df) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    serializable = df.copy()
    serializable["timestamp"] = serializable["timestamp"].astype(str)
    return serializable.to_dict("records")


def _smooth_identity(raw_identity: str, raw_confidence: float, raw_distance: float) -> tuple:
    """
    Apply temporal smoothing to identity recognition (like desktop's smooth_identity).
    Uses weighted voting from recent recognition history.
    
    Returns: (smoothed_identity, smoothed_confidence, was_smoothed)
    """
    global RECOGNITION_HISTORY
    
    current_time = time.time()
    
    with HISTORY_LOCK:
        # Add current recognition to history
        RECOGNITION_HISTORY.append({
            "identity": raw_identity,
            "confidence": raw_confidence,
            "distance": raw_distance,
            "timestamp": current_time,
        })
        
        # Trim to max size
        if len(RECOGNITION_HISTORY) > RECOGNITION_HISTORY_SIZE:
            RECOGNITION_HISTORY = RECOGNITION_HISTORY[-RECOGNITION_HISTORY_SIZE:]
        
        # Need at least 2 entries to smooth
        if len(RECOGNITION_HISTORY) < 2:
            return raw_identity, raw_confidence, False
        
        # Weighted voting
        identity_scores: Dict[str, Dict[str, float]] = {}
        
        for entry in RECOGNITION_HISTORY:
            hist_id = entry["identity"]
            hist_conf = entry["confidence"]
            
            # Weight by confidence
            weight = hist_conf / 100.0
            
            # Boost registered identities (not "Unknown" / "Not Registered" / None)
            if hist_id and hist_id not in ("Not Registered", "Not Registered (Low Confidence)", "Unknown", "Spoof Detected"):
                weight *= SMOOTHING_WEIGHT_BOOST
            
            if hist_id not in identity_scores:
                identity_scores[hist_id] = {"weight": 0.0, "count": 0, "best_conf": 0.0}
            
            identity_scores[hist_id]["weight"] += weight
            identity_scores[hist_id]["count"] += 1
            identity_scores[hist_id]["best_conf"] = max(identity_scores[hist_id]["best_conf"], hist_conf)
        
        # Find best identity
        best_identity = None
        best_score = 0.0
        best_conf = 0.0
        
        for identity, data in identity_scores.items():
            # Require minimum count (remove high-confidence bypass to prevent single-frame false positives)
            if data["count"] >= SMOOTHING_MIN_COUNT:
                if data["weight"] > best_score:
                    best_score = data["weight"]
                    best_identity = identity
                    best_conf = data["best_conf"]
        
        # If smoothed identity is valid registered user, use it
        if best_identity and best_identity not in ("Not Registered", "Not Registered (Low Confidence)", "Unknown", "Spoof Detected", None):
            was_smoothed = (best_identity != raw_identity)
            if was_smoothed:
                print(f"[SMOOTH] Corrected '{raw_identity}' -> '{best_identity}' (score: {best_score:.2f}, count: {identity_scores[best_identity]['count']})")
            return best_identity, best_conf, was_smoothed
        
        return raw_identity, raw_confidence, False


def _check_majority_lock(current_identity: str, has_blinked: bool) -> tuple:
    """
    Check if identity should be locked based on majority voting over time window.
    
    Returns: (should_lock, majority_identity, vote_percentage)
    """
    if not current_identity or current_identity in ("Unknown", "Not Registered", "Spoof Detected"):
        return False, None, 0.0
    
    current_time = time.time()
    
    with HISTORY_LOCK:
        # Filter history to voting window
        recent = [h for h in RECOGNITION_HISTORY if current_time - h["timestamp"] <= VOTING_WINDOW]
        
        if len(recent) < 3:  # Need minimum samples
            return False, None, 0.0
        
        # Count votes per identity
        votes: Dict[str, int] = {}
        for h in recent:
            identity = h.get("identity")
            if identity and identity not in ("Unknown", "Not Registered", "Spoof Detected"):
                votes[identity] = votes.get(identity, 0) + 1
        
        if not votes:
            return False, None, 0.0
        
        # Find majority
        total_valid = sum(votes.values())
        best_identity = max(votes, key=votes.get)
        best_count = votes[best_identity]
        percentage = best_count / total_valid if total_valid > 0 else 0
        
        # Lock if majority threshold met AND blink passed
        should_lock = percentage >= MAJORITY_THRESHOLD and has_blinked
        
        if should_lock:
            print(f"[LOCK] Majority confirmed: '{best_identity}' with {percentage:.1%} ({best_count}/{total_valid})")
        
        return should_lock, best_identity, percentage


def _clear_verification_lock(reason: str = ""):
    """Clear the verification lock and related state."""
    global VERIFICATION_LOCK, RECOGNITION_HISTORY
    
    with LOCK_STATE_LOCK:
        old_identity = VERIFICATION_LOCK.get("identity")
        VERIFICATION_LOCK = {"identity": None, "locked_at": 0.0, "confidence": 0.0, "thumbnail_b64": None}
    
    with HISTORY_LOCK:
        RECOGNITION_HISTORY.clear()
    
    if old_identity:
        print(f"[LOCK] Cleared lock for '{old_identity}' ({reason})")


@app.on_event("startup")
def _startup() -> None:
    global MODEL, VAL_TRANSFORM, EMPLOYEE_DB, THRESHOLD_VALUE
    _, VAL_TRANSFORM = get_transforms()
    MODEL = load_verification_model(MODEL_METRIC_PATH, DEVICE)
    EMPLOYEE_DB = load_employee_db(EMPLOYEE_DB_PATH)
    if EMPLOYEE_DB_PATH.exists():
        LAST_DB_MTIME = EMPLOYEE_DB_PATH.stat().st_mtime
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
def verify(request: VerifyRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    frame = _decode_base64_image(request.image_b64)
    frame_height, frame_width = frame.shape[:2]
    faces = detect_faces(frame)
    
    # Check if no faces → clear lock and return error
    if not faces:
        _clear_verification_lock("no face detected")
        print(f"[VERIFY] No faces detected in frame ({frame_width}x{frame_height})")
        raise HTTPException(400, detail="No faces detected in the provided image")
    
    # Check if identity is locked → return cached result immediately
    current_time = time.time()
    with LOCK_STATE_LOCK:
        if VERIFICATION_LOCK.get("identity") and (current_time - VERIFICATION_LOCK["locked_at"]) < LOCK_DURATION:
            locked_identity = VERIFICATION_LOCK["identity"]
            locked_confidence = VERIFICATION_LOCK["confidence"]
            print(f"[VERIFY] Returning locked identity: '{locked_identity}'")
            
            # Still return face detections but with locked identity
            return {
                "identity": locked_identity,
                "distance": 0.0,
                "confidence": float(locked_confidence),
                "threshold": float(THRESHOLD_VALUE),
                "liveness": "Real",
                "locked": True,
                "blink": {"has_blinked": True, "blink_score": 1.0, "frames_processed": 0, "blinks_needed": 0, "min_ear": None, "max_ear": None, "current_ear": None, "threshold": 0.22},
                "all_distances": [],
                "detections": [
                    {
                        "bbox": {"x": max(0.0, min(1.0, x / frame_width)), "y": max(0.0, min(1.0, y / frame_height)), "width": max(0.0, min(1.0, w / frame_width)), "height": max(0.0, min(1.0, h / frame_height))},
                        "is_primary": idx == 0,
                        "identity": locked_identity if idx == 0 else None,
                        "confidence": locked_confidence if idx == 0 else None,
                        "liveness": "Real" if idx == 0 else "Unknown",
                    }
                    for idx, (x, y, w, h) in enumerate(faces)
                ],
            }
        
    print(f"[VERIFY] Detected {len(faces)} faces")
    
    face_crop, faces, primary_idx = _select_primary_face(frame, faces)
    embedding = _prepare_embedding(face_crop)

    print(f"[VERIFY] Processing verification request. Blink sequence: {len(request.blink_sequence) if request.blink_sequence else 0} frames")
    
    # Handle case when no employees are registered
    if EMPLOYEE_INDEX is None:
        print("[VERIFY] No employees registered - returning Unregistered")
        threshold = round(request.threshold or THRESHOLD_VALUE, 4)
        blink_details = {"has_blinked": False, "blink_score": 1.0, "frames_processed": 0, "blinks_needed": 0}
        return {
            "identity": "Unregistered",
            "distance": 1.0,
            "confidence": 0.0,
            "threshold": float(threshold),
            "liveness": "Real",
            "blink": blink_details,
            "all_distances": [],
            "detections": [
                {
                    "bbox": {
                        "x": max(0.0, min(1.0, x / frame_width)),
                        "y": max(0.0, min(1.0, y / frame_height)),
                        "width": max(0.0, min(1.0, w / frame_width)),
                        "height": max(0.0, min(1.0, h / frame_height)),
                    },
                    "is_primary": idx == primary_idx,
                    "identity": "Unregistered" if idx == primary_idx else None,
                    "confidence": 0.0 if idx == primary_idx else None,
                    "liveness": "Real" if idx == primary_idx else "Unknown",
                }
                for idx, (x, y, w, h) in enumerate(faces)
            ],
        }

    best_name, min_distance, all_distances = verify_embedding_fast(embedding)
    threshold = round(request.threshold or THRESHOLD_VALUE, 4)
    
    # Adjust threshold for small databases (stricter to prevent false positives)
    db_size = len(EMPLOYEE_DB)
    effective_threshold = threshold
    if db_size <= SMALL_DB_MAX_SIZE:
        effective_threshold = threshold / SMALL_DB_THRESHOLD_DIVISOR
        print(f"[VERIFY] Small DB ({db_size} users) - using stricter threshold: {effective_threshold:.3f}")
    
    confidence = max(0.0, min(100.0, (1.0 - (min_distance / threshold)) * 100.0)) if threshold > 0 else 0.0
    
    # EARLY CHECK: If user already checked in today, bypass strict threshold
    # This prevents flicker when distance temporarily exceeds threshold for verified users
    already_verified_today = best_name and attendance_logger.has_checked_in_today(best_name)
    if already_verified_today:
        print(f"[VERIFY] User '{best_name}' already verified today - bypassing strict threshold")
        matched = best_name
        # Skip distance gating entirely for already-verified users
    else:
        # GATE 1: Only accept if distance is below EFFECTIVE threshold
        raw_matched = best_name if min_distance < effective_threshold else None
        
        # GATE 2: Reject low confidence matches
        if raw_matched and confidence < CONFIDENCE_REJECTION_THRESHOLD * 100:
            print(f"[VERIFY] Rejecting '{raw_matched}' - low confidence: {confidence:.1f}%")
            raw_matched = None
        
        smoothed_identity, smoothed_confidence, was_smoothed = _smooth_identity(
            raw_matched if raw_matched else "Unknown",
            confidence if raw_matched else 0.0,
            min_distance
        )
        
        # Final identity: use smoothed result (which now only contains gated entries)
        if smoothed_identity and smoothed_identity not in ("Unknown", "Not Registered"):
            matched = smoothed_identity
            confidence = smoothed_confidence
        else:
            matched = raw_matched
    
    print(f"[VERIFY] Raw: '{best_name}' dist={min_distance:.4f}, eff_thresh={effective_threshold:.3f}, matched: '{matched}' (verified_today={already_verified_today})")

    # Face change detection: check if current face is different from last verified
    global LAST_VERIFIED_EMBEDDING, LAST_VERIFIED_IDENTITY, RECOGNITION_HISTORY
    face_changed = False
    
    with EMBEDDING_LOCK:
        if LAST_VERIFIED_EMBEDDING is not None and matched:
            # Compare current embedding with last verified
            emb_distance = torch.cdist(embedding, LAST_VERIFIED_EMBEDDING, p=2)[0, 0].item()
            if emb_distance > FACE_CHANGE_THRESHOLD:
                face_changed = True
                print(f"[FACE-CHANGE] Detected new face! Embedding distance: {emb_distance:.4f} > {FACE_CHANGE_THRESHOLD}")
                # Reset cache for the previous identity to force blink re-verification
                if LAST_VERIFIED_IDENTITY:
                    attendance_logger.reset_liveness_cache(LAST_VERIFIED_IDENTITY)
                    print(f"[FACE-CHANGE] Reset cache for '{LAST_VERIFIED_IDENTITY}'")
                LAST_VERIFIED_EMBEDDING = None
                LAST_VERIFIED_IDENTITY = None
                # Clear recognition history when face changes
                with HISTORY_LOCK:
                    RECOGNITION_HISTORY.clear()
                    print("[FACE-CHANGE] Cleared recognition history")

    blink_details = None
    liveness_status = "Unknown"
    identity_to_mark = None  # Track who to mark attendance for
    
    # Determine if blink is REQUIRED (for attendance) vs just for debug
    blink_required = False
    
    if matched:
        if attendance_logger.has_checked_in_today(matched):
            # User already checked in today - skip blink requirement, instant Real
            liveness_status = "Real"
            blink_details = {"has_blinked": False, "blink_score": 1.0, "frames_processed": 0, "blinks_needed": 0, "min_ear": None, "max_ear": None, "current_ear": None, "threshold": 0.25}
            print(f"[VERIFY] User '{matched}' already checked in today. Skipping blink detection.")
        else:
            # User matched but not checked in - REQUIRE blink verification
            blink_required = True
    else:
        # User not matched - don't require blink for security, but still process for debug
        # We'll process blink below for debug visualization
        pass
    
    # ALWAYS process blink if we have frames (for debug visualization)
    if request.blink_sequence and blink_details is None:
        frames = [_decode_base64_image(b64) for b64 in request.blink_sequence]
        blink_details = _process_blink_sequence(frames)
        print(f"[DEBUG] Blink processed: score={blink_details['blink_score']}, matched={matched}, blink_required={blink_required}")
        
        # Determine liveness status based on whether blink was required
        if blink_required:
            # User is matched and hasn't checked in - REQUIRE blink
            liveness_status = "Real" if blink_details["blink_score"] >= 0.5 else "Spoof"
            if liveness_status == "Real":
                identity_to_mark = matched
                print(f"[DEBUG] Set identity_to_mark = {identity_to_mark}")
        else:
            # User not matched - show EAR debug but don't require blink for security
            # Liveness is "Real" (no security concern), but still set blinks_needed for UI feedback
            liveness_status = "Real"
            # Keep blinks_needed from the actual detection for debug purposes
    elif blink_required and blink_details is None:
        # Need blink but no sequence provided yet
        liveness_status = "Spoof"
        blink_details = {"has_blinked": False, "blink_score": 0.0, "frames_processed": 0, "blinks_needed": 1, "min_ear": None, "max_ear": None, "current_ear": None, "threshold": 0.25}
    
    # Mark attendance BEFORE nullifying matched (for blink pass case)
    print(f"[DEBUG] Checking attendance: identity_to_mark={identity_to_mark}, mark_attendance={request.mark_attendance}")
    
    # Check if we should lock the identity (majority voting passed + blink passed)
    has_blinked = blink_details.get("has_blinked", False) if blink_details else False
    should_lock, majority_identity, vote_pct = _check_majority_lock(matched, has_blinked)
    
    if identity_to_mark and request.mark_attendance:
        print(f"[ATTENDANCE] Marking attendance for {identity_to_mark} after blink verification")
        background_tasks.add_task(attendance_logger.mark_attendance, identity_to_mark, min_distance, "Unknown", liveness_status)
        
        # Store embedding for face change detection on next request
        with EMBEDDING_LOCK:
            LAST_VERIFIED_EMBEDDING = embedding.clone().detach()
            LAST_VERIFIED_IDENTITY = identity_to_mark
            print(f"[FACE-TRACK] Stored embedding for '{identity_to_mark}'")
        
        # Apply verification lock if majority confirmed
        if should_lock and majority_identity:
            with LOCK_STATE_LOCK:
                VERIFICATION_LOCK["identity"] = majority_identity
                VERIFICATION_LOCK["locked_at"] = time.time()
                VERIFICATION_LOCK["confidence"] = confidence
                print(f"[LOCK] Locked identity: '{majority_identity}' for {LOCK_DURATION}s (majority: {vote_pct:.1%})")
    
    # Mask identity for Spoof responses
    if liveness_status == "Spoof":
        matched = None

    return {
        "identity": matched if liveness_status == "Real" else None,
        "distance": float(min_distance),
        "confidence": float(confidence),
        "threshold": float(threshold),
        "liveness": liveness_status,
        "locked": should_lock,
        "blink": blink_details,
        "all_distances": [
            {"name": name, "distance": float(dist)} for name, dist in sorted(all_distances, key=lambda x: x[1])[:5]
        ],
        "detections": [
            {
                "bbox": {
                    "x": max(0.0, min(1.0, x / frame_width)),
                    "y": max(0.0, min(1.0, y / frame_height)),
                    "width": max(0.0, min(1.0, w / frame_width)),
                    "height": max(0.0, min(1.0, h / frame_height)),
                },
                "is_primary": idx == primary_idx,
                # Only show identity when liveness is Real (not during "blink to verify" state)
                "identity": matched if idx == primary_idx and liveness_status == "Real" else None,
                "confidence": float(confidence) if idx == primary_idx and matched and liveness_status == "Real" else None,
                "liveness": liveness_status if idx == primary_idx else "Unknown",
            }
            for idx, (x, y, w, h) in enumerate(faces)
        ],
    }


@app.post("/verify/reset")
def verify_reset() -> Dict[str, Any]:
    """Reset verification lock for next person."""
    _clear_verification_lock("manual reset")
    return {"success": True, "message": "Verification lock cleared"}


@app.post("/register")
def register(request: RegisterRequest) -> Dict[str, Any]:
    name = request.name.strip()
    print(f"[REGISTER] Received registration request for '{name}' with {len(request.images_b64)} images")
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
    # Check for external updates before listing
    check_and_reload_db_if_changed()
    return {"count": len(EMPLOYEE_DB), "employees": sorted(EMPLOYEE_DB.keys())}


@app.delete("/employees/{name}")
def delete_employee(name: str) -> Dict[str, Any]:
    """Delete an employee from the database."""
    if name not in EMPLOYEE_DB:
        raise HTTPException(404, detail=f"Employee '{name}' not found")
    
    del EMPLOYEE_DB[name]
    save_employee_db(EMPLOYEE_DB, EMPLOYEE_DB_PATH)
    refresh_employee_index()
    
    print(f"[DELETE] Removed employee '{name}'")
    return {"success": True, "name": name, "remaining": len(EMPLOYEE_DB)}


@app.post("/liveness/reset")
def reset_liveness_cache(employee_name: str = None) -> Dict[str, Any]:
    """
    Reset the liveness/attendance cache for testing or multi-user scenarios.
    
    Args:
        employee_name: Optional - reset only this employee. If None, reset all.
    
    Returns:
        Success status and count of cleared entries.
    """
    success, message, count = attendance_logger.reset_liveness_cache(employee_name)
    return {
        "success": success,
        "message": message,
        "cleared_count": count
    }


@app.post("/pose/validate")
def validate_pose(request: PoseValidateRequest) -> Dict[str, Any]:
    """Validate if the user's head pose matches the target pose."""
    from src.utils import estimate_head_pose_angles, validate_pose_for_target
    
    frame = _decode_base64_image(request.image_b64)
    faces = detect_faces(frame)
    
    if not faces:
        return {
            "valid": False,
            "detected_pose": "no_face",
            "target_pose": request.target_pose,
            "feedback": "No face detected. Please position your face in the camera.",
            "yaw": 0.0,
            "pitch": 0.0,
        }
    
    # Get pose angles
    yaw, pitch, roll, detected_label, _, _ = estimate_head_pose_angles(frame)
    
    # Debug: Log raw pose angles
    print(f"[POSE] Raw angles - Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}, Detected: {detected_label}")
    
    # Map target_pose to expected format
    target = request.target_pose.lower()
    
    # Align with Python GUI logic from utils.py:
    # pose_label = "left" if yaw < 0 else "right"
    # So: yaw < 0 = LEFT, yaw > 0 = RIGHT
    # Threshold in Python GUI: abs(yaw) > 20
    
    valid = False
    feedback = "Adjust your pose"
    
    if target == "center":
        # Accept if yaw is within ±15 degrees (slightly lenient)
        valid = abs(yaw) <= 15
        feedback = "✓ Hold steady!" if valid else "Look straight at camera"
    elif target == "left":
        # Python GUI: yaw < 0 = left
        # Accept yaw between -20 and -60 degrees
        valid = yaw < -15 and yaw > -60
        if not valid:
            if yaw >= -15:
                feedback = "Turn more to your RIGHT"
            elif yaw <= -60:
                feedback = "Too far right, come back a bit"
    elif target == "right":
        # Python GUI: yaw > 0 = right
        # Accept yaw between 20 and 60 degrees
        valid = yaw > 15 and yaw < 60
        if not valid:
            if yaw <= 15:
                feedback = "Turn more to your LEFT"
            elif yaw >= 60:
                feedback = "Too far left, come back a bit"
    
    face_image_b64 = None
    if valid:
        feedback = "✓ Hold steady!"
        # Return the cropped face so frontend can display exactly what is captured
        try:
            cropped, _, _ = _select_primary_face(frame, faces)
            _, buffer = cv2.imencode('.jpg', cropped)
            face_image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[POSE] Failed to crop face for display: {e}")
    
    print(f"[POSE] Target: {target}, Yaw: {yaw:.1f}, Valid: {valid}, Feedback: {feedback}")
    
    return {
        "valid": valid,
        "detected_pose": detected_label,
        "target_pose": target,
        "feedback": feedback,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "face_image": face_image_b64
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
