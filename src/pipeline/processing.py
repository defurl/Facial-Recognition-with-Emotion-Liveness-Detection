"""Processing helpers for per-face verification orchestration."""
from __future__ import annotations

from typing import Any, Dict

import time

import cv2
import torch
from PIL import Image

from src.config import (
    DEVICE,
    OPTIMAL_THRESHOLD_GUI,
    UNRECOGNIZED_DISTANCE_MULTIPLIER,
)
from src.pipeline.verification import verify_embedding_fast
from src.runtime.settings import load_gui_threshold


def verify_face(
    cropped_face_resized,
    verification_model,
    val_transform,
    employee_db,
    *,
    face_idx: int,
    frame_count: int,
    total_faces: int,
    is_primary_face: bool,
) -> Dict[str, Any]:
    """Run embedding extraction + indexed verification for one face.

    Returns a dict containing embeddings, distances, thresholds, and timing.
    """
    # Avoid disk I/O: convert cv2 image to PIL directly
    preprocess_start = time.time()
    rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).convert("RGB")
    image_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)
    preprocess_time = (time.time() - preprocess_start) * 1000

    embedding_start = time.time()
    with torch.no_grad():
        trial_embedding = verification_model(image_tensor, mode="metric")
    embedding_time = (time.time() - embedding_start) * 1000

    current_threshold = load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
    adjusted_threshold = current_threshold
    if len(employee_db) <= 2:
        adjusted_threshold = current_threshold * UNRECOGNIZED_DISTANCE_MULTIPLIER

    comparison_start = time.time()
    best_match, min_distance, distance_results = verify_embedding_fast(trial_embedding)
    comparison_time = (time.time() - comparison_start) * 1000

    if best_match is None:
        min_distance = float("inf")
    best_match_data = employee_db.get(best_match)

    confidence = max(0, min(100, (1 - min_distance / current_threshold) * 100))

    # Periodic debug for primary face only
    if is_primary_face and frame_count % 30 == 0:
        print(f"\n=== IDENTITY DEBUG (Frame {frame_count}) ===")
        print(f"Best match: {best_match or 'None'}")
        print(f"Min distance: {min_distance:.4f}")
        print(f"Current threshold: {current_threshold:.4f}")
        print(f"Adjusted threshold: {adjusted_threshold:.4f} (small DB)")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Confidence rejection threshold: {100 * 0.05:.1f}%")
        print(f"Database size: {len(employee_db)} employees")
        print("All distances:")
        for name, dist in distance_results:
            conf = max(0, min(100, (1 - dist / current_threshold) * 100))
            status = "MATCH" if dist < adjusted_threshold else "REJECT"
            print(f"  {name}: dist={dist:.4f}, conf={conf:.1f}% [{status}]")
        print("=" * 50)

    if len(employee_db) <= 2 and is_primary_face and frame_count % 60 == 0:
        print(
            f"[THRESHOLD-WARNING] Small DB ({len(employee_db)} users) - Using strict threshold: {adjusted_threshold:.4f}"
        )

    return {
        "rgb": rgb,
        "image_tensor": image_tensor,
        "trial_embedding": trial_embedding,
        "best_match": best_match,
        "best_match_data": best_match_data,
        "min_distance": min_distance,
        "distance_results": distance_results,
        "current_threshold": current_threshold,
        "adjusted_threshold": adjusted_threshold,
        "confidence": confidence,
        "preprocess_ms": preprocess_time,
        "embedding_ms": embedding_time,
        "comparison_ms": comparison_time,
    }


def resolve_raw_identity(
    *,
    face_idx: int,
    faces_to_process_count: int,
    best_match: str,
    min_distance: float,
    adjusted_threshold: float,
    confidence: float,
    confidence_rejection_threshold: float,
    current_frame_verifications: list,
):
    """Determine raw identity with rejection and conflict checks."""
    if confidence < confidence_rejection_threshold * 100:
        return "Not Registered (Low Confidence)"

    if min_distance < adjusted_threshold:
        margin_to_threshold = adjusted_threshold - min_distance
        margin_ratio = margin_to_threshold / adjusted_threshold if adjusted_threshold else 0.0

        if faces_to_process_count > 1 and current_frame_verifications:
            for other in current_frame_verifications:
                if (
                    other.get("best_match") == best_match
                    and other.get("face_idx") != face_idx
                    and other.get("confidence", 0) > confidence + 10
                ):
                    return "Ambiguous Match"

        return best_match

    return "Not Registered"


def update_identity_lock(
    *,
    is_primary_face: bool,
    raw_identity: str,
    confidence: float,
    current_time: float,
    lock_state: dict,
    lock_duration: float,
    lock_display_time: float,
    lock_min_verifications: int,
    last_liveness: str,
    on_reset_spoof,
    on_mark_attendance,
    on_log_checkin,
):
    """Apply identity lock logic and return (frame_identity, box_color, last_identity_message).

    lock_state carries: locked_identity, lock_timestamp, identity_lock_buffer, last_identity, last_attendance_message.
    """
    locked_identity = lock_state.get("locked_identity")
    lock_timestamp = lock_state.get("lock_timestamp", 0)
    identity_lock_buffer = lock_state.setdefault("identity_lock_buffer", [])
    last_identity = lock_state.get("last_identity", "Not Registered")

    if not is_primary_face:
        return raw_identity, (0, 255, 0), last_identity

    if locked_identity is not None:
        if current_time - lock_timestamp > lock_display_time:
            lock_state["locked_identity"] = None
            lock_state["identity_lock_buffer"] = []
            lock_state["last_identity"] = "Not Registered"
            on_reset_spoof("identity lock timeout")
            return "Not Registered", (0, 0, 255), lock_state["last_identity"]
        return locked_identity, (0, 255, 0), locked_identity

    # Immediate lock on a recognized identity (no verifying countdown)
    if raw_identity not in ["Not Registered", "Not Registered (Low Confidence)", "Processing...", "Error"]:
        lock_state["locked_identity"] = raw_identity
        lock_state["lock_timestamp"] = current_time
        lock_state["last_identity"] = raw_identity
        on_mark_attendance(raw_identity, confidence)
        on_log_checkin(raw_identity)
        return raw_identity, (0, 255, 0), raw_identity

    lock_state["last_identity"] = "Not Registered"
    return "Not Registered", (0, 0, 255), lock_state["last_identity"]


def smooth_identity(
    *,
    raw_identity: str,
    confidence: float,
    face_history: list,
    recognition_history: list,
    smoothing_window: int,
    is_primary_face: bool,
):
    """Apply face-specific smoothing and primary-face fallback smoothing."""
    frame_identity = raw_identity

    if len(face_history) >= 2:
        identity_scores = {}
        for result in face_history:
            hist_id = result.get("identity")
            hist_conf = result.get("confidence", 0)
            weight = hist_conf / 100.0
            if hist_id not in ("Not Registered", "Not Registered (Low Confidence)"):
                weight *= 1.3
            if hist_id not in identity_scores:
                identity_scores[hist_id] = {"weight": 0, "count": 0}
            identity_scores[hist_id]["weight"] += weight
            identity_scores[hist_id]["count"] += 1

        best_identity = None
        best_score = 0
        for identity, data in identity_scores.items():
            avg_confidence = (data["weight"] / data["count"]) * 100
            if data["count"] >= 2 or avg_confidence > 75:
                if data["weight"] > best_score:
                    best_score = data["weight"]
                    best_identity = identity

        if best_identity and best_identity not in ["Not Registered", "Not Registered (Low Confidence)"]:
            frame_identity = best_identity

    if is_primary_face and frame_identity == raw_identity:
        recognition_history.append((raw_identity, confidence))
        if len(recognition_history) > smoothing_window:
            recognition_history.pop(0)

    return frame_identity


def update_face_lists(
    *,
    face_idx: int,
    frame_identity: str,
    confidence: float,
    is_primary_face: bool,
    last_emotion: str,
    last_liveness: str,
    face_identities: list,
    face_confidences: list,
    face_emotions: list,
    face_liveness: list,
    enable_multi_face_verification: bool,
    current_face_id: int,
    face_verification_history: dict,
    face_identities_verified: dict,
    face_confidences_verified: dict,
    current_time: float,
):
    """Ensure per-face arrays are sized and updated; track per-face history when enabled."""
    while len(face_identities) <= face_idx:
        face_identities.append("Processing...")
        face_confidences.append(0.0)
        face_emotions.append("Unknown")
        face_liveness.append("Unknown")

    face_identities[face_idx] = frame_identity
    face_confidences[face_idx] = confidence

    if is_primary_face:
        face_emotions[face_idx] = last_emotion
        face_liveness[face_idx] = last_liveness
    else:
        face_emotions[face_idx] = "N/A"
        face_liveness[face_idx] = "Verified" if confidence > 50 else "Unknown"

    if enable_multi_face_verification and current_face_id >= 0:
        face_identities_verified[current_face_id] = frame_identity
        face_confidences_verified[current_face_id] = confidence
        if current_face_id not in face_verification_history:
            face_verification_history[current_face_id] = []
        face_verification_history[current_face_id].append({
            "identity": frame_identity,
            "confidence": confidence,
            "timestamp": current_time,
        })
        if len(face_verification_history[current_face_id]) > 10:
            face_verification_history[current_face_id].pop(0)


def process_frame_shell(frame, *, detect_faces_cb, process_face_cb, update_ui_cb):
    """Orchestrate a single frame through detection, per-face processing, and UI update.

    The callbacks provide flexibility for UI/state management while keeping orchestration
    predictable. The detection callback can return either a list of face bboxes or a
    context dict containing ``faces`` plus any extra metadata. The processing callback
    receives the frame, the face index, the bbox, and the detection context. The UI
    callback receives the frame, the list of per-face results, and the detection context.
    """

    detection_ctx = detect_faces_cb(frame)

    # Normalize detection output to a context dict with a faces list
    if isinstance(detection_ctx, dict):
        faces = detection_ctx.get("faces", [])
        context = detection_ctx
        context.setdefault("faces", faces)
    else:
        faces = detection_ctx or []
        context = {"faces": faces}

    results = []
    for face_idx, bbox in enumerate(context["faces"]):
        try:
            result = process_face_cb(frame, face_idx, bbox, context)
        except Exception as exc:  # Keep the pipeline resilient to per-face errors
            result = {"error": exc, "face_idx": face_idx, "bbox": bbox}
        results.append(result)

    update_ui_cb(frame, results, context)
    return results
