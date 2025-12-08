"""Per-face processing helper to keep the GUI loop thin."""
from __future__ import annotations

import datetime
import threading
import time
from typing import Iterable, Tuple

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import (
    CONFIDENCE_REJECTION_THRESHOLD,
    IMG_SIZE,
    MAX_CONCURRENT_FACES,
    OPTIMAL_THRESHOLD_GUI,
    USE_MULTI_EMBEDDING,
)
from src.pipeline.processing import (
    resolve_raw_identity,
    smooth_identity,
    update_face_lists,
    update_identity_lock,
    verify_face,
)
from src.runtime.settings import load_gui_threshold
from src.utils import crop_face_with_padding, face_mesh_detector


def face_roles(face_idx: int, current_face_id: int, primary_face_id: int, enable_multi_face: bool) -> Tuple[bool, bool]:
    """Return (is_primary, is_secondary) for the current face."""
    if enable_multi_face:
        is_primary = current_face_id == primary_face_id
        is_secondary = (not is_primary and current_face_id >= 0)
    else:
        is_primary = face_idx == 0
        is_secondary = face_idx > 0
    return is_primary, is_secondary


def role_box_color(is_primary_face: bool, is_secondary_face: bool):
    """Color coding for primary/secondary/unknown faces."""
    if is_primary_face:
        return (0, 255, 0)
    if is_secondary_face:
        return (255, 165, 0)
    return (0, 0, 255)


def interpolate_color(color1, color2, factor):
    """Smoothly interpolate between two colors for transitions."""
    return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))


def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=15):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2

    if thickness < 0:  # Filled
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:  # Outline
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


class FaceProcessor:
    """Encapsulates per-face processing steps (landmarks, liveness, verify, UI)."""

    def __init__(self, gui):
        self.gui = gui

    def process_face(
        self,
        frame,
        face_idx: int,
        bbox: Tuple[int, int, int, int],
        faces: Iterable[Tuple[int, int, int, int]],
        faces_to_process: Iterable[Tuple[int, int, int, int]],
        face_assignments,
        verification_model,
        val_transform,
        employee_db,
    ):
        g = self.gui
        x, y, w, h = bbox
        is_processing_face = face_idx < MAX_CONCURRENT_FACES
        current_face_id = face_assignments.get(face_idx, -1)

        is_primary_face, is_secondary_face = face_roles(
            face_idx,
            current_face_id,
            g.primary_face_id,
            g.ENABLE_MULTI_FACE_VERIFICATION,
        )
        box_color = role_box_color(is_primary_face, is_secondary_face)

        if g.registration_mode:
            if face_idx == 0:
                g.last_identity = "Registering..."
                box_color = (255, 165, 0)
        elif is_primary_face and is_processing_face and g.frame_count % g.PROCESS_EVERY_N_FRAMES == 0:
            process_start = time.time()

            cropped_face = crop_face_with_padding(frame, x, y, w, h)
            if cropped_face.size > 0 and cropped_face.shape[0] >= 50:
                cropped_face_resized = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))

                # Landmarks
                face_landmarks = None
                if is_primary_face:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks_results = face_mesh_detector.process(frame_rgb)
                    face_landmarks = g.match_landmarks_to_face((x, y, w, h), landmarks_results)

                    if face_landmarks and g.frame_count % 30 == 0:
                        print(f"[LANDMARKS] ✓ Matched landmarks to primary face {face_idx}")
                    elif g.frame_count % 30 == 0:
                        print(f"[LANDMARKS] ✗ No landmark match for primary face {face_idx}")

                if face_landmarks is None:
                    rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    landmarks_results = face_mesh_detector.process(rgb_face)
                    if landmarks_results and landmarks_results.multi_face_landmarks:
                        face_landmarks = landmarks_results.multi_face_landmarks[0]
                        if g.frame_count % 30 == 0:
                            print(f"[LANDMARKS] ✓ Detected landmarks on cropped face {face_idx} (fallback)")
                    elif g.frame_count % 30 == 0:
                        print(f"[LANDMARKS] ✗ FAILED to detect landmarks for face {face_idx} (crop size: {cropped_face.shape})")

                # Early identity for spoof cache
                early_identity = None
                if is_primary_face:
                    try:
                        early_identity = g.liveness_adapter.compute_early_identity(
                            cropped_face_resized,
                            verification_model,
                            val_transform,
                            employee_db,
                            current_threshold=load_gui_threshold(OPTIMAL_THRESHOLD_GUI),
                            use_multi_embedding=USE_MULTI_EMBEDDING,
                        )
                        if early_identity and g.frame_count % 15 == 0:
                            print(f"[EARLY-ID] Detected {early_identity} for spoof cache check")
                    except Exception as exc:
                        if g.frame_count % 60 == 0:
                            print(f"[EARLY-ID] Error in early identity check: {exc}")

                # Liveness / emotion
                if is_primary_face and g.emotion_analysis_enabled and (
                    g.frame_count % g.EMOTION_EVERY_N_FRAMES == 0
                ):
                    try:
                        emotion = "Neutral"
                        is_live = False
                        liveness_confidence = 0.0
                        liveness_details = {"error": "No analysis performed"}

                        cache_hit_identity = g.liveness_adapter.pick_cached_identity(
                            early_identity,
                            g.session_spoof_passed,
                            employee_db,
                        )
                        if cache_hit_identity and g.frame_count % 30 == 0:
                            print(f"[SPOOF-CACHE] Cache hit: {cache_hit_identity}")

                        if cache_hit_identity:
                            is_live = True
                            liveness_confidence = 0.95
                            emotion = "Neutral"
                            liveness_details = {"method": "cached_spoof_pass", "cached_user": cache_hit_identity}
                            g.last_identity = cache_hit_identity
                        else:
                            if g.lightweight_liveness:
                                if g.verification_start_time is None:
                                    g.verification_start_time = time.time()

                                current_time = time.time()

                                if face_landmarks is not None:
                                    try:
                                        blink_detected, current_ear, total_blinks = g.blink_detector.detect_blink(face_landmarks)
                                        has_blinked, blinks_needed = g.blink_detector.requires_blink(
                                            g.verification_start_time, current_time, min_blinks=1
                                        )

                                        elapsed = current_time - g.verification_start_time

                                        if has_blinked:
                                            is_live = True
                                            liveness_confidence = 0.95
                                            emotion = "Neutral"
                                        elif elapsed < 10.0:
                                            is_live = None
                                            liveness_confidence = 0.5
                                            emotion = "Neutral"
                                        else:
                                            g.verification_start_time = time.time()
                                            if hasattr(g, "blink_detector") and g.blink_detector:
                                                g.blink_detector.reset()
                                                print("[BLINK] Reset after 10s timeout - fresh start")
                                            is_live = False
                                            liveness_confidence = 0.1
                                            emotion = "Neutral"

                                        liveness_details = {
                                            "method": "lightweight_blink_only",
                                            "blink": {
                                                "current_ear": current_ear,
                                                "total_blinks": total_blinks,
                                                "has_blinked": has_blinked,
                                                "blinks_needed": blinks_needed,
                                                "elapsed_time": elapsed,
                                            },
                                        }

                                        if blink_detected:
                                            print(
                                                f"[BLINK DETECTED!] Frame {len(g.face_trackers)}, Total blinks: {total_blinks}, EAR: {current_ear:.3f}"
                                            )

                                    except Exception as blink_error:
                                        is_live = False
                                        liveness_confidence = 0.0
                                        current_ear = 0.0
                                        total_blinks = 0
                                        liveness_details = {"error": f"Blink detection failed: {blink_error}"}
                                else:
                                    is_live = False
                                    liveness_confidence = 0.0
                                    emotion = "Neutral"
                                    current_ear = 0.0
                                    total_blinks = 0
                                    liveness_details = {"error": "No landmarks available"}
                            else:
                                analysis = g.liveness_adapter.analyze(cropped_face_resized, face_landmarks)
                                is_live = analysis.get("is_live")
                                liveness_confidence = analysis.get("confidence", 0.0)
                                liveness_details = analysis.get("details", {})
                                emotion = liveness_details.get("emotion", "Neutral") if isinstance(liveness_details, dict) else "Neutral"

                        with g.liveness_state_lock:
                            g.last_emotion = emotion
                            if is_live is True:
                                g.last_liveness = "Real"
                            elif is_live is False:
                                g.last_liveness = "Spoof"
                            else:
                                g.last_liveness = "Waiting"

                            g.last_liveness_confidence = liveness_confidence

                            if isinstance(liveness_details, dict) and "blink" in liveness_details:
                                blink_data = liveness_details["blink"]
                                if isinstance(blink_data, dict) and "current_ear" in blink_data:
                                    current_ear = blink_data["current_ear"]
                                    with g.ear_lock:
                                        g.ear_history.append(current_ear)
                                        print(f"[EAR] Stored EAR={current_ear:.3f}, history size: {len(g.ear_history)}")

                                    blink_count = blink_data.get("total_blinks", 0)
                                    g.current_ear = current_ear
                                    g.current_blinks = blink_count
                                    g.window.after(0, g.update_ear_debug_display, current_ear, blink_count)

                        print(f"[ANALYSIS] Emotion: {emotion}, Liveness: {g.last_liveness} ({liveness_confidence:.1%})")

                    except Exception as exc:
                        print(f"[ANALYSIS ERROR] {exc}")
                        g.emotion_failure_count += 1
                        if g.emotion_failure_count >= 5:
                            print("[ANALYSIS] Too many failures, disabling emotion analysis")
                            g.emotion_analysis_enabled = False
                            g.window.after(0, g.disable_blink_detection_ui)

                        with g.liveness_state_lock:
                            g.last_emotion = "Neutral"
                            g.last_liveness = "Unknown"
                            g.last_liveness_confidence = 0.0

                current_time = time.time()

                with g.liveness_state_lock:
                    if g.last_liveness == "Spoof":
                        g.consec_spoof_count += 1
                        g.consec_real_count = 0
                    elif g.last_liveness == "Real":
                        g.consec_real_count += 1
                        g.consec_spoof_count = 0

                    stable_liveness = g.last_liveness
                    if g.consec_spoof_count >= g.CONSEC_REQUIRED:
                        stable_liveness = "Spoof"
                    elif g.consec_real_count >= g.CONSEC_REQUIRED:
                        stable_liveness = "Real"

                    if stable_liveness == "Spoof":
                        time_since_detection = current_time - g.last_spoof_detection_time
                        if time_since_detection > (g.SPOOF_WARNING_DISPLAY_TIME * 3):
                            print("[SPOOF] Auto-recovery: clearing spoof state after extended timeout")
                            g.reset_spoof_detection()
                            stable_liveness = "Unknown"

                    is_live = stable_liveness == "Real"
                    liveness_status = stable_liveness

                if hasattr(g, "last_identity") and g.last_identity in g.session_spoof_passed:
                    print(f"[SPOOF-CACHE] Safety check: {g.last_identity} is cached, forcing Real")
                    is_live = True
                    liveness_status = "Real"
                    stable_liveness = "Real"

                if not is_live:
                    g.last_identity = f"SPOOF - {liveness_status}"
                    box_color = (0, 0, 255)
                    with g.liveness_state_lock:
                        if not g.spoof_warning_shown:
                            g.last_spoof_detection_time = current_time
                            g.spoof_warning_shown = True
                else:
                    try:
                        verify_res = verify_face(
                            cropped_face_resized,
                            verification_model,
                            val_transform,
                            employee_db,
                            face_idx=face_idx,
                            frame_count=g.frame_count,
                            total_faces=len(faces_to_process),
                            is_primary_face=is_primary_face,
                        )

                        rgb = verify_res["rgb"]
                        image_tensor = verify_res["image_tensor"]
                        trial_embedding = verify_res["trial_embedding"]
                        best_match = verify_res["best_match"]
                        best_match_data = verify_res["best_match_data"]
                        min_distance = verify_res["min_distance"]
                        distance_results = verify_res["distance_results"]
                        current_threshold = verify_res["current_threshold"]
                        adjusted_threshold = verify_res["adjusted_threshold"]
                        confidence = verify_res["confidence"]
                        g.last_identity = "Not Registered"

                        verification_result = {
                            "face_idx": face_idx,
                            "best_match": best_match,
                            "distance": min_distance,
                            "confidence": confidence,
                            "threshold": adjusted_threshold,
                            "bbox": (x, y, w, h),
                        }
                        if not hasattr(g, "current_frame_verifications"):
                            g.current_frame_verifications = []
                        g.current_frame_verifications.append(verification_result)

                        raw_identity = resolve_raw_identity(
                            face_idx=face_idx,
                            faces_to_process_count=len(faces_to_process),
                            best_match=best_match,
                            min_distance=min_distance,
                            adjusted_threshold=adjusted_threshold,
                            confidence=confidence,
                            confidence_rejection_threshold=CONFIDENCE_REJECTION_THRESHOLD,
                            current_frame_verifications=g.current_frame_verifications,
                        )
                        if raw_identity == "Not Registered (Low Confidence)" and face_idx == 0:
                            print(f"Rejected: Low confidence ({confidence:.0f}%)")

                        if current_face_id >= 0:
                            g.update_face_recognition(current_face_id, raw_identity, confidence)
                            face_history = g.get_face_recognition_history(current_face_id)
                        else:
                            face_history = []

                        frame_identity = smooth_identity(
                            raw_identity=raw_identity,
                            confidence=confidence,
                            face_history=face_history,
                            recognition_history=g.recognition_history,
                            smoothing_window=g.SMOOTHING_WINDOW,
                            is_primary_face=is_primary_face,
                        )
                        if frame_identity != raw_identity:
                            print(f"[TRACKING] Face {current_face_id}: Using tracked identity '{frame_identity}'")

                        current_time = time.time()

                        def _reset_spoof(reason: str):
                            g.reset_spoof_detection(reason)

                        def _mark_attendance_async(name: str, avg_confidence: float):
                            def mark_async():
                                try:
                                    success, message = g.attendance_logger.mark_attendance(
                                        name, min_distance, g.last_emotion, g.last_liveness
                                    )
                                    if success:
                                        print(f"✓ {message}")
                                except Exception as exc:
                                    print(f"Attendance marking error: {exc}")

                            threading.Thread(target=mark_async, daemon=True).start()
                            g.last_attendance_message = f"Checked in: {name}"

                        def _log_checkin(name: str):
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            log_entry = f"[{timestamp}] {name}"
                            g.log_listbox.insert(0, log_entry)
                            print(f"✓ Added to check-in log: {log_entry}")

                        lock_state = {
                            "locked_identity": g.locked_identity,
                            "lock_timestamp": g.lock_timestamp,
                            "identity_lock_buffer": g.identity_lock_buffer,
                            "last_identity": g.last_identity,
                        }

                        frame_identity, box_color, last_identity_val = update_identity_lock(
                            is_primary_face=is_primary_face,
                            raw_identity=raw_identity,
                            confidence=confidence,
                            current_time=current_time,
                            lock_state=lock_state,
                            lock_duration=g.LOCK_DURATION,
                            lock_display_time=g.LOCK_DISPLAY_TIME,
                            lock_min_verifications=g.LOCK_MIN_VERIFICATIONS,
                            last_liveness=g.last_liveness,
                            on_reset_spoof=_reset_spoof,
                            on_mark_attendance=_mark_attendance_async,
                            on_log_checkin=_log_checkin,
                        )

                        g.locked_identity = lock_state.get("locked_identity")
                        g.lock_timestamp = lock_state.get("lock_timestamp", 0)
                        g.identity_lock_buffer = lock_state.get("identity_lock_buffer", [])
                        g.last_identity = last_identity_val

                        if face_idx == 0:
                            g.last_confidence = confidence
                            g.last_distance = min_distance
                            g.last_identity = frame_identity

                            if (
                                g.last_identity in employee_db
                                and g.last_identity not in g.session_spoof_passed
                                and g.last_liveness == "Real"
                            ):
                                g.session_spoof_passed.add(g.last_identity)
                                print(f"[SPOOF-CACHE] Added {g.last_identity} to spoof cache")

                        update_face_lists(
                            face_idx=face_idx,
                            frame_identity=frame_identity,
                            confidence=confidence,
                            is_primary_face=is_primary_face,
                            last_emotion=g.last_emotion,
                            last_liveness=g.last_liveness,
                            face_identities=g.face_identities,
                            face_confidences=g.face_confidences,
                            face_emotions=g.face_emotions,
                            face_liveness=g.face_liveness,
                            enable_multi_face_verification=g.ENABLE_MULTI_FACE_VERIFICATION,
                            current_face_id=current_face_id,
                            face_verification_history=g.face_verification_history,
                            face_identities_verified=g.face_identities_verified,
                            face_confidences_verified=g.face_confidences_verified,
                            current_time=current_time,
                        )

                        g.current_face_tensor = image_tensor
                        g.current_face_image = rgb

                        if g.explainer is not None:
                            try:
                                threshold = adjusted_threshold
                                g.current_explanation = g.explainer.explain_distance(min_distance, threshold)
                                quality_exp = g.explainer.explain_quality_factors(rgb)
                                g.current_explanation["quality_message"] = quality_exp["overall_message"]
                            except Exception as exc:
                                print(f"[WARNING] Explanation generation failed: {exc}")

                        g.knn_neighbors = ([best_match] * 5, [min_distance] * 5)

                        if best_match_data and USE_MULTI_EMBEDDING and isinstance(best_match_data, list):
                            trial_device = trial_embedding.device
                            distances_with_idx = []
                            for idx_emb, emb in enumerate(best_match_data):
                                emb_on_device = emb.to(trial_device, non_blocking=True)
                                distances_with_idx.append((F.pairwise_distance(trial_embedding, emb_on_device).item(), idx_emb))
                            _, g.matched_pose_index = min(distances_with_idx, key=lambda x: x[0])
                        else:
                            g.matched_pose_index = 0

                        total_process_time = (time.time() - process_start) * 1000
                        if total_process_time > 100:
                            print(f"  Processing time: {total_process_time:.0f}ms")
                    except Exception as exc:
                        print(f"Verification error: {exc}")
            else:
                g.last_identity = "Face too small"
                box_color = (0, 0, 255)

        elif is_secondary_face and is_processing_face and g.frame_count % (g.PROCESS_EVERY_N_FRAMES * 2) == 0:
            cropped_face = crop_face_with_padding(frame, x, y, w, h)
            if cropped_face.size > 0 and cropped_face.shape[0] >= 50:
                while len(g.face_identities) <= face_idx:
                    g.face_identities.append("Processing...")
                g.face_identities[face_idx] = "Detected"
                while len(g.face_confidences) <= face_idx:
                    g.face_confidences.append(0)
                g.face_confidences[face_idx] = 0
            else:
                while len(g.face_identities) <= face_idx:
                    g.face_identities.append("Processing...")
                g.face_identities[face_idx] = "Too Small"
                while len(g.face_confidences) <= face_idx:
                    g.face_confidences.append(0)
                g.face_confidences[face_idx] = 0

        # Display label / UI drawing
        if face_idx < MAX_CONCURRENT_FACES:
            if g.registration_mode and face_idx == 0:
                display_text = "Registering..."
            elif is_primary_face:
                with g.liveness_state_lock:
                    display_text = f"PRIMARY: {g.last_identity} ({g.last_emotion} | {g.last_liveness})"
            elif is_secondary_face:
                if hasattr(g, "face_identities") and face_idx < len(g.face_identities):
                    face_identity = g.face_identities[face_idx]
                    face_confidence = g.face_confidences[face_idx] if face_idx < len(g.face_confidences) else 0
                    display_text = f"Others: {face_identity} ({face_confidence:.0f}%)"
                else:
                    display_text = "Others: Processing..."
            else:
                if hasattr(g, "face_identities") and face_idx < len(g.face_identities):
                    display_text = f"UNTRACKED: {g.face_identities[face_idx]}"
                else:
                    display_text = f"Face {face_idx+1}: Processing..."
        else:
            display_text = "Face Limit Exceeded"

        if face_idx < MAX_CONCURRENT_FACES:
            if face_idx == 0 and hasattr(g, "box_color_transition"):
                trans = g.box_color_transition
                trans["target_color"] = box_color

                if trans["current_color"] != trans["target_color"]:
                    trans["frame"] += 1
                    if trans["frame"] >= trans["transition_frames"]:
                        trans["current_color"] = trans["target_color"]
                        trans["frame"] = 0
                    else:
                        factor = trans["frame"] / trans["transition_frames"]
                        trans["current_color"] = interpolate_color(
                            trans["current_color"], trans["target_color"], factor
                        )
                box_color = trans["current_color"]
            elif face_idx > 0:
                face_colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
                box_color = face_colors[face_idx % len(face_colors)]
        else:
            box_color = (128, 128, 128)

        current_identity = g.face_identities[face_idx] if face_idx < len(g.face_identities) else "Processing..."
        is_recognized = current_identity not in [
            "Not Registered",
            "Not Registered (Low Confidence)",
            "Error",
            "Spoof Detected",
            "Face too small",
            "Registering...",
            "Processing...",
        ]
        thickness = 3 if is_recognized else 2
        draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, thickness, radius=12)

        font = cv2.FONT_HERSHEY_DUPLEX
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, 0.6, 2)
        label_y = max(y - text_h - 18, 10)

        label_overlay = frame.copy()
        draw_rounded_rectangle(
            label_overlay,
            (x - 2, label_y),
            (x + text_w + 24, label_y + text_h + 12),
            (20, 20, 30),
            -1,
            radius=8,
        )
        cv2.addWeighted(label_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, display_text, (x + 10, label_y + text_h + 6), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if face_idx < MAX_CONCURRENT_FACES and g.frame_count % g.PROCESS_EVERY_N_FRAMES == 0:
            g.recognition_stats["total_detections"] += 1
            current_identity = g.face_identities[face_idx] if face_idx < len(g.face_identities) else "Processing..."
            if current_identity not in [
                "Not Registered",
                "Not Registered (Low Confidence)",
                "Error",
                "Spoof Detected",
                "Face too small",
                "Processing...",
            ]:
                g.recognition_stats["successful_recognitions"] += 1
                g.recognition_stats["unique_faces_today"].add(current_identity)

        return box_color
