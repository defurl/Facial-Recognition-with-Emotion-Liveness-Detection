"""Liveness adapter to unify blink-only fast path and optional heavy modes."""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from src.liveness import LivenessDetector
from src.blink_detector import BlinkDetector
from src.config import OPTIMAL_THRESHOLD_GUI, USE_MULTI_EMBEDDING
from src.runtime.settings import load_gui_threshold


class LivenessAdapter:
    """Wraps liveness detection with a lightweight blink-only fast path.

    The heavy path can be swapped in later without touching callers.
    """

    def __init__(self, use_blink_only: bool = True):
        self.use_blink_only = use_blink_only
        self.blink_detector: Optional[BlinkDetector] = None
        self.heavy_liveness: Optional[LivenessDetector] = None

    def ensure_initialized(self):
        if self.use_blink_only and self.blink_detector is None:
            self.blink_detector = BlinkDetector()
        if not self.use_blink_only and self.heavy_liveness is None:
            self.heavy_liveness = LivenessDetector()

    def reset(self):
        if self.blink_detector:
            self.blink_detector.reset()
        if self.heavy_liveness:
            self.heavy_liveness.reset()

    def compute_early_identity(
        self,
        cropped_face_resized,
        verification_model,
        val_transform,
        employee_db,
        *,
        current_threshold: Optional[float] = None,
        relaxed_multiplier: float = 1.5,
        confidence_floor: float = 30.0,
        use_multi_embedding: Optional[bool] = None,
    ) -> Optional[str]:
        """Lightweight identity guess for cache checks.

        Returns a best-match name if confidence crosses the relaxed threshold; otherwise None.
        """
        threshold = current_threshold or load_gui_threshold(OPTIMAL_THRESHOLD_GUI)
        multi = USE_MULTI_EMBEDDING if use_multi_embedding is None else use_multi_embedding

        rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).convert("RGB")
        image_tensor = val_transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            trial_embedding = verification_model(image_tensor, mode="metric").cpu()

        min_distance = float("inf")
        best_match = None
        relaxed_threshold = threshold * relaxed_multiplier

        for name, saved_data in employee_db.items():
            if multi and isinstance(saved_data, list):
                distances = [F.pairwise_distance(trial_embedding, emb).item() for emb in saved_data]
                distance = min(distances) if distances else float("inf")
            else:
                saved_embedding = saved_data[0] if isinstance(saved_data, list) else saved_data
                distance = F.pairwise_distance(trial_embedding, saved_embedding).item()

            if distance < min_distance:
                min_distance = distance
                best_match = name

        if best_match is None:
            return None

        confidence = max(0, min(100, (1 - min_distance / relaxed_threshold) * 100))
        if min_distance < relaxed_threshold and confidence >= confidence_floor:
            return best_match
        return None

    def pick_cached_identity(self, early_identity: Optional[str], session_cache, employee_db) -> Optional[str]:
        """Return a cached identity if present; prefer early_identity when in cache."""
        if not session_cache:
            return None

        if early_identity and early_identity in session_cache:
            return early_identity

        for cached_user in session_cache:
            if cached_user in employee_db:
                return cached_user
        return None

    def analyze(self, frame_rgb, face_landmarks) -> Dict[str, Any]:
        """Run liveness analysis and return a normalized result dict.

        Returns keys: is_live (bool/None), confidence (float), details (dict).
        """
        self.ensure_initialized()

        if self.use_blink_only and self.blink_detector:
            is_blink, blink_conf = self.blink_detector.detect_blink(face_landmarks)
            is_live = True if is_blink else None  # None means still waiting
            details = {
                "blink": {
                    "current_ear": self.blink_detector.current_ear,
                    "total_blinks": self.blink_detector.total_blinks,
                }
            }
            return {"is_live": is_live, "confidence": blink_conf or 0.0, "details": details}

        if self.heavy_liveness:
            is_live, confidence, details = self.heavy_liveness.analyze(frame_rgb, face_landmarks)
            return {"is_live": is_live, "confidence": confidence, "details": details}

        return {"is_live": None, "confidence": 0.0, "details": {}}
