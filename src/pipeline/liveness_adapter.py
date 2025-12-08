"""Liveness adapter to unify blink-only fast path and optional heavy modes."""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.liveness import LivenessDetector
from src.blink_detector import BlinkDetector


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
