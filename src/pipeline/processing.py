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
