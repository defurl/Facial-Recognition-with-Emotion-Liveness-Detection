import time

import pytest

from src.pipeline.processing import (
    resolve_raw_identity,
    smooth_identity,
    update_identity_lock,
)


def test_resolve_raw_identity_conflict_and_confidence():
    # Low confidence rejects
    rid = resolve_raw_identity(
        face_idx=0,
        faces_to_process_count=1,
        best_match="alice",
        min_distance=0.5,
        adjusted_threshold=1.0,
        confidence=4.0,
        confidence_rejection_threshold=0.05,
        current_frame_verifications=[],
    )
    assert rid == "Not Registered (Low Confidence)"

    # Conflict selects ambiguous
    rid = resolve_raw_identity(
        face_idx=0,
        faces_to_process_count=2,
        best_match="alice",
        min_distance=0.5,
        adjusted_threshold=1.0,
        confidence=80.0,
        confidence_rejection_threshold=0.05,
        current_frame_verifications=[{"best_match": "alice", "face_idx": 1, "confidence": 95}],
    )
    assert rid == "Ambiguous Match"

    # Accept
    rid = resolve_raw_identity(
        face_idx=0,
        faces_to_process_count=1,
        best_match="alice",
        min_distance=0.5,
        adjusted_threshold=1.0,
        confidence=80.0,
        confidence_rejection_threshold=0.05,
        current_frame_verifications=[],
    )
    assert rid == "alice"


def test_smooth_identity_prefers_tracked():
    history = [
        {"identity": "alice", "confidence": 90},
        {"identity": "alice", "confidence": 85},
        {"identity": "bob", "confidence": 60},
    ]
    recognition_history = []
    frame_id = smooth_identity(
        raw_identity="Not Registered",
        confidence=50,
        face_history=history,
        recognition_history=recognition_history,
        smoothing_window=5,
        is_primary_face=False,
    )
    assert frame_id == "alice"


def test_update_identity_lock_immediate_lock_and_timeout():
    now = time.time()
    lock_state = {"locked_identity": None, "lock_timestamp": 0, "identity_lock_buffer": [], "last_identity": "Not Registered"}

    resets = []
    marks = []
    logs = []

    def on_reset(reason):
        resets.append(reason)

    def on_mark(name, conf):
        marks.append((name, conf))

    def on_log(name):
        logs.append(name)

    # Immediate lock
    frame_id, box_color, last_id = update_identity_lock(
        is_primary_face=True,
        raw_identity="alice",
        confidence=80,
        current_time=now,
        lock_state=lock_state,
        lock_duration=5.0,
        lock_display_time=2.0,
        lock_min_verifications=3,
        last_liveness="Real",
        on_reset_spoof=on_reset,
        on_mark_attendance=on_mark,
        on_log_checkin=on_log,
    )
    assert frame_id == "alice"
    assert marks == [("alice", 80)]
    assert logs == ["alice"]

    # Timeout clears
    frame_id, box_color, last_id = update_identity_lock(
        is_primary_face=True,
        raw_identity="bob",
        confidence=80,
        current_time=now + 3.0,
        lock_state=lock_state,
        lock_duration=5.0,
        lock_display_time=2.0,
        lock_min_verifications=3,
        last_liveness="Real",
        on_reset_spoof=on_reset,
        on_mark_attendance=on_mark,
        on_log_checkin=on_log,
    )
    assert frame_id == "Not Registered"
    assert resets == ["identity lock timeout"]
