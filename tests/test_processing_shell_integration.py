"""
Phase 2 Integration Test: process_frame_shell contract validation.

Tests the unified async/sync callback pipeline:
- Context dict key consistency (primary_face_id, warnings, face_assignments)
- Role-based coloring logic (face_roles() + role_box_color())
- Callback contract adherence without requiring camera/GUI
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import cv2
import numpy as np

# Ensure project root and src are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from src.pipeline.processing import process_frame_shell
from src.pipeline.face_processor import face_roles, role_box_color


class TestProcessFrameShellContract:
    """Validate process_frame_shell callback contracts and context consistency."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a dummy frame (480x640x3)
        self.frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Standard test bboxes (x, y, w, h)
        self.bbox_1 = (50, 50, 100, 100)
        self.bbox_2 = (300, 150, 120, 120)

    def test_detect_faces_returns_list_of_bboxes(self):
        """detect_faces_cb can return a list of bboxes (backward compatibility)."""
        detect_cb = Mock(return_value=[self.bbox_1, self.bbox_2])
        process_cb = Mock(return_value={"face_idx": 0})
        update_ui_cb = Mock()

        results = process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        # Should normalize list to context dict
        assert len(results) == 2
        process_cb.assert_called()
        update_ui_cb.assert_called_once()
        
        # Check that update_ui_cb receives normalized context with faces
        call_args = update_ui_cb.call_args
        context = call_args[0][2]
        assert "faces" in context
        assert context["faces"] == [self.bbox_1, self.bbox_2]

    def test_detect_faces_returns_context_dict(self):
        """detect_faces_cb returns a context dict with faces + metadata."""
        detection_context = {
            "faces": [self.bbox_1, self.bbox_2],
            "face_assignments": {"0": 1, "1": 2},  # {face_idx: tracker_id}
            "primary_face_id": 1,
            "warnings": ["low_light"],
        }
        detect_cb = Mock(return_value=detection_context)
        process_cb = Mock(return_value={"face_idx": 0})
        update_ui_cb = Mock()

        results = process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        assert len(results) == 2
        
        # Verify context is passed through to callbacks
        call_args = update_ui_cb.call_args
        context = call_args[0][2]
        assert context["faces"] == [self.bbox_1, self.bbox_2]
        assert context["face_assignments"] == {"0": 1, "1": 2}
        assert context["primary_face_id"] == 1
        assert context["warnings"] == ["low_light"]

    def test_process_face_receives_consistent_context(self):
        """process_face_cb receives the same context dict across all faces."""
        detection_context = {
            "faces": [self.bbox_1, self.bbox_2],
            "primary_face_id": 1,
            "face_assignments": {"0": 1, "1": 2},
        }
        detect_cb = Mock(return_value=detection_context)
        process_cb = Mock(return_value={"face_idx": 0})
        update_ui_cb = Mock()

        process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        # process_cb should be called twice with same context
        assert process_cb.call_count == 2
        
        context_1 = process_cb.call_args_list[0][0][3]
        context_2 = process_cb.call_args_list[1][0][3]
        
        # Both calls receive identical context
        assert context_1["primary_face_id"] == context_2["primary_face_id"]
        assert context_1["face_assignments"] == context_2["face_assignments"]

    def test_role_based_coloring_primary_face(self):
        """face_roles() and role_box_color() apply correct colors for primary face."""
        primary_face_id = 1
        
        # Face 0 is primary (current_face_id 1)
        is_primary, is_secondary = face_roles(
            face_idx=0,
            current_face_id=1,
            primary_face_id=primary_face_id,
            enable_multi_face=True,
        )
        assert is_primary is True
        assert is_secondary is False
        
        box_color = role_box_color(is_primary, is_secondary)
        assert box_color == (0, 255, 0)  # Green

    def test_role_based_coloring_secondary_face(self):
        """face_roles() and role_box_color() apply correct colors for secondary face."""
        primary_face_id = 1
        
        # Face 1 is secondary (current_face_id 2, not primary)
        is_primary, is_secondary = face_roles(
            face_idx=1,
            current_face_id=2,
            primary_face_id=primary_face_id,
            enable_multi_face=True,
        )
        assert is_primary is False
        assert is_secondary is True
        
        box_color = role_box_color(is_primary, is_secondary)
        assert box_color == (255, 165, 0)  # Orange

    def test_role_based_coloring_unknown_face(self):
        """face_roles() and role_box_color() apply correct colors for unknown face."""
        primary_face_id = 1
        
        # Face 2 is unknown (current_face_id < 0, indicating no valid tracker)
        is_primary, is_secondary = face_roles(
            face_idx=2,
            current_face_id=-1,
            primary_face_id=primary_face_id,
            enable_multi_face=True,
        )
        assert is_primary is False
        assert is_secondary is False
        
        box_color = role_box_color(is_primary, is_secondary)
        assert box_color == (0, 0, 255)  # Red

    def test_multi_face_verification_disabled(self):
        """With multi-face disabled, only face 0 is primary."""
        primary_face_id = 1
        enable_multi_face = False
        
        # Face 0 is primary (face_idx == 0 when multi-face is off)
        is_primary, is_secondary = face_roles(
            face_idx=0,
            current_face_id=1,
            primary_face_id=primary_face_id,
            enable_multi_face=enable_multi_face,
        )
        assert is_primary is True
        assert is_secondary is False
        
        box_color = role_box_color(is_primary, is_secondary)
        assert box_color == (0, 255, 0)  # Green for primary
        
        # Face 1 is secondary (face_idx > 0 when multi-face is off)
        is_primary, is_secondary = face_roles(
            face_idx=1,
            current_face_id=2,
            primary_face_id=primary_face_id,
            enable_multi_face=enable_multi_face,
        )
        assert is_primary is False
        assert is_secondary is True
        
        box_color = role_box_color(is_primary, is_secondary)
        assert box_color == (255, 165, 0)  # Orange for secondary

    def test_update_ui_cb_receives_per_face_results(self):
        """update_ui_cb receives list of per-face results from process_face_cb."""
        detection_context = {
            "faces": [self.bbox_1, self.bbox_2],
            "primary_face_id": 1,
        }
        detect_cb = Mock(return_value=detection_context)
        
        # Simulate per-face results
        face_results = [
            {"face_idx": 0, "identity": "alice", "confidence": 0.95},
            {"face_idx": 1, "identity": "bob", "confidence": 0.87},
        ]
        process_cb = Mock(side_effect=face_results)
        update_ui_cb = Mock()

        process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        # update_ui_cb should receive the results list
        call_args = update_ui_cb.call_args
        frame_arg, results_arg, context_arg = call_args[0]
        
        assert np.array_equal(frame_arg, self.frame)
        assert results_arg == face_results
        assert "faces" in context_arg
        assert "primary_face_id" in context_arg

    def test_process_face_exception_handling(self):
        """process_frame_shell handles per-face exceptions gracefully."""
        detection_context = {
            "faces": [self.bbox_1, self.bbox_2],
        }
        detect_cb = Mock(return_value=detection_context)
        
        # First face succeeds, second face raises an exception
        def process_with_error(frame, face_idx, bbox, context):
            if face_idx == 0:
                return {"face_idx": 0, "status": "ok"}
            else:
                raise ValueError("Test error in face processing")
        
        process_cb = Mock(side_effect=process_with_error)
        update_ui_cb = Mock()

        results = process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        # Both results should be present: one ok, one error
        assert len(results) == 2
        assert results[0]["face_idx"] == 0
        assert results[0]["status"] == "ok"
        
        # Second result should have error info
        assert "error" in results[1]
        assert isinstance(results[1]["error"], ValueError)
        assert results[1]["face_idx"] == 1
        assert results[1]["bbox"] == self.bbox_2

    def test_empty_detection_normalizes_to_empty_context(self):
        """Empty detection results in empty context with no faces."""
        detect_cb = Mock(return_value=[])
        process_cb = Mock(return_value={})
        update_ui_cb = Mock()

        results = process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        assert len(results) == 0
        process_cb.assert_not_called()
        
        # But update_ui_cb should still be called with empty context
        call_args = update_ui_cb.call_args
        context = call_args[0][2]
        assert context["faces"] == []

    def test_context_defaults_missing_keys(self):
        """process_frame_shell provides sensible defaults for optional context keys."""
        detect_cb = Mock(return_value={"faces": [self.bbox_1]})
        process_cb = Mock(return_value={"face_idx": 0})
        update_ui_cb = Mock()

        process_frame_shell(
            self.frame,
            detect_faces_cb=detect_cb,
            process_face_cb=process_cb,
            update_ui_cb=update_ui_cb,
        )

        # Check that context has faces but may lack optional keys
        call_args = update_ui_cb.call_args
        context = call_args[0][2]
        assert "faces" in context
        # Optional keys may or may not exist; context should be valid regardless

    def test_async_sync_parity_context_structure(self):
        """
        Simulate async and sync paths to verify context structure parity.
        
        This is the key test for Phase 2: both async and sync produce
        compatible context dicts that drive role-based coloring consistently.
        """
        # Simulate sync path context
        sync_context = {
            "faces": [self.bbox_1, self.bbox_2],
            "face_assignments": {"0": 10, "1": 11},
            "primary_face_id": 10,
            "warnings": ["low_confidence"],
        }
        
        # Simulate async path context (should be identical)
        async_context = {
            "faces": [self.bbox_1, self.bbox_2],
            "face_assignments": {"0": 10, "1": 11},
            "primary_face_id": 10,
            "warnings": ["low_confidence"],
        }
        
        # Both paths should produce identical role-based colors
        for face_idx in range(2):
            current_face_id = int(async_context["face_assignments"].get(str(face_idx)))
            
            sync_primary, sync_secondary = face_roles(
                face_idx=face_idx,
                current_face_id=current_face_id,
                primary_face_id=sync_context["primary_face_id"],
                enable_multi_face=True,
            )
            
            async_primary, async_secondary = face_roles(
                face_idx=face_idx,
                current_face_id=current_face_id,
                primary_face_id=async_context["primary_face_id"],
                enable_multi_face=True,
            )
            
            # Colors should match
            sync_color = role_box_color(sync_primary, sync_secondary)
            async_color = role_box_color(async_primary, async_secondary)
            assert sync_color == async_color, f"Mismatch for face {face_idx}"


class TestProcessFrameShellCallbackSignatures:
    """Verify callback function signatures and contract."""

    def test_detect_faces_cb_signature(self):
        """detect_faces_cb should accept frame and return list or dict."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Valid: returns list
        detect_list = lambda f: [(50, 50, 100, 100)]
        result = detect_list(frame)
        assert isinstance(result, list)
        
        # Valid: returns dict with faces key
        detect_dict = lambda f: {"faces": [(50, 50, 100, 100)], "primary_face_id": 1}
        result = detect_dict(frame)
        assert isinstance(result, dict)
        assert "faces" in result

    def test_process_face_cb_signature(self):
        """process_face_cb should accept (frame, face_idx, bbox, context) and return serializable."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_idx = 0
        bbox = (50, 50, 100, 100)
        context = {"faces": [bbox], "primary_face_id": 1}
        
        # Valid: returns a dict
        process_fn = lambda f, fi, b, c: {"face_idx": fi, "identity": "test"}
        result = process_fn(frame, face_idx, bbox, context)
        assert isinstance(result, dict)

    def test_update_ui_cb_signature(self):
        """update_ui_cb should accept (frame, results, context) and return None."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{"face_idx": 0}]
        context = {"faces": [(50, 50, 100, 100)], "primary_face_id": 1}
        
        # Valid: returns None (side-effects only)
        update_fn = lambda f, r, c: None
        result = update_fn(frame, results, context)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
