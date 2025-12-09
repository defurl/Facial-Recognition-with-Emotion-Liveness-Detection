"""
Session manager: Orchestrates camera session, display updates, and registration.

Bridges extracted components with minimal business logic.
"""
import time
from typing import Optional, Callable
import queue

from src.ui.camera_session import CameraSession
from src.ui.display_layer import DisplayLayer
from src.ui.registration_handler import RegistrationHandler


class SessionManager:
    """
    Orchestrates camera session, display layer, and registration handler.
    
    Provides clean API for main GUI to use extracted components without
    needing to know their internal details.
    """
    
    def __init__(self, display_layer: DisplayLayer):
        self.camera = CameraSession()
        self.display = display_layer
        self.registration = RegistrationHandler()
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.processing_active = False
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
    
    # ===== Camera Control =====
    
    def start_camera(self) -> bool:
        """
        Start camera session.
        
        Returns:
            True if camera started successfully
        """
        if not self.camera.start(self.frame_queue):
            return False
        
        self.processing_active = True
        self.display.show_camera_active()
        self.display.enable_camera_controls()
        
        return True
    
    def stop_camera(self):
        """Stop camera session."""
        self.processing_active = False
        self.camera.stop()
        self.display.show_camera_stopped()
        self.display.disable_camera_controls()
        self.display.reset_detections()
    
    def capture_next_frame(self):
        """
        Capture next frame from camera.
        
        Returns:
            Tuple of (frame, timestamp) or None
        """
        if not self.processing_active:
            return None
        
        result = self.camera.capture_frame()
        if result is not None:
            self.frame_count += 1
            
            # Update FPS every 30 frames
            if self.frame_count % 30 == 0:
                current_time = time.time()
                self.fps = 30.0 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                self.display.update_fps(self.fps)
        
        return result
    
    def is_camera_healthy(self) -> bool:
        """Check if camera is healthy."""
        return self.camera.is_healthy()
    
    # ===== Display Updates =====
    
    def update_identity_display(self, name: str, bg_color: str, text_color: str):
        """Update identity display."""
        self.display.update_identity(name, bg_color, text_color)
    
    def update_liveness_display(self, liveness: str, color: str = '#58a6ff'):
        """Update liveness display."""
        self.display.update_liveness(liveness, color)
    
    def update_detection_info(self, emotion: str, distance: float, confidence: float):
        """Update detection information."""
        self.display.update_emotion(emotion)
        self.display.update_distance(f"{distance:.3f}" if distance != float('inf') else "N/A")
        self.display.update_confidence(f"{confidence:.1f}%")
    
    def add_log_entry(self, entry: str):
        """Add entry to check-ins log."""
        self.display.add_log_entry(entry)
    
    def reset_display(self):
        """Reset all display information."""
        self.display.reset_detections()
    
    # ===== Registration Management =====
    
    def start_registration(self, name: str) -> bool:
        """
        Start registration workflow.
        
        Args:
            name: Employee name
            
        Returns:
            True if registration started
        """
        return self.registration.start_registration(name)
    
    def add_registration_frame(self, frame, embedding=None, pose_instruction=""):
        """Add frame to current registration."""
        return self.registration.add_registration_frame(frame, embedding, pose_instruction)
    
    def complete_registration(self):
        """Complete current registration."""
        return self.registration.complete_registration()
    
    def cancel_registration(self):
        """Cancel current registration."""
        self.registration.cancel_registration()
    
    def is_registering(self) -> bool:
        """Check if registration is active."""
        return self.registration.is_registration_active()
    
    def get_registration_progress(self):
        """Get registration progress info."""
        return self.registration.get_registration_progress()
    
    def update_registration_feedback(self, message: str, color=(0, 255, 0)):
        """Update registration feedback message."""
        self.registration.update_registration_feedback(message, color)
    
    # ===== Session Status =====
    
    def is_processing(self) -> bool:
        """Check if session is actively processing."""
        return self.processing_active
    
    def get_stats(self):
        """Get session statistics."""
        return {
            'frames_captured': self.camera.get_frame_count(),
            'fps': self.fps,
            'camera_healthy': self.camera.is_healthy(),
            'processing_active': self.processing_active,
            'registering': self.registration.is_registration_active(),
        }
    
    def cleanup(self):
        """Cleanup session resources."""
        self.stop_camera()
        self.cancel_registration()
