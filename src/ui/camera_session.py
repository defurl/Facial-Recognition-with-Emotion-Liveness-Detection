"""
Camera session management: Handles camera initialization, frame capture, and cleanup.

Separates camera/threading concerns from UI orchestration.
"""
import cv2
import threading
import time
import asyncio
from pathlib import Path
from typing import Optional, Callable

from src.config import CAMERA_INDEX


class CameraSession:
    """Manages camera lifecycle, frame capture, and streaming."""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_queue = None  # Will be set by caller
        self.camera_lock = threading.Lock()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 30
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
    def open_camera(self, camera_index: int = CAMERA_INDEX) -> bool:
        """
        Try to open camera with fallback indices.
        
        Args:
            camera_index: Preferred camera index
            
        Returns:
            True if camera opened successfully
        """
        preferred_indices = [camera_index, 0, 1, 2]
        
        for idx in preferred_indices:
            try:
                cap = cv2.VideoCapture(idx)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                
                # Try to set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Read test frame
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    continue
                
                self.cap = cap
                print(f"[CAMERA] Successfully opened camera at index {idx}")
                return True
                
            except Exception as e:
                print(f"[CAMERA] Error opening camera {idx}: {e}")
                if cap is not None:
                    cap.release()
        
        print("[ERROR] Failed to open any camera")
        return False
    
    def start(self, frame_queue) -> bool:
        """
        Start camera session.
        
        Args:
            frame_queue: Queue to put captured frames into
            
        Returns:
            True if started successfully
        """
        if not self.open_camera():
            return False
        
        self.frame_queue = frame_queue
        self.running = True
        self.consecutive_errors = 0
        
        # Optimize camera settings
        try:
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        
        return True
    
    def stop(self):
        """Stop camera session and cleanup."""
        self.running = False
        time.sleep(0.2)
        
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
    
    def capture_frame(self) -> Optional[tuple]:
        """
        Capture a single frame from camera.
        
        Returns:
            Tuple of (frame, timestamp) or None if error
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            with self.camera_lock:
                ret, frame = self.cap.read()
                
            if not ret or frame is None:
                self.consecutive_errors += 1
                return None
            
            self.consecutive_errors = 0
            self.frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            if self.frame_count % 30 == 0:
                self.fps = 30.0 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
            
            return (frame, time.time())
            
        except Exception as e:
            print(f"[CAMERA] Error capturing frame: {e}")
            self.consecutive_errors += 1
            return None
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def get_frame_count(self) -> int:
        """Get total frames captured."""
        return self.frame_count
    
    def has_errors(self) -> bool:
        """Check if too many consecutive errors occurred."""
        return self.consecutive_errors >= self.max_consecutive_errors
    
    def is_healthy(self) -> bool:
        """Check if camera session is healthy."""
        return (
            self.cap is not None and
            self.cap.isOpened() and
            self.running and
            not self.has_errors()
        )
