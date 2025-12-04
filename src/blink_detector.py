"""
Eye Blink Detection for Anti-Spoofing
Based on Eye Aspect Ratio (EAR) calculation from facial landmarks
"""

import numpy as np
from collections import deque
import time
import threading


class BlinkDetector:
    """
    Detects eye blinks using Eye Aspect Ratio (EAR) from MediaPipe Face Mesh landmarks.
    
    Eye Aspect Ratio formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Where p1-p6 are 6 key points around the eye:
        - p1, p4: Eye corners (horizontal distance)
        - p2, p3, p5, p6: Upper and lower eyelid points (vertical distances)
    
    A blink is detected when:
        1. EAR drops below threshold (eye closes)
        2. EAR returns above threshold (eye opens)
        3. Duration is within typical blink range (0.1-0.4 seconds)
    """
    
    # MediaPipe Face Mesh landmark indices for eyes
    LEFT_EYE_INDICES = {
        'outer': 33,      # p1 - outer corner
        'inner': 133,     # p4 - inner corner
        'top_1': 159,     # p2 - top outer
        'top_2': 145,     # p3 - top inner
        'bottom_1': 23,   # p6 - bottom outer
        'bottom_2': 133   # p5 - bottom inner (reusing inner corner)
    }
    
    RIGHT_EYE_INDICES = {
        'outer': 362,     # p1 - outer corner
        'inner': 263,     # p4 - inner corner
        'top_1': 386,     # p2 - top outer
        'top_2': 374,     # p3 - top inner
        'bottom_1': 253,  # p6 - bottom outer
        'bottom_2': 263   # p5 - bottom inner (reusing inner corner)
    }
    
    def __init__(self, ear_threshold=0.5, history_size=30, min_blink_duration=0.08, max_blink_duration=0.4):
        """
        Initialize blink detector with thread safety
        
        Args:
            ear_threshold: EAR value below which eye is considered closed (default: 0.5)
            history_size: Number of frames to track (default: 30, ~1 second at 30fps)
            min_blink_duration: Minimum blink duration in seconds (default: 0.08)
            max_blink_duration: Maximum blink duration in seconds (default: 0.4)
        """
        self.ear_threshold = ear_threshold
        self.history_size = history_size
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        
        # Thread safety locks
        self._state_lock = threading.RLock()  # Recursive lock for nested access
        self._history_lock = threading.Lock()  # Separate lock for history data
        
        # History tracking (thread-safe)
        self.ear_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        self.blink_timestamps = deque(maxlen=100)  # FIX: Limit memory growth!
        
        # Blink tracking (protected by state_lock)
        self.blink_count = 0
        self.last_blink_time = 0
        self.eye_closed_start = None
        self.consecutive_closed_frames = 0
        
        # State (protected by state_lock)
        self.is_eye_closed = False
        
        print(f"[BlinkDetector] Initialized with threshold={ear_threshold}, thread-safe")
        
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks
        
        Args:
            eye_landmarks: dict with keys 'outer', 'inner', 'top_1', 'top_2', 'bottom_1', 'bottom_2'
                          Each value is a tuple/list of (x, y) coordinates
        
        Returns:
            float: Eye Aspect Ratio value
        """
        # Extract points
        p1 = np.array(eye_landmarks['outer'])    # Outer corner
        p4 = np.array(eye_landmarks['inner'])    # Inner corner
        p2 = np.array(eye_landmarks['top_1'])    # Top outer
        p3 = np.array(eye_landmarks['top_2'])    # Top inner
        p5 = np.array(eye_landmarks['bottom_2']) # Bottom inner
        p6 = np.array(eye_landmarks['bottom_1']) # Bottom outer
        
        # Calculate distances
        vertical_1 = np.linalg.norm(p2 - p6)  # Outer vertical
        vertical_2 = np.linalg.norm(p3 - p5)  # Inner vertical
        horizontal = np.linalg.norm(p1 - p4)  # Horizontal
        
        # Avoid division by zero
        if horizontal < 1e-6:
            return 0.0
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def extract_eye_landmarks(self, face_landmarks, eye='left'):
        """
        Extract eye landmark coordinates from MediaPipe Face Mesh results
        
        Args:
            face_landmarks: MediaPipe Face Mesh landmarks object
            eye: 'left' or 'right'
        
        Returns:
            dict: Eye landmark coordinates
        """
        indices = self.LEFT_EYE_INDICES if eye == 'left' else self.RIGHT_EYE_INDICES
        
        eye_coords = {}
        for key, idx in indices.items():
            landmark = face_landmarks.landmark[idx]
            eye_coords[key] = (landmark.x, landmark.y)
        
        return eye_coords
    
    def detect_blink(self, face_landmarks, timestamp=None):
        """
        Detect blink from facial landmarks (thread-safe)
        
        Args:
            face_landmarks: MediaPipe Face Mesh landmarks object (468 points)
            timestamp: Optional timestamp (uses time.time() if not provided)
        
        Returns:
            tuple: (blink_detected: bool, current_ear: float, blink_count: int)
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Extract eye landmarks
            left_eye = self.extract_eye_landmarks(face_landmarks, 'left')
            right_eye = self.extract_eye_landmarks(face_landmarks, 'right')
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Thread-safe history update
            with self._history_lock:
                self.ear_history.append(avg_ear)
                self.timestamp_history.append(timestamp)
            
            # Thread-safe blink detection
            with self._state_lock:
                blink_detected = self._process_ear_value(avg_ear, timestamp)
                current_blink_count = self.blink_count
            
            return blink_detected, avg_ear, current_blink_count
            
        except Exception as e:
            print(f"[BlinkDetector ERROR] detect_blink failed: {e}")
            return False, 0.0, 0
    
    def _process_ear_value(self, ear, timestamp):
        """
        Process EAR value to detect blink sequence
        
        Args:
            ear: Current Eye Aspect Ratio
            timestamp: Current timestamp
        
        Returns:
            bool: True if a complete blink was detected
        """
        blink_detected = False
        
        # Check if eye is closed
        if ear < self.ear_threshold:
            if not self.is_eye_closed:
                # Eye just closed
                self.is_eye_closed = True
                self.eye_closed_start = timestamp
                self.consecutive_closed_frames = 1
            else:
                # Eye still closed
                self.consecutive_closed_frames += 1
        else:
            # Eye is open
            if self.is_eye_closed:
                # Eye just opened - potential blink completed
                blink_duration = timestamp - self.eye_closed_start if self.eye_closed_start else 0
                
                # Validate blink duration
                if self.min_blink_duration <= blink_duration <= self.max_blink_duration:
                    # Valid blink detected!
                    self.blink_count += 1
                    self.last_blink_time = timestamp
                    self.blink_timestamps.append(timestamp)  # Record blink event
                    blink_detected = True
                
                # Reset state
                self.is_eye_closed = False
                self.eye_closed_start = None
                self.consecutive_closed_frames = 0
        
        return blink_detected
    
    def get_blink_rate(self, time_window=5.0):
        """
        Calculate blink rate over a time window
        
        Args:
            time_window: Time window in seconds (default: 5.0)
        
        Returns:
            float: Blinks per minute
        """
        if not self.timestamp_history or len(self.timestamp_history) < 2:
            return 0.0
        
        current_time = time.time()
        recent_blinks = 0
        
        # Count blinks within time window
        for ts in reversed(list(self.timestamp_history)):
            if current_time - ts <= time_window:
                recent_blinks += 1
            else:
                break
        
        # Convert to blinks per minute
        blinks_per_minute = (recent_blinks / time_window) * 60.0
        return blinks_per_minute
    
    def reset(self):
        """Reset detector state (thread-safe)"""
        try:
            with self._history_lock:
                self.ear_history.clear()
                self.timestamp_history.clear()
                self.blink_timestamps.clear()  # Clear blink events
            
            with self._state_lock:
                self.blink_count = 0
                self.last_blink_time = 0
                self.eye_closed_start = None
                self.consecutive_closed_frames = 0
                self.is_eye_closed = False
            
            print(f"[BlinkDetector] State reset successfully")
        except Exception as e:
            print(f"[BlinkDetector ERROR] Reset failed: {e}")
    
    def requires_blink(self, verification_start_time, current_time, min_blinks=1):
        """
        Check if sufficient blinks detected during verification period (thread-safe)
        
        Args:
            verification_start_time: When verification started
            current_time: Current time
            min_blinks: Minimum number of blinks required (default: 1)
        
        Returns:
            tuple: (has_blinked: bool, blinks_needed: int)
        """
        try:
            # Thread-safe access to blink timestamps
            with self._history_lock:
                blinks_in_period = sum(
                    1 for ts in self.blink_timestamps 
                    if ts >= verification_start_time
                )
            
            has_blinked = blinks_in_period >= min_blinks
            blinks_needed = max(0, min_blinks - blinks_in_period)
            
            return has_blinked, blinks_needed
        except Exception as e:
            print(f"[BlinkDetector ERROR] requires_blink failed: {e}")
            return False, min_blinks


def get_blink_detector():
    """Factory function to create blink detector"""
    return BlinkDetector()
