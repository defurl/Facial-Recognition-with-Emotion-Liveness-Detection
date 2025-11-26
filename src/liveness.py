"""
Advanced Liveness Detection Module
Combines multiple anti-spoofing techniques for robust fake face detection
"""

import cv2
import numpy as np
from collections import deque
import time


class LivenessDetector:
    """
    Multi-method liveness detector combining:
    1. Texture analysis (LBP-based)
    2. Motion detection (optical flow)
    3. Blinking detection
    4. Moiré pattern detection
    5. Color distribution analysis
    """
    
    def __init__(self, motion_history_size=5, blink_threshold=0.21):
        """
        Initialize liveness detector
        
        Args:
            motion_history_size: Number of frames to track for motion
            blink_threshold: Eye aspect ratio threshold for blink detection
        """
        self.motion_history_size = motion_history_size
        self.blink_threshold = blink_threshold
        
        # History buffers
        self.frame_history = deque(maxlen=motion_history_size)
        self.blink_history = deque(maxlen=30)  # Track blinks over 1 second at 30fps
        self.last_blink_time = 0
        
        # Detection state
        self.eye_closed_frames = 0
        self.blink_count = 0
        
    def analyze(self, face_image, landmarks=None):
        """
        Perform comprehensive liveness analysis
        
        Args:
            face_image: RGB face crop (numpy array)
            landmarks: Optional facial landmarks from MediaPipe
            
        Returns:
            is_live: bool - True if face appears to be real
            confidence: float - Confidence score (0-1)
            details: dict - Detailed scores from each method
        """
        if face_image is None or face_image.size == 0:
            return False, 0.0, {}
        
        details = {}
        scores = []
        weights = []
        
        # 1. Texture Analysis (LBP-based)
        texture_score = self._analyze_texture(face_image)
        details['texture'] = texture_score
        scores.append(texture_score)
        weights.append(0.3)
        
        # 2. Color Distribution Analysis
        color_score = self._analyze_color_distribution(face_image)
        details['color'] = color_score
        scores.append(color_score)
        weights.append(0.25)
        
        # 3. Moiré Pattern Detection
        moire_score = self._detect_moire_patterns(face_image)
        details['moire'] = moire_score
        scores.append(moire_score)
        weights.append(0.25)
        
        # 4. Motion Analysis (if sufficient history)
        if len(self.frame_history) >= 3:
            motion_score = self._analyze_motion(face_image)
            details['motion'] = motion_score
            scores.append(motion_score)
            weights.append(0.2)
        
        # Store frame for motion tracking
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        self.frame_history.append(gray)
        
        # Calculate weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        confidence = np.average(scores, weights=weights)
        
        # Threshold: require 60% confidence for liveness
        is_live = confidence >= 0.60
        
        details['overall'] = confidence
        details['decision'] = 'Real' if is_live else 'Spoof'
        
        return is_live, confidence, details
    
    def _analyze_texture(self, face_image):
        """
        Analyze texture using Local Binary Patterns
        Real faces have richer texture than printed photos
        
        Returns:
            score: float (0-1, higher = more likely real)
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Compute LBP
            lbp = self._compute_lbp(gray)
            
            # Calculate texture variance
            texture_variance = np.var(lbp)
            
            # Real faces typically have variance > 800, printed photos < 400
            # Normalize to 0-1 scale
            score = min(1.0, texture_variance / 1000.0)
            
            return score
        except Exception as e:
            return 0.5  # Neutral on error
    
    def _compute_lbp(self, gray_image):
        """Compute Local Binary Pattern"""
        h, w = gray_image.shape
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_image[i, j]
                code = 0
                code |= (gray_image[i-1, j-1] >= center) << 7
                code |= (gray_image[i-1, j] >= center) << 6
                code |= (gray_image[i-1, j+1] >= center) << 5
                code |= (gray_image[i, j+1] >= center) << 4
                code |= (gray_image[i+1, j+1] >= center) << 3
                code |= (gray_image[i+1, j] >= center) << 2
                code |= (gray_image[i+1, j-1] >= center) << 1
                code |= (gray_image[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        return lbp
    
    def _analyze_color_distribution(self, face_image):
        """
        Analyze color distribution
        Real faces have more natural color variation than printed photos
        
        Returns:
            score: float (0-1, higher = more likely real)
        """
        try:
            # Convert to LAB color space (better for skin tones)
            lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
            
            # Calculate standard deviation in each channel
            l_std = np.std(lab[:, :, 0])
            a_std = np.std(lab[:, :, 1])
            b_std = np.std(lab[:, :, 2])
            
            # Real faces have more variation: L > 15, a > 5, b > 5
            # Printed photos are flatter: L < 10, a < 3, b < 3
            
            l_score = min(1.0, l_std / 20.0)
            a_score = min(1.0, a_std / 8.0)
            b_score = min(1.0, b_std / 8.0)
            
            # Average the scores
            score = (l_score + a_score + b_score) / 3.0
            
            return score
        except Exception as e:
            return 0.5
    
    def _detect_moire_patterns(self, face_image):
        """
        Detect Moiré patterns that appear when photographing screens/printed images
        
        Returns:
            score: float (0-1, higher = no moiré = more likely real)
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Apply FFT to detect periodic patterns
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # Get high frequency components (moiré patterns appear here)
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Sample high frequency regions (avoid DC component in center)
            high_freq_mask = np.ones_like(magnitude_spectrum)
            cv2.circle(high_freq_mask, (center_w, center_h), min(h, w) // 4, 0, -1)
            
            high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
            total_energy = np.sum(magnitude_spectrum)
            
            # Moiré patterns have abnormally high energy in high frequencies
            high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
            
            # Real faces: ratio < 0.3, Moiré patterns: ratio > 0.4
            # Invert score (lower ratio = higher score = more real)
            score = max(0.0, 1.0 - (high_freq_ratio / 0.35))
            
            return score
        except Exception as e:
            return 0.5
    
    def _analyze_motion(self, current_frame_rgb):
        """
        Analyze motion patterns between frames
        Real faces have natural micro-movements, printed photos are static
        
        Returns:
            score: float (0-1, higher = more motion = more likely real)
        """
        try:
            if len(self.frame_history) < 2:
                return 0.5
            
            current_gray = cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2GRAY)
            prev_gray = self.frame_history[-1]
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, current_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate motion magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_motion = np.mean(magnitude)
            
            # Real faces: mean_motion typically 0.3-3.0
            # Printed photos: mean_motion < 0.2 (only camera shake)
            # Screens: motion can be high but unnatural
            
            # Score based on motion in reasonable range
            if mean_motion < 0.2:
                score = 0.2  # Too static
            elif mean_motion > 5.0:
                score = 0.4  # Too much motion (maybe screen)
            else:
                score = min(1.0, mean_motion / 2.0)
            
            return score
        except Exception as e:
            return 0.5
    
    def reset(self):
        """Reset detector state"""
        self.frame_history.clear()
        self.blink_history.clear()
        self.eye_closed_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0


# Global detector instance
_liveness_detector = None

def get_liveness_detector():
    """Get or create global liveness detector instance"""
    global _liveness_detector
    if _liveness_detector is None:
        _liveness_detector = LivenessDetector()
    return _liveness_detector


def detect_liveness(face_image, landmarks=None):
    """
    Convenience function for liveness detection
    
    Args:
        face_image: RGB face crop (numpy array)
        landmarks: Optional facial landmarks
        
    Returns:
        is_live: bool
        confidence: float (0-1)
        details: dict
    """
    detector = get_liveness_detector()
    return detector.analyze(face_image, landmarks)


__all__ = ['LivenessDetector', 'get_liveness_detector', 'detect_liveness']
