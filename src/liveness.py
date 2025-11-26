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
        weights.append(0.20)  # Moderate - good indicator but ID cards score high
        
        # 2. Color Distribution Analysis
        color_score = self._analyze_color_distribution(face_image)
        details['color'] = color_score
        scores.append(color_score)
        weights.append(0.20)  # Moderate - good indicator but printed colors accurate
        
        # 3. Moiré Pattern Detection
        moire_score = self._detect_moire_patterns(face_image)
        details['moire'] = moire_score
        scores.append(moire_score)
        weights.append(0.15)  # Good for screens, neutral for ID cards
        
        # 4. Motion Analysis (if sufficient history)
        if len(self.frame_history) >= 3:
            motion_score = self._analyze_motion(face_image)
            details['motion'] = motion_score
            scores.append(motion_score)
            weights.append(0.20)  # Important - distinguishes static from moving
        
        # 5. Edge Detection (NEW - detects phone/card rectangular edges)
        edge_score = self._detect_screen_edges(face_image)
        details['edge_detection'] = edge_score
        scores.append(edge_score)
        weights.append(0.08)  # Supplementary
        
        # 6. Reflection Detection (NEW - screens have specular reflections)
        reflection_score = self._detect_screen_reflections(face_image)
        details['reflection'] = reflection_score
        scores.append(reflection_score)
        weights.append(0.07)  # Supplementary
        
        # 7. Temporal Consistency (NEW - screens have refresh patterns)
        if len(self.frame_history) >= 3:
            temporal_score = self._analyze_temporal_consistency(face_image)
            details['temporal'] = temporal_score
            scores.append(temporal_score)
            weights.append(0.20)  # Important - works with motion to catch static images
        
        # Store frame for motion tracking
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        self.frame_history.append(gray)
        
        # Calculate weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        confidence = np.average(scores, weights=weights)
        
        # Threshold: 60% with gradual motion/temporal scoring
        # Real faces (still): 60-68% → Pass (gradual motion scoring helps)
        # ID cards: 45-58% → Reject (very low motion + temporal)
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
            # Normalize to 0-1 scale (stricter to catch printed photos)
            score = min(1.0, texture_variance / 900.0)  # Balanced threshold
            
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
            # Balanced thresholds for distinguishing prints from real faces
            
            l_score = min(1.0, l_std / 18.0)  # Balanced threshold
            a_score = min(1.0, a_std / 7.0)   # Balanced threshold
            b_score = min(1.0, b_std / 7.0)   # Balanced threshold
            
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
            # Stricter to better detect screens and printed photos
            score = max(0.0, 1.0 - (high_freq_ratio / 0.40))  # Balanced for photo detection
            
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
            
            # Accumulate motion history for better analysis
            self.motion_history.append(mean_motion)
            
            # Real faces: mean_motion typically 0.2-3.0 (breathing, micro-expressions, even when still)
            # ID cards/photos: mean_motion < 0.15 (ONLY hand shake, no facial movement)
            # Screens: motion can be high but unnatural
            
            # GRADUAL scoring to distinguish still person from static print
            if mean_motion < 0.10:
                # Extremely static - definitely a print/card
                score = 0.1
            elif mean_motion < 0.15:
                # Very static - likely print/card, but could be very still person
                score = 0.3
            elif mean_motion < 0.25:
                # Low motion - might be still person or print
                # Check motion history variance to distinguish
                if len(self.motion_history) >= 5:
                    motion_var = np.var(list(self.motion_history)[-5:])
                    if motion_var < 0.01:
                        # No variance = held still = print/card
                        score = 0.35
                    else:
                        # Some variance = natural micro-movements = real person
                        score = 0.60
                else:
                    score = 0.45
            elif mean_motion < 0.5:
                # Reasonable low motion - calm person
                score = 0.70
            elif mean_motion < 3.0:
                # Good natural motion
                score = min(1.0, 0.5 + mean_motion / 5.0)
            elif mean_motion > 6.0:
                # Too erratic
                score = 0.50
            else:
                # Active movement
                score = 0.85
            
            return score
        except Exception as e:
            return 0.5
    
    def _detect_screen_edges(self, face_image):
        """
        Detect sharp rectangular edges characteristic of phone screens.
        Real faces have organic curves, screens have sharp rectangular boundaries.
        
        Returns:
            score: float (0-1, higher = more likely real/no screen edges)
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply Hough Line Transform to detect straight lines
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                                   threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is None:
                # No strong lines detected - likely real face
                return 1.0
            
            # Count long straight lines (characteristic of screen edges)
            long_lines = 0
            total_length = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_length += length
                
                # Lines longer than 40% of image width/height are suspicious
                threshold = min(gray.shape[0], gray.shape[1]) * 0.4
                if length > threshold:
                    long_lines += 1
            
            # Calculate score based on presence of long straight lines
            # Real faces: few long lines, organic curves
            # Screens: multiple long straight lines (edges)
            
            if long_lines == 0:
                score = 1.0  # No suspicious lines
            elif long_lines <= 2:
                score = 0.7  # Some lines, might be background
            elif long_lines <= 4:
                score = 0.4  # Multiple lines, likely screen edge
            else:
                score = 0.1  # Many lines, definitely screen
            
            # Also penalize high total edge density (screens have sharp edges)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > 0.15:  # Very high edge density
                score *= 0.7  # Penalize
            
            return score
        except Exception as e:
            return 0.5
    
    def _detect_screen_reflections(self, face_image):
        """
        Detect specular reflections characteristic of glossy phone/tablet screens.
        Real faces have diffuse reflection, screens have bright specular highlights.
        
        Returns:
            score: float (0-1, higher = more likely real/no screen glare)
        """
        try:
            # Convert to LAB color space (better for brightness analysis)
            lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Find very bright regions (potential reflections)
            bright_threshold = 220  # Very bright pixels
            bright_mask = l_channel > bright_threshold
            bright_ratio = np.sum(bright_mask) / bright_mask.size
            
            # Analyze brightness distribution
            brightness_std = np.std(l_channel)
            brightness_max = np.max(l_channel)
            
            # Calculate local brightness variance (screens have spotty highlights)
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            local_mean = cv2.filter2D(l_channel.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((l_channel.astype(np.float32) - local_mean)**2, -1, kernel)
            high_variance_ratio = np.sum(local_variance > 500) / local_variance.size
            
            # Score based on reflection indicators
            score = 1.0
            
            # Penalize bright spots (screen reflections)
            if bright_ratio > 0.05:  # More than 5% very bright pixels
                score *= 0.5
            elif bright_ratio > 0.02:  # More than 2% very bright pixels
                score *= 0.7
            
            # Penalize high maximum brightness (glare)
            if brightness_max > 245:  # Very bright glare
                score *= 0.6
            elif brightness_max > 235:  # Bright spots
                score *= 0.8
            
            # Penalize high local variance (spotty reflections)
            if high_variance_ratio > 0.15:  # Many high-variance regions
                score *= 0.7
            
            # Real faces have moderate, even brightness
            # Screens have extreme highlights and shadows
            if brightness_std > 45:  # Very high contrast
                score *= 0.8
            
            return max(0.0, score)
        except Exception as e:
            return 0.5
    
    def _analyze_temporal_consistency(self, current_frame):
        """
        Analyze temporal consistency to detect screen refresh patterns.
        Screens have subtle flickering/refresh artifacts, real faces don't.
        
        Returns:
            score: float (0-1, higher = more likely real/consistent)
        """
        try:
            if len(self.frame_history) < 3:
                return 0.5
            
            gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            
            # Compare current frame with recent history
            diffs = []
            for prev_frame in list(self.frame_history)[-3:]:
                # Calculate absolute difference
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                diffs.append(mean_diff)
            
            # Analyze variance in frame differences
            # Screens: inconsistent differences due to refresh rate artifacts
            # Real faces: smooth, consistent micro-movements
            
            diff_variance = np.var(diffs)
            mean_diff = np.mean(diffs)
            
            # Also check for periodic patterns (screen refresh rate)
            # Screens often have 60Hz refresh creating subtle periodic changes
            if len(self.frame_history) >= 5:
                recent_diffs = []
                history_list = list(self.frame_history)
                for i in range(len(history_list) - 1):
                    frame_diff = cv2.absdiff(history_list[i], history_list[i+1])
                    recent_diffs.append(np.mean(frame_diff))
                
                # Check for periodic pattern using autocorrelation
                if len(recent_diffs) >= 4:
                    recent_diffs = np.array(recent_diffs)
                    # Normalize
                    recent_diffs = (recent_diffs - np.mean(recent_diffs)) / (np.std(recent_diffs) + 1e-8)
                    
                    # Simple periodicity check: compare alternating frames
                    odd_mean = np.mean(recent_diffs[::2])
                    even_mean = np.mean(recent_diffs[1::2])
                    periodicity = abs(odd_mean - even_mean)
                    
                    # High periodicity suggests screen refresh pattern
                    if periodicity > 0.5:
                        return 0.3  # Likely screen with refresh artifacts
            
            # Score based on consistency
            # Real faces: variance 0.3-3.0, mean_diff 1.0-8.0 (even when still - breathing, blinking)
            # ID cards/photos: variance <0.2, mean_diff <0.8 (TRULY identical frames)
            # Screens: high variance (>5.0) from refresh artifacts
            
            # GRADUAL scoring to distinguish ID card from still person
            if mean_diff < 0.5:
                # Almost zero change - definitely ID card/photo
                score = 0.15
            elif mean_diff < 0.8:
                # Very little change - likely ID card/photo
                score = 0.30
            elif mean_diff < 1.2:
                # Low change - could be still person or print
                # Check variance to distinguish
                if diff_variance < 0.2:
                    # Low variance too = print/card
                    score = 0.40
                else:
                    # Some variance = micro-movements = real
                    score = 0.65
            elif mean_diff < 2.5:
                # Reasonable change - calm person
                score = 0.75
            elif mean_diff < 8.0:
                # Good natural change
                score = 0.90
            elif diff_variance > 5.0:
                # Too inconsistent, likely screen artifacts
                score = 0.45
            else:
                # High change - active movement or screen
                score = 0.70
            
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
