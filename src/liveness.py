"""
Advanced Liveness Detection Module
Combines multiple anti-spoofing techniques for robust fake face detection
"""

import cv2
import numpy as np
from collections import deque
import time

# Import blink detector (handle both relative and absolute imports)
try:
    from .blink_detector import BlinkDetector
except ImportError:
    from blink_detector import BlinkDetector


class LivenessDetector:
    """
    Multi-method liveness detector combining:
    1. Eye blink detection (PRIMARY - anti-spoofing)
    2. Texture analysis (LBP-based)
    3. Motion detection (optical flow)
    4. Moiré pattern detection
    5. Color distribution analysis
    6. Edge/reflection detection
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
        self.landmark_history = deque(maxlen=10)  # Track last 10 landmark positions
        self.motion_history = deque(maxlen=10)  # Track motion magnitudes
        
        # Initialize blink detector
        self.blink_detector = BlinkDetector(ear_threshold=blink_threshold)
        
        # Verification tracking
        self.verification_start_time = None
        self.frame_counter = 0  # Initialize frame counter for diagnostics
        
    def analyze(self, face_image, landmarks=None):
        """
        Perform comprehensive liveness analysis with eye blink detection
        
        Args:
            face_image: RGB face crop (numpy array)
            landmarks: MediaPipe Face Mesh landmarks (required for blink detection)
            
        Returns:
            is_live: bool - True if face appears to be real
            confidence: float - Confidence score (0-1)
            details: dict - Detailed scores from each method
        """
        if face_image is None or face_image.size == 0:
            return False, 0.0, {}
        
        # Initialize verification timer if not started
        if self.verification_start_time is None:
            self.verification_start_time = time.time()
        
        current_time = time.time()
        details = {}
        scores = []
        weights = []
        
        # 1. EYE BLINK DETECTION (PRIMARY - 40% weight)
        blink_score = 0.0
        if landmarks is not None:
            try:
                blink_detected, current_ear, total_blinks = self.blink_detector.detect_blink(landmarks)
                
                # Check if blink requirement met during verification period
                has_blinked, blinks_needed = self.blink_detector.requires_blink(
                    self.verification_start_time, 
                    current_time, 
                    min_blinks=1
                )
                
                # Score based on blink presence
                elapsed = current_time - self.verification_start_time
                if has_blinked:
                    blink_score = 1.0  # Strong positive signal
                    print(f"  [BLINK] ✓ Blink detected! Total: {total_blinks}, EAR: {current_ear:.3f}")
                elif total_blinks > 0:
                    blink_score = 0.7  # Some blinks detected (good sign)
                    print(f"  [BLINK] Partial blinks: {total_blinks}, EAR: {current_ear:.3f}")
                else:
                    # No blinks yet - check if enough time has passed
                    if elapsed < 2.5:
                        blink_score = 0.5  # Neutral - still waiting
                        print(f"  [BLINK] Waiting for blink... ({elapsed:.1f}s / 2.5s)")
                    else:
                        blink_score = 0.0  # SUSPICIOUS - no blinks after 2.5 seconds
                        print(f"  [BLINK] ⚠️ NO BLINKS DETECTED after {elapsed:.1f}s - LIKELY SPOOF!")
                
                details['blink'] = {
                    'score': blink_score,
                    'has_blinked': has_blinked,
                    'total_blinks': total_blinks,
                    'current_ear': current_ear,
                    'blinks_needed': blinks_needed,
                    'elapsed_time': elapsed
                }
                
                scores.append(blink_score)
                weights.append(0.40)  # PRIMARY indicator - 40% weight
                
            except Exception as e:
                details['blink'] = {
                    'error': str(e), 
                    'score': 0.5,
                    'has_blinked': False,
                    'total_blinks': 0,
                    'current_ear': 0.0,
                    'blinks_needed': 1,
                    'elapsed_time': 0.0
                }
                scores.append(0.5)
                weights.append(0.40)
        else:
            # No landmarks - cannot do blink detection (penalize)
            details['blink'] = {
                'error': 'No landmarks provided', 
                'score': 0.3,
                'has_blinked': False,
                'total_blinks': 0,
                'current_ear': 0.0,
                'blinks_needed': 1,
                'elapsed_time': 0.0
            }
            scores.append(0.3)
            weights.append(0.40)
        
        # 2. Texture Analysis (LBP-based)
        texture_score = self._analyze_texture(face_image)
        details['texture'] = texture_score
        scores.append(texture_score)
        weights.append(0.10)  # Reduced weight
        
        # 3. Color Distribution Analysis
        color_score = self._analyze_color_distribution(face_image)
        details['color'] = color_score
        scores.append(color_score)
        weights.append(0.08)  # Reduced weight
        
        # 4. Blue Light Analysis (phone screen detection)
        blue_score = self._detect_blue_light(face_image)
        details['blue_light'] = blue_score
        scores.append(blue_score)
        weights.append(0.08)  # Phone screen indicator
        
        # 5. Moiré Pattern Detection
        moire_score = self._detect_moire_patterns(face_image)
        details['moire'] = moire_score
        scores.append(moire_score)
        weights.append(0.08)  # Reduced weight
        
        # 6. Facial Landmark Motion Analysis (detects expression changes vs rigid movement)
        landmark_motion_score = 0.5  # Default neutral
        if landmarks is not None:
            landmark_motion_score = self._analyze_landmark_motion(landmarks)
            details['landmark_motion'] = landmark_motion_score
            scores.append(landmark_motion_score)
            weights.append(0.22)  # Key differentiator for moving phone vs real face
        
        # 7. Frame Motion Analysis (if sufficient history) - reduced weight
        if len(self.frame_history) >= 3:
            motion_score = self._analyze_motion(face_image)
            details['motion'] = motion_score
            scores.append(motion_score)
            weights.append(0.03)  # Reduced - can be fooled by moving phone
        
        # 8. Screen Edge Detection
        edge_score = self._detect_screen_edges(face_image)
        details['edge_detection'] = edge_score
        scores.append(edge_score)
        weights.append(0.05)  # Increased for phone detection
        
        # 9. Screen Reflection Detection (ENHANCED)
        reflection_score = self._detect_screen_reflections(face_image)
        details['reflection'] = reflection_score
        scores.append(reflection_score)
        weights.append(0.05)  # Increased for phone detection
        
        # 10. Temporal Consistency
        if len(self.frame_history) >= 3:
            temporal_score = self._analyze_temporal_consistency(face_image)
            details['temporal'] = temporal_score
            scores.append(temporal_score)
            weights.append(0.04)
        
        # Store frame for motion tracking
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        self.frame_history.append(gray)
        
        # Calculate weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        confidence = np.average(scores, weights=weights)
        
        # Decision threshold: 58% (stricter to reject phone screens)
        # Real faces with blinks: 70-85%
        # Real faces without blinks yet (waiting): 52-65%
        # Photos/screens (no blinks): 20-45%
        is_live = confidence >= 0.58
        
        # Log decision reasoning
        if not is_live:
            print(f"  [LIVENESS] ⚠️ SPOOF DETECTED: confidence={confidence:.1%} < threshold=58%")
            if 'blink' in details and 'has_blinked' in details['blink']:
                print(f"              Blink score: {details['blink']['score']:.1%}, Has blinked: {details['blink']['has_blinked']}")
            if 'blue_light' in details:
                print(f"              Blue light score: {details['blue_light']:.1%} (low = screen detected)")
            if 'reflection' in details:
                print(f"              Reflection score: {details['reflection']:.1%} (low = glare detected)")
        
        details['overall'] = confidence
        details['decision'] = 'Real' if is_live else 'Spoof'
        details['elapsed_time'] = current_time - self.verification_start_time
        
        # Per-frame diagnostic logging
        self.frame_counter += 1
        self._log_diagnostics(confidence, details)
        
        return is_live, confidence, details
    
    def reset(self):
        """Reset detector state for new verification"""
        self.frame_history.clear()
        self.landmark_history.clear()
        self.motion_history.clear()
        if hasattr(self, 'blink_detector') and self.blink_detector:
            self.blink_detector.reset()
        self.verification_start_time = None
        # Don't reset frame_counter - keep it incrementing for analysis
    
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
    
    def _detect_blue_light(self, face_image):
        """
        Detect blue light characteristic of phone/tablet screens.
        Phone screens emit more blue light (cooler color temperature) than natural faces.
        
        Returns:
            score: float (0-1, higher = more likely real/natural lighting)
        """
        try:
            # Analyze color channels
            b_channel = face_image[:, :, 2].astype(np.float32)
            g_channel = face_image[:, :, 1].astype(np.float32)
            r_channel = face_image[:, :, 0].astype(np.float32)
            
            # Calculate average intensities
            b_mean = np.mean(b_channel)
            g_mean = np.mean(g_channel)
            r_mean = np.mean(r_channel)
            
            # Calculate blue/red ratio (phone screens have higher B/R ratio)
            if r_mean > 10:  # Avoid division by zero
                br_ratio = b_mean / r_mean
            else:
                br_ratio = 1.0
            
            # Calculate blue/green ratio
            if g_mean > 10:
                bg_ratio = b_mean / g_mean
            else:
                bg_ratio = 1.0
            
            # Real faces under natural/warm lighting: B/R ratio 0.7-0.95
            # Phone screens (LED backlight): B/R ratio 0.95-1.3 (more blue)
            # Phone screens: B/G ratio 0.95-1.2
            
            score = 1.0
            
            # Penalize blue-shifted images (phone screens)
            if br_ratio > 1.15:  # Strong blue shift
                score *= 0.4
            elif br_ratio > 1.05:  # Moderate blue shift
                score *= 0.6
            elif br_ratio > 0.98:  # Slight blue shift
                score *= 0.8
            
            # Check blue-green ratio as well
            if bg_ratio > 1.10:  # Strong blue dominance
                score *= 0.5
            elif bg_ratio > 1.00:  # Moderate blue dominance
                score *= 0.75
            
            # Also check for overall color temperature
            # Calculate color temperature indicator
            color_temp = (r_mean + g_mean) / (b_mean + 1)
            
            # Natural faces: color_temp > 1.8 (warmer)
            # Phone screens: color_temp 1.3-1.7 (cooler)
            if color_temp < 1.4:  # Very cool (blue)
                score *= 0.5
            elif color_temp < 1.6:  # Cool
                score *= 0.7
            elif color_temp < 1.8:  # Slightly cool
                score *= 0.85
            
            # Check for blue dominance in bright regions (screen glare is very blue)
            bright_mask = (r_channel + g_channel + b_channel) > 600
            if np.sum(bright_mask) > 0:
                bright_b_mean = np.mean(b_channel[bright_mask])
                bright_r_mean = np.mean(r_channel[bright_mask])
                if bright_r_mean > 10:
                    bright_br_ratio = bright_b_mean / bright_r_mean
                    if bright_br_ratio > 1.2:  # Blue glare (typical of screens)
                        score *= 0.5
            
            return max(0.0, score)
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
    
    def _analyze_landmark_motion(self, landmarks):
        """
        Analyze facial landmark motion to detect INTERNAL facial deformation.
        Real faces: landmarks move independently (expressions, micro-movements)
        Moving phone: ALL landmarks move rigidly together (no deformation)
        
        Returns:
            score: float (0-1, higher = more likely real)
        """
        try:
            # Extract key landmarks as numpy array
            key_indices = [
                33, 263,    # Eye corners
                61, 291,    # Mouth corners
                1,          # Nose tip
                152,        # Chin
                10, 338,    # Forehead points
                199, 428    # Cheek points
            ]
            
            current_landmarks = np.array([
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
                for i in key_indices
            ])
            
            # Store in history
            self.landmark_history.append(current_landmarks)
            
            if len(self.landmark_history) < 3:
                return 0.5  # Not enough history
            
            # Calculate relative movements between landmarks
            prev_landmarks = self.landmark_history[-2]
            
            # Calculate displacement for each landmark
            displacements = current_landmarks - prev_landmarks
            displacement_magnitudes = np.linalg.norm(displacements, axis=1)
            
            # Key insight: Real faces have VARIED landmark movements
            # Moving phone has UNIFORM landmark movements (all points move same amount)
            
            # Calculate variance in displacement magnitudes
            displacement_variance = np.var(displacement_magnitudes)
            mean_displacement = np.mean(displacement_magnitudes)
            
            # Calculate coefficient of variation (CV = std / mean)
            if mean_displacement > 0.001:
                cv = np.std(displacement_magnitudes) / mean_displacement
            else:
                cv = 0
            
            # Real faces: high CV (varied movement) - expressions cause differential motion
            # Moving phone: low CV (uniform movement) - rigid body motion
            
            # IMPROVED: Sharper sigmoid penalty for rigid motion
            # Use sigmoid mapping: score = 1 / (1 + exp(-k*(cv - threshold)))
            # This creates a steep penalty zone for low CV values
            
            if cv > 0.4:
                # High variation - definitely real face with expressions
                score = 1.0
            elif cv > 0.25:
                # Good variation - likely real face
                score = 0.90
            elif cv > 0.15:
                # Moderate variation - acceptable for real faces
                score = 0.70
            elif cv > 0.10:
                # Low-moderate variation - borderline (could be subtle expressions)
                score = 0.45
            elif cv > 0.06:
                # Low variation - suspicious (likely moving phone)
                score = 0.20
            else:
                # Very uniform motion - STRONG penalty for rigid body motion
                score = 0.05
            
            # Additionally check for micro-expressions in mouth/eyes
            # (distance changes between specific landmark pairs)
            mouth_distance = np.linalg.norm(current_landmarks[2] - current_landmarks[3])
            prev_mouth_distance = np.linalg.norm(prev_landmarks[2] - prev_landmarks[3])
            mouth_change = abs(mouth_distance - prev_mouth_distance)
            
            if mouth_change > 0.005:  # Mouth movement detected
                score = min(1.0, score + 0.15)
            
            return score
            
        except Exception as e:
            return 0.5  # Neutral on error
    
    def _log_diagnostics(self, confidence, details):
        """Log per-frame diagnostics to CSV for analysis"""
        try:
            import csv
            import os
            
            outpath = 'outputs/liveness_debug.csv'
            write_header = not os.path.exists(outpath)
            
            # Extract key metrics
            blink_data = details.get('blink', {})
            landmark_data = details.get('landmark_motion', {})
            
            debug_line = {
                'timestamp': time.time(),
                'frame': self.frame_counter,
                'confidence': float(confidence),
                'decision': details.get('decision', 'Unknown'),
                'blink_score': float(blink_data.get('score', 0.0) if isinstance(blink_data, dict) else blink_data),
                'has_blinked': bool(blink_data.get('has_blinked', False)) if isinstance(blink_data, dict) else False,
                'total_blinks': int(blink_data.get('total_blinks', 0)) if isinstance(blink_data, dict) else 0,
                'landmark_score': float(landmark_data if isinstance(landmark_data, (int, float)) else 0.5),
                'texture': float(details.get('texture', 0.0)),
                'color': float(details.get('color', 0.0)),
                'blue_light': float(details.get('blue_light', 0.0)),
                'moire': float(details.get('moire', 0.0)),
                'reflection': float(details.get('reflection', 0.0)),
                'edge_detection': float(details.get('edge_detection', 0.0)),
                'motion': float(details.get('motion', 0.0)),
                'elapsed': float(details.get('elapsed_time', 0.0))
            }
            
            with open(outpath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(debug_line.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(debug_line)
        except Exception as e:
            pass  # Don't fail on logging errors
    
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
        ENHANCED: Detect specular reflections characteristic of glossy phone/tablet screens.
        Real faces have diffuse reflection, screens have bright specular highlights.
        
        Returns:
            score: float (0-1, higher = more likely real/no screen glare)
        """
        try:
            # Convert to LAB color space (better for brightness analysis)
            lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Find very bright regions (potential reflections)
            bright_threshold = 215  # Lowered to catch more reflections
            bright_mask = l_channel > bright_threshold
            bright_ratio = np.sum(bright_mask) / bright_mask.size
            
            # Find extremely bright spots (strong glare)
            extreme_bright_mask = l_channel > 240
            extreme_bright_ratio = np.sum(extreme_bright_mask) / extreme_bright_mask.size
            
            # Analyze brightness distribution
            brightness_std = np.std(l_channel)
            brightness_max = np.max(l_channel)
            brightness_mean = np.mean(l_channel)
            
            # Calculate local brightness variance (screens have spotty highlights)
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            local_mean = cv2.filter2D(l_channel.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((l_channel.astype(np.float32) - local_mean)**2, -1, kernel)
            high_variance_ratio = np.sum(local_variance > 400) / local_variance.size  # Lowered threshold
            
            # Detect concentrated bright spots (typical of screen glare)
            # Use morphological operations to find compact bright regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            bright_dilated = cv2.dilate(bright_mask.astype(np.uint8), kernel)
            bright_eroded = cv2.erode(bright_dilated, kernel)
            compact_bright_ratio = np.sum(bright_eroded > 0) / bright_eroded.size
            
            # Score based on reflection indicators
            score = 1.0
            
            # STRONGER penalties for phone screen reflections
            
            # Penalize bright spots (screen reflections)
            if bright_ratio > 0.08:  # More than 8% very bright
                score *= 0.3  # Strong penalty
            elif bright_ratio > 0.05:  # More than 5% very bright
                score *= 0.5
            elif bright_ratio > 0.03:  # More than 3% very bright
                score *= 0.7
            elif bright_ratio > 0.015:  # More than 1.5% very bright
                score *= 0.85
            
            # Penalize extreme bright spots (strong glare)
            if extreme_bright_ratio > 0.02:  # More than 2% extreme bright
                score *= 0.4  # Very strong penalty
            elif extreme_bright_ratio > 0.01:  # More than 1% extreme bright
                score *= 0.6
            
            # Penalize compact bright spots (concentrated glare typical of screens)
            if compact_bright_ratio > 0.03:
                score *= 0.5
            elif compact_bright_ratio > 0.015:
                score *= 0.75
            
            # Penalize high maximum brightness (glare)
            if brightness_max > 250:  # Extreme glare
                score *= 0.4
            elif brightness_max > 240:  # Very bright glare
                score *= 0.6
            elif brightness_max > 230:  # Bright spots
                score *= 0.8
            
            # Penalize high local variance (spotty reflections)
            if high_variance_ratio > 0.20:  # Many high-variance regions
                score *= 0.5
            elif high_variance_ratio > 0.12:  # Some high-variance regions
                score *= 0.7
            
            # Real faces have moderate, even brightness
            # Screens have extreme highlights and shadows
            if brightness_std > 50:  # Very high contrast
                score *= 0.7
            elif brightness_std > 40:  # High contrast
                score *= 0.85
            
            # Penalize overly bright faces (screens tend to be brighter)
            if brightness_mean > 180:  # Unusually bright
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
    
    # reset() method is defined earlier in the class


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
