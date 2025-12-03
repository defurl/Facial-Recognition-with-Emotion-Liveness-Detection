import cv2
import os
from pathlib import Path
import mediapipe as mp
import numpy as np
import math

# ============= Face Detection =============

# Initialize MediaPipe Face Detection (preferred)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector_mp = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for close-range (< 2m), 1 for full-range
    min_detection_confidence=0.5
)

# Initialize MediaPipe Face Mesh for pose estimation
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Note: Haar cascade fallback removed — MediaPipe-only detector


def detect_faces(frame):
    """
    Detect faces in a frame using MediaPipe Face Detection only.

    args:
        frame: BGR image from OpenCV

    returns:
        List of (x, y, w, h) face bounding boxes
    """
    height, width, _ = frame.shape

    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector_mp.process(rgb_frame)

        faces = []
        if results and results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute pixels
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = max(0, min(w, width - x))
                h = max(0, min(h, height - y))

                # Basic sanity check for reasonable face dimensions
                if w > 10 and h > 10:
                    faces.append((x, y, w, h))

        return faces
    except Exception as e:
        # If MediaPipe fails for an unexpected reason, return empty list.
        print(f"MediaPipe detection failed: {e}")
        return []


def crop_face_with_padding(frame, x, y, w, h, padding_ratio=0.1):
    """
    Crop face from frame with padding.
    
    args:
        frame: BGR image
        x, y, w, h: Face bounding box
        padding_ratio: Padding ratio relative to max(w, h)
    
    returns:
        Cropped face image
    """
    padding = int(padding_ratio * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    
    cropped_face = frame[y1:y2, x1:x2]
    return cropped_face


# ============= Image Processing Helpers =============

def save_temp_image(image, filename="temp_face.jpg"):
    """
    Save temporary image file.
    
    args:
        image: OpenCV BGR image
        filename: Temporary filename
    
    returns:
        Path to saved file
    """
    cv2.imwrite(filename, image)
    return filename


def cleanup_temp_files(*filenames):
    """
    Remove temporary files.
    
    args:
        *filenames: Variable number of filenames to remove
    """
    for filename in filenames:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass


# ============= Drawing Utilities =============

def draw_face_box(frame, x, y, w, h, label, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box and label on frame.
    
    Args:
        frame: BGR image
        x, y, w, h: Bounding box
        label: Text label to display
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Modified frame
    """
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, color, thickness)
    return frame


# ============= Quality Validation =============

def check_image_blur(image, threshold=100):
    """
    Check if image is blurry using Laplacian variance.
    
    Args:
        image: BGR image (numpy array)
        threshold: Variance threshold (lower = blurrier)
    
    Returns:
        tuple: (variance_score: float, passes: bool, message: str)
    """
    try:
        if image is None or image.size == 0:
            return (0.0, False, "Invalid image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        passes = variance >= threshold
        message = "Sharp" if passes else "Blurry"
        
        return (float(variance), passes, message)
    except Exception as e:
        print(f"check_image_blur error: {e}")
        return (0.0, False, "Error checking blur")


def check_image_lighting(image, min_bright=30, max_bright=220, min_contrast=30):
    """
    Check if image has adequate lighting and contrast.
    
    Args:
        image: BGR image (numpy array)
        min_bright: Minimum acceptable mean brightness
        max_bright: Maximum acceptable mean brightness
        min_contrast: Minimum acceptable contrast (std dev)
    
    Returns:
        tuple: (brightness: float, contrast: float, passes: bool, message: str)
    """
    try:
        if image is None or image.size == 0:
            return (0.0, 0.0, False, "Invalid image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute brightness (mean) and contrast (std dev)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Check conditions
        if brightness < min_bright:
            return (brightness, contrast, False, "Too dark")
        elif brightness > max_bright:
            return (brightness, contrast, False, "Too bright")
        elif contrast < min_contrast:
            return (brightness, contrast, False, "Low contrast")
        else:
            return (brightness, contrast, True, "Good")
            
    except Exception as e:
        print(f"check_image_lighting error: {e}")
        return (0.0, 0.0, False, "Error checking lighting")


def estimate_head_pose_angles(image):
    """
    Estimate head pose angles (yaw, pitch, roll) using MediaPipe Face Mesh.
    
    Args:
        image: BGR image (numpy array)
    
    Returns:
        tuple: (yaw: float, pitch: float, roll: float, pose_label: str, 
                matches_target: bool, tolerance_used: float)
        Returns (0, 0, 0, "unknown", False, 0) on failure
    """
    try:
        if image is None or image.size == 0:
            return (0.0, 0.0, 0.0, "unknown", False, 0.0)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with Face Mesh
        results = face_mesh_detector.process(rgb_image)
        
        if not results or not results.multi_face_landmarks:
            return (0.0, 0.0, 0.0, "no_face", False, 0.0)
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract key landmark points for pose estimation
        # Using nose tip, chin, left eye, right eye, left mouth, right mouth
        h, w = image.shape[:2]
        
        # Key landmark indices
        nose_tip = face_landmarks.landmark[1]  # Nose tip
        chin = face_landmarks.landmark[152]  # Chin
        left_eye = face_landmarks.landmark[33]  # Left eye outer corner
        right_eye = face_landmarks.landmark[263]  # Right eye outer corner
        left_mouth = face_landmarks.landmark[61]  # Left mouth corner
        right_mouth = face_landmarks.landmark[291]  # Right mouth corner
        
        # Convert to pixel coordinates
        nose_2d = np.array([nose_tip.x * w, nose_tip.y * h])
        chin_2d = np.array([chin.x * w, chin.y * h])
        left_eye_2d = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_2d = np.array([right_eye.x * w, right_eye.y * h])
        
        # Calculate yaw (horizontal rotation) from eye positions
        eye_center = (left_eye_2d + right_eye_2d) / 2
        face_center_x = w / 2
        horizontal_offset = eye_center[0] - face_center_x
        yaw = math.degrees(math.atan2(horizontal_offset, w * 0.5))
        
        # Calculate pitch (vertical rotation) from nose to chin
        vertical_offset = nose_2d[1] - chin_2d[1]
        pitch = math.degrees(math.atan2(vertical_offset, h * 0.3))
        
        # Calculate roll (tilt) from eye line angle
        eye_diff = right_eye_2d - left_eye_2d
        roll = math.degrees(math.atan2(eye_diff[1], eye_diff[0]))
        
        # Determine pose label
        pose_label = "center"
        if abs(yaw) > 20:
            pose_label = "left" if yaw < 0 else "right"
        elif abs(pitch) > 10:
            pose_label = "up" if pitch > 0 else "down"
        
        # matches_target and tolerance_used will be set by caller based on context
        return (float(yaw), float(pitch), float(roll), pose_label, False, 0.0)
        
    except Exception as e:
        print(f"estimate_head_pose_angles error: {e}")
        return (0.0, 0.0, 0.0, "error", False, 0.0)


def detect_eyewear(image):
    """
    Detect if person is wearing glasses using facial landmarks.
    
    Args:
        image: BGR image (numpy array)
    
    Returns:
        str or None: Warning message if glasses detected, None otherwise
    """
    try:
        if image is None or image.size == 0:
            return None
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with Face Mesh
        results = face_mesh_detector.process(rgb_image)
        
        if not results or not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Check eye region landmarks for unusual patterns
        # Simplified detection: check if eye landmarks have high variance
        # (glasses frames create edge artifacts)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        h, w = image.shape[:2]
        
        # Extract eye region intensities (simplified check)
        # In practice, more sophisticated detection would be needed
        # For now, return None (non-blocking warning only)
        
        return None  # Placeholder - can be enhanced with ML model
        
    except Exception as e:
        print(f"detect_eyewear error: {e}")
        return None


def validate_pose_for_target(yaw, pitch, target_pose, strict_tolerance=10.0, relaxed_tolerance=15.0, 
                              is_strict=True):
    """
    Validate if current pose matches target pose within tolerance.
    
    Args:
        yaw: Current yaw angle
        pitch: Current pitch angle
        target_pose: Target pose ("center", "left", "right", "up", "down")
        strict_tolerance: Tolerance for yaw in strict mode
        relaxed_tolerance: Tolerance for yaw in relaxed mode
        is_strict: Whether to use strict tolerance
    
    Returns:
        tuple: (matches: bool, tolerance_used: float, feedback: str)
    """
    try:
        # Account for horizontally flipped camera - invert yaw for left/right detection
        # When camera is flipped, turning right appears as negative yaw, turning left as positive yaw
        mirrored_yaw = -yaw  # Invert yaw to match user's perspective
        
        yaw_tol = strict_tolerance if is_strict else relaxed_tolerance
        pitch_tol = 60.0 if is_strict else 70.0  # Very lenient for pitch (camera angle variations)
        
        if target_pose == "center":
            matches = abs(mirrored_yaw) <= yaw_tol and abs(pitch) <= pitch_tol
            feedback = "OK" if matches else f"Look straight"
        elif target_pose == "left":
            target_yaw = -30
            matches = (mirrored_yaw < -15) and (mirrored_yaw > -50) and abs(pitch) <= pitch_tol
            feedback = "OK" if matches else f"Turn left"
        elif target_pose == "right":
            target_yaw = 30
            matches = (mirrored_yaw > 15) and (mirrored_yaw < 50) and abs(pitch) <= pitch_tol
            feedback = "OK" if matches else f"Turn right"
        elif target_pose == "up":
            # Looking up: pitch should be MORE NEGATIVE (e.g., -60 to -80)
            matches = (pitch < -55) and (pitch > -85) and abs(mirrored_yaw) <= yaw_tol
            feedback = "OK" if matches else f"Look up"
        elif target_pose == "down":
            # Looking down: pitch should be LESS NEGATIVE (e.g., -20 to -40)
            matches = (pitch > -45) and (pitch < -15) and abs(mirrored_yaw) <= yaw_tol
            feedback = "OK" if matches else f"Look down"
        else:
            matches = False
            feedback = "Unknown target"
        
        tolerance = yaw_tol if target_pose in ["center", "left", "right"] else pitch_tol
        return (matches, tolerance, feedback)
        
    except Exception as e:
        print(f"validate_pose_for_target error: {e}")
        return (False, 0.0, "Error validating pose")


if __name__ == "__main__":
    print("Testing utilities...")
    try:
        print(f"MediaPipe Face Detection loaded: {face_detector_mp is not None}")
    except:
        print("MediaPipe Face Detection: Failed to load")
    print("✓ Utilities loaded successfully!")
