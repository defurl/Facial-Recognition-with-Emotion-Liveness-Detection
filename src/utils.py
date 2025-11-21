import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp

# ============= Face Detection =============

# Initialize MediaPipe Face Detection (preferred)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector_mp = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for close-range (< 2m), 1 for full-range
    min_detection_confidence=0.5
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


# ============= Face Quality Validation =============

# Initialize MediaPipe Face Mesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def calculate_blur_score(image):
    """
    Calculate blur score using Laplacian variance.
    
    Args:
        image: BGR or grayscale image
    
    Returns:
        float: Blur score (higher = sharper, lower = blurrier)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def check_brightness(image):
    """
    Check if image brightness is within acceptable range.
    
    Args:
        image: BGR image
    
    Returns:
        tuple: (is_valid, brightness_value, message)
    """
    # Convert to grayscale and calculate mean brightness
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    brightness = np.mean(gray)
    
    if brightness < 40:
        return False, brightness, "Too dark"
    elif brightness > 220:
        return False, brightness, "Too bright"
    else:
        return True, brightness, "OK"


def check_frontal_pose(image):
    """
    Check if face is in frontal pose using MediaPipe Face Mesh landmarks.
    
    Args:
        image: BGR image (cropped face)
    
    Returns:
        tuple: (is_frontal, message)
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return False, "No face landmarks detected"
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get key landmarks for pose estimation
        # Left eye outer corner: 33, Right eye outer corner: 263
        # Nose tip: 1, Left mouth corner: 61, Right mouth corner: 291
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Calculate face width at eye level
        eye_width = abs(right_eye.x - left_eye.x)
        
        # Calculate horizontal symmetry (nose should be centered)
        nose_to_left = abs(nose_tip.x - left_eye.x)
        nose_to_right = abs(right_eye.x - nose_tip.x)
        symmetry_ratio = min(nose_to_left, nose_to_right) / max(nose_to_left, nose_to_right)
        
        # Calculate mouth symmetry
        mouth_width = abs(right_mouth.x - left_mouth.x)
        mouth_to_eye_ratio = mouth_width / eye_width if eye_width > 0 else 0
        
        # Frontal pose criteria:
        # 1. Nose should be roughly centered (symmetry ratio > 0.8)
        # 2. Mouth width should be proportional to eye width (0.6 to 1.2)
        is_frontal = symmetry_ratio > 0.75 and 0.5 < mouth_to_eye_ratio < 1.3
        
        if not is_frontal:
            if symmetry_ratio <= 0.75:
                return False, "Face camera directly (head turned)"
            else:
                return False, "Adjust face angle"
        
        return True, "OK"
        
    except Exception as e:
        # If landmark detection fails, be conservative
        return False, f"Pose check failed: {str(e)}"


def validate_registration_quality(cropped_face, all_faces_count):
    """
    Comprehensive validation of face quality for registration.
    
    Args:
        cropped_face: BGR image of cropped face
        all_faces_count: Total number of faces detected in frame
    
    Returns:
        tuple: (is_valid, message)
    """
    # Check 1: Multiple faces
    if all_faces_count > 1:
        return False, "MULTIPLE FACES - Only one person allowed"
    
    # Check 2: Image size
    if cropped_face.size == 0 or cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50:
        return False, "Face too small"
    
    # Check 3: Blur detection
    blur_score = calculate_blur_score(cropped_face)
    if blur_score < 100:
        return False, f"Too blurry (score: {blur_score:.1f})"
    
    # Check 4: Brightness
    is_bright_ok, brightness, brightness_msg = check_brightness(cropped_face)
    if not is_bright_ok:
        return False, brightness_msg
    
    # Check 5: Frontal pose
    is_frontal, pose_msg = check_frontal_pose(cropped_face)
    if not is_frontal:
        return False, pose_msg
    
    return True, "OK"


if __name__ == "__main__":
    print("Testing utilities...")
    try:
        print(f"MediaPipe Face Detection loaded: {face_detector_mp is not None}")
    except:
        print("MediaPipe Face Detection: Failed to load")
    print("✓ Utilities loaded successfully!")
