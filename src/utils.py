import cv2
import os
from pathlib import Path
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


if __name__ == "__main__":
    print("Testing utilities...")
    try:
        print(f"MediaPipe Face Detection loaded: {face_detector_mp is not None}")
    except:
        print("MediaPipe Face Detection: Failed to load")
    print("✓ Utilities loaded successfully!")
