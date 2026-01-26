import cv2
import os
from pathlib import Path

# ============= Image Processing Helpers =============

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
    print("✓ Utilities loaded successfully!")
