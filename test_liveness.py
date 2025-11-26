"""
Test script for liveness detection module
Run this to verify liveness detector can distinguish real faces from photos
"""

import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from liveness import detect_liveness, get_liveness_detector
from utils import detect_faces, crop_face_with_padding

def test_liveness_on_camera():
    """Test liveness detector on webcam feed"""
    print("="*60)
    print("LIVENESS DETECTION TEST")
    print("="*60)
    print("\nInstructions:")
    print("1. Show your LIVE face to the camera")
    print("2. Hold up a PHOTO of yourself")
    print("3. Compare the liveness scores")
    print("\nPress 'q' to quit\n")
    
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    detector = get_liveness_detector()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detect_faces(frame)
        
        for x, y, w, h in faces:
            # Crop face
            face_crop = crop_face_with_padding(frame, x, y, w, h)
            
            if face_crop.size > 0 and face_crop.shape[0] >= 50:
                # Resize and convert to RGB
                face_resized = cv2.resize(face_crop, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
                # Analyze liveness every 5 frames
                if frame_count % 5 == 0:
                    is_live, confidence, details = detect_liveness(face_rgb)
                    
                    # Display results
                    status = "REAL" if is_live else "SPOOF"
                    color = (0, 255, 0) if is_live else (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display liveness result
                    text = f"{status}: {confidence:.1%}"
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display detailed scores
                    y_offset = y + h + 20
                    for key, value in details.items():
                        if key != 'decision' and key != 'overall':
                            score_text = f"{key}: {value:.2f}"
                            cv2.putText(frame, score_text, (x, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            y_offset += 15
                    
                    # Print to console
                    print(f"\n[Frame {frame_count}] {status} - Confidence: {confidence:.1%}")
                    print(f"  Texture: {details.get('texture', 0):.2f}")
                    print(f"  Color: {details.get('color', 0):.2f}")
                    print(f"  Moiré: {details.get('moire', 0):.2f}")
                    if 'motion' in details:
                        print(f"  Motion: {details['motion']:.2f}")
        
        # Display frame
        cv2.imshow('Liveness Detection Test', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest completed!")


if __name__ == "__main__":
    test_liveness_on_camera()
