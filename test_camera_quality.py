"""
Real-time Camera Quality Testing Tool

This script captures frames from your camera and displays the quality metrics
in real-time, helping you understand why images might be rejected.

Press 'q' to quit, 's' to save current frame for analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from utils import detect_faces, crop_face_with_padding, check_image_blur, check_image_lighting

def test_camera_quality(camera_index=0):
    """Test camera quality in real-time"""
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return
    
    print("=" * 70)
    print("CAMERA QUALITY TESTING TOOL")
    print("=" * 70)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame for analysis")
    print("  '1' - Lower blur threshold (more lenient)")
    print("  '2' - Raise blur threshold (more strict)")
    print("=" * 70)
    
    # Current thresholds (matching app.py)
    blur_threshold = 80
    lighting_min = 25
    lighting_max = 230
    lighting_contrast = 30
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Detect faces
        faces = detect_faces(frame)
        
        # Info overlay
        info_y = 30
        cv2.putText(display_frame, f"Blur Threshold: {blur_threshold}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        
        if len(faces) == 0:
            cv2.putText(display_frame, "NO FACE DETECTED", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for idx, (x, y, w, h) in enumerate(faces):
                # Draw face box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Crop face for quality checks
                cropped_face = crop_face_with_padding(frame, x, y, w, h)
                
                # Check blur
                blur_var, blur_ok, blur_msg = check_image_blur(cropped_face, threshold=blur_threshold)
                
                # Check lighting
                brightness, contrast, lighting_ok, lighting_msg = check_image_lighting(
                    cropped_face, lighting_min, lighting_max, lighting_contrast
                )
                
                # Overall status
                all_ok = blur_ok and lighting_ok
                status_color = (0, 255, 0) if all_ok else (0, 0, 255)
                status_text = "PASS" if all_ok else "FAIL"
                
                # Display metrics
                text_x = x + w + 10
                text_y = y + 20
                
                cv2.putText(display_frame, f"Face {idx+1}: {status_text}", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                text_y += 30
                
                # Blur info
                blur_color = (0, 255, 0) if blur_ok else (0, 0, 255)
                cv2.putText(display_frame, f"Blur: {blur_var:.1f} ({blur_msg})", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blur_color, 2)
                text_y += 25
                cv2.putText(display_frame, f"  Threshold: {blur_threshold}", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                text_y += 30
                
                # Lighting info
                lighting_color = (0, 255, 0) if lighting_ok else (0, 0, 255)
                cv2.putText(display_frame, f"Lighting: {lighting_msg}", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
                text_y += 25
                cv2.putText(display_frame, f"  Bright: {brightness:.1f} (range: {lighting_min}-{lighting_max})", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                text_y += 20
                cv2.putText(display_frame, f"  Contrast: {contrast:.1f} (min: {lighting_contrast})", 
                           (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Also show on left side for main face
                if idx == 0:
                    info_y = 30
                    cv2.putText(display_frame, f"Status: {status_text}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    info_y += 30
                    cv2.putText(display_frame, f"Blur Score: {blur_var:.1f} (need >= {blur_threshold})", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blur_color, 2)
                    info_y += 30
                    cv2.putText(display_frame, f"Brightness: {brightness:.1f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
                    info_y += 30
                    cv2.putText(display_frame, f"Contrast: {contrast:.1f}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
        
        # Show frame
        cv2.imshow('Camera Quality Test', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"quality_test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\nSaved frame to {filename}")
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cropped = crop_face_with_padding(frame, x, y, w, h)
                crop_filename = f"quality_test_face_{frame_count}.jpg"
                cv2.imwrite(crop_filename, cropped)
                print(f"Saved face crop to {crop_filename}")
        elif key == ord('1'):
            blur_threshold = max(10, blur_threshold - 10)
            print(f"\nBlur threshold lowered to {blur_threshold} (more lenient)")
        elif key == ord('2'):
            blur_threshold = min(200, blur_threshold + 10)
            print(f"\nBlur threshold raised to {blur_threshold} (more strict)")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final blur threshold used: {blur_threshold}")
    print(f"Lighting range: {lighting_min} - {lighting_max}")
    print(f"Minimum contrast: {lighting_contrast}")
    print("=" * 70)
    print("\nNOTE: If your camera is consistently failing blur checks,")
    print("      you may need to adjust the threshold in app.py line 942")
    print("      Current app.py threshold: 80")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test camera quality metrics")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    
    test_camera_quality(args.camera)
