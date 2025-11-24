"""
Simple script to capture a test image from camera for demo purposes.

Usage:
    python capture_test_image.py
    
This will capture an image from your camera and save it as 'test_face.jpg'
"""

import cv2
import sys

print("="*60)
print("TEST IMAGE CAPTURE")
print("="*60)
print("\nInstructions:")
print("1. Position your face in front of the camera")
print("2. Press SPACE to capture")
print("3. Press ESC to cancel")
print("="*60 + "\n")

# Try to open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    print("Please check:")
    print("  - Camera is connected")
    print("  - No other application is using the camera")
    print("  - Camera permissions are granted")
    sys.exit(1)

print("✓ Camera opened successfully")
print("\nShowing live preview...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        break
    
    # Display instructions on frame
    cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Capture Test Image', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\nCapture cancelled")
        break
    elif key == 32:  # SPACE
        filename = 'test_face.jpg'
        cv2.imwrite(filename, frame)
        print(f"\n✓ Image saved as '{filename}'")
        print(f"\nYou can now run:")
        print(f"  python demo_explainability.py --image {filename} --name 'Your Name'")
        break

cap.release()
cv2.destroyAllWindows()
