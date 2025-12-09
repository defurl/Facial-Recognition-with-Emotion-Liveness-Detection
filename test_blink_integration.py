#!/usr/bin/env python3
"""
Test blink detector integration - Check if blink detection is working properly
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import time
from pathlib import Path

# Add paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.blink_detector import BlinkDetector
from src.utils import face_mesh_detector

print("=" * 70)
print("BLINK DETECTOR INTEGRATION TEST")
print("=" * 70)

# Test 1: Import test
print("\n✓ Test 1: Import BlinkDetector")
blink_detector = BlinkDetector()
print(f"  - BlinkDetector initialized: {blink_detector}")
print(f"  - EAR threshold: {blink_detector.ear_threshold}")
print(f"  - Min blink duration: {blink_detector.min_blink_duration}s")
print(f"  - Max blink duration: {blink_detector.max_blink_duration}s")

# Test 2: Face Mesh detector
print("\n✓ Test 2: MediaPipe Face Mesh Detector")
print(f"  - face_mesh_detector: {face_mesh_detector}")

# Test 3: Camera test (real-time blink detection)
print("\n✓ Test 3: Real-time Camera Test")
print("  - Opening camera...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("  ✗ ERROR: Cannot open camera!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
blink_count_detector = 0
landmark_frames = 0
max_frames = 300  # Test for 10 seconds at 30fps

print(f"  - Camera opened, testing for {max_frames} frames...")
print("  - Controls: 'q' to quit, 'r' to reset blink count")
print("  - Instructions: Blink naturally and fully close eyes for 0.1-0.4 seconds")
print()

verification_start = time.time()

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("  ✗ ERROR: Failed to read frame!")
        break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect landmarks
    results = face_mesh_detector.process(frame_rgb)
    
    display = frame.copy()
    
    if results and results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        landmark_frames += 1
        
        # Call blink detector
        try:
            blink_detected, current_ear, total_blinks = blink_detector.detect_blink(landmarks)
            has_blinked, blinks_needed = blink_detector.requires_blink(
                verification_start, time.time(), min_blinks=1
            )
            
            if blink_detected:
                blink_count_detector += 1
                print(f"[BLINK DETECTED!] Frame {frame_count} | Total: {total_blinks} | EAR: {current_ear:.3f}")
            
            # Draw info
            color = (0, 255, 0) if has_blinked else (255, 165, 0) if blinks_needed > 0 else (0, 0, 255)
            status = f"Has Blinked: {has_blinked} | Blinks Detected: {total_blinks} | EAR: {current_ear:.3f}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw EAR value color-coded
            ear_color = (0, 255, 0) if current_ear > 0.5 else (0, 0, 255)
            cv2.putText(display, f"EAR: {current_ear:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            
        except Exception as e:
            print(f"✗ Blink detection error on frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
            cv2.putText(display, f"ERROR: {str(e)[:40]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(display, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw frame count
    cv2.putText(display, f"Frame: {frame_count}/{max_frames} | Landmark Frames: {landmark_frames}", 
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Blink Detector Test", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"\n[USER] Quit requested at frame {frame_count}")
        break
    elif key == ord('r'):
        print(f"\n[RESET] Blink detector reset at frame {frame_count}")
        blink_detector.reset()
        verification_start = time.time()

cap.release()
cv2.destroyAllWindows()

# Results
print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"Total frames processed: {frame_count}")
print(f"Frames with landmarks: {landmark_frames} ({100*landmark_frames//max(frame_count, 1)}%)")
print(f"Blinks detected by detector: {blink_count_detector}")
print(f"Final blink count from detector: {blink_detector.blink_count}")

if landmark_frames == 0:
    print("\n✗ FAILURE: No landmarks detected! Camera or face detection issue.")
elif blink_count_detector == 0:
    print("\n⚠ WARNING: No blinks detected. Check:")
    print("   1. Are you blinking naturally?")
    print("   2. Are blinks lasting 0.1-0.4 seconds?")
    print("   3. Is your face clearly visible?")
else:
    print(f"\n✓ SUCCESS: Blink detection working! Detected {blink_count_detector} blinks.")

print("=" * 70)
