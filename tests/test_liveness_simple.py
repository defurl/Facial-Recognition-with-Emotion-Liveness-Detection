"""
Simple Liveness Test - Minimal UI for Maximum Performance
Focus on FPS and BLINK DETECTION ONLY
Skips expensive texture/color/FFT analysis for speed
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os
import time
from pathlib import Path

# Ensure project root and src are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from blink_detector import BlinkDetector

# Initialize - USE ONLY BLINK DETECTOR (no full liveness analysis)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

blink_detector = BlinkDetector()
verification_start_time = None

print("=" * 60)
print("SIMPLE BLINK TEST - MAXIMUM PERFORMANCE")
print("=" * 60)
print("TESTING: Blink detection ONLY (no texture/color/FFT analysis)")
print("This should run at 20-30 FPS")
print("Controls: 'q' to quit, 'r' to reset")
print("=" * 60)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    sys.exit(1)
X
frame_count = 0
fps_list = []
ear_history = []  # Track EAR values for visualization
last_time = time.time()

print("\nProcessing... (showing main feed + simple overlay)")
print("\n" + "=" * 60)
print("BLINK DETECTION DEBUG INFO")
print("=" * 60)
print("Watch the graph at bottom-left:")
print("  - GREEN line = Eyes OPEN (EAR > 0.5)")
print("  - RED line = Eyes CLOSED (EAR < 0.5)")
print("  - Yellow line = Threshold (0.5)")
print("\nHow it works:")
print("  1. Eye closes: EAR drops below 0.5")
print("  2. Eye opens: EAR rises above 0.5")
print("  3. Duration 0.08-0.4s = Valid blink!")
print("\nTroubleshooting:")
print("  - If EAR never goes below 0.5: Eyes too open/wide")
print("  - If EAR stays below 0.5: Eyes squinting")
print("  - Blink naturally and fully close eyes")
print("=" * 60)
print()

while True:
    loop_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face
    results = face_mesh.process(frame_rgb)
    
    # Draw on frame
    display = frame.copy()
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        # Get face bbox
        h, w = frame.shape[:2]
        x_coords = [int(lm.x * w) for lm in landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in landmarks.landmark]
        x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
        y_min, y_max = max(0, min(y_coords) - 30), min(h, max(y_coords) + 20)
        
        # ONLY DO BLINK DETECTION (skip all expensive analysis)
        if verification_start_time is None:
            verification_start_time = time.time()
        
        current_time = time.time()
        
        # Detect blink
        blink_detected, current_ear, total_blinks = blink_detector.detect_blink(landmarks)
        has_blinked, blinks_needed = blink_detector.requires_blink(
            verification_start_time, current_time, min_blinks=1
        )
        
        elapsed = current_time - verification_start_time
        
        # CONSOLE DEBUG: Print when blink is detected
        if blink_detected:
            print(f"[BLINK DETECTED!] Frame {frame_count}, Total blinks: {total_blinks}, EAR: {current_ear:.3f}")
        
        # Simple decision based ONLY on blinks
        if has_blinked:
            is_live = True
            status = "REAL - Blink detected!"
            color = (0, 255, 0)
        elif elapsed < 3.0:
            is_live = None  # Waiting
            status = f"WAITING for blink... ({elapsed:.1f}s)"
            color = (255, 165, 0)
        else:
            is_live = False
            status = "SPOOF - No blinks!"
            color = (0, 0, 255)
        
        # Draw bbox
        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw status
        cv2.putText(display, status, 
                   (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Store EAR for graphing
        ear_history.append(current_ear)
        if len(ear_history) > 150:  # 5 seconds at 30 FPS
            ear_history.pop(0)
        
        # Draw blink info with color-coded EAR
        ear_color = (0, 255, 0) if current_ear > 0.5 else (0, 0, 255)  # Green=open, Red=closed
        blink_text = f"Blinks: {total_blinks} | EAR: {current_ear:.3f}"
        cv2.putText(display, blink_text, 
                   (x_min, y_max + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 2)
        
        # Draw EAR real-time graph
        if len(ear_history) > 1:
            graph_x = 10
            graph_y = h - 130
            graph_w = 300
            graph_h = 120
            
            # Background
            cv2.rectangle(display, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), 
                         (20, 20, 20), -1)
            cv2.rectangle(display, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), 
                         (100, 100, 100), 2)
            
            # Threshold line (0.5)
            threshold_y = int(graph_y + graph_h - (0.5 / 0.8 * graph_h))  # Scale: 0-0.8
            cv2.line(display, (graph_x, threshold_y), 
                    (graph_x + graph_w, threshold_y), 
                    (0, 255, 255), 2)
            cv2.putText(display, "Threshold: 0.5", (graph_x + 5, threshold_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            
            # Plot EAR values
            for i in range(1, len(ear_history)):
                x1 = graph_x + int((i - 1) * graph_w / 150)
                x2 = graph_x + int(i * graph_w / 150)
                
                # Scale EAR to graph (0-0.8 range)
                y1 = graph_y + graph_h - int(min(ear_history[i-1], 0.8) / 0.8 * graph_h)
                y2 = graph_y + graph_h - int(min(ear_history[i], 0.8) / 0.8 * graph_h)
                
                # Color: green=open, red=closed
                line_color = (0, 255, 0) if ear_history[i] > 0.5 else (0, 0, 255)
                cv2.line(display, (x1, y1), (x2, y2), line_color, 2)
            
            # Labels
            cv2.putText(display, "EAR Over Time", (graph_x + 5, graph_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"Current: {current_ear:.3f}", (graph_x + 5, graph_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, ear_color, 1)
    
    # Calculate FPS
    loop_time = time.time() - loop_start
    current_fps = 1.0 / loop_time if loop_time > 0 else 0
    fps_list.append(current_fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = np.mean(fps_list)
    
    # Draw FPS and detector state
    cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Show eye state indicator
    if results and results.multi_face_landmarks and 'current_ear' in locals():
        state_text = "Eyes: CLOSED" if current_ear < 0.5 else "Eyes: OPEN"
        state_color = (0, 0, 255) if current_ear < 0.5 else (0, 255, 0)
        cv2.putText(display, state_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    
    # Show
    cv2.imshow('Simple Liveness Test', display)
    
    # Keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("\n[RESET]")
        blink_detector.reset()
        verification_start_time = None

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"Total frames: {frame_count}")
print(f"Average FPS: {np.mean(fps_list):.1f}")
print("=" * 60)
