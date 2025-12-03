"""
Liveness Detection Testing and Debugging Tool
Visualizes all detection methods in real-time with detailed overlays
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os

# Add src to path without importing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules directly
import liveness
import blink_detector
import time
from collections import deque

LivenessDetector = liveness.LivenessDetector

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize liveness detector
liveness_detector = LivenessDetector()

# Color scheme for visualization
COLOR_LIVE = (0, 255, 0)  # Green
COLOR_SPOOF = (0, 0, 255)  # Red
COLOR_NEUTRAL = (255, 165, 0)  # Orange
COLOR_INFO = (255, 255, 255)  # White
COLOR_WARNING = (0, 165, 255)  # Orange-yellow


def draw_text_with_background(img, text, pos, font_scale=0.5, thickness=1, 
                              text_color=COLOR_INFO, bg_color=(0, 0, 0), padding=5):
    """Draw text with background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    # Draw background rectangle
    cv2.rectangle(img, 
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + baseline + padding),
                 bg_color, -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    return text_height + baseline + padding * 2


def visualize_texture_analysis(face_crop, texture_score):
    """Visualize LBP texture analysis"""
    h, w = face_crop.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    
    # Compute simple LBP visualization
    lbp = np.zeros_like(gray)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            lbp[i, j] = code
    
    # Convert to heatmap
    lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_JET)
    
    # Add score text
    color = COLOR_LIVE if texture_score > 0.6 else COLOR_SPOOF if texture_score < 0.4 else COLOR_NEUTRAL
    cv2.putText(vis, f"Texture: {texture_score:.1%}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis


def visualize_color_analysis(face_crop, color_score, blue_score):
    """Visualize color distribution and blue light analysis"""
    h, w = face_crop.shape[:2]
    
    # Convert to LAB
    lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
    
    # Show L channel (luminance)
    l_channel = lab[:, :, 0]
    l_vis = cv2.applyColorMap(l_channel, cv2.COLORMAP_BONE)
    
    # Show color temperature visualization
    b_channel = face_crop[:, :, 2].astype(np.float32)
    r_channel = face_crop[:, :, 0].astype(np.float32)
    
    # Calculate B/R ratio per pixel
    br_ratio = np.zeros_like(b_channel)
    mask = r_channel > 10
    br_ratio[mask] = b_channel[mask] / r_channel[mask]
    br_ratio = np.clip(br_ratio, 0, 2)
    
    # Visualize: blue = cool (screen), red = warm (natural)
    br_normalized = cv2.normalize(br_ratio, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_temp_vis = cv2.applyColorMap(br_normalized, cv2.COLORMAP_TWILIGHT)
    
    # Combine visualizations side by side
    vis = np.hstack([l_vis, color_temp_vis])
    
    # Add labels
    color1 = COLOR_LIVE if color_score > 0.6 else COLOR_SPOOF if color_score < 0.4 else COLOR_NEUTRAL
    color2 = COLOR_LIVE if blue_score > 0.6 else COLOR_SPOOF if blue_score < 0.4 else COLOR_NEUTRAL
    
    cv2.putText(vis, f"Color: {color_score:.1%}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
    cv2.putText(vis, f"Blue: {blue_score:.1%}", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)
    
    return vis


def visualize_reflections(face_crop, reflection_score):
    """Visualize screen reflections and bright spots"""
    h, w = face_crop.shape[:2]
    
    # Convert to LAB
    lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Create visualization
    vis = face_crop.copy()
    
    # Highlight very bright regions (potential reflections)
    bright_mask = l_channel > 215
    vis[bright_mask] = [255, 0, 255]  # Magenta for reflections
    
    # Highlight extreme bright spots (glare)
    extreme_bright = l_channel > 240
    vis[extreme_bright] = [255, 255, 0]  # Yellow for extreme glare
    
    # Add score
    color = COLOR_LIVE if reflection_score > 0.6 else COLOR_SPOOF if reflection_score < 0.4 else COLOR_NEUTRAL
    cv2.putText(vis, f"Reflection: {reflection_score:.1%}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, "Magenta=Bright, Yellow=Glare", (10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INFO, 1)
    
    return vis


def visualize_edges(face_crop, edge_score):
    """Visualize edge detection for screen boundaries"""
    h, w = face_crop.shape[:2]
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=50, minLineLength=30, maxLineGap=10)
    
    # Create visualization
    vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Draw detected lines in red
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Color by length: long lines (screen edges) in red, short in green
            threshold = min(h, w) * 0.4
            color = (0, 0, 255) if length > threshold else (0, 255, 0)
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    
    # Add score
    score_color = COLOR_LIVE if edge_score > 0.6 else COLOR_SPOOF if edge_score < 0.4 else COLOR_NEUTRAL
    cv2.putText(vis, f"Edge: {edge_score:.1%}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
    cv2.putText(vis, "Red=Long lines (screen), Green=Short", (10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INFO, 1)
    
    return vis


def visualize_moire(face_crop, moire_score):
    """Visualize moiré pattern detection (frequency domain)"""
    h, w = face_crop.shape[:2]
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    
    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize for visualization
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, 
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_HOT)
    
    # Draw circle to show DC component region (excluded from analysis)
    center_h, center_w = h // 2, w // 2
    cv2.circle(vis, (center_w, center_h), min(h, w) // 4, (0, 255, 0), 2)
    
    # Add score
    color = COLOR_LIVE if moire_score > 0.6 else COLOR_SPOOF if moire_score < 0.4 else COLOR_NEUTRAL
    cv2.putText(vis, f"Moire: {moire_score:.1%}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, "Green circle=DC, Hot=High freq", (10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INFO, 1)
    
    return vis


def visualize_landmark_motion(frame, landmarks, landmark_score, frame_history):
    """Visualize landmark motion tracking"""
    h, w = frame.shape[:2]
    vis = frame.copy()
    
    # Draw key landmarks
    if landmarks is not None:
        key_indices = [33, 263, 61, 291, 1, 152, 10, 338, 199, 428]
        
        for idx in key_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Color based on motion score
            if landmark_score > 0.7:
                color = (0, 255, 0)  # Green - varied motion (real)
            elif landmark_score > 0.5:
                color = (255, 165, 0)  # Orange - moderate
            else:
                color = (0, 0, 255)  # Red - rigid motion (spoof)
            
            cv2.circle(vis, (x, y), 3, color, -1)
        
        # Draw motion trails if we have history
        if len(frame_history) >= 2:
            for idx in key_indices[:4]:  # Show trails for first 4 landmarks
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw small trail
                cv2.circle(vis, (x, y), 8, (255, 255, 0), 1)
    
    # Add score
    score_color = COLOR_LIVE if landmark_score > 0.6 else COLOR_SPOOF if landmark_score < 0.4 else COLOR_NEUTRAL
    cv2.putText(vis, f"Landmark Motion: {landmark_score:.1%}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
    cv2.putText(vis, "Green=Varied, Red=Rigid", (10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INFO, 1)
    
    return vis


def create_dashboard(frame, face_crop, landmarks, details):
    """Create comprehensive debugging dashboard"""
    if face_crop is None or face_crop.size == 0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Resize face crop for display
    face_display_size = 200
    face_crop_resized = cv2.resize(face_crop, (face_display_size, face_display_size))
    
    # Create visualizations for each method
    texture_score = details.get('texture', 0.5)
    color_score = details.get('color', 0.5)
    blue_score = details.get('blue_light', 0.5)
    reflection_score = details.get('reflection', 0.5)
    edge_score = details.get('edge_detection', 0.5)
    moire_score = details.get('moire', 0.5)
    landmark_score = details.get('landmark_motion', 0.5)
    
    # Generate visualizations
    texture_vis = visualize_texture_analysis(face_crop, texture_score)
    texture_vis = cv2.resize(texture_vis, (face_display_size, face_display_size))
    
    color_vis = visualize_color_analysis(face_crop, color_score, blue_score)
    color_vis = cv2.resize(color_vis, (face_display_size * 2, face_display_size))
    
    reflection_vis = visualize_reflections(face_crop, reflection_score)
    reflection_vis = cv2.resize(reflection_vis, (face_display_size, face_display_size))
    
    edge_vis = visualize_edges(face_crop, edge_score)
    edge_vis = cv2.resize(edge_vis, (face_display_size, face_display_size))
    
    moire_vis = visualize_moire(face_crop, moire_score)
    moire_vis = cv2.resize(moire_vis, (face_display_size, face_display_size))
    
    landmark_vis = visualize_landmark_motion(
        cv2.resize(face_crop, (face_display_size, face_display_size)), 
        landmarks, landmark_score, liveness_detector.landmark_history
    )
    
    # Arrange in grid layout
    # Row 1: Original, Texture, Reflection
    row1 = np.hstack([
        cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR),
        texture_vis,
        reflection_vis
    ])
    
    # Row 2: Edge, Moiré, Landmark Motion
    row2 = np.hstack([
        edge_vis,
        moire_vis,
        landmark_vis
    ])
    
    # Row 3: Color analysis (double width)
    # Pad color_vis to match width of rows 1 & 2
    pad_width = face_display_size * 3 - color_vis.shape[1]
    if pad_width > 0:
        padding = np.zeros((face_display_size, pad_width, 3), dtype=np.uint8)
        row3 = np.hstack([color_vis, padding])
    else:
        row3 = color_vis
    
    # Stack rows
    analysis_grid = np.vstack([row1, row2, row3])
    
    # Create info panel
    info_panel = create_info_panel(details, face_display_size * 3, face_display_size)
    
    # Final dashboard: main frame + analysis grid + info panel
    # Resize frame to fit
    frame_display = cv2.resize(frame, (face_display_size * 3, face_display_size * 2))
    
    dashboard = np.vstack([
        frame_display,
        analysis_grid,
        info_panel
    ])
    
    return dashboard


def create_info_panel(details, width, height):
    """Create information panel with all scores"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Extract scores
    confidence = details.get('overall', 0.0)
    decision = details.get('decision', 'Unknown')
    
    blink_data = details.get('blink', {})
    if isinstance(blink_data, dict):
        blink_score = blink_data.get('score', 0.0)
        has_blinked = blink_data.get('has_blinked', False)
        total_blinks = blink_data.get('total_blinks', 0)
        current_ear = blink_data.get('current_ear', 0.0)
    else:
        blink_score = blink_data
        has_blinked = False
        total_blinks = 0
        current_ear = 0.0
    
    # Overall decision at top
    decision_color = COLOR_LIVE if decision == 'Real' else COLOR_SPOOF
    y_offset = 40
    cv2.putText(panel, f"DECISION: {decision} ({confidence:.1%})", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, decision_color, 2)
    
    # Draw confidence bar
    bar_width = int((width - 40) * confidence)
    cv2.rectangle(panel, (20, y_offset + 10), (20 + bar_width, y_offset + 30), decision_color, -1)
    cv2.rectangle(panel, (20, y_offset + 10), (width - 20, y_offset + 30), COLOR_INFO, 1)
    
    # Threshold line
    threshold_x = int(20 + (width - 40) * 0.58)
    cv2.line(panel, (threshold_x, y_offset + 10), (threshold_x, y_offset + 30), (255, 255, 0), 2)
    
    y_offset += 60
    
    # Detailed scores
    scores = [
        ("Blink", blink_score, f"Blinks: {total_blinks}, EAR: {current_ear:.3f}", has_blinked),
        ("Landmark Motion", details.get('landmark_motion', 0.0), "Facial deformation", None),
        ("Blue Light", details.get('blue_light', 0.0), "Screen color temp", None),
        ("Texture (LBP)", details.get('texture', 0.0), "Surface analysis", None),
        ("Color Dist", details.get('color', 0.0), "Natural variation", None),
        ("Moire Pattern", details.get('moire', 0.0), "Screen artifacts", None),
        ("Reflections", details.get('reflection', 0.0), "Screen glare", None),
        ("Edge Detection", details.get('edge_detection', 0.0), "Screen boundaries", None),
    ]
    
    col_width = width // 2
    col = 0
    row = 0
    
    for name, score, description, extra_info in scores:
        x = 20 + col * col_width
        y = y_offset + row * 25
        
        # Score color
        if score > 0.6:
            color = COLOR_LIVE
        elif score > 0.4:
            color = COLOR_NEUTRAL
        else:
            color = COLOR_SPOOF
        
        # Draw score bar
        bar_w = int((col_width - 60) * score)
        cv2.rectangle(panel, (x + 150, y - 12), (x + 150 + bar_w, y - 2), color, -1)
        cv2.rectangle(panel, (x + 150, y - 12), (x + col_width - 40, y - 2), (80, 80, 80), 1)
        
        # Text
        text = f"{name}: {score:.1%}"
        if extra_info is not None:
            text += f" {'✓' if extra_info else '✗'}"
        
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_INFO, 1)
        
        # Move to next position
        col += 1
        if col >= 2:
            col = 0
            row += 1
    
    # Instructions at bottom
    y_bottom = height - 20
    cv2.putText(panel, "Press 'q' to quit, 'r' to reset, SPACE to capture", 
               (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return panel


def main():
    """Main testing loop"""
    print("=" * 60)
    print("LIVENESS DETECTION DEBUG TOOL")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE - Capture and save current frame analysis")
    print("  'r'   - Reset liveness detector")
    print("  'q'   - Quit")
    print("\nAnalysis Methods:")
    print("  1. Blink Detection (40%) - PRIMARY anti-spoofing")
    print("  2. Landmark Motion (22%) - Detects rigid phone movement")
    print("  3. Blue Light (8%) - Phone screen color temperature")
    print("  4. Texture (10%) - Surface analysis via LBP")
    print("  5. Color (8%) - Natural color variation")
    print("  6. Moiré (8%) - Screen frequency artifacts")
    print("  7. Reflections (5%) - Screen glare detection")
    print("  8. Edge Detection (5%) - Screen boundaries")
    print("  9. Others - Motion & temporal consistency")
    print("\nThreshold: 58% confidence required for REAL")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    capture_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Get bounding box
            h, w = frame.shape[:2]
            x_coords = [int(lm.x * w) for lm in landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in landmarks.landmark]
            
            x_min, x_max = max(0, min(x_coords) - 30), min(w, max(x_coords) + 30)
            y_min, y_max = max(0, min(y_coords) - 50), min(h, max(y_coords) + 30)
            
            # Extract face crop
            face_crop = frame_rgb[y_min:y_max, x_min:x_max]
            
            if face_crop.size > 0:
                # Run liveness detection
                is_live, confidence, details = liveness_detector.analyze(face_crop, landmarks)
                
                # Create dashboard
                dashboard = create_dashboard(frame, face_crop, landmarks, details)
                
                # Display
                cv2.imshow('Liveness Detection Debug', dashboard)
        else:
            # No face detected
            display_frame = frame.copy()
            cv2.putText(display_frame, "No face detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('Liveness Detection Debug', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n[RESET] Resetting liveness detector...")
            liveness_detector.reset()
        elif key == ord(' '):
            if results.multi_face_landmarks and 'details' in locals():
                capture_count += 1
                filename = f"outputs/liveness_debug_capture_{capture_count}.png"
                cv2.imwrite(filename, dashboard)
                decision = details.get('decision', 'Unknown')
                print(f"\n[CAPTURE] Saved to {filename}")
                print(f"  Confidence: {confidence:.1%}, Decision: {decision}")
                print(f"  Blink: {details.get('blink', {}).get('score', 0):.1%}")
                print(f"  Blue Light: {details.get('blue_light', 0):.1%}")
                print(f"  Reflection: {details.get('reflection', 0):.1%}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Captures saved: {capture_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
