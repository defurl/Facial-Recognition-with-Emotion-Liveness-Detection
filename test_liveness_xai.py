"""
Liveness Detection Testing with xAI Analysis
Uses xAI to analyze detection patterns and provide insights
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import time
import json
import os
from openai import OpenAI
from datetime import datetime

# Add src to path without importing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules directly
import liveness

LivenessDetector = liveness.LivenessDetector

# Initialize xAI client
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
if not XAI_API_KEY:
    print("Warning: XAI_API_KEY not set. AI analysis will be disabled.")
    print("Set it with: $env:XAI_API_KEY='your-key-here'")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
) if XAI_API_KEY else None

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

# Storage for analysis history
analysis_history = []


def analyze_with_xai(details_history, scenario_description=""):
    """Send detection data to xAI for intelligent analysis"""
    if not client:
        return "xAI client not initialized. Set XAI_API_KEY environment variable."
    
    # Prepare summary statistics
    confidence_values = [d['overall'] for d in details_history]
    decisions = [d['decision'] for d in details_history]
    
    blink_scores = [d.get('blink', {}).get('score', 0) if isinstance(d.get('blink'), dict) else d.get('blink', 0) 
                    for d in details_history]
    landmark_scores = [d.get('landmark_motion', 0) for d in details_history]
    blue_scores = [d.get('blue_light', 0) for d in details_history]
    reflection_scores = [d.get('reflection', 0) for d in details_history]
    texture_scores = [d.get('texture', 0) for d in details_history]
    
    # Calculate statistics
    stats = {
        "total_frames": len(details_history),
        "avg_confidence": np.mean(confidence_values),
        "std_confidence": np.std(confidence_values),
        "min_confidence": np.min(confidence_values),
        "max_confidence": np.max(confidence_values),
        "real_count": decisions.count('Real'),
        "spoof_count": decisions.count('Spoof'),
        "scores": {
            "blink": {"avg": np.mean(blink_scores), "std": np.std(blink_scores), "min": np.min(blink_scores)},
            "landmark_motion": {"avg": np.mean(landmark_scores), "std": np.std(landmark_scores), "min": np.min(landmark_scores)},
            "blue_light": {"avg": np.mean(blue_scores), "std": np.std(blue_scores), "min": np.min(blue_scores)},
            "reflection": {"avg": np.mean(reflection_scores), "std": np.std(reflection_scores), "min": np.min(reflection_scores)},
            "texture": {"avg": np.mean(texture_scores), "std": np.std(texture_scores), "min": np.min(texture_scores)},
        },
        "blink_data": {
            "total_blinks": details_history[-1].get('blink', {}).get('total_blinks', 0) if isinstance(details_history[-1].get('blink'), dict) else 0,
            "has_blinked": details_history[-1].get('blink', {}).get('has_blinked', False) if isinstance(details_history[-1].get('blink'), dict) else False,
        }
    }
    
    # Create prompt for xAI
    prompt = f"""You are an expert in biometric liveness detection and anti-spoofing systems. Analyze this liveness detection session data and provide insights.

Scenario: {scenario_description if scenario_description else "General liveness detection test"}

Detection Statistics (last {len(details_history)} frames):
- Overall Confidence: {stats['avg_confidence']:.1%} ± {stats['std_confidence']:.1%} (range: {stats['min_confidence']:.1%} to {stats['max_confidence']:.1%})
- Decision: {stats['real_count']} Real, {stats['spoof_count']} Spoof
- Threshold: 58% required for "Real"

Detection Method Scores (0-100%, higher = more likely real):
1. BLINK (40% weight): {stats['scores']['blink']['avg']:.1%} ± {stats['scores']['blink']['std']:.1%}
   - Total blinks detected: {stats['blink_data']['total_blinks']}
   - Has blinked: {stats['blink_data']['has_blinked']}
   
2. LANDMARK MOTION (22% weight): {stats['scores']['landmark_motion']['avg']:.1%} ± {stats['scores']['landmark_motion']['std']:.1%}
   - Detects facial deformation vs rigid phone movement
   
3. BLUE LIGHT (8% weight): {stats['scores']['blue_light']['avg']:.1%} ± {stats['scores']['blue_light']['std']:.1%}
   - Detects phone screen blue-shifted LED backlight
   
4. TEXTURE (10% weight): {stats['scores']['texture']['avg']:.1%} ± {stats['scores']['texture']['std']:.1%}
   - LBP-based surface texture analysis
   
5. REFLECTION (5% weight): {stats['scores']['reflection']['avg']:.1%} ± {stats['scores']['reflection']['std']:.1%}
   - Detects screen glare and specular highlights

Analyze:
1. Is this a REAL face or SPOOF (photo on phone/screen)? Provide confidence %.
2. Which detection methods are working well? Which are failing?
3. What is the PRIMARY reason for the classification (real/spoof)?
4. If it's misclassified, what improvements would help?
5. Are there any unusual patterns or anomalies in the data?

Provide a concise analysis (3-5 sentences) followed by specific recommendations."""

    try:
        print("\n[xAI] Analyzing detection patterns...")
        
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are an expert in biometric security, computer vision, and anti-spoofing technology. Provide technical, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused analysis
        )
        
        analysis = response.choices[0].message.content
        return analysis
        
    except Exception as e:
        return f"xAI analysis failed: {str(e)}"


def draw_text_box(img, text, pos, font_scale=0.5, thickness=1, 
                  text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = text.split('\n')
    
    x, y = pos
    line_height = 25
    max_width = 0
    
    # Calculate dimensions
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)
    
    # Draw background
    padding = 10
    bg_height = len(lines) * line_height + padding * 2
    cv2.rectangle(img, (x - padding, y - padding), 
                 (x + max_width + padding, y + bg_height), bg_color, -1)
    
    # Draw text lines
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_height + 20), 
                   font, font_scale, text_color, thickness)
    
    return bg_height


def create_simple_display(frame, face_crop, landmarks, details, xai_analysis=None):
    """Create simple display with key information"""
    h, w = frame.shape[:2]
    display = frame.copy()
    
    # Draw face box if landmarks available
    if landmarks is not None:
        x_coords = [int(lm.x * w) for lm in landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in landmarks.landmark]
        x_min, x_max = max(0, min(x_coords) - 30), min(w, max(x_coords) + 30)
        y_min, y_max = max(0, min(y_coords) - 50), min(h, max(y_coords) + 30)
        
        decision = details.get('decision', 'Unknown')
        confidence = details.get('overall', 0.0)
        
        # Box color based on decision
        box_color = (0, 255, 0) if decision == 'Real' else (0, 0, 255)
        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), box_color, 3)
        
        # Decision label
        label = f"{decision}: {confidence:.1%}"
        cv2.rectangle(display, (x_min, y_min - 35), (x_max, y_min), box_color, -1)
        cv2.putText(display, label, (x_min + 10, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Info panel on left side
    panel_width = 400
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    
    y_pos = 30
    
    # Title
    cv2.putText(panel, "LIVENESS ANALYSIS", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 40
    
    # Overall score
    confidence = details.get('overall', 0.0)
    decision = details.get('decision', 'Unknown')
    decision_color = (0, 255, 0) if decision == 'Real' else (0, 0, 255)
    
    cv2.putText(panel, f"Decision: {decision}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, decision_color, 2)
    y_pos += 30
    
    # Confidence bar
    bar_width = int((panel_width - 20) * confidence)
    cv2.rectangle(panel, (10, y_pos), (10 + bar_width, y_pos + 20), decision_color, -1)
    cv2.rectangle(panel, (10, y_pos), (panel_width - 10, y_pos + 20), (100, 100, 100), 2)
    cv2.putText(panel, f"{confidence:.1%}", (10 + bar_width + 10, y_pos + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += 40
    
    # Threshold line
    threshold_x = int(10 + (panel_width - 20) * 0.58)
    cv2.line(panel, (threshold_x, y_pos - 60), (threshold_x, y_pos - 40), (255, 255, 0), 2)
    cv2.putText(panel, "58%", (threshold_x - 20, y_pos - 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Method scores
    cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
    y_pos += 20
    
    cv2.putText(panel, "Detection Methods:", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 25
    
    # Extract scores
    blink_data = details.get('blink', {})
    if isinstance(blink_data, dict):
        blink_score = blink_data.get('score', 0.0)
        total_blinks = blink_data.get('total_blinks', 0)
        has_blinked = blink_data.get('has_blinked', False)
    else:
        blink_score = blink_data
        total_blinks = 0
        has_blinked = False
    
    scores = [
        ("Blink (40%)", blink_score, f"Blinks: {total_blinks} {'✓' if has_blinked else '✗'}"),
        ("Landmark (22%)", details.get('landmark_motion', 0.0), "Facial motion"),
        ("Blue Light (8%)", details.get('blue_light', 0.0), "Screen detect"),
        ("Texture (10%)", details.get('texture', 0.0), "Surface"),
        ("Reflection (5%)", details.get('reflection', 0.0), "Glare"),
        ("Edge (5%)", details.get('edge_detection', 0.0), "Boundaries"),
        ("Moiré (8%)", details.get('moire', 0.0), "Patterns"),
        ("Color (8%)", details.get('color', 0.0), "Distribution"),
    ]
    
    for name, score, info in scores:
        # Score color
        if score > 0.6:
            color = (0, 255, 0)
        elif score > 0.4:
            color = (255, 165, 0)
        else:
            color = (0, 0, 255)
        
        # Draw mini bar
        mini_bar = int(100 * score)
        cv2.rectangle(panel, (10, y_pos - 8), (10 + mini_bar, y_pos + 2), color, -1)
        cv2.rectangle(panel, (10, y_pos - 8), (110, y_pos + 2), (80, 80, 80), 1)
        
        # Text
        cv2.putText(panel, f"{name}: {score:.0%}", (120, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, info, (120, y_pos + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        y_pos += 30
    
    # xAI Analysis section
    if xai_analysis:
        y_pos += 20
        cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
        y_pos += 20
        
        cv2.putText(panel, "xAI Analysis:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y_pos += 20
        
        # Display first few lines of analysis
        lines = xai_analysis.split('\n')[:8]  # First 8 lines
        for line in lines:
            if len(line) > 50:
                # Wrap long lines
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) < 45:
                        current_line += word + " "
                    else:
                        cv2.putText(panel, current_line, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                        y_pos += 15
                        current_line = word + " "
                if current_line:
                    cv2.putText(panel, current_line, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                    y_pos += 15
            else:
                cv2.putText(panel, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                y_pos += 15
            
            if y_pos > h - 60:
                break
    
    # Controls at bottom
    y_bottom = h - 40
    cv2.putText(panel, "Controls:", (10, y_bottom), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(panel, "SPACE: Analyze with xAI", (10, y_bottom + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(panel, "R: Reset  Q: Quit", (10, y_bottom + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # Combine panel and display
    combined = np.hstack([panel, display])
    
    return combined


def main():
    """Main testing loop with xAI integration"""
    print("=" * 70)
    print("LIVENESS DETECTION TEST WITH xAI ANALYSIS")
    print("=" * 70)
    print("\nThis tool tests liveness detection and uses xAI (Grok) to analyze")
    print("the detection patterns and provide insights on what's working/failing.")
    print("\nControls:")
    print("  SPACE - Analyze current session with xAI")
    print("  'r'   - Reset liveness detector and history")
    print("  's'   - Save analysis report")
    print("  'q'   - Quit")
    print("\nDetection Methods:")
    print("  - Blink Detection (40%): PRIMARY anti-spoofing")
    print("  - Landmark Motion (22%): Detects rigid phone vs facial expressions")
    print("  - Blue Light (8%): Phone screen LED color temperature")
    print("  - Texture/Color/Moiré/Reflection/Edge: Supporting methods")
    print("\nThreshold: 58% confidence required for REAL")
    print("=" * 70)
    
    if not client:
        print("\n⚠️  xAI client not initialized - AI analysis disabled")
        print("Set XAI_API_KEY environment variable to enable")
    
    print("\nStarting camera...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    last_xai_analysis = None
    
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
                
                # Store in history
                analysis_history.append(details)
                if len(analysis_history) > 300:  # Keep last 10 seconds at 30fps
                    analysis_history.pop(0)
                
                # Create display
                display = create_simple_display(frame, face_crop, landmarks, details, last_xai_analysis)
                
                cv2.imshow('Liveness Detection with xAI', display)
        else:
            # No face detected
            display_frame = frame.copy()
            cv2.putText(display_frame, "No face detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('Liveness Detection with xAI', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n[RESET] Resetting detector and analysis history...")
            liveness_detector.reset()
            analysis_history.clear()
            last_xai_analysis = None
        elif key == ord(' '):
            if len(analysis_history) >= 10 and client:
                print("\n" + "=" * 70)
                print(f"[xAI ANALYSIS] Analyzing last {len(analysis_history)} frames...")
                print("=" * 70)
                
                # Get scenario description
                scenario = "Real-time camera feed analysis"
                
                # Run xAI analysis
                analysis = analyze_with_xai(analysis_history, scenario)
                last_xai_analysis = analysis
                
                print("\n" + analysis)
                print("\n" + "=" * 70)
            elif not client:
                print("\n⚠️  xAI client not initialized. Set XAI_API_KEY to enable AI analysis.")
            else:
                print("\n⚠️  Need at least 10 frames of data. Keep camera running...")
        elif key == ord('s'):
            if last_xai_analysis:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"outputs/liveness_xai_analysis_{timestamp}.txt"
                
                with open(filename, 'w') as f:
                    f.write("LIVENESS DETECTION xAI ANALYSIS REPORT\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Frames analyzed: {len(analysis_history)}\n\n")
                    f.write("xAI Analysis:\n")
                    f.write("-" * 70 + "\n")
                    f.write(last_xai_analysis)
                    f.write("\n\n" + "=" * 70 + "\n")
                
                print(f"\n✓ Analysis report saved to {filename}")
            else:
                print("\n⚠️  No xAI analysis available. Press SPACE to analyze first.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print(f"Total frames processed: {frame_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
