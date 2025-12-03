#!/usr/bin/env python3
"""
Comprehensive Emotion Detection Testing Tool
Captures images for each emotion and analyzes them to debug detection issues
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from datetime import datetime
import json
from deepface import DeepFace
from pathlib import Path

# Create output directory for test images
output_dir = Path("emotion_test_captures")
output_dir.mkdir(exist_ok=True)

# Emotions to test
emotions_to_test = [
    ('neutral', 'Make a neutral/relaxed face'),
    ('happy', 'Smile broadly - show teeth'),
    ('sad', 'Make a sad face - frown, look down'),
    ('angry', 'Make an angry face - furrow brows, tighten jaw'),
    ('surprise', 'Look surprised - raise eyebrows, open mouth'),
    ('fear', 'Look scared/worried - wide eyes, raised eyebrows'),
]

def analyze_saved_image(image_path, expected_emotion):
    """Analyze a saved image and return detailed results"""
    results = {}
    
    # Read the image first to ensure it's valid
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': 'Could not read image file'}
    except Exception as e:
        return {'error': f'Error reading image: {str(e)}'}
    
    # Test with different backends - start with most reliable
    backends = ['opencv', 'ssd']  # Limit to most stable backends
    
    for backend in backends:
        try:
            print(f"    Testing {backend}...", end='', flush=True)
            
            analysis = DeepFace.analyze(
                str(image_path),
                actions=['emotion'],
                detector_backend=backend,
                enforce_detection=False,
                silent=True
            )
            
            result = analysis[0] if isinstance(analysis, list) else analysis
            emotion_scores = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'unknown')
            
            # Convert numpy types to native Python types for JSON serialization
            scores_native = {k: float(v) for k, v in emotion_scores.items()}
            
            results[backend] = {
                'dominant': str(dominant),
                'scores': scores_native,
                'success': True
            }
            print(f" ✓")
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            results[backend] = {
                'success': False,
                'error': str(e)[:150]
            }
            print(f" ✗ ({str(e)[:50]})")
    
    return results

def main():
    print("=" * 70)
    print("EMOTION DETECTION COMPREHENSIVE TEST")
    print("=" * 70)
    print()
    print("This test will capture images of different facial expressions")
    print("and analyze them to help debug emotion detection issues.")
    print()
    print("Instructions:")
    print("  - Press SPACE to capture the current emotion")
    print("  - Press 'r' to retry the current emotion")
    print("  - Press 'q' to quit")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    # Set camera resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    captured_images = []
    current_emotion_idx = 0
    
    while current_emotion_idx < len(emotions_to_test):
        emotion_name, instruction = emotions_to_test[current_emotion_idx]
        
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Draw instructions on frame
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw text
        cv2.putText(frame, f"Emotion {current_emotion_idx + 1}/{len(emotions_to_test)}: {emotion_name.upper()}", 
                   (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, instruction, 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press SPACE to capture | 'r' to retry | 'q' to quit", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show countdown in center if we want (optional)
        cv2.circle(frame, (w // 2, h // 2), 100, (0, 255, 0), 3)
        
        cv2.imshow('Emotion Test - Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nTest cancelled by user")
            break
        elif key == ord(' '):  # Space bar - capture
            try:
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{emotion_name}_{timestamp}.jpg"
                filepath = output_dir / filename
                
                # Save original frame (before mirroring back)
                cv2.imwrite(str(filepath), cv2.flip(frame, 1))
                
                print(f"\n✓ Captured: {emotion_name} -> {filename}")
                print("  Analyzing image...")
                
                # Analyze immediately
                analysis_results = analyze_saved_image(filepath, emotion_name)
                
                # Check if there was an error
                if 'error' in analysis_results:
                    print(f"  ERROR: {analysis_results['error']}")
                    print("  Press 'r' to retry or 'q' to quit")
                    continue
                
                # Store results
                captured_images.append({
                    'filename': filename,
                    'expected_emotion': emotion_name,
                    'timestamp': timestamp,
                    'analysis': analysis_results
                })
                
                # Show quick results
                print(f"\n  Quick Results:")
                for backend, result in analysis_results.items():
                    if result.get('success'):
                        detected = result['dominant']
                        correct = "✓" if detected == emotion_name else "✗"
                        print(f"    {backend:12s}: {detected:10s} {correct}")
                    else:
                        error_msg = result.get('error', 'Unknown error')[:40]
                        print(f"    {backend:12s}: FAILED - {error_msg}")
                
                # Move to next emotion
                current_emotion_idx += 1
                print(f"\n  Moving to next emotion...")
                
                # Brief pause to show capture
                cv2.waitKey(1000)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n  ERROR during capture: {str(e)}")
                print("  Press 'r' to retry this emotion")
                cv2.waitKey(2000)
            
        elif key == ord('r'):  # Retry current emotion
            print(f"\n↻ Retrying {emotion_name}...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save detailed results to JSON
    if captured_images:
        results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(captured_images, f, indent=2)
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        print(f"\nCaptured {len(captured_images)} images")
        print(f"Images saved to: {output_dir}")
        print(f"Detailed results saved to: {results_file}")
        
        # Generate summary report
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        
        backend_accuracy = {}
        
        for item in captured_images:
            expected = item['expected_emotion']
            print(f"\n{item['filename']} (Expected: {expected})")
            
            for backend, result in item['analysis'].items():
                if backend not in backend_accuracy:
                    backend_accuracy[backend] = {'correct': 0, 'total': 0}
                
                if result.get('success'):
                    detected = result['dominant']
                    scores = result['scores']
                    
                    backend_accuracy[backend]['total'] += 1
                    if detected == expected:
                        backend_accuracy[backend]['correct'] += 1
                    
                    # Show detailed scores
                    print(f"  {backend}:")
                    print(f"    Detected: {detected} (Expected: {expected})")
                    print(f"    Scores:")
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    for emotion, score in sorted_scores:
                        marker = "★" if emotion == expected else " "
                        print(f"      {marker} {emotion:10s}: {score:6.2f}%")
        
        # Overall accuracy
        print("\n" + "=" * 70)
        print("BACKEND ACCURACY")
        print("=" * 70)
        for backend, stats in backend_accuracy.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                print(f"{backend:12s}: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Review the images in:", output_dir)
        print("2. Check the detailed JSON results:", results_file)
        print("3. Analyze which emotions are confused with which")
        print("4. Share the results file for further analysis")
        print()
    else:
        print("\nNo images captured")

if __name__ == "__main__":
    main()
