#!/usr/bin/env python3
"""
Liveness Detection Evaluation Script
=====================================
Evaluates liveness detector on labeled test images/videos.

Usage:
    python scripts/eval_liveness.py dataset/debug/

Directory structure expected:
    dataset/debug/
        real/           - Real face images/frames
        phone_static/   - Static phone images
        phone_moving/   - Moving phone video frames

Outputs:
    outputs/eval_liveness_results.json - Per-image detailed results
    outputs/eval_liveness_summary.txt - Accuracy, precision, recall, confusion matrix
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.liveness import LivenessDetector
from src.utils import face_mesh_detector


def load_images_from_folder(folder_path, label, max_images=50):
    """Load images from a folder with a specific label"""
    images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return images
    
    for img_path in sorted(folder.iterdir())[:max_images]:
        if img_path.suffix.lower() in valid_extensions:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append({
                    'path': str(img_path),
                    'label': label,
                    'image': img
                })
    
    return images


def evaluate_liveness(data_root='dataset/debug/', output_dir='outputs'):
    """Evaluate liveness detector on labeled test data"""
    
    print("="*70)
    print("LIVENESS DETECTION EVALUATION")
    print("="*70)
    
    # Initialize detector
    print("\nInitializing liveness detector...")
    detector = LivenessDetector()
    
    # Load test images
    print(f"\nLoading test images from {data_root}...")
    test_data = []
    
    # Real faces (label = 1)
    real_images = load_images_from_folder(f"{data_root}/real", label=1)
    test_data.extend(real_images)
    print(f"  Loaded {len(real_images)} real face images")
    
    # Phone static (label = 0)
    phone_static = load_images_from_folder(f"{data_root}/phone_static", label=0)
    test_data.extend(phone_static)
    print(f"  Loaded {len(phone_static)} static phone images")
    
    # Phone moving (label = 0)
    phone_moving = load_images_from_folder(f"{data_root}/phone_moving", label=0)
    test_data.extend(phone_moving)
    print(f"  Loaded {len(phone_moving)} moving phone images")
    
    if len(test_data) == 0:
        print("\nERROR: No test data found!")
        print(f"Please create folders: {data_root}/real, {data_root}/phone_static, {data_root}/phone_moving")
        return
    
    print(f"\nTotal test images: {len(test_data)}")
    
    # Evaluate each image
    print("\nRunning evaluation...")
    results = []
    predictions = []
    ground_truth = []
    
    for i, sample in enumerate(test_data):
        img = sample['image']
        label = sample['label']
        path = sample['path']
        
        # Convert to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        landmarks_result = face_mesh_detector.process(rgb_img)
        landmarks = None
        if landmarks_result and landmarks_result.multi_face_landmarks:
            landmarks = landmarks_result.multi_face_landmarks[0]
        
        # Run liveness detection
        try:
            is_live, confidence, details = detector.analyze(rgb_img, landmarks)
            prediction = 1 if is_live else 0
            
            predictions.append(prediction)
            ground_truth.append(label)
            
            results.append({
                'file': path,
                'ground_truth': 'Real' if label == 1 else 'Spoof',
                'prediction': 'Real' if is_live else 'Spoof',
                'correct': prediction == label,
                'confidence': float(confidence),
                'blink_score': float(details.get('blink', {}).get('score', 0.0)) if isinstance(details.get('blink'), dict) else 0.0,
                'landmark_score': float(details.get('landmark_motion', 0.5)),
                'texture': float(details.get('texture', 0.0)),
                'color': float(details.get('color', 0.0)),
                'moire': float(details.get('moire', 0.0))
            })
            
            status = "✓" if prediction == label else "✗"
            print(f"  [{i+1}/{len(test_data)}] {status} {os.path.basename(path)}: "
                  f"GT={label}, Pred={prediction}, Conf={confidence:.2f}")
        
        except Exception as e:
            print(f"  [ERROR] Failed on {path}: {e}")
            continue
        
        # Reset detector between images
        detector.reset()
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    accuracy = np.mean(predictions == ground_truth)
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Spoof detection rate
    spoof_detection_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Real  Predicted Spoof")
    print(f"  Actual Real        {tp:4d}           {fn:4d}")
    print(f"  Actual Spoof       {fp:4d}           {tn:4d}")
    print(f"\nMetrics:")
    print(f"  Precision (Real):        {precision:.2%}")
    print(f"  Recall (Real):           {recall:.2%}")
    print(f"  F1 Score:                {f1:.2%}")
    print(f"  Spoof Detection Rate:    {spoof_detection_rate:.2%}")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    results_path = f"{output_dir}/eval_liveness_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save summary
    summary_path = f"{output_dir}/eval_liveness_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("LIVENESS DETECTION EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"                  Predicted Real  Predicted Spoof\n")
        f.write(f"  Actual Real        {tp:4d}           {fn:4d}\n")
        f.write(f"  Actual Spoof       {fp:4d}           {tn:4d}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Precision (Real):        {precision:.2%}\n")
        f.write(f"  Recall (Real):           {recall:.2%}\n")
        f.write(f"  F1 Score:                {f1:.2%}\n")
        f.write(f"  Spoof Detection Rate:    {spoof_detection_rate:.2%}\n")
    print(f"Summary saved to: {summary_path}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'dataset/debug/'
    evaluate_liveness(data_root)
