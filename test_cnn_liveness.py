"""
Quick test script to verify CNN liveness model loads and works
"""
import sys
sys.path.insert(0, 'src')

import os
# fix duplicate OpenMP runtime on Windows (libiomp5md.dll)
# set before importing libraries that load OpenMP (e.g., torch, cv2)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Suppress TensorFlow GPU warnings and disable GPU usage
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # Disable GPU for TensorFlow

# Suppress OpenCV warnings (MSMF errors, etc.)
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Disable MediaPipe GPU/hardware acceleration to prevent crashes
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("GLOG_minloglevel", "2")  # Suppress MediaPipe logs

import torch
import numpy as np
from liveness_cnn import get_cnn_liveness_detector
from pathlib import Path

def test_cnn_liveness():
    """Test the trained CNN liveness detector"""
    
    print("=" * 60)
    print("CNN LIVENESS DETECTOR TEST")
    print("=" * 60)
    
    # Check model exists
    model_path = Path('outputs/liveness_detector.pth')
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {model_path}")
        return
    
    print(f"\n✓ Found trained model: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load detector
    print("\n[1/3] Loading CNN liveness detector...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    try:
        detector = get_cnn_liveness_detector(
            model_path=str(model_path),
            device=device
        )
        print("  ✓ Detector loaded successfully!")
    except Exception as e:
        print(f"  ❌ ERROR loading detector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create dummy test image
    print("\n[2/3] Creating test image (random noise)...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    print(f"  Shape: {test_image.shape}, dtype: {test_image.dtype}")
    
    # Test prediction
    print("\n[3/3] Testing prediction...")
    try:
        is_live, confidence, details = detector.analyze(test_image)
        
        print(f"\n  Results:")
        print(f"    Is Live: {is_live}")
        print(f"    Confidence: {confidence:.4f}")
        print(f"    Details:")
        for key, value in details.items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")
            else:
                print(f"      {key}: {value}")
        
        print("\n✓ Model is working correctly!")
        
    except Exception as e:
        print(f"  ❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - CNN Model is functional!")
    print("=" * 60)
    print("\nNow you can:")
    print("  1. Uncomment liveness detection in app.py")
    print("  2. Run: python app.py")
    print("  3. The trained CNN model will be used automatically")

if __name__ == '__main__':
    test_cnn_liveness()
