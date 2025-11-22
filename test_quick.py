"""
Quick test script for core improvements (no attendance logger).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from utils import check_image_blur, check_image_lighting

print("=" * 70)
print("QUICK TEST - Core Improvements")
print("=" * 70)

# Test 1: Quality validation
print("\n[Test 1] Image Quality Validation")
print("-" * 70)

# Create test images
sharp_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
blurry_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)
dark_img = np.ones((200, 200, 3), dtype=np.uint8) * 20
bright_img = np.ones((200, 200, 3), dtype=np.uint8) * 240

# Test blur
sharp_score, sharp_pass, sharp_msg = check_image_blur(sharp_img)
blurry_score, blurry_pass, blurry_msg = check_image_blur(blurry_img)
print(f"Sharp: {sharp_pass} ({sharp_score:.1f}), Blurry: {blurry_pass} ({blurry_score:.1f})")

# Test lighting
dark_b, dark_c, dark_pass, dark_msg = check_image_lighting(dark_img)
bright_b, bright_c, bright_pass, bright_msg = check_image_lighting(bright_img)
print(f"Dark: {dark_pass} ({dark_msg}), Bright: {bright_pass} ({bright_msg})")

assert sharp_pass == True, "Sharp image should pass"
assert blurry_pass == False, "Blurry image should fail"
assert dark_pass == False, "Dark image should fail"
assert bright_pass == False, "Bright image should fail"

print("✓ Quality validation working correctly!")

# Test 2: Primary face selection
print("\n[Test 2] Primary Face Selection")
print("-" * 70)

# Import select_primary_face
import importlib.util
spec = importlib.util.spec_from_file_location("app_module", Path(__file__).parent / "app.py")
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
select_primary_face = app_module.select_primary_face

frame_w, frame_h = 640, 480

# Test: Large center face should be primary
faces = [
    (50, 50, 100, 100),    # Small left
    (200, 150, 200, 200),  # Large center
    (450, 300, 80, 80)     # Small right
]
primary_idx = select_primary_face(faces, frame_w, frame_h)
print(f"Three faces: Primary = {primary_idx} (expected: 1)")
assert primary_idx == 1, "Large center face should be primary"

# Test: Center face with similar sizes
faces2 = [
    (50, 50, 150, 150),
    (250, 150, 150, 150),
    (450, 250, 150, 150)
]
primary_idx2 = select_primary_face(faces2, frame_w, frame_h)
print(f"Similar sizes: Primary = {primary_idx2} (expected: 1)")
assert primary_idx2 == 1, "Center face should be primary"

# Test: Single face
faces3 = [(200, 150, 200, 200)]
primary_idx3 = select_primary_face(faces3, frame_w, frame_h)
print(f"Single face: Primary = {primary_idx3} (expected: 0)")
assert primary_idx3 == 0, "Single face should be primary"

print("✓ Primary face selection working correctly!")

print("\n" + "=" * 70)
print("✓ ALL CORE TESTS PASSED!")
print("=" * 70)
print("\nYou can now test the full application:")
print("  conda activate face_recog")
print("  python app.py")
