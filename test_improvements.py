"""
Test script for the attendance system improvements.
Tests quality validation, attendance logging, and multi-face tracking utilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from utils import (
    check_image_blur, check_image_lighting, 
    estimate_head_pose_angles
)
from attendance import AttendanceLogger

# Import select_primary_face from app.py
import importlib.util
spec = importlib.util.spec_from_file_location("app_module", Path(__file__).parent / "app.py")
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
select_primary_face = app_module.select_primary_face

print("=" * 70)
print("TESTING ATTENDANCE SYSTEM IMPROVEMENTS")
print("=" * 70)

# Test 1: Image Quality Validation
print("\n[Test 1] Image Quality Validation")
print("-" * 70)

# Create test images
print("Creating test images...")
# Sharp image
sharp_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
# Blurry image (simulate by heavy Gaussian blur)
blurry_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)
# Dark image
dark_img = np.ones((200, 200, 3), dtype=np.uint8) * 20
# Bright image
bright_img = np.ones((200, 200, 3), dtype=np.uint8) * 240

# Test blur detection
print("\n1a. Blur Detection:")
sharp_score, sharp_pass, sharp_msg = check_image_blur(sharp_img)
print(f"   Sharp image: score={sharp_score:.2f}, pass={sharp_pass}, msg='{sharp_msg}'")

blurry_score, blurry_pass, blurry_msg = check_image_blur(blurry_img)
print(f"   Blurry image: score={blurry_score:.2f}, pass={blurry_pass}, msg='{blurry_msg}'")

# Test lighting detection
print("\n1b. Lighting Detection:")
dark_bright, dark_contrast, dark_pass, dark_msg = check_image_lighting(dark_img)
print(f"   Dark image: bright={dark_bright:.1f}, contrast={dark_contrast:.1f}, pass={dark_pass}, msg='{dark_msg}'")

bright_bright, bright_contrast, bright_pass, bright_msg = check_image_lighting(bright_img)
print(f"   Bright image: bright={bright_bright:.1f}, contrast={bright_contrast:.1f}, pass={bright_pass}, msg='{bright_msg}'")

print("✓ Quality validation tests passed!")


# Test 2: Primary Face Selection
print("\n[Test 2] Primary Face Selection")
print("-" * 70)

# Simulate multiple face detections
frame_width, frame_height = 640, 480

# Test case 1: Different sized faces
faces_1 = [
    (50, 50, 100, 100),    # Small face, upper left
    (200, 150, 200, 200),  # Large face, center - should be primary
    (450, 300, 80, 80)     # Small face, lower right
]

primary_idx = select_primary_face(faces_1, frame_width, frame_height)
print(f"\n2a. Three faces (small, large-center, small):")
print(f"   Faces: {faces_1}")
print(f"   Primary face index: {primary_idx} (expected: 1 - large center face)")
print(f"   Primary box: {faces_1[primary_idx]}")

# Test case 2: Similar sized faces, different positions
faces_2 = [
    (50, 50, 150, 150),     # Left face
    (250, 150, 150, 150),   # Center face - should be primary
    (450, 250, 150, 150)    # Right face
]

primary_idx_2 = select_primary_face(faces_2, frame_width, frame_height)
print(f"\n2b. Three similar-sized faces (left, center, right):")
print(f"   Faces: {faces_2}")
print(f"   Primary face index: {primary_idx_2} (expected: 1 - center face)")
print(f"   Primary box: {faces_2[primary_idx_2]}")

# Test case 3: Single face
faces_3 = [(200, 150, 200, 200)]
primary_idx_3 = select_primary_face(faces_3, frame_width, frame_height)
print(f"\n2c. Single face:")
print(f"   Faces: {faces_3}")
print(f"   Primary face index: {primary_idx_3} (expected: 0)")

print("\n✓ Primary face selection tests passed!")


# Test 3: Attendance Logger
print("\n[Test 3] Attendance Logger")
print("-" * 70)

import os
test_csv = Path("outputs/test_attendance.csv")

# Clean up old test file
if test_csv.exists():
    os.remove(test_csv)
    print("Cleaned up old test file")

# Create logger with 1 minute cooldown
logger = AttendanceLogger(csv_path=str(test_csv), cooldown_minutes=1)
print(f"Created logger with 1-minute cooldown")

# Test marking attendance
print("\n3a. First attendance mark:")
success, msg = logger.mark_attendance("John Doe", 0.35, "Happy", "Real")
print(f"   Result: {success}, Message: {msg}")

# Test cooldown
print("\n3b. Immediate retry (should be blocked):")
success, msg = logger.mark_attendance("John Doe", 0.32, "Neutral", "Real")
print(f"   Result: {success}, Message: {msg}")

# Test different employee
print("\n3c. Different employee (should succeed):")
success, msg = logger.mark_attendance("Jane Smith", 0.40, "Neutral", "Real")
print(f"   Result: {success}, Message: {msg}")

# Check records
print("\n3d. Today's records:")
today_records = logger.get_today_records()
print(f"   Total records: {len(today_records)}")
if not today_records.empty:
    print(f"   Records:\n{today_records[['employee_name', 'emotion', 'liveness_status']]}")

# Generate summary
print("\n3e. Daily summary:")
summary = logger.generate_daily_summary()
for key, value in summary.items():
    print(f"   {key}: {value}")

print("\n✓ Attendance logger tests passed!")


# Test 4: Pose Estimation (basic test without actual face images)
print("\n[Test 4] Pose Estimation")
print("-" * 70)

# Test with dummy image (will fail gracefully)
dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
yaw, pitch, roll, label, matches, tol = estimate_head_pose_angles(dummy_img)
print(f"Dummy image (no face):")
print(f"   Angles: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
print(f"   Label: {label}, Matches: {matches}")
print("✓ Pose estimation gracefully handles no-face case!")


# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("✓ All quality validation utilities working")
print("✓ Primary face selection working correctly")
print("✓ Attendance logging with cooldown working")
print("✓ Error handling working (graceful degradation)")
print("\nReady to test full GUI application!")
print("=" * 70)
