"""
Performance and Crash Diagnostic Tool

Analyzes common crash causes and performance issues in the Face Recognition app.
"""

print("=" * 70)
print("FACE RECOGNITION APP - DIAGNOSTIC TOOL")
print("=" * 70)

import sys
import os

# 1. Check Python version
print("\n[1] Python Environment Check")
print(f"  Python Version: {sys.version}")
print(f"  Platform: {sys.platform}")

# 2. Check critical imports
print("\n[2] Library Import Check")
critical_libs = {
    'cv2': 'OpenCV',
    'torch': 'PyTorch',
    'tkinter': 'Tkinter GUI',
    'PIL': 'Pillow (Image processing)',
    'numpy': 'NumPy',
    'mediapipe': 'MediaPipe (Face detection)',
    'deepface': 'DeepFace (Emotion analysis)'
}

missing_libs = []
for lib, name in critical_libs.items():
    try:
        __import__(lib)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} - MISSING")
        missing_libs.append(lib)

if missing_libs:
    print(f"\n  ⚠️ WARNING: Missing libraries: {', '.join(missing_libs)}")
else:
    print("\n  ✓ All critical libraries available")

# 3. Check OpenMP/Threading issues
print("\n[3] Threading/OpenMP Check")
omp_lib_ok = os.environ.get('KMP_DUPLICATE_LIB_OK', 'FALSE')
omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')
print(f"  KMP_DUPLICATE_LIB_OK: {omp_lib_ok}")
print(f"  OMP_NUM_THREADS: {omp_threads}")
if omp_lib_ok != 'TRUE':
    print("  ⚠️ WARNING: May cause OpenMP runtime conflict crashes")
if omp_threads != '1':
    print("  ⚠️ WARNING: Multiple OpenMP threads may cause instability")

# 4. Check GPU settings
print("\n[4] GPU Settings Check")
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
tf_cpp_log = os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')
print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
print(f"  TF_CPP_MIN_LOG_LEVEL: {tf_cpp_log}")
if cuda_visible != '-1':
    print("  ⚠️ WARNING: GPU may cause crashes with incompatible drivers")

# 5. Memory check
print("\n[5] System Memory Check")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  Total RAM: {mem.total / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB ({mem.percent}% used)")
    if mem.percent > 90:
        print("  ⚠️ WARNING: Low memory may cause crashes")
    else:
        print("  ✓ Sufficient memory available")
except ImportError:
    print("  ℹ️ psutil not installed (optional)")

# 6. Check camera availability
print("\n[6] Camera Availability Check")
try:
    import cv2
    found_camera = False
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ Camera {idx} detected and working")
                found_camera = True
                cap.release()
                break
            cap.release()
    
    if not found_camera:
        print("  ⚠️ No working camera found")
        print("  Note: Camera issues don't cause crashes, just functionality loss")
except Exception as e:
    print(f"  ✗ Camera check failed: {e}")

# 7. Check for known crash patterns
print("\n[7] Known Crash Pattern Analysis")

crash_patterns = {
    "DeepFace thread conflicts": "DeepFace analyze() called from multiple threads",
    "MediaPipe GPU crash": "MediaPipe trying to use GPU on incompatible system",
    "OpenCV frame read race": "Multiple threads reading from cv2.VideoCapture",
    "Tkinter thread safety": "Tkinter widgets accessed from non-main thread",
    "Memory leak (emotion)": "DeepFace models not properly cached/reused"
}

print("  Common crash causes in this application:")
for pattern, desc in crash_patterns.items():
    print(f"    • {pattern}: {desc}")

# 8. Performance bottleneck analysis
print("\n[8] Performance Bottleneck Identification")

print("  Known bottlenecks:")
print("    1. DeepFace.analyze() - 500-1000ms per call")
print("    2. Model inference - 50-100ms per frame")
print("    3. Face detection (MediaPipe) - 20-40ms per frame")
print("    4. Frame queue backup - causes lag and crashes")
print("    5. Confidence buffer accumulation - O(n) operations")

# 9. Recommendations
print("\n" + "=" * 70)
print("CRASH PREVENTION RECOMMENDATIONS")
print("=" * 70)

recommendations = [
    ("HIGH", "Add try-except around DeepFace.analyze() calls"),
    ("HIGH", "Ensure Tkinter updates only happen on main thread (use window.after())"),
    ("HIGH", "Add frame queue size limit (maxsize=1) to prevent backup"),
    ("MEDIUM", "Cache DeepFace models properly to avoid memory leaks"),
    ("MEDIUM", "Add timeout to emotion analysis to prevent hanging"),
    ("MEDIUM", "Implement graceful degradation when emotion analysis fails"),
    ("LOW", "Reduce PROCESS_EVERY_N_FRAMES if performance is poor"),
    ("LOW", "Add memory monitoring and cleanup")
]

for priority, rec in recommendations:
    print(f"  [{priority:6}] {rec}")

print("\n" + "=" * 70)
print("WARNINGS ANALYSIS")
print("=" * 70)

print("""
The warnings you see are HARMLESS and do NOT cause crashes:

1. RequestsDependencyWarning (chardet/charset_normalizer)
   • Impact: NONE - just a dependency warning
   • Cause: DeepFace uses requests library without charset detection
   • Fix: Can be ignored safely

2. TensorFlow Lite XNNPACK warnings
   • Impact: NONE - informational message
   • Cause: TensorFlow Lite backend initialization
   • Fix: Can be suppressed with TF_CPP_MIN_LOG_LEVEL=3 (not recommended)

3. Feedback manager warnings
   • Impact: NONE - model signature info
   • Cause: DeepFace models don't have feedback tensors
   • Fix: Cannot be suppressed, can be ignored

4. TensorFlow deprecation warnings
   • Impact: NONE - just API deprecation notice
   • Cause: tf_keras using old TensorFlow API
   • Fix: Wait for tf_keras update (no action needed)

ACTUAL CRASH CAUSES (not from warnings):
• Threading violations (Tkinter access from worker thread)
• DeepFace thread conflicts (multiple analyze() calls)
• Memory leaks (models not released)
• Frame queue backup (memory overflow)
• Camera read race conditions
""")

print("=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
To identify the exact crash cause:

1. Run the app and note when it crashes:
   • During startup? → Model loading issue
   • When camera starts? → Camera/threading issue  
   • During recognition? → DeepFace/threading issue
   • After running a while? → Memory leak

2. Check for error messages in terminal before crash

3. Run with more verbose logging:
   Set TF_CPP_MIN_LOG_LEVEL=0 to see all TensorFlow messages

4. Monitor memory usage:
   Use Task Manager to watch Python process memory

5. Apply targeted fixes based on crash timing
""")

print("=" * 70)
