"""
Crash Prevention Patches for app.py

This file contains specific fixes for known crash causes:
1. DeepFace threading issues
2. Tkinter thread safety violations
3. Memory leaks
4. Frame queue backup
"""

# Apply these fixes to app.py to prevent crashes

print("""
=================================================================
CRASH FIX RECOMMENDATIONS FOR app.py
=================================================================

CRITICAL FIXES (Apply immediately):

1. FIX: DeepFace thread safety
   LOCATION: Line ~1135 (emotion analysis in capture_frames)
   PROBLEM: DeepFace.analyze() is not thread-safe
   
   BEFORE:
   ```python
   emotion_data = analyze_emotion_and_liveness(face_img)
   ```
   
   AFTER:
   ```python
   try:
       # Add timeout and error handling
       emotion_data = analyze_emotion_and_liveness(face_img)
   except Exception as e:
       print(f"Emotion analysis error: {e}")
       emotion_data = {'dominant_emotion': 'Error', 'liveness': 'Unknown'}
   ```

2. FIX: Tkinter thread safety
   LOCATION: Multiple locations in capture_frames()
   PROBLEM: Direct widget updates from background thread
   
   ENSURE ALL TKINTER CALLS USE:
   ```python
   self.window.after(0, lambda: self.some_ui_update())
   ```
   
   NOT:
   ```python
   self.some_label.config(text="...")  # WRONG if in worker thread!
   ```

3. FIX: Frame queue backup prevention
   LOCATION: Line ~241 (initialization)
   CURRENT:
   ```python
   self.frame_queue = queue.Queue(maxsize=1)
   ```
   ✓ Already correct! Keep maxsize=1

4. FIX: Camera read race condition
   LOCATION: Line ~1047 (cap.read() in capture_frames)
   ADD LOCK:
   ```python
   # In __init__:
   self.camera_lock = threading.Lock()
   
   # In capture_frames:
   with self.camera_lock:
       ret, frame = self.cap.read()
   ```

5. FIX: Emotion analysis timeout
   LOCATION: src/emotion.py analyze_emotion_and_liveness()
   ADD:
   ```python
   import signal
   from contextlib import contextmanager
   
   @contextmanager
   def timeout(seconds):
       def timeout_handler(signum, frame):
           raise TimeoutError()
       signal.signal(signal.SIGALRM, timeout_handler)
       signal.alarm(seconds)
       try:
           yield
       finally:
           signal.alarm(0)
   
   # In analyze_emotion_and_liveness:
   try:
       with timeout(2):  # 2 second timeout
           result = DeepFace.analyze(...)
   except TimeoutError:
       return {'dominant_emotion': 'Timeout', 'liveness': 'Unknown'}
   ```
   Note: signal.alarm only works on Unix. For Windows, use threading.Timer

6. FIX: Memory leak in emotion analysis
   LOCATION: src/emotion.py
   ENSURE: DeepFace models are cached, not reloaded each time
   
   ADD MODULE-LEVEL CACHE:
   ```python
   _emotion_model_cache = None
   
   def get_cached_emotion_model():
       global _emotion_model_cache
       if _emotion_model_cache is None:
           # Initialize once
           _emotion_model_cache = DeepFace.build_model('Emotion')
       return _emotion_model_cache
   ```

=================================================================
MODERATE FIXES (Improve stability):

7. FIX: Add frame processing timeout
   In capture_frames(), add overall timeout per frame:
   ```python
   frame_start_time = time.time()
   # ... processing ...
   if time.time() - frame_start_time > 0.5:  # 500ms max per frame
       print("Frame processing timeout, skipping...")
       continue
   ```

8. FIX: Memory monitoring
   Add periodic memory check:
   ```python
   import psutil
   
   if self.frame_count % 100 == 0:  # Every 100 frames
       process = psutil.Process()
       mem_mb = process.memory_info().rss / 1024 / 1024
       if mem_mb > 1000:  # 1GB threshold
           print(f"WARNING: High memory usage: {mem_mb:.0f}MB")
   ```

9. FIX: Graceful degradation
   If emotion analysis keeps failing, disable it:
   ```python
   self.emotion_failure_count = 0
   self.emotion_analysis_enabled = True
   
   # In capture_frames:
   if self.emotion_analysis_enabled:
       try:
           emotion_data = analyze_emotion_and_liveness(face_img)
           self.emotion_failure_count = 0
       except Exception as e:
           self.emotion_failure_count += 1
           if self.emotion_failure_count > 10:
               print("Disabling emotion analysis due to repeated failures")
               self.emotion_analysis_enabled = False
   ```

=================================================================
PERFORMANCE IMPROVEMENTS:

10. OPTIMIZE: Reduce emotion analysis frequency
    CURRENT: Every 15 frames
    RECOMMENDED: Every 30 frames (2x faster)
    
    Change: self.EMOTION_EVERY_N_FRAMES = 30

11. OPTIMIZE: Reduce face detection load
    Use smaller frame for detection:
    ```python
    # Resize frame for detection only
    detection_frame = cv2.resize(frame, (320, 240))
    faces = detect_faces(detection_frame)
    # Scale coordinates back to original size
    faces = [(x*2, y*2, w*2, h*2) for x,y,w,h in faces]
    ```

12. OPTIMIZE: Skip frames more aggressively
    CURRENT: PROCESS_EVERY_N_FRAMES = 3
    RECOMMENDED: PROCESS_EVERY_N_FRAMES = 5 (faster GUI)

=================================================================
TESTING CHECKLIST:

After applying fixes, test these scenarios:
□ Start app → Does it crash immediately? (Model loading issue)
□ Start camera → Does it crash? (Camera/threading issue)
□ Recognize face → Does it crash? (DeepFace threading issue)
□ Run for 5 minutes → Does it crash? (Memory leak)
□ Register employee → Does it crash? (Registration logic issue)
□ Multiple rapid recognitions → Does it crash? (Race condition)

=================================================================
IMMEDIATE ACTION PLAN:

Step 1: Add error handling to emotion analysis (Fix #1)
Step 2: Add camera lock (Fix #4)  
Step 3: Test for 5 minutes continuously
Step 4: If stable, apply performance optimizations (10-12)
Step 5: If still crashing, apply remaining fixes (7-9)

=================================================================
""")
