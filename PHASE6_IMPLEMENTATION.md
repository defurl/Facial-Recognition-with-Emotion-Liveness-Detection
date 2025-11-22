# Phase 6: Multi-Capture Registration Implementation

## Overview
Implemented a comprehensive multi-pose registration system to address the flickering issue when users tilt their heads or wear glasses. Instead of capturing a single image, the system now captures 6 different poses per employee.

## Key Features

### 1. Registration Mode Selection
- **Full Mode (6 poses)**: Best accuracy
  - Poses: center, left (~30°), right (~30°), up (~15°), down (~15°), center
- **Quick Mode (3 poses)**: Faster registration
  - Poses: center, left (~30°), right (~30°)

### 2. State Machine Architecture
The registration process is now managed by a state machine in `self.registration_state`:

```python
{
    'step': 0,                      # Current pose step (0-5 for full mode)
    'frames': [],                   # Captured face images
    'embeddings': [],               # Face embeddings for each pose
    'attempt_count': 0,             # Number of capture attempts
    'failure_counts': {             # Track failure types
        'blur': 0,
        'lighting': 0,
        'pose': 0
    },
    'poses_required': [...],        # List of required poses
    'instructions': [...],          # User instructions for each pose
    'show_tooltip': False,          # Display help tooltip
    'tooltip_message': ''           # Tooltip content
}
```

### 3. Quality Validation
Each capture attempt validates three quality criteria:

#### a) Blur Detection
- **Strict Threshold**: 100 (first 2 attempts)
- **Relaxed Threshold**: 80 (after 2 failed attempts)
- Uses Laplacian variance

#### b) Lighting Validation
- **Strict Range**: [30, 220] brightness
- **Relaxed Range**: [25, 230] brightness
- Validates contrast > 30

#### c) Pose Validation
- Uses MediaPipe Face Mesh to estimate yaw/pitch angles
- **Strict Tolerance**: Yaw ±5°, Pitch ±7°
- **Relaxed Tolerance**: Yaw ±8°, Pitch ±10°
- Validates current pose matches target pose

### 4. Adaptive Threshold System
- After 2 failed attempts, automatically relaxes quality thresholds
- Tracks failure types (blur, lighting, pose)
- After 5 failures, shows helpful tooltip with top 2 problem areas

### 5. Real-Time Feedback
During registration, the video overlay displays:
- Current step (e.g., "Step 2/6: Turn your head slightly left")
- Clarity status: ✓/✗ with variance value
- Lighting status: ✓/✗ with brightness value
- Pose status: Current angles (Yaw: X°, Pitch: Y°)
- Color-coded feedback (green = pass, red = fail)
- Tooltip hints after multiple failures

### 6. Confirmation Dialog
After all poses are captured, shows a review dialog with:
- 2×3 thumbnail grid of captured faces
- Pose labels under each thumbnail
- Three action buttons:
  - **✓ Save to Database**: Stores all embeddings
  - **🔄 Discard & Start Over**: Resets and restarts
  - **✗ Cancel Registration**: Exits registration mode

## Files Modified

### app.py
1. **start_registration()** (lines ~550-640)
   - Added mode selection dialog (full vs quick)
   - Radio buttons for mode choice
   - Updated instructions for multi-pose capture

2. **capture_frames()** (lines ~710-840)
   - Replaced simple registration with state machine
   - Added quality validation checks
   - Real-time feedback overlay
   - Adaptive threshold logic
   - Tooltip system

3. **complete_registration()** (NEW method, lines ~705-805)
   - Creates confirmation dialog
   - Displays thumbnail grid using PIL/ImageTk
   - Saves all embeddings to database as list
   - Handles restart and cancel actions

### src/config.py
Added registration-specific constants:
- `REGISTRATION_POSES_FULL` / `REGISTRATION_POSES_QUICK`
- `REGISTRATION_INSTRUCTIONS_FULL` / `REGISTRATION_INSTRUCTIONS_QUICK`
- Quality thresholds (blur, lighting, pose)
- `ADAPTIVE_THRESHOLD_ATTEMPTS = 2`
- `FAILURE_TOOLTIP_THRESHOLD = 5`

## Database Format
Employees are now stored with multiple embeddings:

```python
employee_db = {
    'John Doe': [embedding1, embedding2, ..., embedding6],  # Full mode
    'Jane Smith': [embedding1, embedding2, embedding3],     # Quick mode
    'Old Employee': [embedding1]                            # Backward compatible
}
```

## Verification Process
During face verification, the system:
1. Compares trial embedding against **all** stored embeddings
2. Uses the **minimum distance** among all comparisons
3. This dramatically improves accuracy with pose variations

Example:
```python
# Multi-embedding comparison
distances = [F.pairwise_distance(trial_emb, stored_emb).item() 
             for stored_emb in employee_embeddings]
min_distance = min(distances)  # Best match across all poses
```

## User Experience Flow

1. **Start Registration**
   - Click "Register New Employee"
   - Enter name
   - Select mode (Full/Quick)
   - Click "✓ Start Registration"

2. **Capture Process**
   - Follow on-screen instructions
   - System validates quality automatically
   - Green ✓ = passed, Red ✗ = failed
   - Hold pose until "✓ Captured X/6" appears
   - System advances to next pose automatically

3. **Review & Save**
   - Review all captured thumbnails
   - Click "✓ Save to Database" to complete
   - Or restart if quality is poor

## Benefits

### Problem Solved
- **Before**: Single registration image caused flickering when user tilted head or wore glasses
- **After**: 6 poses provide robustness to head rotations, tilts, and accessories

### Accuracy Improvements
- Handles left/right head turns (±30°)
- Handles up/down head tilts (±15°)
- Works with glasses on/off
- Robust to slight pose variations during verification

### Quality Assurance
- Ensures sharp, well-lit images
- Validates correct pose angles
- Adaptive thresholds prevent frustration
- Helpful tooltips guide users

## Testing Checklist

- [ ] Test Full mode (6 poses) registration
- [ ] Test Quick mode (3 poses) registration
- [ ] Verify all 6 thumbnails appear in confirmation dialog
- [ ] Test adaptive threshold relaxation after 2 failures
- [ ] Test tooltip appears after 5 failures
- [ ] Verify quality feedback is accurate (blur, lighting, pose)
- [ ] Test "Discard & Start Over" functionality
- [ ] Test "Cancel Registration" button
- [ ] Verify database stores all embeddings as list
- [ ] Test verification with head tilts (left/right/up/down)
- [ ] Test verification with glasses on/off
- [ ] Verify backward compatibility with old single-embedding employees

## Next Steps (Phase 7-9)

**Phase 7**: Enhanced Verification Display
- Add confidence percentage calculation
- Show debug panel with distance/threshold
- Add color-coded confidence bar

**Phase 8**: Attendance Logging Integration
- Import AttendanceLogger
- Mark attendance after successful verification
- Display today's attendance count

**Phase 9**: Testing & Validation
- Performance profiling
- Multi-face scenarios
- Error recovery testing
- Thread safety validation

## Configuration Tuning

If captures are too difficult, adjust in `src/config.py`:

```python
# Make blur detection less strict
BLUR_THRESHOLD_STRICT = 80  # Lower = more lenient

# Widen lighting range
LIGHTING_MIN_BRIGHT_STRICT = 25  # Lower min = darker allowed
LIGHTING_MAX_BRIGHT_STRICT = 230  # Higher max = brighter allowed

# Increase pose tolerance
YAW_TOLERANCE_STRICT = 8.0  # Degrees
PITCH_TOLERANCE_STRICT = 10.0  # Degrees

# Reduce attempts before relaxing thresholds
ADAPTIVE_THRESHOLD_ATTEMPTS = 1  # Relax faster
```

## Known Issues & Limitations

1. **Camera Quality**: Low-quality webcams may struggle with blur detection
2. **Lighting**: Poor room lighting may cause consistent lighting failures
3. **Pose Estimation**: MediaPipe Face Mesh requires visible face landmarks
4. **Processing Speed**: Captures every 20 frames (~2 seconds between attempts)

## Technical Notes

- Registration attempts every 20 frames (vs 10 for verification) to reduce processing load
- Quality validation uses same functions as future verification enhancements
- State machine persists across frames, survives camera stutters
- Thumbnails use PIL for cross-platform compatibility
- All embeddings stored on CPU to reduce memory usage
