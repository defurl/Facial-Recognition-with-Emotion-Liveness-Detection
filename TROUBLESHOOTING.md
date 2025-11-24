# 🔧 Explainability Demo Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: "ValueError: Embeddings must have shape [num_samples, dim]"

**Cause**: Employee database contains embeddings in unexpected format or shape.

**Solution**: The demo now automatically handles this by:
- Flattening multi-dimensional embeddings
- Converting to proper numpy arrays
- Validating shape before processing

**If still having issues:**
```python
# Check your employee database format
import torch
db = torch.load('outputs/employee_db.pt')
for name, data in db.items():
    print(f"{name}: {type(data)}, {len(data) if isinstance(data, list) else 'single'}")
```

**Expected format:**
- Dictionary: `{name: [embedding1, embedding2, ...]}` or `{name: embedding}`
- Each embedding should be a 1D tensor of shape (256,)

---

### Issue 2: "Could not read image from face.jpg"

**Cause**: Image file doesn't exist at the specified path.

**Solutions:**

**Option A - Capture from camera:**
```bash
python capture_test_image.py
# Then use the captured image
python demo_explainability.py --image test_face.jpg --name "Your Name"
```

**Option B - Use existing images:**
```bash
# Find some face images in your dataset
ls dataset/classification_data/test_data/*/

# Use one of them
python demo_explainability.py --image dataset/classification_data/test_data/n000003/0001.jpg --name "Test Person"
```

**Option C - Use camera directly:**
```bash
python demo_explainability.py --camera
```

---

### Issue 3: "Need at least 2 embeddings in database"

**Cause**: Employee database has insufficient data for kNN analysis.

**Solution**: Register more employees using the main app:
```bash
python app.py
# Click "Register New Employee"
# Register at least 2 different people
```

**Temporary workaround**: The demo now automatically adjusts k based on available data.

---

### Issue 4: Camera not opening (VideoCapture error)

**Possible causes:**
1. Camera in use by another application
2. Wrong camera index
3. Permission issues

**Solutions:**

**Try different camera index:**
```python
# Edit demo_explainability.py, line ~250
cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
```

**Check camera permissions:**
```bash
# Linux
ls -l /dev/video*

# Test camera directly
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

**Close other camera applications:**
- Close Cheese, Zoom, Teams, etc.
- Check with: `lsof /dev/video0`

---

### Issue 5: TensorFlow/CUDA warnings

**Example:**
```
E0000 00:00:... Unable to register cuFFT factory...
W0000 00:00:... computation placer already registered...
```

**Solution**: These are harmless warnings from DeepFace/TensorFlow. They don't affect functionality.

**To suppress (optional):**
```bash
export TF_CPP_MIN_LOG_LEVEL=3
python demo_explainability.py --camera
```

Or add to top of script:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

### Issue 6: "No face detected in image"

**Causes:**
- Face too small
- Bad lighting
- Face not centered
- Image resolution too low

**Solutions:**

1. **Check image quality:**
```python
import cv2
img = cv2.imread('your_image.jpg')
print(f"Image shape: {img.shape}")
# Should be at least 640x480
```

2. **Pre-process image:**
- Ensure good lighting
- Face should occupy at least 30% of image
- Face should be frontal
- Avoid extreme poses

3. **Use camera for better control:**
```bash
python capture_test_image.py
# Position face properly before capturing
```

---

### Issue 7: Out of memory (CUDA/GPU)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Use CPU instead:**
```python
# Edit src/config.py
DEVICE = torch.device('cpu')
```

2. **Reduce batch processing:**
```python
# In demo, process fewer frames
if frame_count % 10 == 0:  # Instead of 5
```

3. **Clear CUDA cache:**
```python
import torch
torch.cuda.empty_cache()
```

---

### Issue 8: Slow processing

**Symptoms**: 
- Demo lags
- Low FPS

**Solutions:**

1. **Disable attention maps for speed:**
```bash
python demo_explainability.py --camera --no-attention
```

2. **Process fewer frames:**
```python
# In process_camera(), change
if frame_count % 10 == 0:  # Process every 10th frame
```

3. **Use CPU if GPU is slow:**
Sometimes CPU is faster for small batches.

---

### Issue 9: Matplotlib/GUI issues on remote systems

**Symptoms:**
```
UserWarning: Glyph ... missing from font
_tkinter.TclError: no display name
```

**Solutions:**

1. **Use non-interactive backend:**
```python
# Add at top of script
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

2. **For SSH/remote:**
```bash
# Enable X11 forwarding
ssh -X user@host
# Or use Xvfb
xvfb-run python demo_explainability.py --image test.jpg
```

---

## Quick Diagnostic Commands

### Check system setup:
```bash
# Test all components
python test_explainability.py

# Check model exists
ls -lh outputs/best_metric_model.pth

# Check database exists
ls -lh outputs/employee_db.pt

# Check camera
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Check database contents:
```bash
python -c "
import torch
db = torch.load('outputs/employee_db.pt')
print(f'Employees: {len(db)}')
for name, data in db.items():
    count = len(data) if isinstance(data, list) else 1
    print(f'  {name}: {count} embeddings')
"
```

---

## Working Examples

### Minimal working example:
```bash
# 1. Ensure you have registered employees
python app.py
# Register at least 2 people with multiple poses

# 2. Capture a test image
python capture_test_image.py

# 3. Run demo
python demo_explainability.py --image test_face.jpg --name "Test"
```

### Camera demo:
```bash
python demo_explainability.py --camera
# Press 'q' to quit
# Press 's' to save current frame
```

### Batch processing:
```bash
# Process multiple images
for img in dataset/test_data/n000003/*.jpg; do
    python demo_explainability.py --image "$img" --name "Person3"
done
```

---

## Debugging Tips

### Enable verbose output:
```python
# Edit demo_explainability.py
# Add after imports:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check intermediate results:
```python
# Add after embedding extraction:
print(f"Gallery shape: {gallery_embeddings.shape}")
print(f"Gallery dtype: {gallery_embeddings.dtype}")
print(f"Labels unique: {np.unique(gallery_labels)}")
```

### Profile performance:
```python
import time
start = time.time()
# ... your code ...
print(f"Took {time.time() - start:.2f}s")
```

---

## Still Having Issues?

### Collect diagnostic info:
```bash
# Run this and share output
python -c "
import sys
import torch
import cv2
import numpy as np
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('OpenCV:', cv2.__version__)
print('NumPy:', np.__version__)
"

# Check file structure
ls -lh outputs/
```

### Test minimal functionality:
```python
# test_minimal.py
import torch
from src.models import FaceEmbeddingCNN
from src.config import DEVICE, MODEL_METRIC_PATH, EMPLOYEE_DB_PATH

print("Loading model...")
model = FaceEmbeddingCNN(256, 4000).to(DEVICE)
model.load_state_dict(torch.load(MODEL_METRIC_PATH, map_location=DEVICE))
print("✓ Model loaded")

print("Loading database...")
db = torch.load(EMPLOYEE_DB_PATH)
print(f"✓ Database loaded: {len(db)} employees")

print("Testing inference...")
dummy = torch.randn(1, 3, 64, 64).to(DEVICE)
with torch.no_grad():
    emb = model(dummy, mode='metric')
print(f"✓ Inference works: {emb.shape}")
```

---

## Contact & Support

If issues persist:
1. Run `python test_explainability.py` - ensure all tests pass
2. Check you have valid model (`outputs/best_metric_model.pth`)
3. Check you have valid database (`outputs/employee_db.pt`) with 2+ employees
4. Try the minimal example above
5. Review error messages carefully

**Most common fix**: Register more employees in the database using `python app.py`
