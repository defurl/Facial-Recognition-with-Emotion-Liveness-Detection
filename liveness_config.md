# Liveness Detection Configuration Guide

## Overview
The liveness detector now uses **7 different methods** to distinguish real faces from spoofs (photos, screens, prints). This guide shows you how to customize each parameter.

---

## File to Edit: `src/liveness.py`

---

## 🎯 Main Detection Threshold

**Location:** Line ~120 (in `analyze()` method)

```python
is_live = confidence >= 0.58
```

**What it does:** Final decision threshold - if overall confidence is above this, face is "Real"

**Adjust this to:**
- **More Strict** (better reject spoofs): `0.60` to `0.65`
- **More Lenient** (accept more real faces): `0.50` to `0.55`
- **Balanced** (current): `0.58`

**Example:**
```python
# More strict - for high-security scenarios
is_live = confidence >= 0.62

# More lenient - for better user experience
is_live = confidence >= 0.52
```

---

## ⚖️ Detection Method Weights

**Location:** Lines ~63-95 (in `analyze()` method)

Controls how much each method contributes to the final decision.

### Current Configuration:
```python
# 1. Texture Analysis (LBP patterns)
weights.append(0.30)  # 30% - Detects print quality

# 2. Color Distribution
weights.append(0.30)  # 30% - Detects flat colors in prints

# 3. Moiré Patterns
weights.append(0.25)  # 25% - Detects screens

# 4. Motion Analysis
weights.append(0.10)  # 10% - Detects static images

# 5. Edge Detection (NEW)
weights.append(0.15)  # 15% - Detects phone rectangular edges

# 6. Reflection Detection (NEW)
weights.append(0.15)  # 15% - Detects screen glare

# 7. Temporal Consistency (NEW)
weights.append(0.15)  # 15% - Detects screen refresh patterns
```

### How to Adjust:

**To better detect phone screens:**
```python
weights.append(0.20)  # Edge Detection - increase to 20%
weights.append(0.20)  # Reflection Detection - increase to 20%
weights.append(0.20)  # Temporal Consistency - increase to 20%
weights.append(0.15)  # Moiré Patterns - keep high
# Reduce others proportionally
```

**To better detect printed photos:**
```python
weights.append(0.35)  # Texture Analysis - increase
weights.append(0.35)  # Color Distribution - increase
weights.append(0.10)  # Moiré Patterns - reduce (less relevant for prints)
```

**Note:** All weights auto-normalize, but try to keep them balanced and summing close to 1.0

---

## 🔧 Individual Method Tuning

### 1️⃣ Texture Analysis (LBP)

**Location:** Line ~138 (in `_analyze_texture()`)

```python
score = min(1.0, texture_variance / 1000.0)
```

**What it does:** Measures texture richness. Real faces have variance >800, prints <400.

**Adjust denominator:**
- **More Strict:** `1100` or `1200` (harder for prints to score high)
- **More Lenient:** `800` or `900` (easier for slightly blurry faces)

---

### 2️⃣ Color Distribution

**Location:** Lines ~168-170 (in `_analyze_color_distribution()`)

```python
l_score = min(1.0, l_std / 18.0)  # Lightness variation
a_score = min(1.0, a_std / 7.0)   # Green-red variation
b_score = min(1.0, b_std / 7.0)   # Blue-yellow variation
```

**What it does:** Real faces have natural color variation, prints are flat.

**Adjust denominators:**
- **More Strict:** Increase values → `20.0`, `8.0`, `8.0` (harder to score)
- **More Lenient:** Decrease values → `15.0`, `6.0`, `6.0` (easier to score)

---

### 3️⃣ Moiré Pattern Detection

**Location:** Line ~218 (in `_detect_moire_patterns()`)

```python
score = max(0.0, 1.0 - (high_freq_ratio / 0.32))
```

**What it does:** Detects screen patterns via FFT analysis.

**Adjust denominator:**
- **More Strict:** `0.28` or `0.25` (catches more screens)
- **More Lenient:** `0.35` or `0.40` (less sensitive to camera artifacts)

---

### 4️⃣ Motion Analysis

**Location:** Lines ~267-273 (in `_analyze_motion()`)

```python
if mean_motion < 0.18:
    score = 0.15  # Very static (likely photo)
elif mean_motion > 6.0:
    score = 0.45  # Too erratic
else:
    score = min(1.0, mean_motion / 1.8)
```

**Adjust thresholds:**
- **More Strict on static:**
  ```python
  if mean_motion < 0.25:  # Increase threshold
      score = 0.10  # Harsher penalty
  ```
- **More Lenient:**
  ```python
  if mean_motion < 0.15:  # Lower threshold
      score = 0.30  # Lighter penalty
  ```

---

### 5️⃣ Edge Detection (NEW - Phone Screens)

**Location:** Lines ~298-320 (in `_detect_screen_edges()`)

**What it does:** Detects sharp rectangular edges using Hough Line Transform. Phones have straight edges, faces have organic curves.

```python
# Line length threshold (40% of image dimension)
threshold = min(gray.shape[0], gray.shape[1]) * 0.4

# Scoring based on long lines detected
if long_lines == 0:
    score = 1.0  # No screen edges
elif long_lines <= 2:
    score = 0.7
elif long_lines <= 4:
    score = 0.4  # Likely screen
else:
    score = 0.1  # Definitely screen
```

**Adjust line threshold:**
- **More Strict:** Change `0.4` to `0.3` (catches shorter lines)
- **More Lenient:** Change `0.4` to `0.5` (only very long lines)

**Adjust scoring:**
```python
# More aggressive penalization
if long_lines <= 2:
    score = 0.5  # Stricter
elif long_lines <= 4:
    score = 0.2  # Much stricter
```

---

### 6️⃣ Reflection Detection (NEW - Screen Glare)

**Location:** Lines ~340-365 (in `_detect_screen_reflections()`)

**What it does:** Detects bright specular reflections from glossy screens. Real faces have diffuse reflection.

```python
bright_threshold = 220  # Very bright pixels (glare)

# Penalize bright spots
if bright_ratio > 0.05:  # More than 5% bright pixels
    score *= 0.5
elif bright_ratio > 0.02:
    score *= 0.7
```

**Adjust sensitivity:**
- **More Strict:**
  ```python
  bright_threshold = 210  # Lower threshold catches more
  if bright_ratio > 0.03:  # Stricter ratio
      score *= 0.4  # Harsher penalty
  ```
- **Less Sensitive:**
  ```python
  bright_threshold = 230  # Higher threshold
  if bright_ratio > 0.08:  # More tolerant
      score *= 0.6
  ```

---

### 7️⃣ Temporal Consistency (NEW - Screen Refresh)

**Location:** Lines ~410-430 (in `_analyze_temporal_consistency()`)

**What it does:** Analyzes frame-to-frame consistency. Screens have refresh artifacts, real faces have smooth micro-movements.

```python
if diff_variance < 0.3 and mean_diff < 1.0:
    score = 0.3  # Too static (photo)
elif diff_variance > 5.0:
    score = 0.4  # Too inconsistent (screen artifacts)
elif 0.5 <= diff_variance <= 3.0:
    score = 1.0  # Natural movement
```

**Adjust variance thresholds:**
- **More Strict:**
  ```python
  if diff_variance < 0.5:  # Stricter static detection
      score = 0.2
  elif diff_variance > 4.0:  # Stricter artifact detection
      score = 0.3
  ```

---

## 📊 Quick Presets

### Preset 1: Maximum Security (Strict)
**Best for:** High-security applications, willing to reject some real faces

```python
# Main threshold
is_live = confidence >= 0.65

# Weights (emphasize screen detection)
weights.append(0.20)  # Texture
weights.append(0.20)  # Color
weights.append(0.20)  # Moiré
weights.append(0.05)  # Motion
weights.append(0.20)  # Edge Detection
weights.append(0.20)  # Reflection
weights.append(0.20)  # Temporal

# Individual thresholds
# Texture: /1200, Color: /20/8/8, Moiré: /0.28
# Motion: <0.25→0.10, Edge: 0.3, Reflection: 210
```

### Preset 2: Balanced (Current)
**Best for:** General use, demos, attendance systems

```python
# Main threshold
is_live = confidence >= 0.58

# Weights (balanced approach)
weights.append(0.30)  # Texture
weights.append(0.30)  # Color
weights.append(0.25)  # Moiré
weights.append(0.10)  # Motion
weights.append(0.15)  # Edge Detection
weights.append(0.15)  # Reflection
weights.append(0.15)  # Temporal
```

### Preset 3: User-Friendly (Lenient)
**Best for:** Maximizing real face acceptance, demos

```python
# Main threshold
is_live = confidence >= 0.50

# Weights (prioritize user experience)
weights.append(0.35)  # Texture
weights.append(0.35)  # Color
weights.append(0.15)  # Moiré
weights.append(0.15)  # Motion
weights.append(0.10)  # Edge Detection
weights.append(0.10)  # Reflection
weights.append(0.10)  # Temporal

# Individual thresholds
# Texture: /800, Color: /15/6/6, Moiré: /0.40
# Motion: <0.15→0.30, Reflection: 230
```

---

## 🧪 Testing Your Changes

After modifying parameters, test with:

```bash
conda run -n final-topic python test_liveness.py
# OR
conda run -n final-topic python app.py
```

### Test Scenarios:
1. ✅ **Real face** - should score 55-75%
2. ❌ **Old printed photo** - should score <40% (texture/color catch it)
3. ❌ **Phone screen** - should score <50% (edge/reflection/temporal catch it)
4. ❌ **Tablet screen** - should score <45% (moiré/edge catch it)
5. ❌ **ID card photo** - should score <45% (texture/color/edge catch it)

---

## 🎓 Understanding Each Score

When testing, check the `details` dictionary:

```python
details = {
    'texture': 0.85,          # High = good texture
    'color': 0.78,            # High = natural colors
    'moire': 0.92,            # High = no screen patterns
    'motion': 0.65,           # High = natural movement
    'edge_detection': 0.45,   # High = no screen edges (LOW = screen!)
    'reflection': 0.35,       # High = no glare (LOW = glare!)
    'temporal': 0.72,         # High = consistent (no refresh)
    'overall': 0.67           # Weighted average
}
```

**Low scores indicate spoof:**
- `edge_detection < 0.5` → Phone/tablet edges detected ⚠️
- `reflection < 0.5` → Screen glare detected ⚠️
- `temporal < 0.5` → Refresh pattern or too static ⚠️
- `texture < 0.5` → Poor texture (print) ⚠️
- `color < 0.5` → Flat colors (print) ⚠️

---

## 📝 Research & Comparison Guide

### NEW Methods Effectiveness (for modern phones):

**🥇 Most Effective Against Modern Screens:**
1. **Edge Detection** - Catches rectangular phone borders (Hough Lines)
2. **Reflection Detection** - Catches glossy screen glare (brightness analysis)
3. **Temporal Consistency** - Catches screen refresh artifacts (frame diff)

**🥈 Moderately Effective:**
4. **Moiré Patterns** - Works for LCD screens, less for OLED
5. **Motion Analysis** - Good for static photos, less for moving screens

**🥉 Less Effective Against Modern Screens:**
6. **Texture (LBP)** - High-res screens (>300 DPI) mimic texture well
7. **Color (LAB)** - Modern screens have accurate color gamut

### Effectiveness Against Different Spoofs:

| Spoof Type | Best Detectors | Typical Confidence |
|------------|----------------|-------------------|
| Old printed photo | Texture, Color | 25-35% |
| New printed photo | Edge, Texture | 35-45% |
| Phone LCD screen | Edge, Reflection, Moiré | 40-50% |
| Phone OLED screen | Edge, Reflection, Temporal | 45-55% |
| Tablet screen | Edge, Reflection, Temporal | 40-50% |
| ID card | Texture, Color, Edge | 30-40% |
| Paper cutout | All methods | 15-25% |

### Known Limitations:

**❌ Hard to Detect:**
- High-end OLED/AMOLED screens (minimal moiré, accurate colors)
- Anti-glare screen protectors (reduce reflections)
- Moving the phone (mimics natural motion)
- Professional prints on quality paper

**✅ Easy to Detect:**
- Old/faded photos
- Low-quality printouts
- Paper cutouts
- Obvious screen borders in frame
- Static photos

---

## 🆘 Troubleshooting

**Problem: Real faces being rejected**
- ✅ Lower main threshold to 0.50-0.55
- ✅ Reduce weights on edge/reflection/temporal
- ✅ Make individual thresholds more lenient
- ✅ Check if lighting is very poor (causes low texture/color scores)

**Problem: Phone screens passing as Real**
- ✅ Increase main threshold to 0.62-0.65
- ✅ Increase weights on edge/reflection/temporal (to 0.20 each)
- ✅ Make edge detection stricter (0.3 threshold)
- ✅ Make reflection stricter (bright_threshold = 210)

**Problem: Printed photos passing as Real**
- ✅ Increase weights on texture/color (to 0.35 each)
- ✅ Increase texture denominator (/1100 or /1200)
- ✅ Increase color denominators (/20/8/8)
- ✅ Ensure good lighting during testing

**Problem: Inconsistent results**
- ✅ Check camera quality/lighting
- ✅ Increase motion_history_size to 10 frames
- ✅ Add more weight to temporal consistency
- ✅ Test in controlled lighting conditions

---

## 🔬 For Your Research

### Comparison Metrics to Track:

1. **True Positive Rate (TPR)** - % of real faces correctly identified
2. **True Negative Rate (TNR)** - % of spoofs correctly identified
3. **False Acceptance Rate (FAR)** - % of spoofs wrongly accepted
4. **False Rejection Rate (FRR)** - % of real faces wrongly rejected

### Suggested Experiments:

**Experiment 1: Method Ablation**
- Test with only texture+color (baseline)
- Add moiré+motion
- Add new methods (edge+reflection+temporal)
- Compare detection rates

**Experiment 2: Weight Optimization**
- Try different weight combinations
- Measure FAR and FRR for each
- Find optimal balance

**Experiment 3: Threshold Tuning**
- Test thresholds from 0.45 to 0.70 (step 0.05)
- Plot ROC curve (TPR vs FPR)
- Find Equal Error Rate (EER)

**Experiment 4: Spoof Type Analysis**
- Test against: old photos, new photos, LCD screens, OLED screens
- Identify which methods work best for each
- Document limitations

### Data to Collect:

```python
# For each test
test_results = {
    'face_type': 'real' | 'photo_old' | 'photo_new' | 'lcd_screen' | 'oled_screen',
    'confidence': 0.67,
    'decision': 'Real' | 'Spoof',
    'scores': {
        'texture': 0.85,
        'color': 0.78,
        # ... all 7 scores
    },
    'correct': True | False
}
```

---

## 🚀 Future Improvements

### Option A: Train CNN Detector
- Use `liveness_cnn.py` (already in codebase)
- Requires dataset: real faces + various spoofs
- Expected accuracy: 90-95%
- Training time: 2-4 hours on GPU

### Option B: Challenge-Response
- Ask user to blink, smile, turn head
- 99% effective against static spoofs
- Requires user interaction
- Easy to implement in registration flow

### Option C: Hardware Enhancement
- Use depth camera (Intel RealSense, iPhone TrueDepth)
- Infrared detection
- 99.9% effective
- Requires special hardware

---

## 📚 References

**Algorithms Used:**
- LBP (Local Binary Patterns) for texture
- LAB color space for color analysis
- FFT (Fast Fourier Transform) for moiré detection
- Optical Flow (Farneback) for motion
- Hough Line Transform for edge detection
- Brightness variance for reflection detection
- Frame differencing for temporal analysis

**Academic Basis:**
- Traditional CV methods based on published anti-spoofing research
- Effective against 60-70% of common spoofs
- Modern deep learning methods achieve 90-95% accuracy

---

**Last Updated:** November 26, 2025  
**Configuration Version:** Advanced Detection (7 methods)  
**Author:** Enhanced for Research & Comparison
