# Multi-Task Learning (MTL) Model Documentation

## Overview

This document describes the Unified Face Model for multi-task learning, covering:
- Model architecture
- Training procedures
- Dataset details
- Evaluation results
- Benchmark comparisons

---

## Model Architecture

### UnifiedFaceModel

A single CNN backbone with task-specific heads for simultaneous face verification, emotion recognition, and liveness detection.

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedFaceModel                         │
├─────────────────────────────────────────────────────────────┤
│  Input: RGB Image (112 × 112 × 3)                           │
│                        ↓                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            ResNet-18 Backbone + CBAM                │    │
│  │  • 4 residual blocks with channel attention         │    │
│  │  • Output: 512-dim feature maps                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Shared Embedding Layer (256-dim)          │    │
│  │  • AdaptiveAvgPool → Linear(512, 256) → BN → PReLU  │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                ↓                ↓               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Identity   │  │  Emotion    │  │  Liveness   │         │
│  │    Head     │  │    Head     │  │    Head     │         │
│  │ (ArcFace)   │  │ (7 classes) │  │ (2 classes) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│   ↓ Embedding      ↓ Logits         ↓ Logits               │
│  (256-dim)       (7-dim)          (2-dim)                   │
└─────────────────────────────────────────────────────────────┘
```

### Task Heads

| Head | Architecture | Output | Purpose |
|------|--------------|--------|---------|
| **Identity** | ArcFace classifier | 256-dim embedding | Face verification |
| **Emotion** | FC(256,128) → ReLU → Dropout → FC(128,7) | 7 logits | Emotion classification |
| **Liveness** | FC(256,64) → ReLU → Dropout → FC(64,2) | 2 logits | Live/Spoof detection |

### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | 12.1M |
| Trainable (heads only) | 1.16M |
| Model Size | ~48 MB |
| Inference Time | ~8ms/image (GPU) |

---

## Training Procedures

### Phase 1: Emotion Recognition Training

**Dataset**: RAF-DB (Real-world Affective Faces Database)

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 12,271 | 7 emotions |
| Test | 3,068 | 7 emotions |

**Class Distribution (Imbalanced)**:
| Class | Train Samples | Weight |
|-------|---------------|--------|
| Surprise | 1,290 | 1.36 |
| Fear | 281 | 6.24 |
| Disgust | 717 | 2.44 |
| Happy | 4,772 | 0.37 |
| Sad | 1,982 | 0.88 |
| Angry | 705 | 2.49 |
| Neutral | 2,524 | 0.69 |

**Training Configuration**:
```python
{
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'scheduler': 'OneCycleLR',
    'backbone': 'frozen',
    'early_stopping_patience': 10,
    'class_balancing': 'WeightedRandomSampler',
}
```

**Data Augmentation**:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.1)
- Random affine translation (10%)

### Phase 2: Liveness Detection Training

**Dataset**: Improved Synthetic Liveness (created from RAF-DB)

| Split | Live | Spoof | Total |
|-------|------|-------|-------|
| Train | 6,136 | 6,135 | 12,271 |
| Test | 1,534 | 1,534 | 3,068 |

**Spoof Augmentation Techniques**:
1. **Print Attack Simulation**:
   - Halftone patterns
   - Paper texture overlay
   - Reduced skin texture (flat look)
   - Specular reflections
   - Boundary artifacts

2. **Replay Attack Simulation**:
   - Moiré interference patterns
   - Screen scan lines
   - Color channel shifts
   - JPEG compression artifacts

**Training Configuration**:
```python
{
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 1e-3,
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'scheduler': 'OneCycleLR',
    'backbone': 'frozen',
}
```

---

## Loss Functions

### MTLLoss (Multi-Task Loss)

Combines task-specific losses with learnable uncertainty weighting:

```python
L_total = Σ (1/(2σ_i²)) * L_i + log(σ_i)
```

Where:
- `L_i` = task-specific loss
- `σ_i` = learned uncertainty parameter for task i

### Emotion Loss: Label Smoothing Cross-Entropy

```python
L_emotion = -Σ y_smooth * log(p)
```

Where:
- `y_smooth = (1 - ε) * y_hard + ε / K`
- `ε = 0.1` (smoothing factor)
- `K = 7` (number of classes)

### Liveness Loss: Binary Cross-Entropy with Logits

```python
L_liveness = BCE(logits, labels)
```

---

## Evaluation Results

### Emotion Recognition

**Overall Metrics**:
| Metric | Baseline | After Balancing | Change |
|--------|----------|-----------------|--------|
| Accuracy | 44.9% | 38.0% | -6.9pp* |
| Precision | 42.1% | 38.8% | -3.3pp |
| Recall | 44.9% | 38.0% | -6.9pp |
| F1 Score | 37.7% | 37.1% | -0.6pp |

*Note: Overall accuracy decreased but per-class balance improved significantly.

**Per-Class Performance (Recall)**:
| Class | Baseline | After Balancing | Improvement |
|-------|----------|-----------------|-------------|
| Disgust | 0.0% | **21.9%** | +21.9pp ✅ |
| Fear | 5.4% | **39.2%** | +33.8pp ✅ |
| Surprise | 6.4% | **17.6%** | +11.2pp ✅ |
| Angry | 14.8% | **33.3%** | +18.5pp ✅ |
| Neutral | 26.6% | **38.7%** | +12.1pp ✅ |
| Sad | 43.5% | 10.0% | -33.5pp |
| Happy | 83.4% | 57.4% | -26.0pp |

**Key Insight**: The model is no longer biased toward predicting only "Happy". Minority classes are now being recognized.

### Liveness Detection

**Metrics Comparison**:
| Metric | Baseline | After Improvement | Change |
|--------|----------|-------------------|--------|
| Accuracy | 54.5% | **62.3%** | +7.8pp ✅ |
| AUC | 0.583 | **0.673** | +0.09 ✅ |
| EER | 0.460 | **0.377** | -0.083 ✅ |

**Confusion Matrix (After)**:
```
                Predicted
              Live    Spoof
Actual Live    898     636   (Recall: 58.5%)
       Spoof   521    1013   (Recall: 66.0%)
```

**Per-Class Performance**:
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Live | 0.633 | 0.585 | 0.608 |
| Spoof | 0.614 | 0.660 | 0.637 |

---

## Benchmark Summary

### Current Performance

| Task | Metric | Value | Status |
|------|--------|-------|--------|
| Emotion | Accuracy | 38.0% | Balanced ✓ |
| Emotion | Minority Recall | 17-39% | Improved ✓ |
| Liveness | Accuracy | 62.3% | Improved ✓ |
| Liveness | AUC | 0.673 | Improved ✓ |

### Comparison with State-of-the-Art

| Model | Emotion Acc | Liveness AUC | Notes |
|-------|-------------|--------------|-------|
| **Ours (MTL)** | 38.0% | 0.673 | Synthetic liveness data |
| RAF-DB Baseline | 74% | - | Single task, full train |
| FER2013 Winner | 71% | - | Single task |
| CelebA-Spoof SOTA | - | 0.99+ | Real spoof data |

*Note: Our results are with frozen backbone and synthetic/limited data. Performance will improve with steps 2-4 in FUTURE_IMPROVEMENTS.md.*

---

## Model Files

### Checkpoints

| File | Description | Size |
|------|-------------|------|
| `emotion_best.pth` | Best emotion model | ~48MB |
| `liveness_best.pth` | Best liveness model | ~48MB |
| `emotion_latest.pth` | Final emotion checkpoint | ~48MB |
| `liveness_latest.pth` | Final liveness checkpoint | ~48MB |

### Training Logs

| File | Description |
|------|-------------|
| `emotion_history.json` | Epoch-by-epoch metrics |
| `liveness_history.json` | Epoch-by-epoch metrics |
| `training_emotion_balanced.log` | Training stdout |
| `training_liveness_improved.log` | Training stdout |

### Evaluation Outputs

| File | Description |
|------|-------------|
| `emotion_results.json` | Full evaluation metrics |
| `liveness_results.json` | Full evaluation metrics |
| `emotion_confusion_matrix.png` | Visual confusion matrix |
| `liveness_roc_curve.png` | ROC curve plot |

---

## Usage

### Training

```bash
# Train emotion head (with balanced sampling)
python train_mtl.py --phase emotion --epochs 50

# Train liveness head
python train_mtl.py --phase liveness --epochs 50

# Joint training (future)
python train_mtl.py --phase joint --epochs 30
```

### Evaluation

```bash
# Evaluate emotion
python evaluate_mtl.py --task emotion

# Evaluate liveness
python evaluate_mtl.py --task liveness

# Full evaluation
python evaluate_mtl.py --task all
```

### Inference

```python
from models_mtl import UnifiedFaceModel
import torch

# Load model
model = UnifiedFaceModel(num_identities=1000)
model.load_state_dict(torch.load('emotion_best.pth')['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image_tensor, tasks=['emotion', 'liveness'])
    emotion_pred = outputs['emotion'].argmax(dim=1)
    liveness_pred = outputs['liveness'].argmax(dim=1)
```

---

## References

- RAF-DB: [http://www.whdeng.cn/raf/model1.html](http://www.whdeng.cn/raf/model1.html)
- ResNet: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- CBAM: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss", CVPR 2019

---

*Documentation generated: January 11, 2026*
