#!/usr/bin/env python3
"""
MTL Model Diagnostic Analysis

This script analyzes training results and provides recommendations for improvement.

Author: MTL Implementation  
Date: January 2026
"""

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/hieutran/Facial-Recognition-with-Emotion-Liveness-Detection")
OUTPUT_DIR = BASE_DIR / "outputs" / "mtl"

def analyze_training():
    """Analyze training curves for both tasks."""
    
    print("=" * 80)
    print("MTL MODEL DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Load histories
    with open(OUTPUT_DIR / "emotion_history.json") as f:
        emotion_hist = json.load(f)
    
    with open(OUTPUT_DIR / "liveness_history.json") as f:
        liveness_hist = json.load(f)
    
    # ============= EMOTION ANALYSIS =============
    print("\n" + "=" * 80)
    print("1. EMOTION RECOGNITION ANALYSIS")
    print("=" * 80)
    
    emotion_train_acc = emotion_hist['train_acc']
    emotion_val_acc = emotion_hist['val_acc']
    emotion_train_loss = emotion_hist['train_loss']
    emotion_val_loss = emotion_hist['val_loss']
    
    print(f"\nTraining Summary:")
    print(f"  Epochs: {len(emotion_train_acc)}")
    print(f"  Final Train Accuracy: {emotion_train_acc[-1]:.4f}")
    print(f"  Final Val Accuracy: {emotion_val_acc[-1]:.4f}")
    print(f"  Best Val Accuracy: {max(emotion_val_acc):.4f} (epoch {np.argmax(emotion_val_acc)+1})")
    print(f"  Final Train Loss: {emotion_train_loss[-1]:.4f}")
    print(f"  Final Val Loss: {emotion_val_loss[-1]:.4f}")
    
    # Convergence analysis
    early_acc = np.mean(emotion_val_acc[:5])
    late_acc = np.mean(emotion_val_acc[-5:])
    improvement = late_acc - early_acc
    
    print(f"\nConvergence Analysis:")
    print(f"  Early Val Acc (epochs 1-5): {early_acc:.4f}")
    print(f"  Late Val Acc (epochs 46-50): {late_acc:.4f}")
    print(f"  Total Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
    
    # Overfitting check
    train_val_gap = emotion_train_acc[-1] - emotion_val_acc[-1]
    print(f"\nOverfitting Check:")
    print(f"  Train-Val Gap: {train_val_gap:.4f}")
    if train_val_gap > 0.05:
        print(f"  ⚠️  MILD OVERFITTING DETECTED")
    else:
        print(f"  ✓ No significant overfitting")
    
    # Learning rate analysis
    loss_reduction_rate = (emotion_val_loss[0] - emotion_val_loss[-1]) / len(emotion_val_loss)
    print(f"\nLearning Dynamics:")
    print(f"  Avg Val Loss Reduction per Epoch: {loss_reduction_rate:.6f}")
    
    # Per-class issues from evaluation
    print(f"\nPer-Class Issues (from evaluation):")
    print(f"  ❌ DISGUST: 0% recall - Model never predicts this class!")
    print(f"  ❌ FEAR: 5.4% recall - Severely underrepresented")
    print(f"  ❌ SURPRISE: 6.7% recall - Very low detection")
    print(f"  ⚠️  ANGRY: 9.9% recall - Low detection")
    print(f"  ⚠️  SAD: 14.4% recall - Below average")
    print(f"  ✓ HAPPY: 83.4% recall - Dominant class")
    print(f"  ✓ NEUTRAL: 41.0% recall - Reasonable")
    
    # ============= LIVENESS ANALYSIS =============
    print("\n" + "=" * 80)
    print("2. LIVENESS DETECTION ANALYSIS")
    print("=" * 80)
    
    liveness_train_acc = liveness_hist['train_acc']
    liveness_val_acc = liveness_hist['val_acc']
    liveness_train_loss = liveness_hist['train_loss']
    liveness_val_loss = liveness_hist['val_loss']
    
    print(f"\nTraining Summary:")
    print(f"  Epochs: {len(liveness_train_acc)}")
    print(f"  Final Train Accuracy: {liveness_train_acc[-1]:.4f}")
    print(f"  Final Val Accuracy: {liveness_val_acc[-1]:.4f}")
    print(f"  Best Val Accuracy: {max(liveness_val_acc):.4f} (epoch {np.argmax(liveness_val_acc)+1})")
    print(f"  AUC: 0.5830")
    print(f"  EER: 0.4597")
    
    print(f"\nConvergence Analysis:")
    early_acc = np.mean(liveness_val_acc[:5])
    late_acc = np.mean(liveness_val_acc[-5:])
    improvement = late_acc - early_acc
    print(f"  Early Val Acc (epochs 1-5): {early_acc:.4f}")
    print(f"  Late Val Acc (epochs 26-30): {late_acc:.4f}")
    print(f"  Total Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
    
    print(f"\nCritical Issue:")
    print(f"  ❌ ACCURACY ~54% is barely above random (50%)!")
    print(f"  ❌ AUC 0.58 indicates poor discriminative power")
    print(f"  ❌ EER 0.46 is very high (good models have EER < 0.1)")
    
    # ============= EMBEDDING ANALYSIS =============
    print("\n" + "=" * 80)
    print("3. EMBEDDING QUALITY ANALYSIS")
    print("=" * 80)
    
    print(f"\nFrom Evaluation:")
    print(f"  Embedding Dimension: 256")
    print(f"  Embedding Norm: 1.0000 (correctly L2-normalized)")
    print(f"  Intra-class Distance: 0.1263 ± 0.0751")
    print(f"  Inter-class Distance: 0.1385 ± 0.0806")
    
    print(f"\nCritical Issue:")
    print(f"  ❌ Inter-class distance (0.1385) is only slightly > intra-class (0.1263)")
    print(f"  ❌ Gap of only 0.0122 indicates poor class separation")
    print(f"  ⚠️  This explains why classification performance is limited")
    
    # ============= ROOT CAUSE ANALYSIS =============
    print("\n" + "=" * 80)
    print("4. ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DIAGNOSED PROBLEMS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EMOTION RECOGNITION:                                                        │
│  ────────────────────                                                        │
│  1. SEVERE CLASS IMBALANCE                                                   │
│     - Happy: 1185 samples (38.6%) → Model biased toward predicting happy     │
│     - Fear: 74 samples (2.4%) → Almost never predicted                       │
│     - Disgust: 160 samples (5.2%) → Never predicted (0% recall)              │
│                                                                              │
│  2. FROZEN BACKBONE LIMITATION                                               │
│     - Backbone trained on VGGFace2 for identity, not emotions                │
│     - Features may not capture emotion-discriminative patterns               │
│     - Only training heads limits representational capacity                   │
│                                                                              │
│  3. SMALL DATASET                                                            │
│     - Only 12,271 training images for 7 classes                              │
│     - ~1,750 images per class on average (very few for deep learning)        │
│                                                                              │
│  LIVENESS DETECTION:                                                         │
│  ───────────────────                                                         │
│  1. SYNTHETIC DATA PROBLEM                                                   │
│     - Dataset was synthetically created from emotion images                  │
│     - Simple augmentations (blur, noise) don't mimic real spoofs             │
│     - No actual printed photos, screen replays, or 3D masks                  │
│                                                                              │
│  2. DOMAIN MISMATCH                                                          │
│     - Model cannot learn real spoof patterns from synthetic data             │
│     - Real liveness requires texture analysis, moire patterns, etc.          │
│                                                                              │
│  3. FROZEN BACKBONE                                                          │
│     - Liveness features (texture, depth cues) differ from face identity      │
│     - Pretrained features not suitable for liveness detection                │
│                                                                              │
│  EMBEDDING QUALITY:                                                          │
│  ──────────────────                                                          │
│  - Poor class separation indicates backbone needs fine-tuning                │
│  - Current embeddings optimized for identity, not emotion/liveness           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    # ============= RECOMMENDATIONS =============
    print("\n" + "=" * 80)
    print("5. IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRIORITY 1: IMMEDIATE FIXES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  A. ADDRESS CLASS IMBALANCE (Emotion):                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Option 1: Weighted Loss (already using, but weights need tuning)│     │
│     │   - Set class_weight inversely proportional to frequency        │     │
│     │   - Fear weight: 1185/74 = 16x                                  │     │
│     │   - Disgust weight: 1185/160 = 7.4x                             │     │
│     │                                                                 │     │
│     │ Option 2: Oversampling Minority Classes                         │     │
│     │   - Use WeightedRandomSampler in DataLoader                     │     │
│     │   - Oversample fear, disgust, surprise, angry                   │     │
│     │                                                                 │     │
│     │ Option 3: Data Augmentation                                     │     │
│     │   - More aggressive augmentation for minority classes           │     │
│     │   - Mixup/CutMix between classes                                │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  B. UNFREEZE BACKBONE (Both tasks):                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Current: Backbone frozen, only heads training                   │     │
│     │                                                                 │     │
│     │ Recommended Strategy:                                           │     │
│     │   Phase 1 (current): Freeze backbone, train heads (warmup)      │     │
│     │   Phase 2 (NEW): Unfreeze last 2 blocks, fine-tune              │     │
│     │   Phase 3 (NEW): Unfreeze all, low LR full fine-tuning          │     │
│     │                                                                 │     │
│     │ Learning Rates:                                                 │     │
│     │   - Backbone layers: 1e-5 to 1e-4 (10x lower than heads)        │     │
│     │   - Head layers: 1e-3 to 1e-4                                   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                       PRIORITY 2: DATASET IMPROVEMENTS                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  C. GET REAL LIVENESS DATA:                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ The synthetic dataset is fundamentally flawed for liveness.     │     │
│     │                                                                 │     │
│     │ Options:                                                        │     │
│     │   1. Free disk space for CelebA-Spoof (78GB) - Best option      │     │
│     │   2. Use NUAA dataset (4GB) - Smaller but real spoofs           │     │
│     │   3. Use CASIA-FASD (2GB) - Another real option                 │     │
│     │   4. Use Replay-Attack (2GB) - Good for replay attacks          │     │
│     │                                                                 │     │
│     │ Without real spoof data, liveness model will NOT generalize!    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  D. EXPAND EMOTION DATASET:                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Consider combining datasets:                                    │     │
│     │   - RAF-DB (current): 15K images                                │     │
│     │   - FER2013: 35K images                                         │     │
│     │   - AffectNet: 400K+ images (largest, but needs license)        │     │
│     │   - ExpW: 91K images (web collected)                            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                       PRIORITY 3: ARCHITECTURE IMPROVEMENTS                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  E. TASK-SPECIFIC ATTENTION:                                                 │
│     - Add separate attention modules for each task                           │
│     - Emotion: Focus on mouth, eyebrows, eyes                                │
│     - Liveness: Focus on texture, edges, frequency patterns                  │
│                                                                              │
│  F. AUXILIARY LOSSES:                                                        │
│     - Emotion: Add center loss for better class separation                   │
│     - Liveness: Add binary cross-entropy + texture consistency loss          │
│                                                                              │
│  G. LARGER INPUT SIZE:                                                       │
│     - Current: 64x64 (very small for subtle expression/liveness cues)        │
│     - Recommended: 112x112 or 224x224                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    # ============= ACTION PLAN =============
    print("\n" + "=" * 80)
    print("6. RECOMMENDED ACTION PLAN")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Fix Emotion Class Imbalance (Quick Win)                            │
│  ───────────────────────────────────────────────────────────────────────────│
│  • Add class weights to loss function                                        │
│  • Use WeightedRandomSampler                                                 │
│  • Expected improvement: +10-15% accuracy                                    │
│  • Time: 1-2 hours                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 2: Unfreeze and Fine-tune Backbone                                    │
│  ───────────────────────────────────────────────────────────────────────────│
│  • Unfreeze last 2 conv blocks                                               │
│  • Use differential learning rates                                           │
│  • Expected improvement: +5-10% accuracy                                     │
│  • Time: 2-3 hours training                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 3: Replace Synthetic Liveness Data (Critical)                         │
│  ───────────────────────────────────────────────────────────────────────────│
│  • Download NUAA or CASIA-FASD dataset                                       │
│  • Retrain liveness head with real data                                      │
│  • Expected improvement: +20-30% accuracy                                    │
│  • Time: 3-4 hours (download + training)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 4: Joint Fine-tuning (Final Polish)                                   │
│  ───────────────────────────────────────────────────────────────────────────│
│  • Train all tasks together                                                  │
│  • Use uncertainty weighting for task balancing                              │
│  • Expected: Unified model with good multi-task performance                  │
│  • Time: 4-6 hours training                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_training()
