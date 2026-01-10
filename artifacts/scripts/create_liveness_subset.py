#!/usr/bin/env python3
"""
Download a subset of CelebA-Spoof dataset for liveness detection training.

Since the full dataset is 78GB, we download only a subset (~10-15GB).
Strategy: Download metadata first, then selectively download images.

Alternative: Use NUAA or Replay-Attack datasets which are smaller.

Author: MTL Implementation
Date: January 2026
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Check for kaggle
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False
    print("Kaggle API not installed. Install with: pip install kaggle")

# Configuration
BASE_DIR = Path("/home/hieutran/Facial-Recognition-with-Emotion-Liveness-Detection")
LIVENESS_DIR = BASE_DIR / "dataset" / "liveness"
SUBSET_SIZE = 50000  # Number of images to use (balanced: 25k live, 25k spoof)


def create_synthetic_liveness_dataset():
    """
    Create a synthetic liveness dataset for initial testing.
    Uses existing face images with augmentation to simulate live/spoof.
    
    This allows training to proceed while waiting for real dataset.
    """
    import shutil
    from PIL import Image
    import numpy as np
    
    print("\n" + "="*60)
    print("Creating Synthetic Liveness Dataset for Testing")
    print("="*60)
    
    # Use emotion dataset images as base
    emotion_dir = BASE_DIR / "dataset" / "emotion" / "DATASET" / "train"
    
    if not emotion_dir.exists():
        print(f"ERROR: Emotion dataset not found at {emotion_dir}")
        return False
    
    # Create output directories
    train_live = LIVENESS_DIR / "synthetic" / "train" / "live"
    train_spoof = LIVENESS_DIR / "synthetic" / "train" / "spoof"
    test_live = LIVENESS_DIR / "synthetic" / "test" / "live"
    test_spoof = LIVENESS_DIR / "synthetic" / "test" / "spoof"
    
    for d in [train_live, train_spoof, test_live, test_spoof]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Collect source images
    source_images = list(emotion_dir.rglob("*.jpg"))
    print(f"Found {len(source_images)} source images")
    
    if len(source_images) < 1000:
        print("ERROR: Not enough source images")
        return False
    
    random.shuffle(source_images)
    
    # Split: 80% train, 20% test
    train_images = source_images[:int(len(source_images) * 0.8)]
    test_images = source_images[int(len(source_images) * 0.8):]
    
    def apply_spoof_augmentation(img_path, output_path):
        """Apply transformations to simulate spoof images."""
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # Random spoof simulation techniques:
            aug_type = random.choice(['blur', 'noise', 'color', 'screen'])
            
            if aug_type == 'blur':
                # Gaussian blur (printed photo effect)
                from PIL import ImageFilter
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
            
            elif aug_type == 'noise':
                # Add noise (low quality replay)
                noise = np.random.normal(0, 15, img_np.shape).astype(np.int16)
                img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            
            elif aug_type == 'color':
                # Color distortion (screen display)
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(random.uniform(0.5, 1.5))
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.7, 1.3))
            
            elif aug_type == 'screen':
                # Screen pattern (moire effect simulation)
                pattern = np.zeros(img_np.shape[:2], dtype=np.uint8)
                pattern[::2, :] = 10
                pattern[:, ::2] += 10
                img_np = np.clip(img_np.astype(np.int16) - pattern[:,:,np.newaxis], 0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            
            img.save(output_path, quality=85)
            return True
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            return False
    
    # Process train images
    print(f"\nProcessing {len(train_images)} training images...")
    live_count = spoof_count = 0
    
    for i, img_path in enumerate(train_images):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(train_images)}")
        
        # 50% live, 50% spoof
        if i % 2 == 0:
            # Live: just copy
            dst = train_live / f"live_{i:05d}.jpg"
            try:
                shutil.copy(img_path, dst)
                live_count += 1
            except:
                pass
        else:
            # Spoof: apply augmentation
            dst = train_spoof / f"spoof_{i:05d}.jpg"
            if apply_spoof_augmentation(img_path, dst):
                spoof_count += 1
    
    print(f"  Train: {live_count} live, {spoof_count} spoof")
    
    # Process test images
    print(f"Processing {len(test_images)} test images...")
    test_live_count = test_spoof_count = 0
    
    for i, img_path in enumerate(test_images):
        if i % 2 == 0:
            dst = test_live / f"live_{i:05d}.jpg"
            try:
                shutil.copy(img_path, dst)
                test_live_count += 1
            except:
                pass
        else:
            dst = test_spoof / f"spoof_{i:05d}.jpg"
            if apply_spoof_augmentation(img_path, dst):
                test_spoof_count += 1
    
    print(f"  Test: {test_live_count} live, {test_spoof_count} spoof")
    
    # Create metadata
    metadata = {
        "dataset": "synthetic_liveness",
        "created": datetime.now().isoformat(),
        "train": {"live": live_count, "spoof": spoof_count},
        "test": {"live": test_live_count, "spoof": test_spoof_count},
        "note": "Synthetic dataset created from RAF-DB for initial testing"
    }
    
    with open(LIVENESS_DIR / "synthetic" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Synthetic Liveness Dataset Created!")
    print(f"Location: {LIVENESS_DIR / 'synthetic'}")
    print(f"Train: {live_count + spoof_count} images")
    print(f"Test: {test_live_count + test_spoof_count} images")
    print("="*60)
    
    return True


def download_celeba_spoof_subset():
    """
    Download a subset of CelebA-Spoof from Kaggle.
    
    Note: Due to the dataset structure, we need to download the full
    archive and then select a subset. This function provides instructions.
    """
    print("\n" + "="*60)
    print("CelebA-Spoof Subset Download")
    print("="*60)
    
    print("""
The CelebA-Spoof dataset is 78GB and requires significant disk space.
    
Options:
1. SYNTHETIC (Recommended for testing):
   Run this script with --synthetic to create a synthetic liveness
   dataset from existing emotion images. Good for initial development.

2. NUAA Dataset (~4GB):
   A smaller real liveness dataset. Download from:
   https://www.kaggle.com/datasets/chensteve/nuaa-imposter-dataset
   
3. CASIA-FASD (~2GB):
   Another small liveness dataset.
   
4. Full CelebA-Spoof:
   Ensure you have 80GB+ free space and run:
   kaggle datasets download -d attrell/celeba-spoof-2020
   
Current disk space: """)
    
    import subprocess
    subprocess.run(["df", "-h", str(LIVENESS_DIR.parent)])
    
    print("\nTo proceed with synthetic dataset for testing, run:")
    print(f"  python {__file__} --synthetic")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download/Create Liveness Dataset")
    parser.add_argument("--synthetic", action="store_true",
                       help="Create synthetic liveness dataset from existing images")
    parser.add_argument("--info", action="store_true",
                       help="Show download options and disk space info")
    
    args = parser.parse_args()
    
    LIVENESS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.synthetic:
        create_synthetic_liveness_dataset()
    elif args.info:
        download_celeba_spoof_subset()
    else:
        # Default: create synthetic for quick start
        print("No option specified. Creating synthetic dataset for quick start...")
        create_synthetic_liveness_dataset()


if __name__ == "__main__":
    main()
