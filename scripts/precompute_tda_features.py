#!/usr/bin/env python3
"""
Precompute TDA features for all dataset splits.

This script computes TDA features for train, val, and test sets
and saves them to disk for fast loading during training.

Usage:
    conda run -n face_recog python scripts/precompute_tda_features.py
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, OUTPUT_DIR
from tda_features import TDAFeatureExtractor, precompute_tda_features, load_image_as_array


def load_cached_features(cache_path: Path):
    """Load features and paths from cache file."""
    data = np.load(cache_path, allow_pickle=True)
    return data['features'], data['paths'].tolist()


def main():
    print("=" * 70)
    print("PRECOMPUTING TDA FEATURES FOR ALL DATASETS")
    print("=" * 70)
    
    # Configuration
    TDA_OUTPUT_DIR = OUTPUT_DIR / "tda_cache"
    TDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_cache = TDA_OUTPUT_DIR / "tda_train.npz"
    val_cache = TDA_OUTPUT_DIR / "tda_val.npz"
    test_cache = TDA_OUTPUT_DIR / "tda_test.npz"
    extractor_cache = TDA_OUTPUT_DIR / "tda_extractor.pkl"
    
    start_time = time.time()
    
    # Check if training cache exists
    if train_cache.exists() and train_cache.stat().st_size > 0:
        print("\n[1/4] Loading existing TRAINING cache...")
        train_features, train_paths = load_cached_features(train_cache)
        print(f"  Loaded {len(train_paths):,} cached training images")
        
        # Need to create a fitted extractor from the cached data
        # Auto-detect im_range from cached features
        print("\n[2/4] Reconstructing TDA extractor from cached data...")
        
        # Load sample images to fit extractor
        sample_images = []
        for p in train_paths[:2000]:
            img = load_image_as_array(p)
            if img is not None:
                sample_images.append(img)
        sample_images = np.array(sample_images)
        
        extractor = TDAFeatureExtractor(
            homology_dimensions=0,
            resolution=(20, 20),
            bandwidth=50.0,
            n_jobs=-2,
        )
        extractor.fit(sample_images)
        print(f"  Extractor fitted with im_range: {extractor.im_range}")
    else:
        # Process training set from scratch
        print("\n[1/4] Initializing TDA extractor...")
        extractor = TDAFeatureExtractor(
            homology_dimensions=0,
            resolution=(20, 20),
            bandwidth=50.0,
            n_jobs=-2,
        )
        
        print("\n[2/4] Processing TRAINING set...")
        train_features, train_paths, extractor = precompute_tda_features(
            data_dir=TRAIN_DIR,
            output_path=train_cache,
            extractor=extractor,
            fit_on_sample=True,
            sample_size=2000,
            batch_size=500,
        )
        print(f"  Training: {train_features.shape[0]:,} images → {train_features.shape[1]}-dim features")
    
    # Save the fitted extractor
    print(f"\n  Saving extractor to {extractor_cache}...")
    extractor.save(extractor_cache)
    
    # Process validation set
    if val_cache.exists() and val_cache.stat().st_size > 0:
        print("\n[3/4] Loading existing VALIDATION cache...")
        val_features, val_paths = load_cached_features(val_cache)
        print(f"  Loaded {len(val_paths):,} cached validation images")
    else:
        print("\n[3/4] Processing VALIDATION set...")
        val_features, val_paths, _ = precompute_tda_features(
            data_dir=VAL_DIR,
            output_path=val_cache,
            extractor=extractor,
            fit_on_sample=False,
            batch_size=500,
        )
        print(f"  Validation: {val_features.shape[0]:,} images → {val_features.shape[1]}-dim features")
    
    # Process test set
    if test_cache.exists() and test_cache.stat().st_size > 0:
        print("\n[4/4] Loading existing TEST cache...")
        test_features, test_paths = load_cached_features(test_cache)
        print(f"  Loaded {len(test_paths):,} cached test images")
    else:
        print("\n[4/4] Processing TEST set...")
        test_features, test_paths, _ = precompute_tda_features(
            data_dir=TEST_DIR,
            output_path=test_cache,
            extractor=extractor,
            fit_on_sample=False,
            batch_size=500,
        )
        print(f"  Test: {test_features.shape[0]:,} images → {test_features.shape[1]}-dim features")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TDA FEATURE PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"\nSaved files:")
    print(f"  - {train_cache}")
    print(f"  - {val_cache}")
    print(f"  - {test_cache}")
    print(f"  - {extractor_cache}")
    
    # Summary stats
    total_images = len(train_paths) + len(val_paths) + len(test_paths)
    print(f"\nTotal images processed: {total_images:,}")
    print(f"Feature dimensionality: {extractor.feature_dim}")
    print(f"im_range: {extractor.im_range}")
    print("=" * 70)


if __name__ == "__main__":
    main()
