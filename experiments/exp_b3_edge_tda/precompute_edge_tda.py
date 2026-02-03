#!/usr/bin/env python3
"""
Experiment B3: Edge-based TDA Feature Precomputation

This script computes TDA features on Canny edge maps instead of raw intensity.
Edges capture facial structure better and may provide more identity-relevant topology.

Process:
1. Load training images
2. Convert to grayscale
3. Apply Canny edge detection
4. Compute persistence diagrams using cubical complexes
5. Vectorize into persistence images/landscapes
6. Cache for training

Usage:
    conda run -n face_recog python experiments/exp_b3_edge_tda/precompute_edge_tda.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import TRAIN_DIR
from data_loader import load_classification_data

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "exp_b3_edge_tda"
EXPERIMENT_DIR = Path(__file__).parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TDA Configuration
IMAGE_SIZE = 64
CANNY_LOW = 50       # Low threshold for Canny edge detection
CANNY_HIGH = 150     # High threshold for Canny edge detection
PERSISTENCE_DIM = 400  # Output feature dimension (same as original TDA)

# Output paths
TDA_CACHE_PATH = OUTPUT_DIR / "tda_edge_train.npz"
LOG_FILE = OUTPUT_DIR / "precompute.log"

# Parallel processing
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ============================================================================


def log(msg, log_file=LOG_FILE):
    """Print and log to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(log_file, 'a') as f:
        f.write(full_msg + '\n')


def compute_persistence_features(image_array):
    """
    Compute persistence features from a 2D image array using cubical complexes.
    
    Args:
        image_array: 2D numpy array (grayscale image or edge map)
    
    Returns:
        1D numpy array of persistence features
    """
    try:
        from gtda.homology import CubicalPersistence
        from gtda.diagrams import PersistenceImage
        
        # Ensure correct shape for gtda (samples, height, width)
        img = image_array.reshape(1, image_array.shape[0], image_array.shape[1])
        
        # Compute persistence diagram
        cubical = CubicalPersistence(homology_dimensions=[0, 1], n_jobs=1)
        diagrams = cubical.fit_transform(img)
        
        # Convert to persistence image
        pi = PersistenceImage(sigma=0.1, n_bins=20, n_jobs=1)
        features = pi.fit_transform(diagrams)
        
        return features.flatten()
    except Exception as e:
        # Return zeros if computation fails
        return np.zeros(PERSISTENCE_DIM, dtype=np.float32)


def process_single_image(args):
    """Process a single image to extract edge-based TDA features"""
    path, idx = args
    try:
        # Load image
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img, dtype=np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(img_array, CANNY_LOW, CANNY_HIGH)
        
        # Normalize to [0, 1] for TDA
        edges_norm = edges.astype(np.float32) / 255.0
        
        # Compute persistence features
        features = compute_persistence_features(edges_norm)
        
        # Ensure correct dimension
        if len(features) != PERSISTENCE_DIM:
            features = np.zeros(PERSISTENCE_DIM, dtype=np.float32)
        
        return idx, features, str(path)
    except Exception as e:
        return idx, np.zeros(PERSISTENCE_DIM, dtype=np.float32), str(path)


def main():
    log("=" * 70)
    log(f"EXPERIMENT: {EXPERIMENT_NAME} - Edge TDA Precomputation")
    log("=" * 70)
    log(f"Canny thresholds: low={CANNY_LOW}, high={CANNY_HIGH}")
    log(f"Output: {TDA_CACHE_PATH}")
    
    # Check for required libraries
    try:
        import cv2
        from gtda.homology import CubicalPersistence
        from gtda.diagrams import PersistenceImage
        log("✓ Required libraries available (cv2, giotto-tda)")
    except ImportError as e:
        log(f"ERROR: Missing required library: {e}")
        log("Install with: pip install opencv-python giotto-tda")
        return
    
    # Load training data paths
    log("\n" + "-" * 70)
    log("Loading training data paths...")
    
    train_paths, train_labels, _, num_classes = load_classification_data(TRAIN_DIR)
    log(f"  {len(train_paths)} images from {num_classes} identities")
    
    # Prepare arguments for parallel processing
    args_list = [(path, idx) for idx, path in enumerate(train_paths)]
    
    # Initialize arrays
    all_features = np.zeros((len(train_paths), PERSISTENCE_DIM), dtype=np.float32)
    all_paths = [None] * len(train_paths)
    
    # Process images
    log("\n" + "-" * 70)
    log(f"Computing edge-based TDA features (workers: {NUM_WORKERS})...")
    start_time = time.time()
    
    # Use sequential processing for stability (TDA can be memory-intensive)
    # For faster processing, uncomment the parallel version below
    
    for args in tqdm(args_list, desc="Processing"):
        idx, features, path = process_single_image(args)
        all_features[idx] = features
        all_paths[idx] = path
    
    # # Parallel version (uncomment if memory allows):
    # with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     futures = {executor.submit(process_single_image, args): args[1] for args in args_list}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
    #         idx, features, path = future.result()
    #         all_features[idx] = features
    #         all_paths[idx] = path
    
    elapsed = time.time() - start_time
    log(f"\nProcessing complete in {elapsed/60:.1f} minutes")
    
    # Normalize features
    log("\nNormalizing features...")
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True) + 1e-8
    all_features = (all_features - mean) / std
    
    # Check for any failed computations
    zero_count = np.sum(np.all(all_features == 0, axis=1))
    if zero_count > 0:
        log(f"WARNING: {zero_count} images failed TDA computation")
    
    # Save cache
    log("\n" + "-" * 70)
    log(f"Saving TDA cache to {TDA_CACHE_PATH}...")
    
    np.savez_compressed(
        TDA_CACHE_PATH,
        features=all_features,
        paths=np.array(all_paths),
        mean=mean,
        std=std,
        config={
            'image_size': IMAGE_SIZE,
            'canny_low': CANNY_LOW,
            'canny_high': CANNY_HIGH,
            'feature_dim': PERSISTENCE_DIM,
        }
    )
    
    file_size = TDA_CACHE_PATH.stat().st_size / (1024 * 1024)
    log(f"✓ Saved {len(train_paths)} samples ({file_size:.1f} MB)")
    
    log("\n" + "=" * 70)
    log("PRECOMPUTATION COMPLETE")
    log("=" * 70)
    log(f"\nFeature shape: {all_features.shape}")
    log(f"Feature stats: mean={all_features.mean():.4f}, std={all_features.std():.4f}")
    log(f"\nNext step: Run training with edge-based TDA")
    log(f"  python experiments/exp_b3_edge_tda/train.py")


if __name__ == "__main__":
    main()
