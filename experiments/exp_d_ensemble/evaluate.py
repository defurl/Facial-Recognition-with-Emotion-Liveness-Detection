#!/usr/bin/env python3
"""
Experiment D: Ensemble of Exp A1 + CNN Baseline

This experiment evaluates inference-time embedding fusion:
- No training required
- Combines embeddings from best TDA-regularized model (Exp A1) and CNN baseline
- Tests different fusion strategies: averaging, weighted average, concatenation

Fusion Methods:
1. Simple Average: (emb_a1 + emb_cnn) / 2
2. Weighted Average: α * emb_a1 + (1-α) * emb_cnn  (test α = 0.3, 0.5, 0.7)
3. Concatenation: [emb_a1 || emb_cnn] → 512-dim

Usage:
    conda run -n face_recog python experiments/exp_d_ensemble/evaluate.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import DEVICE, VAL_DIR
from models import FaceEmbeddingCNN
from data_loader import load_classification_data, get_transforms

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_NAME = "exp_d_ensemble"
EXPERIMENT_DIR = Path(__file__).parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
CNN_BASELINE_PATH = PROJECT_ROOT / "outputs" / "best_cnn_baseline_model.pth"
EXP_A1_PATH = PROJECT_ROOT / "experiments" / "exp_a1_lambda_005" / "outputs" / "best_model.pth"

# Output paths
RESULTS_SAVE_PATH = OUTPUT_DIR / "ensemble_results.json"
LOG_FILE = OUTPUT_DIR / "evaluate.log"

# Fusion weights to test
ALPHA_VALUES = [0.3, 0.5, 0.7]

# ============================================================================


def log(msg, log_file=LOG_FILE):
    """Print and log to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(log_file, 'a') as f:
        f.write(full_msg + '\n')


class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]


def load_model(model_path, device):
    """Load a trained model"""
    model = FaceEmbeddingCNN(embedding_dim=256)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def compute_embeddings(model, dataloader, device):
    """Compute embeddings for all images"""
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Computing embeddings", leave=False):
            imgs = imgs.to(device)
            embeddings = model(imgs)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.numpy())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = np.array(all_labels)
    
    return embeddings, labels


def compute_verification_auc(embeddings, labels, n_pairs=5000, seed=42):
    """Compute verification AUC using random pairs"""
    np.random.seed(seed)
    
    same_pairs = []
    diff_pairs = []
    unique_labels = np.unique(labels)
    
    for _ in range(n_pairs // 2):
        # Same class pair
        lbl = np.random.choice(unique_labels)
        idxs = np.where(labels == lbl)[0]
        if len(idxs) >= 2:
            i, j = np.random.choice(idxs, 2, replace=False)
            same_pairs.append((i, j, 1))
        
        # Different class pair
        lbl1, lbl2 = np.random.choice(unique_labels, 2, replace=False)
        idx1 = np.random.choice(np.where(labels == lbl1)[0])
        idx2 = np.random.choice(np.where(labels == lbl2)[0])
        diff_pairs.append((idx1, idx2, 0))
    
    pairs = same_pairs + diff_pairs
    np.random.shuffle(pairs)
    
    # Compute similarities
    similarities = []
    true_labels = []
    for i, j, label in pairs:
        emb1 = embeddings[i]
        emb2 = embeddings[j]
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        similarities.append(sim)
        true_labels.append(label)
    
    auc = roc_auc_score(true_labels, similarities)
    return auc


def fuse_embeddings_average(emb1, emb2):
    """Simple averaging of embeddings"""
    # L2 normalize before averaging
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    fused = (emb1_norm + emb2_norm) / 2
    return F.normalize(fused, p=2, dim=1)


def fuse_embeddings_weighted(emb1, emb2, alpha):
    """Weighted averaging: alpha * emb1 + (1-alpha) * emb2"""
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    fused = alpha * emb1_norm + (1 - alpha) * emb2_norm
    return F.normalize(fused, p=2, dim=1)


def fuse_embeddings_concat(emb1, emb2):
    """Concatenation: [emb1 || emb2]"""
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    return torch.cat([emb1_norm, emb2_norm], dim=1)


def main():
    log("=" * 70)
    log(f"EXPERIMENT: {EXPERIMENT_NAME}")
    log("=" * 70)
    log("Ensemble Evaluation: Exp A1 + CNN Baseline")
    log(f"Output directory: {OUTPUT_DIR}")
    
    # Check model availability
    if not CNN_BASELINE_PATH.exists():
        log(f"ERROR: CNN baseline not found at {CNN_BASELINE_PATH}")
        return
    if not EXP_A1_PATH.exists():
        log(f"ERROR: Exp A1 model not found at {EXP_A1_PATH}")
        return
    
    log(f"\nModels:")
    log(f"  CNN Baseline: {CNN_BASELINE_PATH}")
    log(f"  Exp A1: {EXP_A1_PATH}")
    log(f"  Device: {DEVICE}")
    
    # Load models
    log("\n" + "-" * 70)
    log("Loading models...")
    model_cnn = load_model(CNN_BASELINE_PATH, DEVICE)
    model_a1 = load_model(EXP_A1_PATH, DEVICE)
    log("✓ Models loaded successfully")
    
    # Load validation data
    log("\n" + "-" * 70)
    log("Loading validation data...")
    val_paths, val_labels, _, num_classes = load_classification_data(VAL_DIR)
    log(f"  {len(val_paths)} images, {num_classes} classes")
    
    _, val_transform = get_transforms()
    val_dataset = SimpleDataset(val_paths, val_labels, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Compute embeddings for both models
    log("\n" + "-" * 70)
    log("Computing embeddings...")
    
    log("  CNN Baseline...")
    emb_cnn, labels = compute_embeddings(model_cnn, val_loader, DEVICE)
    log(f"    Shape: {emb_cnn.shape}")
    
    log("  Exp A1...")
    emb_a1, _ = compute_embeddings(model_a1, val_loader, DEVICE)
    log(f"    Shape: {emb_a1.shape}")
    
    # Evaluate individual models
    log("\n" + "-" * 70)
    log("Evaluating individual models...")
    
    auc_cnn = compute_verification_auc(emb_cnn, labels)
    log(f"  CNN Baseline AUC: {auc_cnn*100:.2f}%")
    
    auc_a1 = compute_verification_auc(emb_a1, labels)
    log(f"  Exp A1 AUC: {auc_a1*100:.2f}%")
    
    # Evaluate fusion methods
    log("\n" + "-" * 70)
    log("Evaluating fusion methods...")
    
    results = {
        'individual': {
            'cnn_baseline': {'auc': auc_cnn},
            'exp_a1': {'auc': auc_a1},
        },
        'fusion': {}
    }
    
    # Simple average
    log("\n  Simple Average:")
    emb_avg = fuse_embeddings_average(emb_a1, emb_cnn)
    auc_avg = compute_verification_auc(emb_avg, labels)
    log(f"    AUC: {auc_avg*100:.2f}%")
    results['fusion']['simple_average'] = {'auc': auc_avg}
    
    # Weighted averages
    for alpha in ALPHA_VALUES:
        log(f"\n  Weighted (α={alpha}, A1-weighted):")
        emb_weighted = fuse_embeddings_weighted(emb_a1, emb_cnn, alpha)
        auc_weighted = compute_verification_auc(emb_weighted, labels)
        log(f"    AUC: {auc_weighted*100:.2f}%")
        results['fusion'][f'weighted_alpha_{alpha}'] = {'alpha': alpha, 'auc': auc_weighted}
    
    # Concatenation
    log("\n  Concatenation (512-dim):")
    emb_concat = fuse_embeddings_concat(emb_a1, emb_cnn)
    auc_concat = compute_verification_auc(emb_concat, labels)
    log(f"    AUC: {auc_concat*100:.2f}%")
    results['fusion']['concatenation'] = {'auc': auc_concat, 'dim': 512}
    
    # Find best fusion
    best_fusion = max(results['fusion'].items(), key=lambda x: x[1]['auc'])
    best_individual = max([auc_cnn, auc_a1])
    
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"\n{'Method':<30} {'AUC':<12}")
    log("-" * 42)
    log(f"{'CNN Baseline':<30} {auc_cnn*100:.2f}%")
    log(f"{'Exp A1 (λ=0.05)':<30} {auc_a1*100:.2f}%")
    log("-" * 42)
    log(f"{'Simple Average':<30} {results['fusion']['simple_average']['auc']*100:.2f}%")
    for alpha in ALPHA_VALUES:
        key = f'weighted_alpha_{alpha}'
        log(f"{'Weighted (α=' + str(alpha) + ')':<30} {results['fusion'][key]['auc']*100:.2f}%")
    log(f"{'Concatenation':<30} {results['fusion']['concatenation']['auc']*100:.2f}%")
    
    log("\n" + "-" * 42)
    log(f"Best Individual: {best_individual*100:.2f}%")
    log(f"Best Fusion: {best_fusion[0]} ({best_fusion[1]['auc']*100:.2f}%)")
    
    improvement = best_fusion[1]['auc'] - best_individual
    if improvement > 0:
        log(f"✓ Fusion improvement: +{improvement*100:.2f}%")
    else:
        log(f"✗ No fusion improvement: {improvement*100:.2f}%")
    
    # Save results
    results['best_fusion'] = {
        'method': best_fusion[0],
        'auc': best_fusion[1]['auc'],
        'improvement_over_best_individual': improvement
    }
    results['timestamp'] = datetime.now().isoformat()
    
    with open(RESULTS_SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {RESULTS_SAVE_PATH}")
    
    log("\n" + "=" * 70)


if __name__ == "__main__":
    main()
