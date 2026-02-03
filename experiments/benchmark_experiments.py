#!/usr/bin/env python3
"""
Benchmark All Experiments

Run this script after training to benchmark all experiment models
and compare with baselines.

Usage:
    conda run -n face_recog python experiments/benchmark_experiments.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
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


# Experiments to benchmark
EXPERIMENTS = [
    # === Baselines ===
    {
        "name": "CNN Baseline",
        "model_path": PROJECT_ROOT / "outputs" / "best_cnn_baseline_model.pth",
        "is_baseline": True,
    },
    {
        "name": "TDA Regularized (λ=0.10)",
        "model_path": PROJECT_ROOT / "outputs" / "best_tda_regularized_model.pth",
        "is_baseline": True,
    },
    
    # === Series A: Lambda Optimization (Intensity-based TDA) ===
    {
        "name": "Exp A1 (λ=0.05)",
        "model_path": PROJECT_ROOT / "experiments" / "exp_a1_lambda_005" / "outputs" / "best_model.pth",
        "history_path": PROJECT_ROOT / "experiments" / "exp_a1_lambda_005" / "outputs" / "training_history.json",
        "is_baseline": False,
    },
    {
        "name": "Exp A2 (λ=0.02)",
        "model_path": PROJECT_ROOT / "experiments" / "exp_a2_lambda_002" / "outputs" / "best_model.pth",
        "history_path": PROJECT_ROOT / "experiments" / "exp_a2_lambda_002" / "outputs" / "training_history.json",
        "is_baseline": False,
    },
    {
        "name": "Exp A3 (λ=0.03)",
        "model_path": PROJECT_ROOT / "experiments" / "exp_a3_lambda_003" / "outputs" / "best_model.pth",
        "history_path": PROJECT_ROOT / "experiments" / "exp_a3_lambda_003" / "outputs" / "training_history.json",
        "is_baseline": False,
    },
    
    # === Series B: TDA Feature Engineering ===
    {
        "name": "Exp B3 (Edge TDA λ=0.05)",
        "model_path": PROJECT_ROOT / "experiments" / "exp_b3_edge_tda" / "outputs" / "best_model.pth",
        "history_path": PROJECT_ROOT / "experiments" / "exp_b3_edge_tda" / "outputs" / "training_history.json",
        "is_baseline": False,
    },
    
    # === Series D: Ensemble Methods ===
    # Note: Exp D is ensemble evaluation only (no trained model), results stored separately
]


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


def compute_embeddings(model, dataloader, device):
    """Compute embeddings for all images"""
    model.eval()
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


def get_best_val_acc_from_history(history_path):
    """Extract best validation accuracy from training history"""
    if not history_path.exists():
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        return None
    
    best_val_acc = max(h['val_acc'] for h in history)
    return best_val_acc


def benchmark_model(exp_config, val_loader, device):
    """Benchmark a single model"""
    model_path = exp_config["model_path"]
    
    if not model_path.exists():
        return None
    
    # Load model
    model = FaceEmbeddingCNN(embedding_dim=256)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Compute embeddings
    embeddings, labels = compute_embeddings(model, val_loader, device)
    
    # Compute verification AUC
    auc = compute_verification_auc(embeddings, labels)
    
    # Get val accuracy from history if available
    val_acc = None
    if "history_path" in exp_config and exp_config["history_path"]:
        val_acc = get_best_val_acc_from_history(exp_config["history_path"])
    
    return {
        "auc": auc,
        "val_acc": val_acc,
    }


def main():
    print("=" * 70)
    print("BENCHMARK ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load validation data
    print("Loading validation data...")
    val_paths, val_labels, _, num_classes = load_classification_data(VAL_DIR)
    print(f"  {len(val_paths)} images, {num_classes} classes")
    
    _, val_transform = get_transforms()
    val_dataset = SimpleDataset(val_paths, val_labels, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Benchmark each experiment
    results = []
    
    print("\n" + "-" * 70)
    print("Benchmarking models...")
    print("-" * 70)
    
    for exp in EXPERIMENTS:
        print(f"\n  {exp['name']}...")
        
        if not exp["model_path"].exists():
            print(f"    ✗ Model not found: {exp['model_path']}")
            results.append({
                "name": exp["name"],
                "status": "not_found",
            })
            continue
        
        metrics = benchmark_model(exp, val_loader, DEVICE)
        
        if metrics:
            print(f"    ✓ AUC: {metrics['auc']*100:.2f}%", end="")
            if metrics['val_acc']:
                print(f" | Val Acc: {metrics['val_acc']*100:.2f}%")
            else:
                print()
            
            results.append({
                "name": exp["name"],
                "status": "success",
                "auc": metrics["auc"],
                "val_acc": metrics["val_acc"],
                "is_baseline": exp.get("is_baseline", False),
            })
        else:
            print(f"    ✗ Benchmark failed")
            results.append({
                "name": exp["name"],
                "status": "failed",
            })
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<30} {'Val Acc':<12} {'AUC':<12} {'Status':<10}")
    print("-" * 70)
    
    for r in results:
        name = r["name"][:30]
        if r["status"] == "success":
            val_acc = f"{r['val_acc']*100:.2f}%" if r['val_acc'] else "N/A"
            auc = f"{r['auc']*100:.2f}%"
            status = "✓"
        else:
            val_acc = "N/A"
            auc = "N/A"
            status = "✗"
        
        print(f"{name:<30} {val_acc:<12} {auc:<12} {status:<10}")
    
    # Find best model
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        best_by_auc = max(successful, key=lambda x: x["auc"])
        print("\n" + "-" * 70)
        print(f"Best by AUC: {best_by_auc['name']} ({best_by_auc['auc']*100:.2f}%)")
        
        # Compare with baseline
        baseline_auc = 0.8836  # CNN baseline
        improvement = (best_by_auc["auc"] - baseline_auc) * 100
        print(f"Improvement over CNN baseline: {improvement:+.2f}%")
    
    # Save results
    results_path = PROJECT_ROOT / "experiments" / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
