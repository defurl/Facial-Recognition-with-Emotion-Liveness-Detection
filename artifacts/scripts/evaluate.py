"""Robust evaluation script for verification pairs.

This script is a drop-in replacement for older `scripts/evaluate.py` but handles
large pair lists by using a DataLoader and batching, computes similarity scores
correctly for both Euclidean and Cosine metrics, and saves ROC plots + JSON.

Usage:
    python artifacts/scripts/evaluate.py

Optional args:
    --softmax-model PATH
    --metric-model PATH
    --pairs PATH
    --batch-size N
    --metric euclidean|cosine (similarity metric to evaluate)
    --device auto|cpu|cuda
    --output-dir artifacts/outputs

"""
from __future__ import annotations

import os

# fix duplicate OpenMP runtime on Windows (libiomp5md.dll)
# set before importing libraries that load OpenMP (e.g., torch, cv2)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

# Add src to path via relative import assumptions
ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ARTIFACTS_ROOT.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config import DEVICE as CONFIG_DEVICE, IMG_SIZE, OUTPUT_DIR  # type: ignore
from data_loader import get_transforms, load_verification_pairs  # type: ignore
from models import FaceEmbeddingCNN  # type: ignore


class VerificationPairsDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, int]], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, int(label)


def compute_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_mode: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for b1, b2, lbl in loader:
            b1 = b1.to(device)
            b2 = b2.to(device)

            out1 = model(b1, mode=model_mode)
            out2 = model(b2, mode=model_mode)

            if metric == 'euclidean':
                # distance, lower=more similar -> convert to similarity by negation
                dist = F.pairwise_distance(out1, out2, p=2)
                sim = -dist
            elif metric == 'cosine':
                cos = F.cosine_similarity(out1, out2)
                sim = cos
            else:
                raise ValueError('Unsupported metric: ' + str(metric))

            scores.extend(sim.cpu().numpy().tolist())
            labels.extend(lbl.numpy().tolist())

    return np.array(scores), np.array(labels)


def evaluate_and_plot(results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['labels'], res['scores'])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle=':')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / 'roc_comparison_fixed.png'
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close()

    json_path = out_dir / 'roc_results_fixed.json'
    serializable = {
        name: {
            'auc': float(auc(roc_curve(res['labels'], res['scores'])[0], roc_curve(res['labels'], res['scores'])[1]))
        }
        for name, res in results.items()
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved ROC plot to: {plot_path}")
    print(f"Saved ROC summary to: {json_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--softmax-model', type=Path, default=ARTIFACTS_ROOT / 'outputs' / 'best_softmax_model.pth')
    p.add_argument('--metric-model', type=Path, default=ARTIFACTS_ROOT / 'outputs' / 'best_metric_model.pth')
    p.add_argument('--pairs', type=Path, default=ARTIFACTS_ROOT / 'dataset' / 'verification_pairs_val.txt')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--out-dir', type=Path, default=ARTIFACTS_ROOT / 'outputs')
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if args.device == 'cuda' else 'cpu')

    print('Device:', device)

    pairs = load_verification_pairs(args.pairs)
    if len(pairs) == 0:
        print('No verification pairs found; check path:', args.pairs)
        return

    _, val_transform = get_transforms()
    dataset = VerificationPairsDataset(pairs, val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results = {}
    
    # Test both euclidean and cosine metrics
    for metric in ['euclidean', 'cosine']:
        print(f'\n--- Evaluating with {metric} metric ---')
        
        # Softmax model (use 'metric' mode for normalized embeddings)
        soft_path = args.softmax_model
        if soft_path.exists():
            print(f'Loading softmax model from {soft_path}')
            model_soft = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000).to(device)
            model_soft.load_state_dict(torch.load(soft_path, map_location=device))
            model_soft.eval()
            scores, labels = compute_scores(model_soft, loader, device, model_mode='metric', metric=metric)
            results[f'Softmax - {metric.capitalize()}'] = {'scores': scores, 'labels': labels}
        else:
            print('Softmax model not found at', soft_path)

        # Metric model (use 'metric' mode)
        metric_path = args.metric_model
        if metric_path.exists():
            print(f'Loading metric model from {metric_path}')
            model_metric = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000).to(device)
            model_metric.load_state_dict(torch.load(metric_path, map_location=device))
            model_metric.eval()
            scores, labels = compute_scores(model_metric, loader, device, model_mode='metric', metric=metric)
            results[f'Triplet - {metric.capitalize()}'] = {'scores': scores, 'labels': labels}
        else:
            print('Metric model not found at', metric_path)

    if not results:
        print('No models evaluated.')
        return

    evaluate_and_plot(results, args.out_dir)


if __name__ == '__main__':
    main()
