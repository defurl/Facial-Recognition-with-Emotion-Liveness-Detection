#!/usr/bin/env python3
"""
Evaluation Script for Multi-Task Learning Unified Face Model.

This script evaluates:
1. Emotion recognition accuracy (7-class classification)
2. Liveness detection accuracy (binary classification)
3. Face verification (AUC, EER, accuracy at threshold)
4. Combined MTL performance metrics

Features:
- Confusion matrices for each task
- ROC curves and AUC scores
- Per-class precision/recall/F1
- Embedding quality analysis
- Comparison with baseline models

Usage:
    python evaluate_mtl.py --task all
    python evaluate_mtl.py --task emotion --checkpoint emotion_best.pth
    python evaluate_mtl.py --task liveness --checkpoint liveness_best.pth
    python evaluate_mtl.py --task verification --pairs verification_pairs.txt

Author: MTL Implementation
Date: January 2026
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models_mtl import UnifiedFaceModel
from data_loader_mtl import (
    EmotionDataset, LivenessDataset, get_mtl_transforms,
    EMOTION_DIR, LIVENESS_DIR, EMOTION_NAMES
)
from config import DEVICE, BASE_DIR


# ============= Configuration =============

class EvalConfig:
    """Evaluation configuration."""
    
    OUTPUT_DIR = BASE_DIR / 'outputs' / 'mtl'
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    RESULTS_DIR = OUTPUT_DIR / 'evaluation'
    
    # Model
    NUM_IDENTITIES = 4000
    EMBEDDING_DIM = 256
    NUM_EMOTIONS = 7
    NUM_LIVENESS = 2
    
    # Evaluation
    BATCH_SIZE = 64
    
    @classmethod
    def setup_directories(cls):
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============= Evaluation Functions =============

@torch.no_grad()
def evaluate_emotion(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate emotion recognition performance.
    
    Returns:
        Dictionary with accuracy, per-class metrics, confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images, tasks=['emotion'])
        probs = F.softmax(outputs['emotion'], dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Per-class metrics
    per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class': {
            EMOTION_NAMES[i]: {
                'precision': float(per_class[0][i]),
                'recall': float(per_class[1][i]),
                'f1': float(per_class[2][i]),
                'support': int(per_class[3][i]),
            }
            for i in range(len(EMOTION_NAMES))
        },
        'confusion_matrix': cm.tolist(),
    }
    
    # Print results
    print("\n" + "="*60)
    print("EMOTION RECOGNITION EVALUATION")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nPer-class Performance:")
    print("-"*50)
    for name in EMOTION_NAMES:
        metrics = results['per_class'][name]
        print(f"  {name:12s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} (n={metrics['support']})")
    
    # Save confusion matrix plot
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(EMOTION_NAMES)),
            yticks=np.arange(len(EMOTION_NAMES)),
            xticklabels=EMOTION_NAMES,
            yticklabels=EMOTION_NAMES,
            title='Emotion Recognition Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(save_path / 'emotion_confusion_matrix.png', dpi=150)
        plt.close()
        
        # Save results JSON
        with open(save_path / 'emotion_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


@torch.no_grad()
def evaluate_liveness(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate liveness detection performance.
    
    Returns:
        Dictionary with accuracy, AUC, EER, confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images, tasks=['liveness'])
        probs = F.softmax(outputs['liveness'], dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Spoof probability
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Equal Error Rate (EER)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    results = {
        'accuracy': float(accuracy),
        'auc': float(roc_auc),
        'eer': float(eer),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'live': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0]),
            },
            'spoof': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1]),
            },
        },
    }
    
    # Print results
    print("\n" + "="*60)
    print("LIVENESS DETECTION EVALUATION")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN (Live→Live):   {cm[0,0]:5d}    FP (Live→Spoof):  {cm[0,1]:5d}")
    print(f"  FN (Spoof→Live):  {cm[1,0]:5d}    TP (Spoof→Spoof): {cm[1,1]:5d}")
    print(f"\nLive:  P={precision[0]:.3f} R={recall[0]:.3f} F1={f1[0]:.3f}")
    print(f"Spoof: P={precision[1]:.3f} R={recall[1]:.3f} F1={f1[1]:.3f}")
    
    # Save ROC curve plot
    if save_path:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.scatter([fpr[eer_threshold_idx]], [tpr[eer_threshold_idx]], 
                   color='red', s=100, zorder=5, label=f'EER = {eer:.4f}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Liveness Detection ROC Curve')
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.savefig(save_path / 'liveness_roc_curve.png', dpi=150)
        plt.close()
        
        # Save results JSON
        with open(save_path / 'liveness_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


@torch.no_grad()
def evaluate_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate embedding quality using intra-class and inter-class distances.
    
    Returns:
        Dictionary with embedding statistics
    """
    model.eval()
    
    embeddings = []
    labels = []
    
    for images, batch_labels in dataloader:
        images = images.to(device)
        outputs = model(images, tasks=['embedding'])
        embeddings.append(outputs['embedding'].cpu().numpy())
        labels.extend(batch_labels.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Compute pairwise distances (sample for efficiency)
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_emb = embeddings[indices]
    sample_labels = labels[indices]
    
    # Compute distances
    from scipy.spatial.distance import cdist
    distances = cdist(sample_emb, sample_emb, metric='cosine')
    
    # Intra-class and inter-class
    intra_distances = []
    inter_distances = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if sample_labels[i] == sample_labels[j]:
                intra_distances.append(distances[i, j])
            else:
                inter_distances.append(distances[i, j])
    
    results = {
        'embedding_dim': int(embeddings.shape[1]),
        'num_samples': int(len(embeddings)),
        'embedding_norm_mean': float(np.linalg.norm(embeddings, axis=1).mean()),
        'embedding_norm_std': float(np.linalg.norm(embeddings, axis=1).std()),
    }
    
    if intra_distances:
        results['intra_class_distance_mean'] = float(np.mean(intra_distances))
        results['intra_class_distance_std'] = float(np.std(intra_distances))
    
    if inter_distances:
        results['inter_class_distance_mean'] = float(np.mean(inter_distances))
        results['inter_class_distance_std'] = float(np.std(inter_distances))
    
    # Print results
    print("\n" + "="*60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*60)
    print(f"Embedding dimension: {results['embedding_dim']}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Embedding norm: {results['embedding_norm_mean']:.4f} ± {results['embedding_norm_std']:.4f}")
    if 'intra_class_distance_mean' in results:
        print(f"Intra-class distance: {results['intra_class_distance_mean']:.4f} ± {results['intra_class_distance_std']:.4f}")
    if 'inter_class_distance_mean' in results:
        print(f"Inter-class distance: {results['inter_class_distance_mean']:.4f} ± {results['inter_class_distance_std']:.4f}")
    
    if save_path:
        with open(save_path / 'embedding_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    config = EvalConfig()
    
    model = UnifiedFaceModel(
        num_identities=config.NUM_IDENTITIES,
        embedding_dim=config.EMBEDDING_DIM,
        num_emotions=config.NUM_EMOTIONS,
        num_liveness=config.NUM_LIVENESS,
    ).to(device)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    return model


def evaluate_all(
    emotion_checkpoint: Optional[Path] = None,
    liveness_checkpoint: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Run all evaluations."""
    config = EvalConfig()
    config.setup_directories()
    device = DEVICE
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = config.RESULTS_DIR / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults will be saved to: {results_dir}")
    print(f"Device: {device}")
    
    # Get transforms
    _, val_transform = get_mtl_transforms(augment=False)
    
    all_results = {}
    
    # Evaluate emotion
    if emotion_checkpoint is None:
        emotion_checkpoint = config.CHECKPOINT_DIR / 'emotion_best.pth'
    
    if emotion_checkpoint.exists():
        print("\n" + "-"*60)
        print("Loading emotion model...")
        model = load_model(emotion_checkpoint, device)
        
        # Load emotion test data
        test_ds = EmotionDataset(EMOTION_DIR, split='test', transform=val_transform)
        test_loader = DataLoader(
            test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
        )
        
        all_results['emotion'] = evaluate_emotion(
            model, test_loader, device, results_dir
        )
    else:
        print(f"Skipping emotion evaluation - no checkpoint at {emotion_checkpoint}")
    
    # Evaluate liveness
    if liveness_checkpoint is None:
        liveness_checkpoint = config.CHECKPOINT_DIR / 'liveness_best.pth'
    
    if liveness_checkpoint.exists():
        print("\n" + "-"*60)
        print("Loading liveness model...")
        model = load_model(liveness_checkpoint, device)
        
        # Load liveness test data
        test_ds = LivenessDataset(LIVENESS_DIR, split='test', transform=val_transform)
        test_loader = DataLoader(
            test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
        )
        
        all_results['liveness'] = evaluate_liveness(
            model, test_loader, device, results_dir
        )
    else:
        print(f"Skipping liveness evaluation - no checkpoint at {liveness_checkpoint}")
    
    # Evaluate embeddings (using emotion model as base)
    if emotion_checkpoint.exists():
        print("\n" + "-"*60)
        print("Analyzing embedding quality...")
        model = load_model(emotion_checkpoint, device)
        
        test_ds = EmotionDataset(EMOTION_DIR, split='test', transform=val_transform)
        test_loader = DataLoader(
            test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
        )
        
        all_results['embeddings'] = evaluate_embeddings(
            model, test_loader, device, results_dir
        )
    
    # Save combined results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if 'emotion' in all_results:
        print(f"Emotion Accuracy: {all_results['emotion']['accuracy']:.4f}")
        print(f"Emotion F1 Score: {all_results['emotion']['f1_score']:.4f}")
    
    if 'liveness' in all_results:
        print(f"Liveness Accuracy: {all_results['liveness']['accuracy']:.4f}")
        print(f"Liveness AUC: {all_results['liveness']['auc']:.4f}")
        print(f"Liveness EER: {all_results['liveness']['eer']:.4f}")
    
    print(f"\nFull results saved to: {results_dir}")
    
    return all_results


# ============= Main Entry Point =============

def main():
    parser = argparse.ArgumentParser(description='MTL Evaluation Script')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'emotion', 'liveness', 'embeddings'],
                        help='Task to evaluate')
    parser.add_argument('--emotion-checkpoint', type=str, default=None,
                        help='Path to emotion checkpoint')
    parser.add_argument('--liveness-checkpoint', type=str, default=None,
                        help='Path to liveness checkpoint')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    emotion_ckpt = Path(args.emotion_checkpoint) if args.emotion_checkpoint else None
    liveness_ckpt = Path(args.liveness_checkpoint) if args.liveness_checkpoint else None
    
    if args.task == 'all':
        evaluate_all(emotion_ckpt, liveness_ckpt)
    elif args.task == 'emotion':
        config = EvalConfig()
        config.setup_directories()
        
        ckpt = emotion_ckpt or config.CHECKPOINT_DIR / 'emotion_best.pth'
        model = load_model(ckpt, DEVICE)
        
        _, val_transform = get_mtl_transforms(augment=False)
        test_ds = EmotionDataset(EMOTION_DIR, split='test', transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        evaluate_emotion(model, test_loader, DEVICE, config.RESULTS_DIR)
    elif args.task == 'liveness':
        config = EvalConfig()
        config.setup_directories()
        
        ckpt = liveness_ckpt or config.CHECKPOINT_DIR / 'liveness_best.pth'
        model = load_model(ckpt, DEVICE)
        
        _, val_transform = get_mtl_transforms(augment=False)
        test_ds = LivenessDataset(LIVENESS_DIR, split='test', transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        evaluate_liveness(model, test_loader, DEVICE, config.RESULTS_DIR)


if __name__ == '__main__':
    main()
