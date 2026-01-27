#!/usr/bin/env python3
"""
Model Benchmark Comparison Script

Comprehensive comparison of Baseline vs TDA-regularized models:
1. Verification accuracy on validation pairs
2. Embedding quality analysis
3. Training efficiency comparison
4. ROC curve analysis
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from models import FaceEmbeddingCNN, count_parameters
from config import VAL_DIR, OUTPUT_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def convert_old_state_dict(old_state_dict: dict) -> dict:
    """Convert old backbone-style state dict to new block-style."""
    # Mapping from old backbone indices to new block names
    # Need to be careful with the order - longer prefixes first to avoid partial matches
    mapping = [
        ('backbone.18.', 'block4_cbam.'),
        ('backbone.16.', 'block4_conv.1.'),
        ('backbone.15.', 'block4_conv.0.'),
        ('backbone.13.', 'block3_cbam.'),
        ('backbone.11.', 'block3_conv.1.'),
        ('backbone.10.', 'block3_conv.0.'),
        ('backbone.8.', 'block2_cbam.'),
        ('backbone.6.', 'block2_conv.1.'),
        ('backbone.5.', 'block2_conv.0.'),
        ('backbone.3.', 'block1_cbam.'),
        ('backbone.1.', 'block1_conv.1.'),
        ('backbone.0.', 'block1_conv.0.'),
    ]
    
    new_state_dict = {}
    for old_key, value in old_state_dict.items():
        new_key = old_key
        for old_prefix, new_prefix in mapping:
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix):]
                break
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_model(model_path: Path, device: torch.device) -> Optional[FaceEmbeddingCNN]:
    """Load a model from checkpoint."""
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        return None
    
    model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000, use_cbam=True)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Check if conversion is needed (old backbone.* format)
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        print(f"  Converting old model format to new format...")
        state_dict = convert_old_state_dict(state_dict)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_verification_pairs(val_dir: Path) -> List[Tuple[Path, Path, int]]:
    """
    Create verification pairs from validation directory.
    Returns list of (img1_path, img2_path, label) where label=1 for same identity.
    """
    identity_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    
    pairs = []
    
    # Positive pairs (same identity)
    for identity_dir in identity_dirs:
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if len(images) >= 2:
            # Take first 2 images as positive pair
            pairs.append((images[0], images[1], 1))
    
    # Negative pairs (different identities)
    for i in range(len(identity_dirs) - 1):
        imgs1 = list(identity_dirs[i].glob("*.jpg")) + list(identity_dirs[i].glob("*.png"))
        imgs2 = list(identity_dirs[i + 1].glob("*.jpg")) + list(identity_dirs[i + 1].glob("*.png"))
        if imgs1 and imgs2:
            pairs.append((imgs1[0], imgs2[0], 0))
    
    return pairs


def compute_embeddings(
    model: FaceEmbeddingCNN,
    image_paths: List[Path],
    device: torch.device,
    transform: transforms.Compose,
    batch_size: int = 64
) -> torch.Tensor:
    """Compute embeddings for a list of images."""
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            batch_embeddings = model(batch_tensor, mode='metric')
        
        embeddings.append(batch_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)


def compute_verification_metrics(
    model: FaceEmbeddingCNN,
    pairs: List[Tuple[Path, Path, int]],
    device: torch.device,
    transform: transforms.Compose
) -> Dict:
    """Compute verification metrics for a model."""
    # Get all unique images
    all_images = set()
    for img1, img2, _ in pairs:
        all_images.add(img1)
        all_images.add(img2)
    all_images = list(all_images)
    
    # Compute embeddings
    print(f"  Computing embeddings for {len(all_images)} images...")
    embeddings = compute_embeddings(model, all_images, device, transform)
    
    # Create lookup
    img_to_idx = {img: i for i, img in enumerate(all_images)}
    
    # Compute similarities
    similarities = []
    labels = []
    
    for img1, img2, label in pairs:
        emb1 = embeddings[img_to_idx[img1]]
        emb2 = embeddings[img_to_idx[img2]]
        
        # Cosine similarity
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        similarities.append(sim)
        labels.append(label)
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # ROC analysis
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute accuracy at optimal threshold
    predictions = (similarities >= optimal_threshold).astype(int)
    accuracy = (predictions == labels).mean()
    
    # TPR at specific FPR values
    tpr_at_fpr_001 = np.interp(0.01, fpr, tpr)
    tpr_at_fpr_01 = np.interp(0.1, fpr, tpr)
    
    # Precision-Recall
    precision, recall, _ = precision_recall_curve(labels, similarities)
    ap = average_precision_score(labels, similarities)
    
    return {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'tpr_at_fpr_001': float(tpr_at_fpr_001),
        'tpr_at_fpr_01': float(tpr_at_fpr_01),
        'average_precision': float(ap),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'similarities': similarities.tolist(),
        'labels': labels.tolist()
    }


def analyze_embedding_quality(
    model: FaceEmbeddingCNN,
    val_dir: Path,
    device: torch.device,
    transform: transforms.Compose,
    n_identities: int = 50
) -> Dict:
    """Analyze embedding quality: intra-class vs inter-class distances."""
    identity_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])[:n_identities]
    
    identity_embeddings = {}
    
    print(f"  Computing embeddings for {n_identities} identities...")
    for identity_dir in tqdm(identity_dirs, desc="  Identities"):
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))[:5]
        if len(images) >= 2:
            embeddings = compute_embeddings(model, images, device, transform)
            identity_embeddings[identity_dir.name] = embeddings
    
    # Compute intra-class distances
    intra_distances = []
    for identity, embeddings in identity_embeddings.items():
        if len(embeddings) >= 2:
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = 1 - F.cosine_similarity(
                        embeddings[i].unsqueeze(0), 
                        embeddings[j].unsqueeze(0)
                    ).item()
                    intra_distances.append(dist)
    
    # Compute inter-class distances (sample)
    inter_distances = []
    identities = list(identity_embeddings.keys())
    for i in range(min(100, len(identities))):
        for j in range(i + 1, min(100, len(identities))):
            emb1 = identity_embeddings[identities[i]][0]
            emb2 = identity_embeddings[identities[j]][0]
            dist = 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            inter_distances.append(dist)
    
    return {
        'intra_class_mean': float(np.mean(intra_distances)),
        'intra_class_std': float(np.std(intra_distances)),
        'inter_class_mean': float(np.mean(inter_distances)),
        'inter_class_std': float(np.std(inter_distances)),
        'separation_ratio': float(np.mean(inter_distances) / (np.mean(intra_distances) + 1e-8)),
        'intra_distances': intra_distances,
        'inter_distances': inter_distances
    }


def load_training_history(history_path: Path) -> Optional[List[Dict]]:
    """Load training history from JSON file."""
    if not history_path.exists():
        return None
    with open(history_path, 'r') as f:
        return json.load(f)


def plot_roc_comparison(baseline_metrics: Dict, tda_metrics: Dict, save_path: Path):
    """Plot ROC curve comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax = axes[0]
    ax.plot(baseline_metrics['fpr'], baseline_metrics['tpr'], 
            'b-', linewidth=2, label=f"Baseline (AUC={baseline_metrics['roc_auc']:.4f})")
    ax.plot(tda_metrics['fpr'], tda_metrics['tpr'], 
            'r-', linewidth=2, label=f"TDA (AUC={tda_metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    # Precision-Recall Curve
    ax = axes[1]
    ax.plot(baseline_metrics['recall'], baseline_metrics['precision'], 
            'b-', linewidth=2, label=f"Baseline (AP={baseline_metrics['average_precision']:.4f})")
    ax.plot(tda_metrics['recall'], tda_metrics['precision'], 
            'r-', linewidth=2, label=f"TDA (AP={tda_metrics['average_precision']:.4f})")
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC comparison to {save_path}")


def plot_embedding_analysis(baseline_quality: Dict, tda_quality: Dict, save_path: Path):
    """Plot embedding quality analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Distance distributions
    ax = axes[0]
    ax.hist(baseline_quality['intra_distances'], bins=50, alpha=0.5, 
            label='Baseline Intra', color='blue', density=True)
    ax.hist(baseline_quality['inter_distances'], bins=50, alpha=0.5, 
            label='Baseline Inter', color='lightblue', density=True)
    ax.hist(tda_quality['intra_distances'], bins=50, alpha=0.5, 
            label='TDA Intra', color='red', density=True)
    ax.hist(tda_quality['inter_distances'], bins=50, alpha=0.5, 
            label='TDA Inter', color='lightsalmon', density=True)
    ax.set_xlabel('Cosine Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Embedding Distance Distributions', fontsize=14)
    ax.legend(fontsize=9)
    
    # Separation metrics
    ax = axes[1]
    metrics = ['Intra Mean', 'Inter Mean', 'Separation Ratio']
    baseline_vals = [
        baseline_quality['intra_class_mean'],
        baseline_quality['inter_class_mean'],
        baseline_quality['separation_ratio']
    ]
    tda_vals = [
        tda_quality['intra_class_mean'],
        tda_quality['inter_class_mean'],
        tda_quality['separation_ratio']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue')
    ax.bar(x + width/2, tda_vals, width, label='TDA', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Embedding Quality Metrics', fontsize=14)
    ax.legend()
    
    # Box plot comparison
    ax = axes[2]
    data = [
        baseline_quality['intra_distances'],
        tda_quality['intra_distances'],
        baseline_quality['inter_distances'],
        tda_quality['inter_distances']
    ]
    positions = [1, 2, 4, 5]
    colors = ['steelblue', 'coral', 'steelblue', 'coral']
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['Intra-Class', 'Inter-Class'])
    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_title('Distance Distribution Comparison', fontsize=14)
    ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Baseline', 'TDA'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved embedding analysis to {save_path}")


def plot_training_comparison(baseline_history: List[Dict], tda_history: List[Dict], save_path: Path):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    baseline_epochs = [h['epoch'] for h in baseline_history]
    tda_epochs = [h['epoch'] for h in tda_history]
    
    # Validation Loss
    ax = axes[0, 0]
    ax.plot(baseline_epochs, [h['val_loss'] for h in baseline_history], 
            'b-', linewidth=2, label='Baseline')
    ax.plot(tda_epochs, [h['val_loss'] for h in tda_history], 
            'r-', linewidth=2, label='TDA')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[0, 1]
    ax.plot(baseline_epochs, [h['val_acc'] * 100 for h in baseline_history], 
            'b-', linewidth=2, label='Baseline')
    ax.plot(tda_epochs, [h['val_acc'] * 100 for h in tda_history], 
            'r-', linewidth=2, label='TDA')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training Loss
    ax = axes[1, 0]
    # Baseline uses 'train_loss' or 'train_triplet_loss'
    baseline_train_loss = [h.get('train_triplet_loss', h.get('train_loss', 0)) for h in baseline_history]
    tda_train_loss = [h.get('train_triplet_loss', h.get('train_loss', 0)) for h in tda_history]
    ax.plot(baseline_epochs, baseline_train_loss, 'b-', linewidth=2, label='Baseline')
    ax.plot(tda_epochs, tda_train_loss, 'r-', linewidth=2, label='TDA')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Triplet Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Epoch Time
    ax = axes[1, 1]
    ax.plot(baseline_epochs, [h['time'] for h in baseline_history], 
            'b-', linewidth=2, label='Baseline')
    ax.plot(tda_epochs, [h['time'] for h in tda_history], 
            'r-', linewidth=2, label='TDA')
    baseline_avg_time = np.mean([h['time'] for h in baseline_history])
    tda_avg_time = np.mean([h['time'] for h in tda_history])
    ax.axhline(y=baseline_avg_time, color='b', linestyle='--', alpha=0.5, 
               label=f'Baseline Avg: {baseline_avg_time:.1f}s')
    ax.axhline(y=tda_avg_time, color='r', linestyle='--', alpha=0.5, 
               label=f'TDA Avg: {tda_avg_time:.1f}s')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time per Epoch', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training comparison to {save_path}")


def generate_summary_report(
    baseline_metrics: Dict,
    tda_metrics: Dict,
    baseline_quality: Dict,
    tda_quality: Dict,
    baseline_history: List[Dict],
    tda_history: List[Dict],
    save_path: Path
):
    """Generate a comprehensive summary report."""
    
    # Find best epochs
    baseline_best_epoch = min(baseline_history, key=lambda x: x['val_loss'])
    tda_best_epoch = min(tda_history, key=lambda x: x['val_loss'])
    
    report = {
        'summary': {
            'baseline_best_val_acc': baseline_best_epoch['val_acc'],
            'tda_best_val_acc': tda_best_epoch['val_acc'],
            'accuracy_diff': tda_best_epoch['val_acc'] - baseline_best_epoch['val_acc'],
            'baseline_best_epoch': baseline_best_epoch['epoch'],
            'tda_best_epoch': tda_best_epoch['epoch'],
            'baseline_total_epochs': len(baseline_history),
            'tda_total_epochs': len(tda_history),
        },
        'verification_metrics': {
            'baseline_accuracy': baseline_metrics['accuracy'],
            'tda_accuracy': tda_metrics['accuracy'],
            'baseline_roc_auc': baseline_metrics['roc_auc'],
            'tda_roc_auc': tda_metrics['roc_auc'],
            'baseline_tpr_at_fpr_001': baseline_metrics['tpr_at_fpr_001'],
            'tda_tpr_at_fpr_001': tda_metrics['tpr_at_fpr_001'],
        },
        'embedding_quality': {
            'baseline_separation_ratio': baseline_quality['separation_ratio'],
            'tda_separation_ratio': tda_quality['separation_ratio'],
            'baseline_intra_mean': baseline_quality['intra_class_mean'],
            'tda_intra_mean': tda_quality['intra_class_mean'],
        },
        'training_efficiency': {
            'baseline_avg_epoch_time': np.mean([h['time'] for h in baseline_history]),
            'tda_avg_epoch_time': np.mean([h['time'] for h in tda_history]),
            'tda_overhead_percent': (
                (np.mean([h['time'] for h in tda_history]) - 
                 np.mean([h['time'] for h in baseline_history])) / 
                np.mean([h['time'] for h in baseline_history]) * 100
            ),
        },
        'conclusion': {}
    }
    
    # Determine conclusion
    acc_diff = report['summary']['accuracy_diff'] * 100
    if acc_diff > 0.5:
        report['conclusion']['verdict'] = 'TDA_IMPROVED'
        report['conclusion']['explanation'] = f"TDA improved accuracy by {acc_diff:.2f}%"
    elif acc_diff < -0.5:
        report['conclusion']['verdict'] = 'TDA_DEGRADED'
        report['conclusion']['explanation'] = f"TDA decreased accuracy by {-acc_diff:.2f}%"
    else:
        report['conclusion']['verdict'] = 'TDA_NEUTRAL'
        report['conclusion']['explanation'] = f"TDA had minimal effect ({acc_diff:.2f}% change)"
    
    # Add recommendations
    recommendations = []
    if report['training_efficiency']['tda_overhead_percent'] > 20:
        recommendations.append("Consider reducing TDA computation frequency (increase tda_apply_every_n)")
    if abs(acc_diff) < 0.5:
        recommendations.append("Try increasing tda_loss_weight (0.1-0.2) for stronger regularization")
        recommendations.append("Experiment with different target_entropy values")
    report['conclusion']['recommendations'] = recommendations
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Main benchmark pipeline."""
    print("=" * 70)
    print("Model Benchmark Comparison: Baseline vs TDA")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create analysis directory
    analysis_dir = OUTPUT_DIR / "tda_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Load models
    print("\n1. Loading models...")
    baseline_model = load_model(OUTPUT_DIR / "best_metric_model.pth", device)
    tda_model = load_model(OUTPUT_DIR / "best_tda_metric_model.pth", device)
    
    if baseline_model is None or tda_model is None:
        print("Error: Both models required for comparison")
        return
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Load verification pairs
    print("\n2. Creating verification pairs...")
    pairs = load_verification_pairs(VAL_DIR)
    print(f"   Created {len(pairs)} verification pairs")
    print(f"   Positive pairs: {sum(1 for _, _, l in pairs if l == 1)}")
    print(f"   Negative pairs: {sum(1 for _, _, l in pairs if l == 0)}")
    
    # Compute verification metrics
    print("\n3. Computing verification metrics...")
    print("   Baseline model:")
    baseline_metrics = compute_verification_metrics(baseline_model, pairs, device, transform)
    print("   TDA model:")
    tda_metrics = compute_verification_metrics(tda_model, pairs, device, transform)
    
    # Analyze embedding quality
    print("\n4. Analyzing embedding quality...")
    print("   Baseline model:")
    baseline_quality = analyze_embedding_quality(baseline_model, VAL_DIR, device, transform)
    print("   TDA model:")
    tda_quality = analyze_embedding_quality(tda_model, VAL_DIR, device, transform)
    
    # Load training histories
    print("\n5. Loading training histories...")
    baseline_history = load_training_history(OUTPUT_DIR / "training_history.json")
    tda_history = load_training_history(OUTPUT_DIR / "tda_training_history.json")
    
    if baseline_history is None:
        print("   Warning: Baseline history not found")
        baseline_history = []
    if tda_history is None:
        print("   Warning: TDA history not found")
        tda_history = []
    
    # Generate plots
    print("\n6. Generating comparison plots...")
    
    # ROC comparison
    plot_roc_comparison(baseline_metrics, tda_metrics, 
                       analysis_dir / "roc_comparison.png")
    
    # Embedding analysis
    plot_embedding_analysis(baseline_quality, tda_quality,
                           analysis_dir / "embedding_analysis.png")
    
    # Training comparison
    if baseline_history and tda_history:
        plot_training_comparison(baseline_history, tda_history,
                                analysis_dir / "training_comparison.png")
    
    # Generate summary report
    print("\n7. Generating summary report...")
    report = generate_summary_report(
        baseline_metrics, tda_metrics,
        baseline_quality, tda_quality,
        baseline_history if baseline_history else [{'epoch': 0, 'val_loss': 0, 'val_acc': 0, 'time': 0}],
        tda_history if tda_history else [{'epoch': 0, 'val_loss': 0, 'val_acc': 0, 'time': 0}],
        analysis_dir / "benchmark_report.json"
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\n📊 Verification Performance:")
    print(f"   Baseline Accuracy: {baseline_metrics['accuracy']*100:.2f}%")
    print(f"   TDA Accuracy:      {tda_metrics['accuracy']*100:.2f}%")
    print(f"   Difference:        {(tda_metrics['accuracy'] - baseline_metrics['accuracy'])*100:+.2f}%")
    
    print(f"\n   Baseline ROC-AUC:  {baseline_metrics['roc_auc']:.4f}")
    print(f"   TDA ROC-AUC:       {tda_metrics['roc_auc']:.4f}")
    
    print(f"\n   Baseline TPR@FPR=0.01: {baseline_metrics['tpr_at_fpr_001']:.4f}")
    print(f"   TDA TPR@FPR=0.01:      {tda_metrics['tpr_at_fpr_001']:.4f}")
    
    print("\n📈 Embedding Quality:")
    print(f"   Baseline Separation Ratio: {baseline_quality['separation_ratio']:.4f}")
    print(f"   TDA Separation Ratio:      {tda_quality['separation_ratio']:.4f}")
    
    if baseline_history and tda_history:
        print("\n⏱️  Training Efficiency:")
        print(f"   Baseline Avg Time/Epoch: {report['training_efficiency']['baseline_avg_epoch_time']:.1f}s")
        print(f"   TDA Avg Time/Epoch:      {report['training_efficiency']['tda_avg_epoch_time']:.1f}s")
        print(f"   TDA Overhead:            {report['training_efficiency']['tda_overhead_percent']:.1f}%")
    
    print(f"\n🎯 VERDICT: {report['conclusion']['verdict']}")
    print(f"   {report['conclusion']['explanation']}")
    
    if report['conclusion']['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['conclusion']['recommendations']:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {analysis_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
