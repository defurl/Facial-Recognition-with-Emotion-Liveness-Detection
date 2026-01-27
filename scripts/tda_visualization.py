#!/usr/bin/env python3
"""
TDA Visualization Script

Visualizes topological features from CBAM attention maps to explain
"why" TDA regularization works (or doesn't work).

Outputs:
1. Attention map comparison (Baseline vs TDA model)
2. Persistence diagrams showing topological features
3. Betti curves comparison
4. Statistical analysis of topological properties
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

# Setup paths
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from models import FaceEmbeddingCNN
from config import VAL_DIR, OUTPUT_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Import TDA
try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import BettiCurve, PersistenceEntropy, NumberOfPoints
    from gtda.plotting import plot_diagram
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("Warning: giotto-tda not available")


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


def load_models(device: torch.device) -> Tuple[Optional[FaceEmbeddingCNN], Optional[FaceEmbeddingCNN]]:
    """Load baseline and TDA models."""
    baseline_path = OUTPUT_DIR / "best_metric_model.pth"
    tda_path = OUTPUT_DIR / "best_tda_metric_model.pth"
    
    baseline_model = None
    tda_model = None
    
    if baseline_path.exists():
        print(f"Loading baseline model from {baseline_path}")
        baseline_model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000, use_cbam=True)
        state_dict = torch.load(baseline_path, map_location=device, weights_only=True)
        
        # Check if conversion is needed (old backbone.* format)
        if any(k.startswith('backbone.') for k in state_dict.keys()):
            print("  Converting old model format to new format...")
            state_dict = convert_old_state_dict(state_dict)
        
        baseline_model.load_state_dict(state_dict)
        baseline_model.to(device)
        baseline_model.eval()
    else:
        print(f"Warning: Baseline model not found at {baseline_path}")
    
    if tda_path.exists():
        print(f"Loading TDA model from {tda_path}")
        tda_model = FaceEmbeddingCNN(embedding_dim=256, num_classes=4000, use_cbam=True)
        state_dict = torch.load(tda_path, map_location=device, weights_only=True)
        
        # Check if conversion is needed
        if any(k.startswith('backbone.') for k in state_dict.keys()):
            print("  Converting old model format to new format...")
            state_dict = convert_old_state_dict(state_dict)
        
        tda_model.load_state_dict(state_dict)
        tda_model.to(device)
        tda_model.eval()
    else:
        print(f"Warning: TDA model not found at {tda_path}")
    
    return baseline_model, tda_model


def get_sample_images(val_dir: Path, n_samples: int = 16) -> List[Path]:
    """Get sample images from validation set."""
    identity_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    
    images = []
    for identity_dir in identity_dirs[:n_samples]:
        img_files = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if img_files:
            images.append(img_files[0])
    
    return images


def extract_attention_maps(
    model: FaceEmbeddingCNN, 
    images: List[Path], 
    device: torch.device,
    transform: transforms.Compose
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract attention maps from model for given images."""
    attention_maps = []
    original_images = []
    
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, attention = model(img_tensor, mode='metric', return_attention=True)
        
        attention_maps.append(attention.cpu())
        original_images.append(img_tensor.cpu())
    
    return torch.cat(attention_maps, dim=0), torch.cat(original_images, dim=0)


def compute_persistence_features(attention_maps: np.ndarray) -> Dict:
    """Compute persistence diagrams and features."""
    if not TDA_AVAILABLE:
        return {}
    
    # Ensure shape is (B, H, W)
    if attention_maps.ndim == 4:
        attention_maps = attention_maps.squeeze(1)
    
    # Compute persistence
    cubical = CubicalPersistence(homology_dimensions=[0, 1], infinity_values=1.0)
    diagrams = cubical.fit_transform(attention_maps)
    
    # Extract features
    entropy_extractor = PersistenceEntropy()
    num_points_extractor = NumberOfPoints()
    betti_curve_extractor = BettiCurve(n_bins=20)
    
    entropy = entropy_extractor.fit_transform(diagrams)
    num_points = num_points_extractor.fit_transform(diagrams)
    betti_curves = betti_curve_extractor.fit_transform(diagrams)
    
    return {
        'diagrams': diagrams,
        'entropy': entropy,
        'num_points': num_points,
        'betti_curves': betti_curves
    }


def plot_attention_comparison(
    baseline_att: np.ndarray,
    tda_att: np.ndarray,
    original_images: np.ndarray,
    save_path: Path,
    n_samples: int = 8
):
    """Plot attention maps comparison between baseline and TDA models."""
    fig = plt.figure(figsize=(20, 3 * n_samples))
    gs = gridspec.GridSpec(n_samples, 4, figure=fig, hspace=0.3, wspace=0.1)
    
    for i in range(min(n_samples, len(baseline_att))):
        # Original image
        ax1 = fig.add_subplot(gs[i, 0])
        img = original_images[i].transpose(1, 2, 0)
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = np.clip(img, 0, 1)
        ax1.imshow(img)
        ax1.set_title(f"Sample {i+1}" if i == 0 else "")
        ax1.axis('off')
        if i == 0:
            ax1.set_ylabel("Original", fontsize=12)
        
        # Baseline attention
        ax2 = fig.add_subplot(gs[i, 1])
        att_baseline = baseline_att[i].squeeze()
        im2 = ax2.imshow(att_baseline, cmap='hot', vmin=0, vmax=1)
        ax2.set_title(f"Baseline Attention" if i == 0 else "")
        ax2.axis('off')
        
        # TDA attention
        ax3 = fig.add_subplot(gs[i, 2])
        att_tda = tda_att[i].squeeze()
        im3 = ax3.imshow(att_tda, cmap='hot', vmin=0, vmax=1)
        ax3.set_title(f"TDA Attention" if i == 0 else "")
        ax3.axis('off')
        
        # Difference
        ax4 = fig.add_subplot(gs[i, 3])
        diff = att_tda - att_baseline
        im4 = ax4.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax4.set_title(f"Difference (TDA - Baseline)" if i == 0 else "")
        ax4.axis('off')
    
    # Add colorbars
    fig.colorbar(im2, ax=fig.axes[1::4], shrink=0.6, label='Attention')
    fig.colorbar(im4, ax=fig.axes[3::4], shrink=0.6, label='Difference')
    
    plt.suptitle("Attention Map Comparison: Baseline vs TDA-Regularized Model", fontsize=16, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention comparison to {save_path}")


def plot_persistence_diagrams(
    baseline_features: Dict,
    tda_features: Dict,
    save_path: Path,
    n_samples: int = 4
):
    """Plot persistence diagrams for both models."""
    if not TDA_AVAILABLE:
        print("Skipping persistence diagrams - giotto-tda not available")
        return
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    for i in range(n_samples):
        # Baseline H0 (connected components)
        ax = axes[i, 0]
        diagram = baseline_features['diagrams'][i]
        h0_points = diagram[diagram[:, 2] == 0][:, :2]
        if len(h0_points) > 0:
            ax.scatter(h0_points[:, 0], h0_points[:, 1], c='blue', alpha=0.6, s=30)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title(f"Baseline H0" if i == 0 else "")
        ax.set_aspect('equal')
        
        # TDA H0
        ax = axes[i, 1]
        diagram = tda_features['diagrams'][i]
        h0_points = diagram[diagram[:, 2] == 0][:, :2]
        if len(h0_points) > 0:
            ax.scatter(h0_points[:, 0], h0_points[:, 1], c='red', alpha=0.6, s=30)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Birth')
        ax.set_title(f"TDA H0" if i == 0 else "")
        ax.set_aspect('equal')
        
        # Baseline H1 (holes)
        ax = axes[i, 2]
        diagram = baseline_features['diagrams'][i]
        h1_points = diagram[diagram[:, 2] == 1][:, :2]
        if len(h1_points) > 0:
            ax.scatter(h1_points[:, 0], h1_points[:, 1], c='blue', alpha=0.6, s=30, marker='^')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Birth')
        ax.set_title(f"Baseline H1" if i == 0 else "")
        ax.set_aspect('equal')
        
        # TDA H1
        ax = axes[i, 3]
        diagram = tda_features['diagrams'][i]
        h1_points = diagram[diagram[:, 2] == 1][:, :2]
        if len(h1_points) > 0:
            ax.scatter(h1_points[:, 0], h1_points[:, 1], c='red', alpha=0.6, s=30, marker='^')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Birth')
        ax.set_title(f"TDA H1" if i == 0 else "")
        ax.set_aspect('equal')
    
    plt.suptitle("Persistence Diagrams: H0 (Connected Components) and H1 (Holes)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved persistence diagrams to {save_path}")


def plot_topological_statistics(
    baseline_features: Dict,
    tda_features: Dict,
    save_path: Path
):
    """Plot statistical comparison of topological features."""
    if not TDA_AVAILABLE:
        print("Skipping topological statistics - giotto-tda not available")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Entropy comparison
    ax = axes[0, 0]
    data = {
        'Baseline H0': baseline_features['entropy'][:, 0],
        'TDA H0': tda_features['entropy'][:, 0],
        'Baseline H1': baseline_features['entropy'][:, 1],
        'TDA H1': tda_features['entropy'][:, 1]
    }
    positions = [1, 2, 4, 5]
    colors = ['steelblue', 'coral', 'steelblue', 'coral']
    bp = ax.boxplot([data[k] for k in data.keys()], positions=positions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['H0 Entropy', 'H1 Entropy'])
    ax.set_ylabel('Persistence Entropy')
    ax.set_title('Persistence Entropy Distribution')
    ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Baseline', 'TDA'], loc='upper right')
    
    # Number of topological features
    ax = axes[0, 1]
    data = {
        'Baseline H0': baseline_features['num_points'][:, 0],
        'TDA H0': tda_features['num_points'][:, 0],
        'Baseline H1': baseline_features['num_points'][:, 1],
        'TDA H1': tda_features['num_points'][:, 1]
    }
    bp = ax.boxplot([data[k] for k in data.keys()], positions=positions, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['H0 Features', 'H1 Features'])
    ax.set_ylabel('Number of Features')
    ax.set_title('Topological Feature Count')
    
    # Mean Betti curves
    ax = axes[0, 2]
    n_bins = baseline_features['betti_curves'].shape[1] // 2
    baseline_betti_h0 = baseline_features['betti_curves'][:, :n_bins].mean(axis=0)
    tda_betti_h0 = tda_features['betti_curves'][:, :n_bins].mean(axis=0)
    x = np.linspace(0, 1, n_bins)
    ax.plot(x, baseline_betti_h0, 'b-', linewidth=2, label='Baseline H0')
    ax.plot(x, tda_betti_h0, 'r-', linewidth=2, label='TDA H0')
    ax.set_xlabel('Filtration Value')
    ax.set_ylabel('Betti Number')
    ax.set_title('Mean Betti Curves (H0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Attention statistics
    ax = axes[1, 0]
    # Will be filled later with actual attention values
    ax.text(0.5, 0.5, 'Attention Statistics\n(See detailed analysis)', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Attention Map Statistics')
    
    # Summary metrics
    ax = axes[1, 1]
    metrics = [
        ('H0 Entropy', baseline_features['entropy'][:, 0].mean(), tda_features['entropy'][:, 0].mean()),
        ('H1 Entropy', baseline_features['entropy'][:, 1].mean(), tda_features['entropy'][:, 1].mean()),
        ('H0 Features', baseline_features['num_points'][:, 0].mean(), tda_features['num_points'][:, 0].mean()),
        ('H1 Features', baseline_features['num_points'][:, 1].mean(), tda_features['num_points'][:, 1].mean()),
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    baseline_vals = [m[1] for m in metrics]
    tda_vals = [m[2] for m in metrics]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue')
    ax.bar(x + width/2, tda_vals, width, label='TDA', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=45, ha='right')
    ax.set_ylabel('Mean Value')
    ax.set_title('Summary Metrics Comparison')
    ax.legend()
    
    # Improvement analysis
    ax = axes[1, 2]
    improvements = [(tda_vals[i] - baseline_vals[i]) / (baseline_vals[i] + 1e-8) * 100 
                    for i in range(len(metrics))]
    colors = ['green' if imp < 0 else 'red' for imp in improvements]
    ax.barh(range(len(metrics)), improvements, color=colors, alpha=0.7)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlabel('% Change (negative = improvement)')
    ax.set_title('TDA vs Baseline Change')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle("Topological Feature Statistics: Baseline vs TDA Model", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved topological statistics to {save_path}")


def analyze_tda_effectiveness(
    baseline_att: np.ndarray,
    tda_att: np.ndarray,
    baseline_features: Dict,
    tda_features: Dict
) -> Dict:
    """Analyze why TDA might not be improving results."""
    analysis = {
        'attention_stats': {},
        'topology_stats': {},
        'issues': [],
        'recommendations': []
    }
    
    # Attention map statistics
    baseline_mean = baseline_att.mean()
    baseline_std = baseline_att.std()
    baseline_sparsity = (baseline_att < 0.5).mean()
    
    tda_mean = tda_att.mean()
    tda_std = tda_att.std()
    tda_sparsity = (tda_att < 0.5).mean()
    
    analysis['attention_stats'] = {
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(baseline_std),
        'baseline_sparsity': float(baseline_sparsity),
        'tda_mean': float(tda_mean),
        'tda_std': float(tda_std),
        'tda_sparsity': float(tda_sparsity),
        'mean_diff': float(tda_mean - baseline_mean),
        'std_diff': float(tda_std - baseline_std),
    }
    
    # Topology statistics
    if TDA_AVAILABLE and baseline_features and tda_features:
        analysis['topology_stats'] = {
            'baseline_h0_entropy': float(baseline_features['entropy'][:, 0].mean()),
            'baseline_h1_entropy': float(baseline_features['entropy'][:, 1].mean()),
            'tda_h0_entropy': float(tda_features['entropy'][:, 0].mean()),
            'tda_h1_entropy': float(tda_features['entropy'][:, 1].mean()),
            'baseline_h0_count': float(baseline_features['num_points'][:, 0].mean()),
            'baseline_h1_count': float(baseline_features['num_points'][:, 1].mean()),
            'tda_h0_count': float(tda_features['num_points'][:, 0].mean()),
            'tda_h1_count': float(tda_features['num_points'][:, 1].mean()),
        }
        
        # Detect issues
        h0_entropy_change = analysis['topology_stats']['tda_h0_entropy'] - analysis['topology_stats']['baseline_h0_entropy']
        h0_count_change = analysis['topology_stats']['tda_h0_count'] - analysis['topology_stats']['baseline_h0_count']
        
        if abs(h0_entropy_change) < 0.1:
            analysis['issues'].append(
                "TDA regularization had minimal effect on H0 entropy "
                f"(change: {h0_entropy_change:.3f}). The loss weight may be too small."
            )
            analysis['recommendations'].append(
                "Try increasing tda_loss_weight from 0.05 to 0.1-0.2"
            )
        
        if abs(h0_count_change) < 1:
            analysis['issues'].append(
                "Number of connected components barely changed. "
                "TDA is not significantly simplifying attention structure."
            )
            analysis['recommendations'].append(
                "Consider using a stronger complexity_weight (0.2-0.5)"
            )
        
        # Check if attention maps are already well-structured
        if baseline_features['entropy'][:, 0].mean() < 0.3:
            analysis['issues'].append(
                "Baseline attention already has low entropy (well-focused). "
                "TDA may not provide additional benefit."
            )
            analysis['recommendations'].append(
                "For already well-focused attention, TDA offers diminishing returns. "
                "Consider TDA for noisier or more complex architectures."
            )
    
    # Check attention map differences
    att_diff = np.abs(tda_att - baseline_att).mean()
    if att_diff < 0.05:
        analysis['issues'].append(
            f"Attention maps are nearly identical (mean diff: {att_diff:.4f}). "
            "TDA regularization is not affecting attention learning."
        )
        analysis['recommendations'].append(
            "The TDA loss might be too weak relative to triplet loss. "
            "Or the model architecture already produces optimal attention."
        )
    
    return analysis


def main():
    """Main visualization and analysis pipeline."""
    print("=" * 70)
    print("TDA Visualization and Analysis")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create analysis output directory
    analysis_dir = OUTPUT_DIR / "tda_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Load models
    baseline_model, tda_model = load_models(device)
    
    if baseline_model is None or tda_model is None:
        print("Error: Both models are required for comparison")
        return
    
    # Get sample images
    print(f"\nLoading sample images from {VAL_DIR}")
    sample_images = get_sample_images(VAL_DIR, n_samples=32)
    print(f"Found {len(sample_images)} sample images")
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Extract attention maps
    print("\nExtracting attention maps...")
    baseline_att, orig_imgs = extract_attention_maps(baseline_model, sample_images, device, transform)
    tda_att, _ = extract_attention_maps(tda_model, sample_images, device, transform)
    
    baseline_att_np = baseline_att.numpy()
    tda_att_np = tda_att.numpy()
    orig_imgs_np = orig_imgs.numpy()
    
    print(f"Attention map shape: {baseline_att_np.shape}")
    
    # Compute persistence features
    print("\nComputing topological features...")
    baseline_features = compute_persistence_features(baseline_att_np)
    tda_features = compute_persistence_features(tda_att_np)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Attention comparison
    plot_attention_comparison(
        baseline_att_np, tda_att_np, orig_imgs_np,
        analysis_dir / "attention_comparison.png",
        n_samples=8
    )
    
    # 2. Persistence diagrams
    plot_persistence_diagrams(
        baseline_features, tda_features,
        analysis_dir / "persistence_diagrams.png",
        n_samples=4
    )
    
    # 3. Topological statistics
    plot_topological_statistics(
        baseline_features, tda_features,
        analysis_dir / "topological_statistics.png"
    )
    
    # 4. Effectiveness analysis
    print("\nAnalyzing TDA effectiveness...")
    analysis = analyze_tda_effectiveness(
        baseline_att_np, tda_att_np,
        baseline_features, tda_features
    )
    
    # Save analysis
    with open(analysis_dir / "tda_effectiveness_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print analysis summary
    print("\n" + "=" * 70)
    print("TDA EFFECTIVENESS ANALYSIS")
    print("=" * 70)
    
    print("\nAttention Statistics:")
    for key, val in analysis['attention_stats'].items():
        print(f"  {key}: {val:.4f}")
    
    if analysis['topology_stats']:
        print("\nTopology Statistics:")
        for key, val in analysis['topology_stats'].items():
            print(f"  {key}: {val:.4f}")
    
    if analysis['issues']:
        print("\n⚠️  IDENTIFIED ISSUES:")
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"  {i}. {issue}")
    
    if analysis['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {analysis_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
